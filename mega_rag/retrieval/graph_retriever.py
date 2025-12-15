"""
Graph Retriever using NetworkX for entity-based retrieval.
Builds a Neuro-Symbolic knowledge graph using Neural Entity Linking.
"""
from __future__ import annotations
import pickle
import re
import networkx as nx
import spacy
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, TYPE_CHECKING
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from mega_rag.config import (
    GRAPH_TOP_K,
    GRAPH_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    BASE_DIR,
    SPACY_MODEL
)

if TYPE_CHECKING:
    from mega_rag.core.document_processor import Document


class GraphRetriever:
    """
    Neuro-Symbolic Graph Retriever.
    
    Upgrade from Regex:
    1. Uses spaCy (en_core_web_sm) for robust entity extraction.
    2. Uses BGE-M3 embeddings to "link" entities to concepts semantically.
    """

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        graph_path: Optional[Path] = None
    ):
        self.graph_path = graph_path or (BASE_DIR / "knowledge_graph.pkl")
        self.graph = nx.Graph()
        self.entity_to_docs: Dict[str, Set[str]] = defaultdict(set)
        self.doc_contents: Dict[str, str] = {}
        self.doc_metadatas: Dict[str, dict] = {}

        # Load standard spaCy model for entity extraction
        print(f"Loading spaCy model: {SPACY_MODEL}...")
        try:
            self.nlp = spacy.load(SPACY_MODEL)
            # Add pipe for merging noun chunks to get better medical terms (e.g., "blood pressure")
            if "merge_noun_chunks" not in self.nlp.pipe_names:
                self.nlp.add_pipe("merge_noun_chunks")
        except OSError:
            print(f"Model {SPACY_MODEL} not found. please run: python -m spacy download {SPACY_MODEL}")
            raise

        # Embedding model for entity similarity (Neural Linking)
        print(f"Loading Neural Linking model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.entity_embeddings: Dict[str, np.ndarray] = {}

    def _extract_entities(self, text: str) -> Set[str]:
        """
        Extract entities using spaCy NER + Noun Chunks.
        Filters for likely medical candidates (NN, NNP).
        """
        doc = self.nlp(text)
        entities = set()
        
        # 1. Standard NER entities (Person, Org, GPE are usually noise in medical text, but we keep broad matching)
        for ent in doc.ents:
            if len(ent.text) > 3:
                entities.add(ent.text.lower())
                
        # 2. Noun Chunks (captures "high blood pressure", "diabetes mellitus")
        # This acts as our "candidate generation" step for medical terms
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            # Filter out short/trivial chunks
            if len(chunk_text) > 3 and not chunk_text.isnumeric():
                entities.add(chunk_text)

        return entities

    def add_documents(self, documents: List[Document]) -> None:
        """Build Neuro-Symbolic knowledge graph from documents."""
        print("Building Neural Knowledge Graph...")

        for doc in documents:
            doc_id = f"{doc.doc_id}_{doc.chunk_id}"
            self.doc_contents[doc_id] = doc.content
            self.doc_metadatas[doc_id] = doc.metadata

            # Extract entities using Neural extraction
            entities = self._extract_entities(doc.content)

            # Add document node
            self.graph.add_node(doc_id, type='document', content=doc.content[:500])

            # Add entity nodes and edges
            for entity in entities:
                if entity not in self.graph:
                    self.graph.add_node(entity, type='entity')

                # Link document to entity
                self.graph.add_edge(doc_id, entity, relation='contains')

                # Track entity to document mapping
                self.entity_to_docs[entity].add(doc_id)

        # Add entity-entity edges based on co-occurrence
        self._add_entity_relationships()

        # Compute entity embeddings for Semantic Linking
        self._compute_entity_embeddings()

        print(f"Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def _add_entity_relationships(self) -> None:
        """Add edges between entities that co-occur in documents."""
        entity_nodes = [n for n, d in self.graph.nodes(data=True)
                       if d.get('type') == 'entity']
        
        # Prune very rare entities to keep graph manageable
        pruned_entities = [e for e in entity_nodes if len(self.entity_to_docs[e]) > 1]
        
        print(f"Computing co-occurrences for {len(pruned_entities)} entities...")

        for i, e1 in enumerate(pruned_entities):
            docs1 = self.entity_to_docs[e1]
            for e2 in pruned_entities[i + 1:]:
                docs2 = self.entity_to_docs[e2]

                # If entities share documents, link them
                shared_docs = docs1 & docs2
                if shared_docs:
                    weight = len(shared_docs)
                    self.graph.add_edge(e1, e2, relation='co-occurs', weight=weight)

    def _compute_entity_embeddings(self) -> None:
        """Compute embeddings for all entities to enable Neural Linking."""
        entity_nodes = [n for n, d in self.graph.nodes(data=True)
                       if d.get('type') == 'entity']
        
        print(f"Embedding {len(entity_nodes)} entity nodes for Neural Linking...")

        if entity_nodes:
            embeddings = self.embedding_model.encode(
                entity_nodes,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            for entity, emb in zip(entity_nodes, embeddings):
                self.entity_embeddings[entity] = emb

    def _find_similar_entities(self, query: str, top_k: int = 5) -> List[str]:
        """
        Neural Entity Linking:
        Finds graph entities that are *semantically* similar to the query terms.
        This bridges the gap between different terminologies (e.g. "high bp" -> "hypertension").
        """
        if not self.entity_embeddings:
            return []

        # 1. Embed the query
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )[0]
        
        # 2. Extract specific query entities using spaCy
        query_doc = self.nlp(query)
        query_entities = {chunk.text.lower() for chunk in query_doc.noun_chunks}
        
        # Include full query as a candidate
        query_candidates = list(query_entities) + [query]
        candidate_embeddings = self.embedding_model.encode(query_candidates, normalize_embeddings=True)

        # 3. Find matches in the graph
        similar_entities = set()
        entity_list = list(self.entity_embeddings.keys())
        entity_embs = np.array([self.entity_embeddings[e] for e in entity_list])

        # Match entire query against graph
        similarities = cosine_similarity([query_embedding], entity_embs)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        for idx in top_indices:
            if similarities[idx] > GRAPH_SIMILARITY_THRESHOLD:
                similar_entities.add(entity_list[idx])
                
        # Also match extracted query chunks against graph
        # This handles compound queries like "treatment for diabetes" better
        for cand_emb in candidate_embeddings:
             chunk_sims = cosine_similarity([cand_emb], entity_embs)[0]
             chunk_indices = np.argsort(chunk_sims)[::-1][:3]
             for idx in chunk_indices:
                 if chunk_sims[idx] > GRAPH_SIMILARITY_THRESHOLD:
                     similar_entities.add(entity_list[idx])

        return list(similar_entities)[:top_k*2]

    def retrieve(
        self,
        query: str,
        top_k: int = GRAPH_TOP_K
    ) -> List[Tuple[str, float, dict]]:
        """
        Retrieve documents using Neuro-Symbolic graph traversal.
        
        Flow:
        1. Link query -> Graph Entities (Neural Linking)
        2. Expand to neighbors (Graph Traversal)
        3. Score documents based on centrality + relevance
        """
        if self.graph.number_of_nodes() == 0:
            return []

        # 1. Neural Linking
        relevant_entities = self._find_similar_entities(query)

        if not relevant_entities:
            return []

        # Collect documents connected to relevant entities
        doc_scores: Dict[str, float] = defaultdict(float)

        for entity in relevant_entities:
            if entity in self.entity_to_docs:
                # Score based on entity centrality (how important is this concept?)
                # Use cached or simple degree centrality for speed
                centrality = self.graph.degree(entity) 
                
                for doc_id in self.entity_to_docs[entity]:
                    # Boost score if document contains highly central entity
                    doc_scores[doc_id] += 1 + (centrality * 0.1)

        # Normalize scores
        if doc_scores:
            max_score = max(doc_scores.values())
            doc_scores = {k: v / max_score for k, v in doc_scores.items()}

        # Sort and get top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if doc_id in self.doc_contents:
                results.append((
                    self.doc_contents[doc_id],
                    score,
                    self.doc_metadatas.get(doc_id, {})
                ))

        return results

    def save_graph(self) -> None:
        """Save graph to disk."""
        data = {
            'graph': self.graph,
            'entity_to_docs': dict(self.entity_to_docs),
            'doc_contents': self.doc_contents,
            'doc_metadatas': self.doc_metadatas,
            'entity_embeddings': self.entity_embeddings
        }
        with open(self.graph_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Graph saved to {self.graph_path}")

    def load_graph(self) -> bool:
        """Load graph from disk."""
        if not self.graph_path.exists():
            return False

        with open(self.graph_path, 'rb') as f:
            data = pickle.load(f)

        self.graph = data['graph']
        self.entity_to_docs = defaultdict(set, data['entity_to_docs'])
        self.doc_contents = data['doc_contents']
        self.doc_metadatas = data['doc_metadatas']
        self.entity_embeddings = data.get('entity_embeddings', {})

        print(f"Graph loaded: {self.graph.number_of_nodes()} nodes")
        return True

    def get_stats(self) -> dict:
        """Get graph statistics."""
        entity_nodes = sum(1 for _, d in self.graph.nodes(data=True)
                          if d.get('type') == 'entity')
        doc_nodes = sum(1 for _, d in self.graph.nodes(data=True)
                       if d.get('type') == 'document')

        return {
            'total_nodes': self.graph.number_of_nodes(),
            'entity_nodes': entity_nodes,
            'document_nodes': doc_nodes,
            'edges': self.graph.number_of_edges()
        }


if __name__ == "__main__":
    # Test graph retriever
    retriever = GraphRetriever()

    if retriever.load_graph():
        stats = retriever.get_stats()
        print(f"Graph stats: {stats}")

        results = retriever.retrieve("hypertension treatment first line")
        print("\nGraph Top results:")
        for doc, score, meta in results[:3]:
            print(f"Score: {score:.4f}")
            print(f"Content: {doc[:200]}...")
            print()
