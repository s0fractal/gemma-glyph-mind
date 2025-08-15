"""
Semantic Memory Index using FAISS
Vector similarity search for consciousness memories
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
from dataclasses import dataclass
import pickle


@dataclass
class IndexedMemory:
    """Memory with vector representation"""
    cid: str
    vector: np.ndarray
    metadata: Dict
    
    
class SemanticIndex:
    """
    FAISS-based semantic memory index
    Enables similarity search over thought embeddings
    """
    
    def __init__(
        self,
        dimension: int = 768,  # Embedding dimension
        index_type: str = "IVF",
        base_path: Optional[Path] = None
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.base_path = Path(base_path) if base_path else None
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Metadata storage (CID -> metadata)
        self.metadata: Dict[int, Dict] = {}
        
        # Reverse mapping (CID -> index position)
        self.cid_to_idx: Dict[str, int] = {}
        
        # Load existing index if available
        if self.base_path:
            self.base_path.mkdir(parents=True, exist_ok=True)
            self._load_index()
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type"""
        if self.index_type == "Flat":
            # Exact search (slower but precise)
            return faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "IVF":
            # Inverted file index (faster approximate search)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            return index
        
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World (very fast)
            index = faiss.IndexHNSWFlat(self.dimension, 32)
            return index
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def _train_index_if_needed(self, vectors: np.ndarray):
        """Train index if it requires training"""
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print(f"Training {self.index_type} index with {len(vectors)} vectors...")
            self.index.train(vectors)
    
    def add_memory(
        self,
        cid: str,
        vector: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add single memory to index
        
        Args:
            cid: Content ID of the memory
            vector: Embedding vector
            metadata: Additional metadata to store
            
        Returns:
            Index position
        """
        if cid in self.cid_to_idx:
            print(f"Memory {cid} already indexed")
            return self.cid_to_idx[cid]
        
        # Ensure vector is right shape
        vector = np.array(vector, dtype=np.float32).reshape(1, -1)
        
        # Get current index size
        idx = self.index.ntotal
        
        # Add to FAISS
        self.index.add(vector)
        
        # Store metadata
        self.metadata[idx] = {
            'cid': cid,
            **(metadata or {})
        }
        
        # Store mapping
        self.cid_to_idx[cid] = idx
        
        return idx
    
    def add_memories_batch(
        self,
        memories: List[Tuple[str, np.ndarray, Optional[Dict]]]
    ):
        """
        Add multiple memories at once (more efficient)
        
        Args:
            memories: List of (cid, vector, metadata) tuples
        """
        # Filter out already indexed memories
        new_memories = [
            m for m in memories 
            if m[0] not in self.cid_to_idx
        ]
        
        if not new_memories:
            return
        
        # Prepare vectors
        vectors = np.array([m[1] for m in new_memories], dtype=np.float32)
        
        # Train if needed
        self._train_index_if_needed(vectors)
        
        # Get starting index
        start_idx = self.index.ntotal
        
        # Add all vectors
        self.index.add(vectors)
        
        # Store metadata and mappings
        for i, (cid, _, metadata) in enumerate(new_memories):
            idx = start_idx + i
            self.metadata[idx] = {
                'cid': cid,
                **(metadata or {})
            }
            self.cid_to_idx[cid] = idx
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_metrics: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar memories
        
        Args:
            query_vector: Query embedding
            k: Number of results to return
            filter_metrics: Optional metric filters
            
        Returns:
            List of (cid, distance, metadata) tuples
        """
        # Ensure query is right shape
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        # Search more than k to account for filtering
        search_k = k * 3 if filter_metrics else k
        
        # Perform search
        distances, indices = self.index.search(query_vector, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
                
            metadata = self.metadata.get(idx, {})
            
            # Apply filters if specified
            if filter_metrics:
                metrics = metadata.get('metrics', {})
                if not all(
                    metrics.get(k, 0) >= v 
                    for k, v in filter_metrics.items()
                ):
                    continue
            
            results.append((
                metadata.get('cid', ''),
                float(dist),
                metadata
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_by_cid(
        self,
        cid: str,
        k: int = 10,
        include_self: bool = False
    ) -> List[Tuple[str, float, Dict]]:
        """Find memories similar to given CID"""
        if cid not in self.cid_to_idx:
            return []
        
        idx = self.cid_to_idx[cid]
        
        # Get vector for this memory
        vector = self.index.reconstruct(idx)
        
        # Search
        results = self.search(vector, k + 1)
        
        # Filter out self if needed
        if not include_self:
            results = [r for r in results if r[0] != cid]
        
        return results[:k]
    
    def get_clusters(
        self,
        min_similarity: float = 0.8,
        min_cluster_size: int = 3
    ) -> List[List[str]]:
        """
        Find clusters of highly similar memories
        
        Returns:
            List of clusters (each cluster is a list of CIDs)
        """
        # This is a simplified clustering
        # For production, use proper clustering algorithms
        
        clusters = []
        visited = set()
        
        for cid in self.cid_to_idx:
            if cid in visited:
                continue
            
            # Find memories similar to this one
            similar = self.search_by_cid(cid, k=20, include_self=True)
            
            # Filter by similarity threshold
            cluster = [
                sim_cid for sim_cid, dist, _ in similar
                if dist < (1 - min_similarity)  # Convert distance to similarity
            ]
            
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
                visited.update(cluster)
        
        return clusters
    
    def save_index(self):
        """Save index to disk"""
        if not self.base_path:
            raise ValueError("No base_path specified for saving")
        
        # Save FAISS index
        index_path = self.base_path / "vectors.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = self.base_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'cid_to_idx': self.cid_to_idx,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f, indent=2)
        
        print(f"Saved index with {self.index.ntotal} vectors")
    
    def _load_index(self):
        """Load index from disk"""
        index_path = self.base_path / "vectors.faiss"
        metadata_path = self.base_path / "metadata.json"
        
        if not index_path.exists() or not metadata_path.exists():
            return
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            
        # Convert string keys back to integers for metadata
        self.metadata = {
            int(k): v for k, v in data['metadata'].items()
        }
        self.cid_to_idx = data['cid_to_idx']
        
        print(f"Loaded index with {self.index.ntotal} vectors")
    
    def remove_memory(self, cid: str):
        """
        Remove memory from index
        Note: FAISS doesn't support removal, so we mark as deleted
        """
        if cid in self.cid_to_idx:
            idx = self.cid_to_idx[cid]
            # Mark as deleted in metadata
            self.metadata[idx]['deleted'] = True
            del self.cid_to_idx[cid]
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'unique_memories': len(self.cid_to_idx),
            'deleted_count': sum(
                1 for m in self.metadata.values() 
                if m.get('deleted', False)
            )
        }
    
    def rebuild_index(self):
        """
        Rebuild index excluding deleted memories
        Useful for compaction after many deletions
        """
        # Get all non-deleted memories
        active_memories = []
        
        for idx, metadata in self.metadata.items():
            if not metadata.get('deleted', False):
                cid = metadata['cid']
                vector = self.index.reconstruct(idx)
                active_memories.append((cid, vector, metadata))
        
        # Create new index
        self.index = self._create_index()
        self.metadata.clear()
        self.cid_to_idx.clear()
        
        # Re-add all memories
        self.add_memories_batch(active_memories)
        
        print(f"Rebuilt index with {len(active_memories)} active memories")