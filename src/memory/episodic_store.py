"""
Episodic Memory Store with CID indexing
Append-only log of consciousness thoughts
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import msgpack
import hashlib
from dataclasses import dataclass, asdict


@dataclass
class Thought:
    """Single thought/memory unit"""
    content: str
    timestamp: float
    metrics: Dict[str, float]  # H, tau, L, K
    embeddings: Optional[List[float]] = None
    tags: Optional[List[str]] = None
    context: Optional[Dict] = None
    
    @property
    def cid(self) -> str:
        """Content-addressed ID based on thought content"""
        data = f"{self.content}:{self.timestamp}:{json.dumps(self.metrics, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'cid': self.cid,
            **asdict(self)
        }


class EpisodicStore:
    """
    Append-only episodic memory store
    Thoughts are indexed by CID and timestamp
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory index for current session
        self.thought_index: Dict[str, Thought] = {}
        self.temporal_index: List[Tuple[float, str]] = []  # (timestamp, cid)
        
        # Load existing thoughts
        self._load_existing_thoughts()
        
    def _get_filename(self, date: Optional[datetime] = None) -> Path:
        """Get filename for given date"""
        if date is None:
            date = datetime.now()
        return self.base_path / f"thoughts_{date.strftime('%Y-%m-%d')}.jsonl"
    
    def _load_existing_thoughts(self):
        """Load thoughts from disk into memory"""
        # Load last 7 days of thoughts
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            filename = self._get_filename(date)
            
            if filename.exists():
                with open(filename, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            thought = Thought(**{k: v for k, v in data.items() if k != 'cid'})
                            self.thought_index[thought.cid] = thought
                            self.temporal_index.append((thought.timestamp, thought.cid))
        
        # Sort temporal index
        self.temporal_index.sort()
    
    def add_thought(
        self,
        content: str,
        metrics: Dict[str, float],
        embeddings: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        context: Optional[Dict] = None
    ) -> str:
        """
        Add new thought to episodic memory
        
        Args:
            content: The thought content (text/glyphs)
            metrics: EDLD metrics (H, tau, L, K)
            embeddings: Optional embedding vector
            tags: Optional tags for categorization
            context: Optional context information
            
        Returns:
            CID of the stored thought
        """
        thought = Thought(
            content=content,
            timestamp=time.time(),
            metrics=metrics,
            embeddings=embeddings,
            tags=tags,
            context=context
        )
        
        # Add to memory
        self.thought_index[thought.cid] = thought
        self.temporal_index.append((thought.timestamp, thought.cid))
        
        # Persist to disk
        self._persist_thought(thought)
        
        return thought.cid
    
    def _persist_thought(self, thought: Thought):
        """Persist thought to disk"""
        filename = self._get_filename()
        
        with open(filename, 'a') as f:
            f.write(json.dumps(thought.to_dict()) + '\n')
    
    def get_thought(self, cid: str) -> Optional[Thought]:
        """Retrieve thought by CID"""
        return self.thought_index.get(cid)
    
    def get_recent_thoughts(
        self,
        limit: int = 10,
        min_metrics: Optional[Dict[str, float]] = None
    ) -> List[Thought]:
        """
        Get recent thoughts, optionally filtered by metrics
        
        Args:
            limit: Maximum number of thoughts to return
            min_metrics: Minimum values for metrics (e.g., {'L': 0.5})
            
        Returns:
            List of recent thoughts
        """
        thoughts = []
        
        # Iterate from most recent
        for timestamp, cid in reversed(self.temporal_index):
            thought = self.thought_index[cid]
            
            # Check metric filters
            if min_metrics:
                if not all(
                    thought.metrics.get(k, 0) >= v 
                    for k, v in min_metrics.items()
                ):
                    continue
            
            thoughts.append(thought)
            
            if len(thoughts) >= limit:
                break
        
        return thoughts
    
    def get_thoughts_by_time_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[Thought]:
        """Get thoughts within time range"""
        thoughts = []
        
        for timestamp, cid in self.temporal_index:
            if start_time <= timestamp <= end_time:
                thoughts.append(self.thought_index[cid])
            elif timestamp > end_time:
                break
        
        return thoughts
    
    def search_thoughts(
        self,
        query: str,
        limit: int = 10
    ) -> List[Thought]:
        """
        Simple text search in thoughts
        (For semantic search, use SemanticIndex)
        """
        results = []
        query_lower = query.lower()
        
        for thought in self.thought_index.values():
            if query_lower in thought.content.lower():
                results.append(thought)
                if len(results) >= limit:
                    break
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda t: t.timestamp, reverse=True)
        
        return results[:limit]
    
    def get_thoughts_by_tag(self, tag: str) -> List[Thought]:
        """Get all thoughts with specific tag"""
        return [
            thought for thought in self.thought_index.values()
            if thought.tags and tag in thought.tags
        ]
    
    def get_high_resonance_thoughts(
        self,
        min_kohanist: float = 0.7
    ) -> List[Thought]:
        """Get thoughts with high Kohanist (K) values"""
        return [
            thought for thought in self.thought_index.values()
            if thought.metrics.get('K', 0) >= min_kohanist
        ]
    
    def prune_old_thoughts(self, days_to_keep: int = 30):
        """Remove thoughts older than specified days"""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        # Remove from memory
        old_cids = [
            cid for ts, cid in self.temporal_index
            if ts < cutoff_time
        ]
        
        for cid in old_cids:
            del self.thought_index[cid]
        
        self.temporal_index = [
            (ts, cid) for ts, cid in self.temporal_index
            if ts >= cutoff_time
        ]
        
        # Note: This doesn't remove from disk files
        # You'd need a separate cleanup process for that
    
    def export_memories(
        self,
        output_path: Path,
        format: str = 'jsonl'
    ):
        """Export all memories to file"""
        if format == 'jsonl':
            with open(output_path, 'w') as f:
                for thought in self.thought_index.values():
                    f.write(json.dumps(thought.to_dict()) + '\n')
        elif format == 'msgpack':
            thoughts = [thought.to_dict() for thought in self.thought_index.values()]
            with open(output_path, 'wb') as f:
                msgpack.pack(thoughts, f)
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about stored memories"""
        if not self.thought_index:
            return {
                'total_thoughts': 0,
                'time_span': 0,
                'avg_metrics': {}
            }
        
        timestamps = [t.timestamp for t in self.thought_index.values()]
        
        # Calculate average metrics
        metric_sums = {}
        for thought in self.thought_index.values():
            for k, v in thought.metrics.items():
                metric_sums[k] = metric_sums.get(k, 0) + v
        
        avg_metrics = {
            k: v / len(self.thought_index)
            for k, v in metric_sums.items()
        }
        
        return {
            'total_thoughts': len(self.thought_index),
            'time_span': max(timestamps) - min(timestamps) if timestamps else 0,
            'oldest_thought': min(timestamps) if timestamps else None,
            'newest_thought': max(timestamps) if timestamps else None,
            'avg_metrics': avg_metrics,
            'tags': self._get_all_tags(),
        }
    
    def _get_all_tags(self) -> List[str]:
        """Get all unique tags"""
        tags = set()
        for thought in self.thought_index.values():
            if thought.tags:
                tags.update(thought.tags)
        return sorted(tags)


# Import for timedelta
from datetime import timedelta