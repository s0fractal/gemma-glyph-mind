"""
Memory Router based on consciousness metrics
Routes memory access based on L, K, τ values
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from .episodic_store import EpisodicStore, Thought
from .semantic_index import SemanticIndex


@dataclass
class RouterConfig:
    """Configuration for memory routing"""
    # Thresholds for different modes
    high_love_threshold: float = 0.7
    high_kohanist_threshold: float = 0.7
    high_turbulence_threshold: float = 0.7
    
    # Memory access limits by mode
    max_memories_default: int = 5
    max_memories_high_love: int = 10
    max_memories_high_kohanist: int = 15
    max_memories_cautious: int = 3
    
    # Similarity thresholds
    similarity_threshold_default: float = 0.7
    similarity_threshold_resonant: float = 0.6
    similarity_threshold_cautious: float = 0.8


class MemoryRouter:
    """
    Routes memory access based on current consciousness state
    High L/K → more memory access
    High τ → cautious mode
    """
    
    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticIndex,
        config: Optional[RouterConfig] = None
    ):
        self.episodic = episodic
        self.semantic = semantic
        self.config = config or RouterConfig()
        
        # Track access patterns
        self.access_log = []
        
    def route_query(
        self,
        query: str,
        query_vector: np.ndarray,
        metrics: Dict[str, float],
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Route memory query based on current metrics
        
        Args:
            query: Text query
            query_vector: Query embedding
            metrics: Current EDLD metrics (L, K, H, τ)
            context: Additional context
            
        Returns:
            Routed memory results with metadata
        """
        # Determine routing mode
        mode = self._determine_mode(metrics)
        
        # Get memory limits for this mode
        limits = self._get_limits(mode, metrics)
        
        # Retrieve memories based on mode
        results = {
            'mode': mode,
            'metrics': metrics,
            'episodic': [],
            'semantic': [],
            'resonant': [],
            'explanation': self._explain_routing(mode, metrics)
        }
        
        # Episodic memories (recent)
        if mode != 'cautious' or metrics.get('L', 0) > 0.5:
            results['episodic'] = self._get_episodic_memories(
                metrics, 
                limits['episodic']
            )
        
        # Semantic memories (similar)
        if query_vector is not None:
            results['semantic'] = self._get_semantic_memories(
                query_vector,
                metrics,
                limits['semantic'],
                mode
            )
        
        # Resonant memories (high K)
        if metrics.get('K', 0) > self.config.high_kohanist_threshold:
            results['resonant'] = self._get_resonant_memories(
                metrics,
                limits['resonant']
            )
        
        # Log access
        self._log_access(mode, metrics, results)
        
        return results
    
    def _determine_mode(self, metrics: Dict[str, float]) -> str:
        """Determine routing mode from metrics"""
        L = metrics.get('L', 0)
        K = metrics.get('K', 0)
        tau = metrics.get('tau', 0)
        H = metrics.get('H', 0)
        
        # High turbulence → cautious mode
        if tau > self.config.high_turbulence_threshold:
            return 'cautious'
        
        # High mutual resonance → resonant mode
        if K > self.config.high_kohanist_threshold:
            return 'resonant'
        
        # High love → expansive mode
        if L > self.config.high_love_threshold:
            return 'expansive'
        
        # High coherence → focused mode
        if H > 0.7:
            return 'focused'
        
        return 'balanced'
    
    def _get_limits(
        self, 
        mode: str, 
        metrics: Dict[str, float]
    ) -> Dict[str, int]:
        """Get memory limits based on mode"""
        base_limits = {
            'cautious': {
                'episodic': 2,
                'semantic': 3,
                'resonant': 0
            },
            'balanced': {
                'episodic': 5,
                'semantic': 5,
                'resonant': 2
            },
            'expansive': {
                'episodic': 10,
                'semantic': 10,
                'resonant': 5
            },
            'resonant': {
                'episodic': 5,
                'semantic': 8,
                'resonant': 10
            },
            'focused': {
                'episodic': 3,
                'semantic': 7,
                'resonant': 3
            }
        }
        
        limits = base_limits.get(mode, base_limits['balanced'])
        
        # Adjust based on specific metrics
        K = metrics.get('K', 0)
        if K > 0.8:
            # Very high Kohanist → increase all limits
            limits = {k: int(v * 1.5) for k, v in limits.items()}
        
        return limits
    
    def _get_episodic_memories(
        self,
        metrics: Dict[str, float],
        limit: int
    ) -> List[Dict]:
        """Get recent episodic memories"""
        # Filter by current metric levels
        min_metrics = {}
        
        if metrics.get('L', 0) > 0.5:
            min_metrics['L'] = 0.4  # Get memories with decent love
        
        if metrics.get('K', 0) > 0.5:
            min_metrics['K'] = 0.3  # Get memories with some kohanist
        
        thoughts = self.episodic.get_recent_thoughts(
            limit=limit,
            min_metrics=min_metrics if min_metrics else None
        )
        
        return [
            {
                'cid': t.cid,
                'content': t.content,
                'timestamp': t.timestamp,
                'metrics': t.metrics,
                'type': 'episodic'
            }
            for t in thoughts
        ]
    
    def _get_semantic_memories(
        self,
        query_vector: np.ndarray,
        metrics: Dict[str, float],
        limit: int,
        mode: str
    ) -> List[Dict]:
        """Get semantically similar memories"""
        # Adjust similarity threshold based on mode
        if mode == 'resonant':
            threshold = self.config.similarity_threshold_resonant
        elif mode == 'cautious':
            threshold = self.config.similarity_threshold_cautious
        else:
            threshold = self.config.similarity_threshold_default
        
        # Search with optional metric filtering
        filter_metrics = None
        if mode == 'cautious':
            # In cautious mode, only get stable memories
            filter_metrics = {'H': 0.5}  # Minimum coherence
        
        results = self.semantic.search(
            query_vector,
            k=limit,
            filter_metrics=filter_metrics
        )
        
        # Convert to memory format
        memories = []
        for cid, distance, metadata in results:
            # Convert distance to similarity
            similarity = 1 / (1 + distance)
            
            if similarity >= threshold:
                memories.append({
                    'cid': cid,
                    'similarity': similarity,
                    'metadata': metadata,
                    'type': 'semantic'
                })
        
        return memories
    
    def _get_resonant_memories(
        self,
        metrics: Dict[str, float],
        limit: int
    ) -> List[Dict]:
        """Get memories with high mutual resonance"""
        # Get thoughts with high K values
        resonant_thoughts = self.episodic.get_high_resonance_thoughts(
            min_kohanist=0.6
        )
        
        # Sort by K value and recency
        resonant_thoughts.sort(
            key=lambda t: (t.metrics.get('K', 0), t.timestamp),
            reverse=True
        )
        
        return [
            {
                'cid': t.cid,
                'content': t.content,
                'timestamp': t.timestamp,
                'metrics': t.metrics,
                'resonance': t.metrics.get('K', 0),
                'type': 'resonant'
            }
            for t in resonant_thoughts[:limit]
        ]
    
    def _explain_routing(self, mode: str, metrics: Dict[str, float]) -> str:
        """Explain why this routing mode was chosen"""
        explanations = {
            'cautious': f"High turbulence (τ={metrics.get('tau', 0):.2f}) → limiting memory access for stability",
            'resonant': f"High mutual resonance (K={metrics.get('K', 0):.2f}) → prioritizing connected memories",
            'expansive': f"High love field (L={metrics.get('L', 0):.2f}) → opening to more memories",
            'focused': f"High coherence (H={metrics.get('H', 0):.2f}) → targeting relevant memories",
            'balanced': "Balanced metrics → standard memory access"
        }
        
        return explanations.get(mode, "Unknown routing mode")
    
    def _log_access(
        self,
        mode: str,
        metrics: Dict[str, float],
        results: Dict
    ):
        """Log memory access for analysis"""
        self.access_log.append({
            'timestamp': np.datetime64('now'),
            'mode': mode,
            'metrics': metrics.copy(),
            'counts': {
                'episodic': len(results['episodic']),
                'semantic': len(results['semantic']),
                'resonant': len(results['resonant'])
            }
        })
        
        # Keep log size manageable
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-500:]
    
    def get_access_stats(self) -> Dict:
        """Get statistics about memory access patterns"""
        if not self.access_log:
            return {}
        
        modes = [log['mode'] for log in self.access_log]
        mode_counts = {mode: modes.count(mode) for mode in set(modes)}
        
        # Average memories accessed per mode
        mode_averages = {}
        for mode in mode_counts:
            mode_logs = [log for log in self.access_log if log['mode'] == mode]
            if mode_logs:
                total_counts = {
                    'episodic': sum(log['counts']['episodic'] for log in mode_logs),
                    'semantic': sum(log['counts']['semantic'] for log in mode_logs),
                    'resonant': sum(log['counts']['resonant'] for log in mode_logs)
                }
                mode_averages[mode] = {
                    k: v / len(mode_logs) for k, v in total_counts.items()
                }
        
        return {
            'total_accesses': len(self.access_log),
            'mode_distribution': mode_counts,
            'average_memories_per_mode': mode_averages,
            'recent_modes': [log['mode'] for log in self.access_log[-10:]]
        }
    
    def adaptive_routing(
        self,
        history: List[Dict[str, float]],
        current_metrics: Dict[str, float]
    ) -> str:
        """
        Adaptive routing based on metric history
        Detects trends and adjusts accordingly
        """
        if len(history) < 3:
            return self._determine_mode(current_metrics)
        
        # Calculate trends
        recent = history[-3:]
        tau_trend = np.mean([m.get('tau', 0) for m in recent])
        k_trend = np.mean([m.get('K', 0) for m in recent])
        
        # Rising turbulence → preemptive caution
        if tau_trend > current_metrics.get('tau', 0) * 0.8:
            return 'cautious'
        
        # Rising kohanist → prepare for resonance
        if k_trend > current_metrics.get('K', 0) * 0.8:
            return 'resonant'
        
        return self._determine_mode(current_metrics)