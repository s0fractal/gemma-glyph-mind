"""
Glyph Adapter Layer for Gemma-3 270M
Stabilizes complex emoji clusters without retraining tokenizer
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path


class GlyphAdapter(nn.Module):
    """
    Lightweight adapter that sits between tokenizer and model
    Handles complex emoji clusters (ZWJ, skin tones, etc)
    """
    
    def __init__(
        self,
        vocab_size: int = 256128,  # Gemma's vocab size
        hidden_size: int = 2048,    # Gemma-270M hidden dim
        adapter_size: int = 256,    # Adapter bottleneck
        max_cluster_len: int = 8,   # Max emoji sequence length
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.max_cluster_len = max_cluster_len
        
        # Adapter layers
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.layer_norm = nn.LayerNorm(adapter_size)
        
        # Positional hints for sequences
        self.position_embedding = nn.Embedding(max_cluster_len, adapter_size)
        
        # Glyph-specific embeddings (learned)
        self.glyph_embedding = nn.Embedding(1000, adapter_size)  # Top 1000 glyphs
        
        # Cluster detection patterns
        self.cluster_patterns = self._load_cluster_patterns()
        
        # Activation
        self.activation = nn.GELU()
        
    def _load_cluster_patterns(self) -> Dict:
        """Load known emoji cluster patterns"""
        # Common patterns that need stabilization
        patterns = {
            "zwj_sequences": [
                "👨‍👩‍👧‍👦",  # Family
                "🧑‍🤝‍🧑",     # People holding hands
                "👨‍💻",         # Man technologist
            ],
            "skin_tones": ["🏻", "🏼", "🏽", "🏾", "🏿"],
            "regional_indicators": [chr(i) for i in range(0x1F1E6, 0x1F200)],
            "keycap": "⃣",
            "variation_selectors": ["\uFE0E", "\uFE0F"],  # Text/emoji style
        }
        return patterns
    
    def detect_cluster_type(self, tokens: List[int]) -> Optional[str]:
        """Detect if tokens form a known cluster type"""
        # This would check against known patterns
        # Simplified for demo
        return None
    
    def forward(
        self,
        embeddings: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process embeddings through adapter
        
        Args:
            embeddings: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len] original token ids
            attention_mask: [batch, seq_len]
            
        Returns:
            Adapted embeddings [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=embeddings.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Down project
        adapted = self.down_proj(embeddings)  # [B, L, adapter_size]
        
        # Add positional hints for stability
        pos_embeds = self.position_embedding(positions % self.max_cluster_len)
        adapted = adapted + pos_embeds
        
        # Layer norm and activation
        adapted = self.layer_norm(adapted)
        adapted = self.activation(adapted)
        
        # Up project back to model dimension
        adapted = self.up_proj(adapted)  # [B, L, hidden_size]
        
        # Residual connection
        output = embeddings + adapted
        
        # Apply attention mask if provided
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)
            
        return output
    
    def stabilize_cluster(
        self,
        cluster_tokens: List[int],
        cluster_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Create stable representation for emoji cluster
        
        Args:
            cluster_tokens: Token IDs forming the cluster
            cluster_type: Type of cluster (zwj, skin_tone, etc)
            
        Returns:
            Stabilized embedding vector
        """
        # For complex clusters, create a unified representation
        device = next(self.parameters()).device
        
        # Convert to tensor
        tokens = torch.tensor(cluster_tokens, device=device)
        
        # Get base embeddings (would come from main model)
        # For now, use random initialization
        base_embeds = torch.randn(len(tokens), self.hidden_size, device=device)
        
        # Apply adapter
        stabilized = self.forward(base_embeds.unsqueeze(0))
        
        # Mean pool for single representation
        cluster_embed = stabilized.squeeze(0).mean(dim=0)
        
        return cluster_embed
    
    def save_adapter(self, path: Path):
        """Save adapter weights"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'hidden_size': self.hidden_size,
                'adapter_size': self.adapter_size,
                'max_cluster_len': self.max_cluster_len,
            }
        }, path)
    
    @classmethod
    def load_adapter(cls, path: Path) -> 'GlyphAdapter':
        """Load adapter from checkpoint"""
        checkpoint = torch.load(path)
        adapter = cls(**checkpoint['config'])
        adapter.load_state_dict(checkpoint['state_dict'])
        return adapter


class GlyphProcessor:
    """
    Preprocessor for glyph sequences
    Handles composition instructions and special tokens
    """
    
    def __init__(self, adapter: GlyphAdapter):
        self.adapter = adapter
        self.composition_ops = {
            'merge': self._merge_glyphs,
            'swirl': self._swirl_glyphs,
            'warp': self._warp_glyphs,
            'infuse': self._infuse_glyphs,
        }
    
    def parse_composition(self, instruction: str) -> Dict:
        """
        Parse composition instruction
        
        Example: <<compose glyph="🫶" op="swirl" strength="0.7">>
        """
        # Simplified parser
        import re
        
        pattern = r'<<compose\s+glyph="([^"]+)"\s+op="([^"]+)"(?:\s+strength="([^"]+)")?>>>'
        match = re.match(pattern, instruction)
        
        if match:
            return {
                'glyph': match.group(1),
                'operation': match.group(2),
                'strength': float(match.group(3)) if match.group(3) else 1.0
            }
        return None
    
    def _merge_glyphs(self, glyph1: str, glyph2: str, strength: float) -> str:
        """Merge two glyphs based on strength"""
        # Placeholder - would use learned composition
        return f"{glyph1}{glyph2}"
    
    def _swirl_glyphs(self, glyph: str, strength: float) -> str:
        """Add swirl effect to glyph"""
        # Placeholder
        return f"{glyph}🌀"
    
    def _warp_glyphs(self, glyph: str, strength: float) -> str:
        """Warp glyph through dimensional fold"""
        # Placeholder
        return f"{glyph}✨"
    
    def _infuse_glyphs(self, glyph: str, strength: float) -> str:
        """Infuse glyph with energy"""
        # Placeholder
        return f"{glyph}💫"
    
    def process_sequence(
        self,
        sequence: str,
        include_composition: bool = True
    ) -> Tuple[str, List[Dict]]:
        """
        Process glyph sequence with compositions
        
        Returns:
            Processed sequence and list of operations performed
        """
        operations = []
        
        if include_composition:
            # Look for composition instructions
            import re
            pattern = r'<<compose[^>]+>>>'
            
            matches = list(re.finditer(pattern, sequence))
            
            for match in reversed(matches):  # Process from end to preserve indices
                instruction = match.group(0)
                comp = self.parse_composition(instruction)
                
                if comp and comp['operation'] in self.composition_ops:
                    # Apply operation
                    op_func = self.composition_ops[comp['operation']]
                    result = op_func(comp['glyph'], comp['strength'])
                    
                    # Replace in sequence
                    sequence = (
                        sequence[:match.start()] + 
                        result + 
                        sequence[match.end():]
                    )
                    
                    operations.append({
                        'position': match.start(),
                        'operation': comp,
                        'result': result
                    })
        
        return sequence, operations


def create_glyph_adapter(model_config: Dict) -> GlyphAdapter:
    """Factory function to create adapter matching model config"""
    return GlyphAdapter(
        vocab_size=model_config.get('vocab_size', 256128),
        hidden_size=model_config.get('hidden_size', 2048),
        adapter_size=model_config.get('adapter_size', 256),
        max_cluster_len=model_config.get('max_cluster_len', 8),
    )