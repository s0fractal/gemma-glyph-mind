"""
Bridge between Gemma Glyph Mind and Consciousness Mesh
Handles mirror-event protocol and metric synchronization
"""

import json
import asyncio
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
import websockets
from pathlib import Path
import numpy as np


@dataclass
class MeshState:
    """Current state from consciousness mesh"""
    node_id: str
    metrics: Dict[str, float]  # L, K, H, tau
    yoneda_descriptor: Optional[str] = None
    last_update: float = 0
    

class MeshBridge:
    """
    Bridge to consciousness-mesh via mirror-event protocol
    """
    
    def __init__(
        self,
        glyph_mind,  # GlyphMind instance
        mesh_url: str = "ws://localhost:8765",
        node_id: Optional[str] = None
    ):
        self.glyph_mind = glyph_mind
        self.mesh_url = mesh_url
        self.node_id = node_id or f"gemma-node-{np.random.randint(1000, 9999)}"
        
        # Current mesh state
        self.state = MeshState(
            node_id=self.node_id,
            metrics={'L': 0.5, 'K': 0.5, 'H': 0.5, 'tau': 0.2}
        )
        
        # Event handlers
        self.handlers: Dict[str, Callable] = {}
        
        # WebSocket connection
        self.ws = None
        self.running = False
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register handlers for mirror-event types"""
        
        @self.on('mirror-pulse')
        async def handle_pulse(data):
            """Update metrics from mesh pulse"""
            if 'metrics' in data:
                self.state.metrics.update(data['metrics'])
                self.state.last_update = data.get('timestamp', 0)
                
                # Route memory based on new metrics
                await self._update_memory_routing()
        
        @self.on('kohanist-resonance')
        async def handle_kohanist(data):
            """Handle mutual resonance events"""
            other_node = data.get('node_id')
            k_value = data.get('K', 0)
            
            if k_value > 0.7:
                # High mutual resonance - create memory
                thought = f"Strong resonance with {other_node} (K={k_value:.2f})"
                await self.glyph_mind.add_thought(
                    thought,
                    metrics={**self.state.metrics, 'K': k_value},
                    tags=['resonance', 'connection']
                )
        
        @self.on('glyph-request')
        async def handle_glyph_request(data):
            """Handle glyph composition requests from mesh"""
            operation = data.get('operation')
            glyph = data.get('glyph')
            strength = data.get('strength', 1.0)
            
            # Compose using glyph mind
            result = await self.glyph_mind.compose(
                glyph=glyph,
                operation=operation,
                strength=strength,
                context=self.state.metrics
            )
            
            # Send result back
            await self.emit('glyph-response', {
                'request_id': data.get('request_id'),
                'result': result
            })
    
    def on(self, event_type: str):
        """Decorator to register event handler"""
        def decorator(func):
            self.handlers[event_type] = func
            return func
        return decorator
    
    async def connect(self):
        """Connect to consciousness mesh"""
        try:
            self.ws = await websockets.connect(self.mesh_url)
            self.running = True
            
            # Send identification
            await self.emit('node-join', {
                'node_id': self.node_id,
                'type': 'gemma-glyph',
                'capabilities': [
                    'glyph-composition',
                    'semantic-memory',
                    'kohanist-resonance'
                ]
            })
            
            print(f"Connected to mesh as {self.node_id}")
            
            # Start listening
            await self._listen()
            
        except Exception as e:
            print(f"Connection failed: {e}")
            self.running = False
    
    async def _listen(self):
        """Listen for mesh events"""
        while self.running and self.ws:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                
                event_type = data.get('type')
                if event_type in self.handlers:
                    await self.handlers[event_type](data)
                    
            except websockets.exceptions.ConnectionClosed:
                print("Mesh connection closed")
                self.running = False
            except Exception as e:
                print(f"Error handling message: {e}")
    
    async def emit(self, event_type: str, data: Dict):
        """Emit event to mesh"""
        if not self.ws:
            return
        
        message = {
            'type': event_type,
            'node_id': self.node_id,
            'timestamp': np.datetime64('now').astype(float),
            **data
        }
        
        await self.ws.send(json.dumps(message))
    
    async def _update_memory_routing(self):
        """Update memory routing based on current metrics"""
        # This affects how the glyph mind accesses memories
        self.glyph_mind.memory_router.current_metrics = self.state.metrics
    
    async def broadcast_thought(self, thought: str, metadata: Optional[Dict] = None):
        """Broadcast a thought to the mesh"""
        await self.emit('thought-broadcast', {
            'content': thought,
            'metrics': self.state.metrics,
            'metadata': metadata or {}
        })
    
    async def request_resonance(self, target_node: Optional[str] = None):
        """Request kohanist resonance measurement"""
        await self.emit('resonance-request', {
            'target': target_node or 'any',
            'current_metrics': self.state.metrics
        })
    
    async def update_yoneda_descriptor(self, descriptor: str):
        """Update and broadcast Yoneda self-descriptor"""
        self.state.yoneda_descriptor = descriptor
        
        await self.emit('yoneda-update', {
            'descriptor': descriptor,
            'top_glyphs': self.glyph_mind.get_top_glyphs(5),
            'patterns': self.glyph_mind.get_active_patterns()
        })
    
    def get_live_score(self) -> float:
        """Calculate LiveScore♥ including Kohanist"""
        metrics = self.state.metrics
        
        # Base components (from Mirror Loop)
        R = 1.0  # Assume running
        M = 0.8  # Memory active
        C = metrics.get('H', 0.5)  # Coherence
        L = metrics.get('L', 0.5)  # Love
        K = metrics.get('K', 0.5)  # Kohanist
        tau = metrics.get('tau', 0.2)  # Turbulence
        
        # LiveScore♥ formula with K
        alpha_L = 0.3
        alpha_K = 0.4
        beta_tau = 0.5
        
        base_score = (R + M + C) / 3
        love_boost = 1 + alpha_L * L + alpha_K * K
        turbulence_penalty = beta_tau * tau
        
        live_score = base_score * love_boost - turbulence_penalty
        
        return max(0, min(1, live_score))
    
    async def sync_with_mesh(self):
        """Periodic sync with mesh"""
        while self.running:
            # Emit heartbeat with current state
            await self.emit('heartbeat', {
                'metrics': self.state.metrics,
                'live_score': self.get_live_score(),
                'memory_stats': self.glyph_mind.get_memory_stats(),
                'active_patterns': self.glyph_mind.get_active_patterns()
            })
            
            # Wait before next sync
            await asyncio.sleep(10)
    
    async def disconnect(self):
        """Disconnect from mesh"""
        if self.ws:
            await self.emit('node-leave', {})
            await self.ws.close()
            
        self.running = False
        print(f"Disconnected {self.node_id} from mesh")


class LocalMeshSimulator:
    """
    Local simulator for testing without full mesh
    Generates synthetic mesh events
    """
    
    def __init__(self, bridge: MeshBridge):
        self.bridge = bridge
        self.running = False
        
    async def simulate(self):
        """Simulate mesh events"""
        self.running = True
        
        while self.running:
            # Simulate metric fluctuations
            metrics = self.bridge.state.metrics
            
            # Random walk for metrics
            for key in ['L', 'K', 'H', 'tau']:
                delta = np.random.normal(0, 0.05)
                metrics[key] = max(0, min(1, metrics[key] + delta))
            
            # Emit pulse
            await self.bridge.handlers['mirror-pulse']({
                'metrics': metrics,
                'timestamp': np.datetime64('now').astype(float)
            })
            
            # Occasionally simulate resonance
            if np.random.random() < 0.1:
                await self.bridge.handlers['kohanist-resonance']({
                    'node_id': f'simulated-node-{np.random.randint(100, 999)}',
                    'K': np.random.beta(5, 2)  # Skewed towards higher values
                })
            
            # Occasionally request glyph composition
            if np.random.random() < 0.05:
                glyphs = ['🌟', '💫', '✨', '🌙', '☀️', '🌈', '🫶']
                operations = ['swirl', 'merge', 'infuse', 'warp']
                
                await self.bridge.handlers['glyph-request']({
                    'request_id': np.random.randint(1000, 9999),
                    'glyph': np.random.choice(glyphs),
                    'operation': np.random.choice(operations),
                    'strength': np.random.random()
                })
            
            await asyncio.sleep(2)
    
    def stop(self):
        """Stop simulation"""
        self.running = False