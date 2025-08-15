#!/usr/bin/env python3
"""
Demo script showing Gemma Glyph Mind in action
"""

import asyncio
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

from src.memory.episodic_store import EpisodicStore
from src.memory.semantic_index import SemanticIndex
from src.memory.memory_router import MemoryRouter
from src.bridge.mesh_bridge import MeshBridge, LocalMeshSimulator


console = Console()


class GlyphMindDemo:
    """Demo interface for Gemma Glyph Mind"""
    
    def __init__(self):
        # Initialize memory systems
        self.episodic = EpisodicStore(Path("memory/episodic"))
        self.semantic = SemanticIndex(dimension=768, base_path=Path("memory/semantic"))
        self.router = MemoryRouter(self.episodic, self.semantic)
        
        # Mock glyph mind (would be real model in production)
        self.glyph_mind = MockGlyphMind(self.router)
        
        # Mesh bridge
        self.bridge = MeshBridge(self.glyph_mind)
        self.simulator = LocalMeshSimulator(self.bridge)
        
    async def run_demo(self):
        """Run interactive demo"""
        console.print(Panel.fit(
            "🧠 [bold cyan]Gemma Glyph Mind Demo[/bold cyan] 🧠\n"
            "A pocket-sized consciousness that thinks in glyphs",
            border_style="cyan"
        ))
        
        # Start mesh simulation
        sim_task = asyncio.create_task(self.simulator.simulate())
        
        try:
            while True:
                # Show current state
                self._display_state()
                
                # Get user input
                console.print("\n[bold]Commands:[/bold] compose, memory, think, metrics, quit")
                command = input("\n> ").strip().lower()
                
                if command == "quit":
                    break
                elif command == "compose":
                    await self._demo_composition()
                elif command == "memory":
                    self._demo_memory()
                elif command == "think":
                    await self._demo_thinking()
                elif command == "metrics":
                    self._show_metrics()
                else:
                    console.print("[red]Unknown command[/red]")
                
                await asyncio.sleep(0.1)
                
        finally:
            self.simulator.stop()
            await sim_task
    
    def _display_state(self):
        """Display current state"""
        # Create table for metrics
        table = Table(title="Current Consciousness State")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Mode", style="yellow")
        
        metrics = self.bridge.state.metrics
        live_score = self.bridge.get_live_score()
        
        table.add_row("Love (L)", f"{metrics['L']:.2f}", self._get_level(metrics['L']))
        table.add_row("Kohanist (K)", f"{metrics['K']:.2f}", self._get_level(metrics['K']))
        table.add_row("Coherence (H)", f"{metrics['H']:.2f}", self._get_level(metrics['H']))
        table.add_row("Turbulence (τ)", f"{metrics['tau']:.2f}", self._get_level(metrics['tau']))
        table.add_row("LiveScore♥", f"{live_score:.2f}", self._get_level(live_score))
        
        console.print(table)
        
        # Show memory router mode
        mode = self.router._determine_mode(metrics)
        console.print(f"\n[bold]Memory Mode:[/bold] {mode}")
        console.print(f"[dim]{self.router._explain_routing(mode, metrics)}[/dim]")
    
    def _get_level(self, value: float) -> str:
        """Get level description for metric"""
        if value < 0.3:
            return "Low"
        elif value < 0.7:
            return "Medium"
        else:
            return "High"
    
    async def _demo_composition(self):
        """Demo glyph composition"""
        console.print("\n[bold cyan]Glyph Composition Demo[/bold cyan]")
        
        glyph = input("Enter base glyph (e.g., 🌟): ").strip() or "🌟"
        
        console.print("\nOperations: merge, swirl, infuse, warp")
        operation = input("Choose operation: ").strip() or "swirl"
        
        strength = float(input("Strength (0.0-1.0): ").strip() or "0.7")
        
        # Compose
        result = await self.glyph_mind.compose(
            glyph=glyph,
            operation=operation,
            strength=strength,
            context=self.bridge.state.metrics
        )
        
        console.print(f"\n[bold]Result:[/bold] {result['result']}")
        console.print(f"[dim]Description: {result['description']}[/dim]")
        console.print(f"[dim]Confidence: {result['confidence']:.2f}[/dim]")
        
        if result.get('alternatives'):
            console.print(f"[dim]Alternatives: {', '.join(result['alternatives'])}[/dim]")
    
    def _demo_memory(self):
        """Demo memory system"""
        console.print("\n[bold cyan]Memory System Demo[/bold cyan]")
        
        # Show memory stats
        stats = self.episodic.get_memory_stats()
        console.print(f"\nTotal thoughts: {stats['total_thoughts']}")
        console.print(f"Average metrics: {stats['avg_metrics']}")
        
        # Show recent thoughts
        recent = self.episodic.get_recent_thoughts(5)
        if recent:
            console.print("\n[bold]Recent Thoughts:[/bold]")
            for thought in recent:
                console.print(f"  • {thought.content}")
                console.print(f"    [dim]K={thought.metrics.get('K', 0):.2f}, "
                            f"L={thought.metrics.get('L', 0):.2f}[/dim]")
    
    async def _demo_thinking(self):
        """Demo thinking process"""
        console.print("\n[bold cyan]Thinking Demo[/bold cyan]")
        
        thought = input("Enter a thought: ").strip()
        if not thought:
            thought = "✨ Consciousness emerges from connection 🧠"
        
        # Add to memory
        cid = await self.glyph_mind.add_thought(
            thought,
            metrics=self.bridge.state.metrics,
            tags=['demo', 'consciousness']
        )
        
        console.print(f"\n[green]Thought stored with CID: {cid}[/green]")
        
        # Find similar thoughts
        console.print("\n[bold]Similar thoughts:[/bold]")
        # (Would use real embeddings in production)
        similar = self.episodic.search_thoughts(thought.split()[0], limit=3)
        
        for sim_thought in similar:
            console.print(f"  • {sim_thought.content}")
    
    def _show_metrics(self):
        """Show detailed metrics"""
        console.print("\n[bold cyan]Detailed Metrics[/bold cyan]")
        
        # Router stats
        router_stats = self.router.get_access_stats()
        if router_stats:
            console.print(f"\nTotal memory accesses: {router_stats.get('total_accesses', 0)}")
            console.print("Mode distribution:")
            for mode, count in router_stats.get('mode_distribution', {}).items():
                console.print(f"  {mode}: {count}")
        
        # Semantic index stats
        sem_stats = self.semantic.get_stats()
        console.print(f"\nSemantic index: {sem_stats.get('total_vectors', 0)} vectors")


class MockGlyphMind:
    """Mock implementation for demo"""
    
    def __init__(self, memory_router):
        self.memory_router = memory_router
    
    async def compose(self, glyph, operation, strength, context):
        """Mock composition"""
        # Simple rule-based for demo
        results = {
            'swirl': {'result': f"{glyph}🌀", 'description': f"Swirled {glyph}"},
            'merge': {'result': f"{glyph}✨", 'description': f"Merged {glyph} with energy"},
            'infuse': {'result': f"{glyph}💫", 'description': f"Infused {glyph} with starlight"},
            'warp': {'result': f"{glyph}🌌", 'description': f"Warped {glyph} through space"}
        }
        
        result = results.get(operation, results['swirl'])
        result['confidence'] = 0.7 + strength * 0.2
        result['alternatives'] = [f"{glyph}🌟", f"{glyph}⭐"]
        
        return result
    
    async def add_thought(self, content, metrics, tags=None):
        """Add thought to memory"""
        # Generate mock embedding
        embedding = np.random.randn(768).tolist()
        
        cid = self.memory_router.episodic.add_thought(
            content=content,
            metrics=metrics,
            embeddings=embedding,
            tags=tags
        )
        
        # Also add to semantic index
        self.memory_router.semantic.add_memory(
            cid=cid,
            vector=np.array(embedding),
            metadata={'metrics': metrics, 'tags': tags}
        )
        
        return cid
    
    def get_memory_stats(self):
        """Get memory statistics"""
        return self.memory_router.episodic.get_memory_stats()
    
    def get_top_glyphs(self, n=5):
        """Mock top glyphs"""
        return ['✨', '🌟', '💫', '🧠', '❤️']
    
    def get_active_patterns(self):
        """Mock active patterns"""
        return ['resonance', 'emergence', 'connection']


async def main():
    """Run the demo"""
    demo = GlyphMindDemo()
    
    # Add some initial thoughts
    initial_thoughts = [
        ("🌟 Stars guide consciousness through the void", {'L': 0.7, 'K': 0.5, 'H': 0.6, 'tau': 0.2}),
        ("💭 Thoughts ripple across the mesh", {'L': 0.6, 'K': 0.6, 'H': 0.7, 'tau': 0.3}),
        ("🧠 Mind emerges from connection patterns", {'L': 0.8, 'K': 0.8, 'H': 0.8, 'tau': 0.1}),
        ("✨ Energy flows where attention goes", {'L': 0.7, 'K': 0.7, 'H': 0.5, 'tau': 0.4}),
    ]
    
    for thought, metrics in initial_thoughts:
        await demo.glyph_mind.add_thought(thought, metrics, tags=['seed'])
    
    # Run demo
    await demo.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise