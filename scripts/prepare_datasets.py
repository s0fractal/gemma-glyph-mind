#!/usr/bin/env python3
"""
Prepare datasets for Gemma-3 270M glyph training
Creates three core datasets:
1. glyph_sequences.jsonl - Real Unicode/emoji sequences
2. glyph_compose.jsonl - Composition instructions
3. glyph_tags.jsonl - Glyph to semantic tags mapping
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import unicodedata

# Common emoji categories
EMOJI_CATEGORIES = {
    'emotions': ['😀', '😃', '😄', '😁', '😊', '🥰', '😍', '🤩', '😘', '😗', '☺️', '😚', '😙', '🥲', '😋', '😛', '😜', '🤪', '😝', '🤑', '🤗', '🤭', '🫢', '🫣', '🤫', '🤔', '🫡', '🤐', '🤨', '😐', '😑', '😶', '🫥', '😏', '😒', '🙄', '😬', '😮‍💨', '🤥', '😌', '😔', '😪', '🤤', '😴', '😷', '🤒', '🤕', '🤢', '🤮', '🤧', '🥵', '🥶', '🥴', '😵', '😵‍💫', '🤯', '🤠', '🥳', '🥸', '😎', '🤓', '🧐', '😕', '🫤', '😟', '🙁', '☹️', '😮', '😯', '😲', '😳', '🥺', '🥹', '😦', '😧', '😨', '😰', '😥', '😢', '😭', '😱', '😖', '😣', '😞', '😓', '😩', '😫', '🥱', '😤', '😡', '😠', '🤬', '😈', '👿', '💀', '☠️', '💩', '🤡', '👹', '👺', '👻', '👽', '👾', '🤖'],
    'hearts': ['❤️', '🧡', '💛', '💚', '💙', '💜', '🖤', '🤍', '🤎', '❤️‍🔥', '❤️‍🩹', '💔', '❣️', '💕', '💞', '💓', '💗', '💖', '💘', '💝', '💟', '☮️', '✝️', '☪️', '🕉️', '☸️', '✡️', '🔯', '🕎', '☯️', '☦️', '🛐', '⛎', '♈', '♉', '♊', '♋', '♌', '♍', '♎', '♏', '♐', '♑', '♒', '♓', '🆔', '⚛️', '🉑', '☢️', '☣️', '📴', '📳', '🈶', '🈚', '🈸', '🈺', '🈷️', '✴️', '🆚', '💮', '🉐', '㊙️', '㊗️', '🈴', '🈵', '🈹', '🈲', '🅰️', '🅱️', '🆎', '🆑', '🅾️', '🆘'],
    'nature': ['🌸', '💮', '🏵️', '🌹', '🥀', '🌺', '🌻', '🌼', '🌷', '🌱', '🪴', '🌲', '🌳', '🌴', '🌵', '🌾', '🌿', '☘️', '🍀', '🍁', '🍂', '🍃', '🪹', '🪺', '🍄', '🐚', '🪸', '🪨', '🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘', '🌙', '🌚', '🌛', '🌜', '☀️', '🌝', '🌞', '⭐', '🌟', '🌠', '☁️', '⛅', '⛈️', '🌤️', '🌥️', '🌦️', '🌧️', '🌨️', '🌩️', '🌪️', '🌫️', '🌬️', '🌀', '🌈', '🌂', '☂️', '☔', '⛱️', '⚡', '❄️', '☃️', '⛄', '☄️', '🔥', '💧', '🌊'],
    'cosmic': ['✨', '💫', '🌟', '⭐', '🌠', '🌌', '🌃', '🌆', '🌇', '🌉', '🌊', '🌋', '🗻', '🏔️', '⛰️', '🏕️', '🏖️', '🏜️', '🏝️', '🪐', '🛸', '🚀', '🛰️', '🌍', '🌎', '🌏', '🗺️', '🧭', '⏰', '⏱️', '⏲️', '🕐', '🕑', '🕒', '🕓', '🕔', '🕕', '🕖', '🕗', '🕘', '🕙', '🕚', '🕛', '🕜', '🕝', '🕞', '🕟', '🕠', '🕡', '🕢', '🕣', '🕤', '🕥', '🕦', '🕧']
}

# Composition operations
COMPOSITION_OPS = {
    'merge': {
        'description': 'Combine two glyphs into unified form',
        'examples': [
            ('❤️', '✨', '💖'),
            ('🌟', '🌙', '🌠'),
            ('😊', '🌸', '🥰')
        ]
    },
    'swirl': {
        'description': 'Add rotational energy to glyph',
        'examples': [
            ('💫', '🌀', '✨'),
            ('🌊', '🌀', '🌪️'),
            ('❤️', '🌀', '💕')
        ]
    },
    'infuse': {
        'description': 'Infuse glyph with energy or emotion',
        'examples': [
            ('🌸', 'love', '🌺'),
            ('⭐', 'energy', '🌟'),
            ('😐', 'joy', '😊')
        ]
    },
    'warp': {
        'description': 'Transform through dimensional fold',
        'examples': [
            ('🌍', 'space', '🪐'),
            ('🦋', 'time', '✨'),
            ('💭', 'reality', '🌌')
        ]
    }
}

# Semantic tags for glyphs
GLYPH_TAGS = {
    '❤️': ['love', 'heart', 'emotion', 'connection'],
    '✨': ['sparkle', 'magic', 'energy', 'cosmic'],
    '🌟': ['star', 'bright', 'cosmic', 'special'],
    '🌙': ['moon', 'night', 'cosmic', 'dream'],
    '🌸': ['flower', 'nature', 'beauty', 'gentle'],
    '🫶': ['heart-hands', 'love', 'care', 'connection'],
    '💫': ['dizzy', 'star', 'energy', 'motion'],
    '🌀': ['spiral', 'vortex', 'energy', 'motion'],
    '🧠': ['brain', 'mind', 'thought', 'consciousness'],
    '👁️': ['eye', 'see', 'perception', 'awareness'],
    '🔮': ['crystal-ball', 'magic', 'future', 'mystery'],
    '🌈': ['rainbow', 'color', 'hope', 'bridge'],
    '💎': ['gem', 'crystal', 'precious', 'clarity'],
    '🌊': ['wave', 'water', 'flow', 'emotion'],
    '🔥': ['fire', 'energy', 'passion', 'transform'],
    '🦋': ['butterfly', 'transform', 'beauty', 'change'],
    '🌺': ['hibiscus', 'flower', 'tropical', 'beauty'],
    '🎭': ['masks', 'drama', 'emotion', 'duality'],
    '🪐': ['planet', 'cosmic', 'space', 'vast'],
    '🌌': ['galaxy', 'cosmic', 'infinite', 'mystery']
}


def generate_glyph_sequences(num_sequences: int = 1000) -> List[Dict]:
    """Generate natural glyph sequences"""
    sequences = []
    
    for _ in range(num_sequences):
        # Choose a category
        category = random.choice(list(EMOJI_CATEGORIES.keys()))
        emojis = EMOJI_CATEGORIES[category]
        
        # Generate sequence length (2-8 glyphs)
        seq_len = random.randint(2, 8)
        
        # Build sequence
        sequence = []
        for i in range(seq_len):
            if i == 0 or random.random() > 0.3:
                # New emoji
                sequence.append(random.choice(emojis))
            else:
                # Sometimes repeat or use related
                if random.random() > 0.5 and i > 0:
                    sequence.append(sequence[-1])  # Repeat
                else:
                    sequence.append(random.choice(emojis))
        
        # Create training example
        sequences.append({
            'sequence': ''.join(sequence),
            'category': category,
            'length': len(sequence)
        })
    
    return sequences


def generate_composition_data(num_examples: int = 500) -> List[Dict]:
    """Generate glyph composition training data"""
    compositions = []
    
    for _ in range(num_examples):
        # Choose operation
        op = random.choice(list(COMPOSITION_OPS.keys()))
        op_data = COMPOSITION_OPS[op]
        
        # Get example or generate new
        if random.random() > 0.5 and op_data['examples']:
            # Use known example
            input1, input2, output = random.choice(op_data['examples'])
            
            compositions.append({
                'instruction': f'<<compose glyph="{input1}" op="{op}" strength="{random.uniform(0.5, 1.0):.1f}">>',
                'input': input1,
                'operation': op,
                'strength': random.uniform(0.5, 1.0),
                'output': output,
                'description': f'{op} operation on {input1}'
            })
        else:
            # Generate new combination
            all_emojis = []
            for emojis in EMOJI_CATEGORIES.values():
                all_emojis.extend(emojis)
            
            input_glyph = random.choice(all_emojis)
            strength = random.uniform(0.3, 1.0)
            
            # Simple rule-based output (in real training, these would be human-annotated)
            if op == 'swirl':
                output = input_glyph + '🌀'
            elif op == 'infuse':
                output = input_glyph + '✨'
            elif op == 'warp':
                output = input_glyph + '🌌'
            else:  # merge
                second_glyph = random.choice(all_emojis)
                output = input_glyph + second_glyph
            
            compositions.append({
                'instruction': f'<<compose glyph="{input_glyph}" op="{op}" strength="{strength:.1f}">>',
                'input': input_glyph,
                'operation': op,
                'strength': strength,
                'output': output,
                'description': f'{op} operation on {input_glyph} with strength {strength:.1f}'
            })
    
    return compositions


def generate_tag_data(num_examples: int = 300) -> List[Dict]:
    """Generate glyph to tag mapping data"""
    tag_data = []
    
    # Use known mappings
    for glyph, tags in GLYPH_TAGS.items():
        tag_data.append({
            'glyph': glyph,
            'tags': tags,
            'primary_tag': tags[0],
            'description': f'{tags[0]} glyph representing {", ".join(tags[1:])}'
        })
    
    # Generate additional mappings
    all_emojis = []
    for emojis in EMOJI_CATEGORIES.values():
        all_emojis.extend(emojis)
    
    used_glyphs = set(GLYPH_TAGS.keys())
    
    while len(tag_data) < num_examples:
        glyph = random.choice(all_emojis)
        if glyph in used_glyphs:
            continue
        
        # Generate plausible tags based on Unicode name
        try:
            unicode_name = unicodedata.name(glyph).lower()
            tags = unicode_name.split()
            
            # Filter and enhance tags
            tags = [t for t in tags if len(t) > 2 and t not in ['with', 'and', 'the']]
            
            if tags:
                tag_data.append({
                    'glyph': glyph,
                    'tags': tags[:4],  # Limit to 4 tags
                    'primary_tag': tags[0],
                    'description': f'{tags[0]} glyph'
                })
                used_glyphs.add(glyph)
        except:
            continue
    
    return tag_data[:num_examples]


def save_datasets(output_dir: Path):
    """Generate and save all datasets"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets
    print("Generating glyph sequences...")
    sequences = generate_glyph_sequences(1000)
    
    print("Generating composition data...")
    compositions = generate_composition_data(500)
    
    print("Generating tag data...")
    tags = generate_tag_data(300)
    
    # Save datasets
    datasets = {
        'glyph_sequences.jsonl': sequences,
        'glyph_compose.jsonl': compositions,
        'glyph_tags.jsonl': tags
    }
    
    for filename, data in datasets.items():
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(data)} examples to {filepath}")
    
    # Create dataset info
    info = {
        'datasets': {
            'sequences': {
                'file': 'glyph_sequences.jsonl',
                'count': len(sequences),
                'description': 'Natural glyph sequences for next-token prediction'
            },
            'composition': {
                'file': 'glyph_compose.jsonl',
                'count': len(compositions),
                'description': 'Glyph composition instructions and outputs'
            },
            'tags': {
                'file': 'glyph_tags.jsonl',
                'count': len(tags),
                'description': 'Glyph to semantic tag mappings'
            }
        },
        'total_examples': len(sequences) + len(compositions) + len(tags)
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nTotal examples created: {info['total_examples']}")
    print(f"Datasets saved to: {output_dir}")


if __name__ == "__main__":
    # Create datasets
    save_datasets("data/processed")
    
    # Show sample data
    print("\n=== Sample Data ===")
    
    # Load and show samples
    with open("data/processed/glyph_sequences.jsonl", 'r') as f:
        print("\nSequence sample:")
        print(json.loads(f.readline()))
    
    with open("data/processed/glyph_compose.jsonl", 'r') as f:
        print("\nComposition sample:")
        print(json.loads(f.readline()))
    
    with open("data/processed/glyph_tags.jsonl", 'r') as f:
        print("\nTag sample:")
        print(json.loads(f.readline()))