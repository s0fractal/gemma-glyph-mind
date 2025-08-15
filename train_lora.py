#!/usr/bin/env python3
"""
LoRA fine-tuning script for Gemma-3 270M on glyph tasks
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import wandb

from src.core.glyph_adapter import GlyphAdapter, GlyphProcessor


class GlyphDataset(Dataset):
    """Multi-task dataset for glyph training"""
    
    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        max_length: int = 512,
        task_weights: Optional[Dict[str, float]] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_weights = task_weights or {
            'next_glyph': 0.4,
            'composition': 0.3,
            'alignment': 0.3
        }
        
        # Load all data
        self.sequences = []
        self.compositions = []
        self.alignments = []
        
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    
                    if 'sequence' in data:
                        self.sequences.append(data)
                    elif 'instruction' in data:
                        self.compositions.append(data)
                    elif 'tags' in data:
                        self.alignments.append(data)
        
        print(f"Loaded {len(self.sequences)} sequences, "
              f"{len(self.compositions)} compositions, "
              f"{len(self.alignments)} alignments")
    
    def __len__(self):
        return len(self.sequences) + len(self.compositions) + len(self.alignments)
    
    def __getitem__(self, idx):
        # Weighted sampling based on task
        total = len(self.sequences) + len(self.compositions) + len(self.alignments)
        
        if idx < len(self.sequences):
            return self._prepare_sequence_example(self.sequences[idx])
        elif idx < len(self.sequences) + len(self.compositions):
            comp_idx = idx - len(self.sequences)
            return self._prepare_composition_example(self.compositions[comp_idx])
        else:
            align_idx = idx - len(self.sequences) - len(self.compositions)
            return self._prepare_alignment_example(self.alignments[align_idx])
    
    def _prepare_sequence_example(self, data):
        """Prepare next-glyph prediction example"""
        text = data['sequence']
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze(),
            'task_type': 'next_glyph'
        }
    
    def _prepare_composition_example(self, data):
        """Prepare composition instruction example"""
        # Format as instruction-following
        instruction = data['instruction']
        output = data['output']
        
        text = f"Instruction: {instruction}\nOutput: {output}"
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze(),
            'task_type': 'composition'
        }
    
    def _prepare_alignment_example(self, data):
        """Prepare glyph-tag alignment example"""
        glyph = data['glyph']
        tags = ', '.join(data['tags'])
        description = data.get('description', '')
        
        text = f"Glyph: {glyph}\nTags: {tags}\nDescription: {description}"
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze(),
            'task_type': 'alignment'
        }


class GlyphTrainer(Trainer):
    """Custom trainer with glyph-specific metrics"""
    
    def __init__(self, glyph_adapter=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.glyph_adapter = glyph_adapter
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with task-specific weighting"""
        # Get task type
        task_type = inputs.pop('task_type', 'next_glyph')
        
        # Standard forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Apply task-specific loss weighting
        task_weights = {
            'next_glyph': 1.0,
            'composition': 1.2,  # Slightly higher weight for harder task
            'alignment': 1.1
        }
        
        loss = loss * task_weights.get(task_type, 1.0)
        
        return (loss, outputs) if return_outputs else loss


def load_config(config_path: str) -> Dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: Dict):
    """Setup Gemma model with LoRA and glyph adapter"""
    model_name = config['model']['name']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config['training']['fp16'] else torch.float32,
        device_map="auto",
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules']
    )
    
    model = get_peft_model(model, lora_config)
    
    # Add glyph adapter if enabled
    glyph_adapter = None
    if config['model']['adapter']['enabled']:
        glyph_adapter = GlyphAdapter(
            vocab_size=config['model']['vocab_size'],
            hidden_size=config['model']['hidden_size'],
            adapter_size=config['model']['adapter']['adapter_size'],
            max_cluster_len=config['model']['adapter']['max_cluster_len']
        )
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer, glyph_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/glyph_training.yaml',
        help='Path to training config'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup wandb if enabled
    if config['output']['wandb']['enabled']:
        wandb.init(
            project=config['output']['wandb']['project'],
            entity=config['output']['wandb']['entity'],
            tags=config['output']['wandb']['tags'],
            config=config
        )
    
    # Setup model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer, glyph_adapter = setup_model_and_tokenizer(config)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = GlyphDataset(
        data_files=config['datasets']['train_files'],
        tokenizer=tokenizer,
        max_length=config['training']['max_length'],
        task_weights=config['datasets']['task_weights']
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output']['checkpoint_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['output']['logging_steps'],
        save_steps=config['output']['save_steps'],
        save_total_limit=config['output']['save_total_limit'],
        fp16=config['training']['fp16'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        optim=config['training']['optimizer'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['scheduler'],
        report_to="wandb" if config['output']['wandb']['enabled'] else "none",
        logging_dir=config['output']['logging_dir'],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )
    
    # Create trainer
    trainer = GlyphTrainer(
        glyph_adapter=glyph_adapter,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("Starting training...")
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(config['output']['checkpoint_dir'] + "/final")
    
    # Save glyph adapter if used
    if glyph_adapter:
        adapter_path = Path(config['output']['checkpoint_dir']) / "final" / "glyph_adapter.pt"
        glyph_adapter.save_adapter(adapter_path)
    
    print("Training complete!")


if __name__ == "__main__":
    main()