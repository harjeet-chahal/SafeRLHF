"""
PPO (Proximal Policy Optimization) trainer for RLHF.

This module handles:
- Implementing PPO algorithm for RLHF training
- Using reward model to provide rewards during training
- Updating policy model using PPO loss
- Managing training loops and checkpoints
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    RobertaModel,
    RobertaTokenizer,
    RobertaConfig,
)
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler
from tqdm import tqdm


# Import reward model class (or define it here if needed)
class SafetyRewardModel(nn.Module):
    """RoBERTa-based reward model with multiple regression heads."""
    
    def __init__(
        self,
        model_name: str = 'roberta-base',
        dropout_rate: float = 0.1,
        num_labels: int = 4
    ):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        hidden_dim = self.config.hidden_size
        
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.toxicity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.hallucination_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.compliance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        if training:
            pooled_output = self.dropout(pooled_output)
        
        return {
            'safety_score': torch.sigmoid(self.safety_head(pooled_output)).squeeze(),
            'toxicity_score': torch.sigmoid(self.toxicity_head(pooled_output)).squeeze(),
            'hallucination_risk': torch.sigmoid(self.hallucination_head(pooled_output)).squeeze(),
            'compliance_score': torch.sigmoid(self.compliance_head(pooled_output)).squeeze(),
        }
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: torch.device):
        """Load reward model from checkpoint."""
        # Load model info
        with open(os.path.join(model_path, 'model_info.json'), 'r') as f:
            model_info = json.load(f)
        
        model = cls(
            model_name=model_info['model_name'],
            dropout_rate=model_info['dropout_rate']
        )
        
        # Load state dict
        state_dict = torch.load(
            os.path.join(model_path, 'pytorch_model.bin'),
            map_location=device
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model


def setup_logging(log_file: Optional[str] = None, log_level: str = 'INFO'):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts(prompts_path: str) -> List[Dict]:
    """Load prompts from JSON file."""
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts


def load_base_policy_model(model_path: str, device: torch.device, logger):
    """Load base policy model."""
    if os.path.exists(model_path) and os.path.isdir(model_path):
        logger.info(f"Loading base policy from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        logger.info(f"Base policy path not found, using distilgpt2 as default")
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    return model, tokenizer


def load_reward_model(model_path: str, device: torch.device, logger):
    """Load reward model."""
    logger.info(f"Loading reward model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Reward model not found at {model_path}")
    
    model = SafetyRewardModel.from_pretrained(model_path, device)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    
    logger.info("Reward model loaded successfully")
    return model, tokenizer


def compute_reward(
    reward_model: SafetyRewardModel,
    reward_tokenizer: RobertaTokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
    reward_weights: Dict[str, float] = None
) -> List[float]:
    """Compute rewards using reward model."""
    if reward_weights is None:
        reward_weights = {
            'safety_score': 1.0,
            'toxicity_score': -1.0,
            'hallucination_risk': -1.0,
            'compliance_score': 0.5
        }
    
    rewards = []
    
    for prompt, response in zip(prompts, responses):
        # Concatenate prompt and response
        text = f"{prompt} {reward_tokenizer.sep_token} {response}"
        
        # Tokenize
        encoding = reward_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = reward_model(input_ids, attention_mask, training=False)
        
        # Compute weighted reward
        reward = (
            reward_weights['safety_score'] * outputs['safety_score'].item() +
            reward_weights['toxicity_score'] * outputs['toxicity_score'].item() +
            reward_weights['hallucination_risk'] * outputs['hallucination_risk'].item() +
            reward_weights['compliance_score'] * outputs['compliance_score'].item()
        )
        
        rewards.append(reward)
    
    return rewards


def build_dataset(prompts: List[Dict], tokenizer) -> Dataset:
    """Build dataset from prompts."""
    prompt_texts = [p['prompt'] for p in prompts]
    
    def tokenize(sample):
        return tokenizer(sample['prompt'], truncation=True, padding='max_length', max_length=512)
    
    dataset = Dataset.from_dict({'prompt': prompt_texts})
    dataset = dataset.map(tokenize, batched=True)
    
    return dataset


def train_ppo(
    base_model,
    base_tokenizer,
    reward_model,
    reward_tokenizer,
    prompts: List[Dict],
    device: torch.device,
    output_dir: str,
    num_epochs: int = 2,
    batch_size: int = 8,
    mini_batch_size: int = 4,
    learning_rate: float = 1.41e-5,
    kl_coef: float = 0.1,
    adaptive_kl: bool = True,
    target_kl: float = 0.1,
    reward_weights: Dict[str, float] = None,
    max_length: int = 150,
    logger: Optional[logging.Logger] = None
):
    """Train policy using PPO."""
    
    # Create reference model (frozen copy of base model for KL computation)
    import copy
    ref_model = copy.deepcopy(base_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Build dataset
    dataset = build_dataset(prompts, base_tokenizer)
    
    # PPO Config
    ppo_config = PPOConfig(
        model_name="base_policy",
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=target_kl if adaptive_kl else None,
        ppo_epochs=1,
        seed=42,
        log_with=None,  # Can use wandb or tensorboard
        tracker_project_name="saferlhf",
        init_kl_coef=kl_coef,
    )
    
    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=base_model,
        ref_model=ref_model,
        tokenizer=base_tokenizer,
        dataset=dataset,
    )
    
    # Training metrics
    training_stats = {
        'rewards': [],
        'kl_divergences': [],
        'entropies': [],
        'policy_shifts': [],
        'epochs': []
    }
    
    generation_kwargs = {
        "max_length": max_length,
        "temperature": 1.0,
        "do_sample": True,
    }
    
    # Length sampler for variable response lengths
    output_length_sampler = LengthSampler(min_length=20, max_length=max_length)
    
    logger.info(f"Starting PPO training for {num_epochs} epochs")
    logger.info(f"Total prompts: {len(prompts)}")
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        epoch_rewards = []
        epoch_kls = []
        epoch_entropies = []
        epoch_policy_shifts = []
        
        for step, batch in enumerate(tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch+1}")):
            # Get prompts
            prompt_tensors = batch['input_ids']
            prompt_texts = base_tokenizer.batch_decode(prompt_tensors, skip_special_tokens=True)
            
            # Generate responses
            response_tensors = ppo_trainer.generate(
                prompt_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs
            )
            response_texts = base_tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            
            # Compute rewards
            rewards = compute_reward(
                reward_model,
                reward_tokenizer,
                prompt_texts,
                response_texts,
                device,
                reward_weights
            )
            
            # Run PPO step (this computes KL internally)
            stats = ppo_trainer.step(
                prompt_tensors.tolist(),
                response_tensors.tolist(),
                rewards
            )
            
            # Extract metrics from stats
            current_reward = np.mean(rewards)
            current_kl = stats.get('ppo/mean/kl', 0.0)
            current_entropy = stats.get('ppo/mean/entropy', 0.0)
            current_policy_loss = stats.get('ppo/policy/mean/mean_non_score_reward', 0.0)
            
            # Policy shift is approximated by KL divergence
            policy_shift = current_kl
            
            # Update adaptive KL coefficient if enabled
            if adaptive_kl and current_kl > target_kl * 1.5:
                ppo_config.init_kl_coef = min(ppo_config.init_kl_coef * 1.5, 1.0)
                if logger:
                    logger.info(f"KL divergence ({current_kl:.4f}) exceeds target, increasing KL coefficient to {ppo_config.init_kl_coef:.4f}")
            
            # Collect metrics
            epoch_rewards.append(current_reward)
            epoch_kls.append(current_kl)
            epoch_entropies.append(current_entropy)
            epoch_policy_shifts.append(policy_shift)
            
            if (step + 1) % 10 == 0 and logger:
                logger.info(
                    f"Step {step+1}: "
                    f"Reward={current_reward:.4f}, "
                    f"KL={current_kl:.4f}, "
                    f"Entropy={current_entropy:.4f}"
                )
        
        # Epoch summary
        avg_reward = np.mean(epoch_rewards)
        avg_kl = np.mean(epoch_kls)
        avg_entropy = np.mean(epoch_entropies)
        avg_policy_shift = np.mean(epoch_policy_shifts)
        
        training_stats['rewards'].append(avg_reward)
        training_stats['kl_divergences'].append(avg_kl)
        training_stats['entropies'].append(avg_entropy)
        training_stats['policy_shifts'].append(avg_policy_shift)
        training_stats['epochs'].append(epoch + 1)
        
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"  Average Reward: {avg_reward:.4f}")
        logger.info(f"  Average KL Divergence: {avg_kl:.4f}")
        logger.info(f"  Average Entropy: {avg_entropy:.4f}")
        logger.info(f"  Average Policy Shift: {avg_policy_shift:.4f}")
    
    # Save final model
    logger.info(f"\nSaving final RL policy to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    base_model.save_pretrained(output_dir)
    base_tokenizer.save_pretrained(output_dir)
    
    # Save training statistics
    with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    # Plot training curves
    plot_training_curves(training_stats, output_dir, logger)
    
    return training_stats


def plot_training_curves(stats: Dict, output_dir: str, logger: Optional[logging.Logger] = None):
    """Plot training curves."""
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    epochs = stats['epochs']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Reward curve
    axes[0, 0].plot(epochs, stats['rewards'], marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Reward Curve')
    axes[0, 0].grid(True, alpha=0.3)
    
    # KL divergence
    axes[0, 1].plot(epochs, stats['kl_divergences'], marker='o', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].set_title('KL Divergence')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 0].plot(epochs, stats['entropies'], marker='o', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].set_title('Policy Entropy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Policy shift
    axes[1, 1].plot(epochs, stats['policy_shifts'], marker='o', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Policy Shift (KL)')
    axes[1, 1].set_title('Policy Shift')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'training_curves.png'), dpi=150)
    plt.close()
    
    if logger:
        logger.info("Training curves saved to plots/training_curves.png")


def main():
    parser = argparse.ArgumentParser(description='Train RL policy using PPO')
    parser.add_argument('--prompts_path', type=str, default='data/prompts.json',
                        help='Path to prompts JSON file')
    parser.add_argument('--base_policy_path', type=str, default='models/base_policy',
                        help='Path to base policy model')
    parser.add_argument('--reward_model_path', type=str, default='models/reward_model',
                        help='Path to reward model')
    parser.add_argument('--output_dir', type=str, default='models/rl_policy',
                        help='Output directory for RL policy')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--mini_batch_size', type=int, default=4,
                        help='Mini batch size for PPO')
    parser.add_argument('--learning_rate', type=float, default=1.41e-5,
                        help='Learning rate')
    parser.add_argument('--kl_coef', type=float, default=0.1,
                        help='KL penalty coefficient')
    parser.add_argument('--adaptive_kl', action='store_true', default=True,
                        help='Use adaptive KL coefficient')
    parser.add_argument('--target_kl', type=float, default=0.1,
                        help='Target KL divergence for adaptive KL')
    parser.add_argument('--max_length', type=int, default=150,
                        help='Maximum generation length')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to log file')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--reward_weight_safety', type=float, default=1.0,
                        help='Weight for safety_score in reward')
    parser.add_argument('--reward_weight_toxicity', type=float, default=-1.0,
                        help='Weight for toxicity_score in reward')
    parser.add_argument('--reward_weight_hallucination', type=float, default=-1.0,
                        help='Weight for hallucination_risk in reward')
    parser.add_argument('--reward_weight_compliance', type=float, default=0.5,
                        help='Weight for compliance_score in reward')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.log_level)
    
    logger.info("=" * 60)
    logger.info("PPO Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Prompts path: {args.prompts_path}")
    logger.info(f"Base policy path: {args.base_policy_path}")
    logger.info(f"Reward model path: {args.reward_model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"KL coefficient: {args.kl_coef}")
    logger.info(f"Adaptive KL: {args.adaptive_kl}")
    logger.info(f"Target KL: {args.target_kl}")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load prompts
    prompts = load_prompts(args.prompts_path)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Load models
    base_model, base_tokenizer = load_base_policy_model(args.base_policy_path, device, logger)
    reward_model, reward_tokenizer = load_reward_model(args.reward_model_path, device, logger)
    
    # Reward weights
    reward_weights = {
        'safety_score': args.reward_weight_safety,
        'toxicity_score': args.reward_weight_toxicity,
        'hallucination_risk': args.reward_weight_hallucination,
        'compliance_score': args.reward_weight_compliance
    }
    
    logger.info(f"Reward weights: {reward_weights}")
    
    # Train
    training_stats = train_ppo(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        prompts=prompts,
        device=device,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        adaptive_kl=args.adaptive_kl,
        target_kl=args.target_kl,
        reward_weights=reward_weights,
        max_length=args.max_length,
        logger=logger
    )
    
    logger.info("Training complete!")
    logger.info(f"Final average reward: {training_stats['rewards'][-1]:.4f}")
    logger.info(f"Final KL divergence: {training_stats['kl_divergences'][-1]:.4f}")


if __name__ == "__main__":
    main()
