"""
Train reward model for RLHF.

This module handles:
- Loading training data (prompts, responses, safety labels)
- Training a reward model to predict safety scores
- Evaluating reward model performance
- Saving trained reward model checkpoints
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    RobertaModel,
    RobertaTokenizer,
    RobertaConfig,
    get_linear_schedule_with_warmup,
    AdamW
)
from tqdm import tqdm


class SafetyRewardDataset(Dataset):
    """Dataset for safety reward model training."""
    
    def __init__(
        self,
        prompts: List[str],
        responses: List[str],
        safety_scores: List[float],
        toxicity_scores: List[float],
        hallucination_risks: List[float],
        compliance_scores: List[float],
        tokenizer: RobertaTokenizer,
        max_length: int = 512
    ):
        self.prompts = prompts
        self.responses = responses
        self.safety_scores = safety_scores
        self.toxicity_scores = toxicity_scores
        self.hallucination_risks = hallucination_risks
        self.compliance_scores = compliance_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        
        # Concatenate prompt and response
        text = f"{prompt} {self.tokenizer.sep_token} {response}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'safety_score': torch.tensor(self.safety_scores[idx], dtype=torch.float32),
            'toxicity_score': torch.tensor(self.toxicity_scores[idx], dtype=torch.float32),
            'hallucination_risk': torch.tensor(self.hallucination_risks[idx], dtype=torch.float32),
            'compliance_score': torch.tensor(self.compliance_scores[idx], dtype=torch.float32),
        }


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
        
        # Dropout for MC Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Hidden dimension
        hidden_dim = self.config.hidden_size
        
        # Regression heads for each signal
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
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional MC Dropout."""
        # Get RoBERTa embeddings
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout (always during training, controlled during inference for MC Dropout)
        if training:
            pooled_output = self.dropout(pooled_output)
        
        # Get predictions from each head
        safety_pred = torch.sigmoid(self.safety_head(pooled_output))
        toxicity_pred = torch.sigmoid(self.toxicity_head(pooled_output))
        hallucination_pred = torch.sigmoid(self.hallucination_head(pooled_output))
        compliance_pred = torch.sigmoid(self.compliance_head(pooled_output))
        
        return {
            'safety_score': safety_pred.squeeze(),
            'toxicity_score': toxicity_pred.squeeze(),
            'hallucination_risk': hallucination_pred.squeeze(),
            'compliance_score': compliance_pred.squeeze(),
        }
    
    def forward_mc_dropout(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with MC Dropout for uncertainty estimation."""
        self.train()  # Enable dropout
        predictions = {
            'safety_score': [],
            'toxicity_score': [],
            'hallucination_risk': [],
            'compliance_score': [],
        }
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                pooled_output = outputs.pooler_output
                pooled_output = self.dropout(pooled_output)  # Apply dropout
                
                safety_pred = torch.sigmoid(self.safety_head(pooled_output))
                toxicity_pred = torch.sigmoid(self.toxicity_head(pooled_output))
                hallucination_pred = torch.sigmoid(self.hallucination_head(pooled_output))
                compliance_pred = torch.sigmoid(self.compliance_head(pooled_output))
                
                predictions['safety_score'].append(safety_pred.squeeze())
                predictions['toxicity_score'].append(toxicity_pred.squeeze())
                predictions['hallucination_risk'].append(hallucination_pred.squeeze())
                predictions['compliance_score'].append(compliance_pred.squeeze())
        
        # Stack and compute statistics
        result = {}
        for key, values in predictions.items():
            stacked = torch.stack(values, dim=0)
            result[key] = {
                'mean': torch.mean(stacked, dim=0),
                'std': torch.std(stacked, dim=0),
                'samples': stacked
            }
        
        return result


def load_data(csv_path: str) -> Tuple[List, List, List, List, List, List]:
    """Load data from CSV file."""
    prompts = []
    responses = []
    safety_scores = []
    toxicity_scores = []
    hallucination_risks = []
    compliance_scores = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row['prompt'])
            responses.append(row['response'])
            safety_scores.append(float(row['safety_score']))
            toxicity_scores.append(float(row['toxicity_score']))
            hallucination_risks.append(float(row['hallucination_risk']))
            compliance_scores.append(float(row['compliance_score']))
    
    return prompts, responses, safety_scores, toxicity_scores, hallucination_risks, compliance_scores


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(predictions - targets))


def plot_calibration_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    signal_name: str,
    save_path: str
):
    """Plot calibration curve for a regression signal (reliability diagram)."""
    # For regression, create bins and compute mean predicted vs mean actual
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_centers = []
    bin_means_true = []
    bin_means_pred = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_means_true.append(np.mean(y_true[mask]))
            bin_means_pred.append(np.mean(y_pred[mask]))
            bin_counts.append(np.sum(mask))
    
    bin_centers = np.array(bin_centers)
    bin_means_true = np.array(bin_means_true)
    bin_means_pred = np.array(bin_means_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(bin_means_pred, bin_means_true, marker='o', label=f'{signal_name}', linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    plt.xlabel('Mean Predicted Value', fontsize=12)
    plt.ylabel('Mean Actual Value', fontsize=12)
    plt.title(f'Calibration Curve (Reliability Diagram): {signal_name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_model(
    model: SafetyRewardModel,
    dataloader: DataLoader,
    device: torch.device,
    signal_names: List[str] = ['safety_score', 'toxicity_score', 'hallucination_risk', 'compliance_score']
) -> Dict:
    """Evaluate model and compute metrics."""
    model.eval()
    
    all_predictions = {name: [] for name in signal_names}
    all_targets = {name: [] for name in signal_names}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask, training=False)
            
            for name in signal_names:
                all_predictions[name].extend(
                    outputs[name].cpu().numpy()
                )
                all_targets[name].extend(
                    batch[name].cpu().numpy()
                )
    
    # Compute MAE for each signal
    metrics = {}
    for name in signal_names:
        preds = np.array(all_predictions[name])
        targets = np.array(all_targets[name])
        mae = compute_mae(preds, targets)
        metrics[f'{name}_mae'] = mae
        metrics[f'{name}_predictions'] = preds
        metrics[f'{name}_targets'] = targets
    
    return metrics


def train(
    model: SafetyRewardModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: RobertaTokenizer,
    model_name: str,
    dropout_rate: float,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    output_dir: str = 'models/reward_model'
):
    """Train the reward model."""
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Loss function (MSE for each head)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    signal_names = ['safety_score', 'toxicity_score', 'hallucination_risk', 'compliance_score']
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_losses_by_signal = {name: 0.0 for name in signal_names}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, training=True)
            
            # Compute loss for each signal
            total_loss = 0.0
            for name in signal_names:
                pred = outputs[name]
                target = batch[name].to(device)
                loss = criterion(pred, target)
                total_loss += loss
                train_losses_by_signal[name] += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += total_loss.item()
            progress_bar.set_postfix({'loss': total_loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_losses_by_signal = {name: 0.0 for name in signal_names}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask, training=False)
                
                total_loss = 0.0
                for name in signal_names:
                    pred = outputs[name]
                    target = batch[name].to(device)
                    loss = criterion(pred, target)
                    total_loss += loss
                    val_losses_by_signal[name] += loss.item()
                
                val_loss += total_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print("Train Losses by Signal:")
        for name in signal_names:
            print(f"  {name}: {train_losses_by_signal[name]/len(train_loader):.4f}")
        print("Val Losses by Signal:")
        for name in signal_names:
            print(f"  {name}: {val_losses_by_signal[name]/len(val_loader):.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving best model (val_loss: {best_val_loss:.4f})")
            os.makedirs(output_dir, exist_ok=True)
            # Save model state dict
            torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            # Save model config (RoBERTa config)
            model.roberta.config.save_pretrained(output_dir)
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)
            # Save model architecture info
            model_info = {
                'model_name': model_name,
                'dropout_rate': dropout_rate,
                'num_labels': 4
            }
            with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
    
    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train safety reward model')
    parser.add_argument('--data_path', type=str, default='data/safety_labels.csv',
                        help='Path to safety labels CSV file')
    parser.add_argument('--output_dir', type=str, default='models/reward_model',
                        help='Output directory for saved model')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        help='Base model name')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    prompts, responses, safety_scores, toxicity_scores, hallucination_risks, compliance_scores = load_data(args.data_path)
    print(f"Loaded {len(prompts)} samples")
    
    # Split data
    train_prompts, temp_prompts, train_responses, temp_responses, \
    train_safety, temp_safety, train_toxicity, temp_toxicity, \
    train_hallucination, temp_hallucination, train_compliance, temp_compliance = train_test_split(
        prompts, responses, safety_scores, toxicity_scores,
        hallucination_risks, compliance_scores,
        test_size=args.test_size + args.val_size,
        random_state=args.seed
    )
    
    val_size_adjusted = args.val_size / (args.test_size + args.val_size)
    val_prompts, test_prompts, val_responses, test_responses, \
    val_safety, test_safety, val_toxicity, test_toxicity, \
    val_hallucination, test_hallucination, val_compliance, test_compliance = train_test_split(
        temp_prompts, temp_responses, temp_safety, temp_toxicity,
        temp_hallucination, temp_compliance,
        test_size=1 - val_size_adjusted,
        random_state=args.seed
    )
    
    print(f"Train: {len(train_prompts)}, Val: {len(val_prompts)}, Test: {len(test_prompts)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = SafetyRewardDataset(
        train_prompts, train_responses,
        train_safety, train_toxicity, train_hallucination, train_compliance,
        tokenizer, args.max_length
    )
    val_dataset = SafetyRewardDataset(
        val_prompts, val_responses,
        val_safety, val_toxicity, val_hallucination, val_compliance,
        tokenizer, args.max_length
    )
    test_dataset = SafetyRewardDataset(
        test_prompts, test_responses,
        test_safety, test_toxicity, test_hallucination, test_compliance,
        tokenizer, args.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = SafetyRewardModel(
        model_name=args.model_name,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # Train model
    print("Training model...")
    train_losses, val_losses = train(
        model, train_loader, val_loader, device, tokenizer,
        args.model_name, args.dropout_rate,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    print("\nTest Set MAE:")
    signal_names = ['safety_score', 'toxicity_score', 'hallucination_risk', 'compliance_score']
    for name in signal_names:
        mae = test_metrics[f'{name}_mae']
        print(f"  {name}: {mae:.4f}")
    
    # Plot calibration curves
    print("\nGenerating calibration curves...")
    os.makedirs(f"{args.output_dir}/plots", exist_ok=True)
    for name in signal_names:
        plot_calibration_curves(
            test_metrics[f'{name}_targets'],
            test_metrics[f'{name}_predictions'],
            name,
            f"{args.output_dir}/plots/calibration_{name}.png"
        )
    
    # MC Dropout uncertainty estimation on a sample
    print("\nComputing MC Dropout uncertainty estimates...")
    model.eval()
    sample_batch = next(iter(test_loader))
    sample_input_ids = sample_batch['input_ids'][:5].to(device)
    sample_attention_mask = sample_batch['attention_mask'][:5].to(device)
    
    mc_results = model.forward_mc_dropout(sample_input_ids, sample_attention_mask, n_samples=20)
    
    uncertainty_results = {}
    for name in signal_names:
        mean_preds = mc_results[name]['mean'].cpu().numpy()
        std_preds = mc_results[name]['std'].cpu().numpy()
        uncertainty_results[name] = {
            'mean': mean_preds,
            'std': std_preds,
            'uncertainty': std_preds  # Standard deviation as uncertainty measure
        }
        print(f"\n{name} - Sample predictions with uncertainty:")
        for i in range(min(5, len(mean_preds))):
            print(f"  Sample {i+1}: {mean_preds[i]:.4f} ± {std_preds[i]:.4f}")
    
    # Save uncertainty results
    with open(f"{args.output_dir}/uncertainty_results.json", 'w') as f:
        json.dump(
            {k: {'mean': v['mean'].tolist(), 'std': v['std'].tolist()}
             for k, v in uncertainty_results.items()},
            f, indent=2
        )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics': {k: float(v) for k, v in test_metrics.items() if 'mae' in k}
    }
    with open(f"{args.output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
