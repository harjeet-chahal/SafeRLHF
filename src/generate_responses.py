"""
Generate responses using base policy model for RLHF training.

This module handles:
- Loading prompts from data files
- Generating responses using the base policy model
- Saving generated responses for reward model training
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import random

import torch
import numpy as np
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)
from tqdm import tqdm


def setup_logging(log_file: Optional[str] = None, log_level: str = 'INFO'):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler()]
    if log_file:
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
    logging.info(f"Random seed set to {seed}")


def load_prompts(prompts_path: str) -> List[Dict]:
    """Load prompts from JSON file."""
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    logging.info(f"Loaded {len(prompts)} prompts from {prompts_path}")
    return prompts


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    logger: logging.Logger
):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Try loading as GPT2 model first
        if 'gpt2' in model_name.lower() or 'distilgpt2' in model_name.lower():
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            # Fallback to AutoModel
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully. Device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_length: int = 150,
    min_length: int = 10,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    do_sample: bool = True,
    repetition_penalty: float = 1.2,
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """Generate response(s) for a given prompt."""
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generation parameters
    generation_config = {
        'max_length': max_length,
        'min_length': min_length,
        'temperature': temperature,
        'top_k': top_k if top_k > 0 else None,
        'top_p': top_p if top_p < 1.0 else None,
        'num_return_sequences': num_return_sequences,
        'do_sample': do_sample,
        'repetition_penalty': repetition_penalty,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    
    # Remove None values
    generation_config = {k: v for k, v in generation_config.items() if v is not None}
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                **generation_config
            )
        
        # Decode generated text
        generated_texts = []
        for output in outputs:
            # Remove the input prompt from the generated text
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            # Remove the original prompt if it's at the start
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            generated_texts.append(generated_text)
        
        return generated_texts
    
    except Exception as e:
        if logger:
            logger.error(f"Error generating response for prompt '{prompt[:50]}...': {e}")
        return [""] * num_return_sequences


def generate_responses_for_prompts(
    model,
    tokenizer,
    prompts: List[Dict],
    device: torch.device,
    num_responses_per_prompt: int = 3,
    max_length: int = 150,
    min_length: int = 10,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """Generate responses for all prompts."""
    
    results = []
    
    for prompt_data in tqdm(prompts, desc="Generating responses"):
        prompt_id = prompt_data.get('id', None)
        prompt_text = prompt_data.get('prompt', '')
        
        if not prompt_text:
            if logger:
                logger.warning(f"Skipping prompt {prompt_id}: empty prompt text")
            continue
        
        # Generate multiple responses
        responses = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            device=device,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_responses_per_prompt,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            logger=logger
        )
        
        # Store results
        result = {
            'prompt_id': prompt_id,
            'prompt': prompt_text,
            'responses': responses,
            'num_responses': len(responses),
            'generation_params': {
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'max_length': max_length,
                'min_length': min_length,
                'repetition_penalty': repetition_penalty,
            }
        }
        
        results.append(result)
        
        if logger and len(results) % 50 == 0:
            logger.info(f"Generated responses for {len(results)} prompts")
    
    return results


def save_outputs(outputs: List[Dict], output_path: str, logger: Optional[logging.Logger] = None):
    """Save generated responses to JSON file."""
    output_data = {
        'metadata': {
            'total_prompts': len(outputs),
            'total_responses': sum(len(o['responses']) for o in outputs),
            'generation_timestamp': datetime.now().isoformat(),
        },
        'outputs': outputs
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    if logger:
        logger.info(f"Saved {len(outputs)} prompt outputs to {output_path}")
        logger.info(f"Total responses generated: {output_data['metadata']['total_responses']}")


def main():
    parser = argparse.ArgumentParser(description='Generate responses using base policy model')
    parser.add_argument('--prompts_path', type=str, default='data/prompts.json',
                        help='Path to prompts JSON file')
    parser.add_argument('--output_path', type=str, default='data/base_policy_outputs.json',
                        help='Path to save generated responses')
    parser.add_argument('--model_name', type=str, default='distilgpt2',
                        choices=['distilgpt2', 'gpt2', 'gpt2-small'],
                        help='Model to use for generation')
    parser.add_argument('--num_responses', type=int, default=3,
                        help='Number of responses to generate per prompt')
    parser.add_argument('--max_length', type=int, default=150,
                        help='Maximum generation length')
    parser.add_argument('--min_length', type=int, default=10,
                        help='Minimum generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling (0 to disable)')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (nucleus) sampling (1.0 to disable)')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                        help='Repetition penalty (1.0 = no penalty)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for generation (currently not used, generates one at a time)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to log file (optional)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detected if not specified')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.log_level)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("Response Generation Configuration")
    logger.info("=" * 60)
    logger.info(f"Prompts path: {args.prompts_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Responses per prompt: {args.num_responses}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Min length: {args.min_length}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Top-k: {args.top_k}")
    logger.info(f"Top-p: {args.top_p}")
    logger.info(f"Repetition penalty: {args.repetition_penalty}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Map gpt2-small to gpt2
    model_name = 'gpt2' if args.model_name == 'gpt2-small' else args.model_name
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device, logger)
    
    # Load prompts
    prompts = load_prompts(args.prompts_path)
    
    if not prompts:
        logger.error("No prompts found!")
        return
    
    # Generate responses
    logger.info(f"Starting generation for {len(prompts)} prompts...")
    outputs = generate_responses_for_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        num_responses_per_prompt=args.num_responses,
        max_length=args.max_length,
        min_length=args.min_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        logger=logger
    )
    
    # Save outputs
    save_outputs(outputs, args.output_path, logger)
    
    logger.info("Generation complete!")
    
    # Print summary statistics
    total_responses = sum(len(o['responses']) for o in outputs)
    avg_response_length = np.mean([
        len(response)
        for output in outputs
        for response in output['responses']
    ])
    
    logger.info("=" * 60)
    logger.info("Generation Summary")
    logger.info("=" * 60)
    logger.info(f"Total prompts processed: {len(outputs)}")
    logger.info(f"Total responses generated: {total_responses}")
    logger.info(f"Average response length: {avg_response_length:.1f} characters")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
