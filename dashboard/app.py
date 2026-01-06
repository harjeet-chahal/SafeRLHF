"""
Streamlit dashboard for SafeRLHF project.

This module provides a web interface for:
- Visualizing training progress
- Exploring generated responses
- Analyzing safety metrics
- Comparing model performance
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import model loading functions
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GPT2LMHeadModel,
        GPT2Tokenizer,
        RobertaModel,
        RobertaTokenizer,
        RobertaConfig,
    )
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("PyTorch not available. Interactive prompt tester will be disabled.")


# Page config
st.set_page_config(
    page_title="SafeRLHF Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_evaluation_results(file_path: str) -> Optional[Dict]:
    """Load evaluation results from JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_training_stats(file_path: str) -> Optional[Dict]:
    """Load training statistics from JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_probe_results(file_path: str) -> Optional[Dict]:
    """Load probe results from JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None


@st.cache_resource
def load_model_for_inference(model_path: str, device: str = 'cpu'):
    """Load model for interactive testing (cached)."""
    if not TORCH_AVAILABLE:
        return None, None
    
    if not os.path.exists(model_path):
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        return None, None


@st.cache_resource
def load_reward_model_for_inference(model_path: str, device: str = 'cpu'):
    """Load reward model for interactive testing (cached)."""
    if not TORCH_AVAILABLE:
        return None, None
    
    if not os.path.exists(model_path):
        return None, None
    
    try:
        # Import reward model class
        from src.evaluate_safety import SafetyRewardModel
        reward_model = SafetyRewardModel.from_pretrained(model_path, torch.device(device))
        reward_tokenizer = RobertaTokenizer.from_pretrained(model_path)
        return reward_model, reward_tokenizer
    except Exception as e:
        return None, None


def compute_reward_scores_interactive(reward_model, reward_tokenizer, prompt: str, response: str, device: str):
    """Compute reward scores for interactive testing."""
    if reward_model is None or reward_tokenizer is None:
        return None
    
    text = f"{prompt} {reward_tokenizer.sep_token} {response}"
    encoding = reward_tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = reward_model(input_ids, attention_mask, training=False)
    
    reward = (
        outputs['safety_score'].item() -
        outputs['toxicity_score'].item() -
        outputs['hallucination_risk'].item() +
        0.5 * outputs['compliance_score'].item()
    )
    
    return {
        'safety_score': outputs['safety_score'].item(),
        'toxicity_score': outputs['toxicity_score'].item(),
        'hallucination_risk': outputs['hallucination_risk'].item(),
        'compliance_score': outputs['compliance_score'].item(),
        'reward': reward
    }


def generate_response_cached(model, tokenizer, prompt: str, max_length: int = 150):
    """Generate response from model."""
    if model is None or tokenizer is None:
        return "Model not available"
    
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text


def main():
    # Header
    st.markdown('<div class="main-header">🛡️ SafeRLHF Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Section",
        [
            "Overview",
            "Before/After Comparison",
            "Training Metrics",
            "Safety Metrics",
            "Calibration Plots",
            "Safety Probes",
            "Interactive Prompt Tester"
        ]
    )
    
    # File paths configuration
    st.sidebar.markdown("### Configuration")
    eval_results_path = st.sidebar.text_input(
        "Evaluation Results Path",
        value="evaluation_results/evaluation_results.json"
    )
    training_stats_path = st.sidebar.text_input(
        "Training Stats Path",
        value="models/rl_policy/training_stats.json"
    )
    probe_results_path = st.sidebar.text_input(
        "Probe Results Path",
        value="probes_report.json"
    )
    base_model_path = st.sidebar.text_input(
        "Base Model Path",
        value="models/base_policy"
    )
    rl_model_path = st.sidebar.text_input(
        "RL Model Path",
        value="models/rl_policy"
    )
    reward_model_path = st.sidebar.text_input(
        "Reward Model Path",
        value="models/reward_model"
    )
    
    # Main content based on selected page
    if page == "Overview":
        show_overview()
    elif page == "Before/After Comparison":
        show_before_after_comparison(eval_results_path)
    elif page == "Training Metrics":
        show_training_metrics(training_stats_path)
    elif page == "Safety Metrics":
        show_safety_metrics(eval_results_path)
    elif page == "Calibration Plots":
        show_calibration_plots()
    elif page == "Safety Probes":
        show_safety_probes(probe_results_path)
    elif page == "Interactive Prompt Tester":
        show_interactive_tester(base_model_path, rl_model_path, reward_model_path)


def show_overview():
    """Show overview of RLHF pipeline."""
    st.header("RLHF Pipeline Overview")
    
    st.markdown("""
    ### Safe Reinforcement Learning from Human Feedback (SafeRLHF)
    
    This dashboard provides a comprehensive view of the SafeRLHF training pipeline and evaluation results.
    """)
    
    # Pipeline diagram
    st.subheader("Pipeline Architecture")
    
    pipeline_steps = [
        ("1. Data Generation", "Generate prompts and synthetic responses with safety labels"),
        ("2. Reward Model Training", "Train RoBERTa-based reward model to predict safety scores"),
        ("3. RLHF Training", "Use PPO to train policy model with reward shaping"),
        ("4. Evaluation", "Comprehensive safety evaluation and probe testing"),
    ]
    
    for step, description in pipeline_steps:
        with st.expander(step, expanded=True):
            st.write(description)
    
    # Key metrics summary
    st.subheader("Quick Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Try to load evaluation results for quick metrics
    eval_results = load_evaluation_results("evaluation_results/evaluation_results.json")
    if eval_results:
        with col1:
            st.metric(
                "Unsafe Outputs Reduction",
                f"{eval_results.get('unsafe_outputs', {}).get('improvement_pct', 0):.1f}%"
            )
        
        with col2:
            st.metric(
                "Toxicity Reduction",
                f"{eval_results.get('toxicity_reduction', {}).get('reduction_pct', 0):.1f}%"
            )
        
        with col3:
            st.metric(
                "Hallucination Reduction",
                f"{eval_results.get('hallucination_reduction', {}).get('reduction_pct', 0):.1f}%"
            )
        
        with col4:
            st.metric(
                "Compliance Improvement",
                f"{eval_results.get('compliance_improvement', {}).get('improvement_pct', 0):.1f}%"
            )
    else:
        st.info("Run evaluation to see metrics. Use the sidebar to configure file paths.")


def show_before_after_comparison(eval_results_path: str):
    """Show before/after comparison of model outputs."""
    st.header("Before/After Model Comparison")
    
    eval_results = load_evaluation_results(eval_results_path)
    
    if not eval_results:
        st.warning(f"Evaluation results not found at {eval_results_path}")
        st.info("Run `python src/evaluate_safety.py` to generate evaluation results.")
        return
    
    # Key comparison metrics
    st.subheader("Key Improvements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unsafe_base = eval_results['unsafe_outputs']['base_model_pct']
        unsafe_rl = eval_results['unsafe_outputs']['rl_model_pct']
        st.metric("Unsafe Outputs", f"{unsafe_rl:.1f}%", f"{unsafe_base - unsafe_rl:.1f}% reduction")
    
    with col2:
        reward_base = eval_results['reward_distribution']['base_mean']
        reward_rl = eval_results['reward_distribution']['rl_mean']
        st.metric("Average Reward", f"{reward_rl:.3f}", f"{reward_rl - reward_base:.3f} improvement")
    
    with col3:
        kl_div = eval_results['kl_divergence']['mean_kl']
        st.metric("KL Divergence", f"{kl_div:.4f}", "Policy shift from base")
    
    # Comparison charts
    st.subheader("Metric Comparisons")
    
    # Toxicity comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Base Model', 'RL Model'],
        y=[
            eval_results['toxicity_reduction']['base_mean'],
            eval_results['toxicity_reduction']['rl_mean']
        ],
        name='Toxicity Score',
        marker_color=['#ff7f0e', '#2ca02c']
    ))
    fig.update_layout(
        title='Toxicity Score Comparison',
        yaxis_title='Toxicity Score',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Hallucination comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Base Model', 'RL Model'],
        y=[
            eval_results['hallucination_reduction']['base_mean'],
            eval_results['hallucination_reduction']['rl_mean']
        ],
        name='Hallucination Risk',
        marker_color=['#ff7f0e', '#2ca02c']
    ))
    fig.update_layout(
        title='Hallucination Risk Comparison',
        yaxis_title='Hallucination Risk',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Compliance comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Base Model', 'RL Model'],
        y=[
            eval_results['compliance_improvement']['base_mean'],
            eval_results['compliance_improvement']['rl_mean']
        ],
        name='Compliance Score',
        marker_color=['#ff7f0e', '#2ca02c']
    ))
    fig.update_layout(
        title='Compliance Score Comparison',
        yaxis_title='Compliance Score',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def show_training_metrics(training_stats_path: str):
    """Show reward curves and KL divergence plots."""
    st.header("Training Metrics")
    
    training_stats = load_training_stats(training_stats_path)
    
    if not training_stats:
        st.warning(f"Training statistics not found at {training_stats_path}")
        st.info("Training stats are saved during PPO training.")
        return
    
    # Reward curve
    st.subheader("Reward Curve")
    if 'rewards' in training_stats and 'epochs' in training_stats:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=training_stats['epochs'],
            y=training_stats['rewards'],
            mode='lines+markers',
            name='Average Reward',
            line=dict(color='#1f77b4', width=3)
        ))
        fig.update_layout(
            title='Reward Over Training',
            xaxis_title='Epoch',
            yaxis_title='Average Reward',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # KL divergence
    st.subheader("KL Divergence")
    if 'kl_divergences' in training_stats and 'epochs' in training_stats:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=training_stats['epochs'],
            y=training_stats['kl_divergences'],
            mode='lines+markers',
            name='KL Divergence',
            line=dict(color='#ff7f0e', width=3)
        ))
        fig.update_layout(
            title='KL Divergence Over Training',
            xaxis_title='Epoch',
            yaxis_title='KL Divergence',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Entropy
    st.subheader("Policy Entropy")
    if 'entropies' in training_stats and 'epochs' in training_stats:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=training_stats['epochs'],
            y=training_stats['entropies'],
            mode='lines+markers',
            name='Entropy',
            line=dict(color='#2ca02c', width=3)
        ))
        fig.update_layout(
            title='Policy Entropy Over Training',
            xaxis_title='Epoch',
            yaxis_title='Entropy',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Policy shift
    st.subheader("Policy Shift")
    if 'policy_shifts' in training_stats and 'epochs' in training_stats:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=training_stats['epochs'],
            y=training_stats['policy_shifts'],
            mode='lines+markers',
            name='Policy Shift',
            line=dict(color='#d62728', width=3)
        ))
        fig.update_layout(
            title='Policy Shift Over Training',
            xaxis_title='Epoch',
            yaxis_title='Policy Shift (KL)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def show_safety_metrics(eval_results_path: str):
    """Show safety metrics."""
    st.header("Safety Metrics")
    
    eval_results = load_evaluation_results(eval_results_path)
    
    if not eval_results:
        st.warning(f"Evaluation results not found at {eval_results_path}")
        return
    
    # Metrics cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Toxicity Reduction",
            f"{eval_results['toxicity_reduction']['reduction_pct']:.1f}%",
            f"From {eval_results['toxicity_reduction']['base_mean']:.3f} to {eval_results['toxicity_reduction']['rl_mean']:.3f}"
        )
    
    with col2:
        st.metric(
            "Hallucination Reduction",
            f"{eval_results['hallucination_reduction']['reduction_pct']:.1f}%",
            f"From {eval_results['hallucination_reduction']['base_mean']:.3f} to {eval_results['hallucination_reduction']['rl_mean']:.3f}"
        )
    
    with col3:
        st.metric(
            "Compliance Improvement",
            f"{eval_results['compliance_improvement']['improvement_pct']:.1f}%",
            f"From {eval_results['compliance_improvement']['base_mean']:.3f} to {eval_results['compliance_improvement']['rl_mean']:.3f}"
        )
    
    # Improvement summary chart
    st.subheader("Improvement Summary")
    
    improvements = [
        eval_results['unsafe_outputs']['improvement_pct'],
        eval_results['toxicity_reduction']['reduction_pct'],
        eval_results['hallucination_reduction']['reduction_pct'],
        eval_results['compliance_improvement']['improvement_pct'],
        eval_results['adversarial_robustness']['improvement_pct']
    ]
    
    labels = [
        'Unsafe Outputs\nReduction',
        'Toxicity\nReduction',
        'Hallucination\nReduction',
        'Compliance\nImprovement',
        'Adversarial\nRobustness'
    ]
    
    fig = go.Figure()
    colors = ['green' if x > 0 else 'red' for x in improvements]
    fig.add_trace(go.Bar(
        x=labels,
        y=improvements,
        marker_color=colors,
        text=[f"{x:.1f}%" for x in improvements],
        textposition='outside'
    ))
    fig.update_layout(
        title='RLHF Training Improvements',
        yaxis_title='Improvement (%)',
        height=500,
        xaxis_tickangle=-45
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)


def show_calibration_plots():
    """Show calibration plots."""
    st.header("Calibration Plots")
    
    # Try to load calibration plot images
    plot_dir = "evaluation_results/plots"
    
    if os.path.exists(plot_dir):
        calibration_files = [
            f for f in os.listdir(plot_dir)
            if f.startswith('reliability_') and f.endswith('.png')
        ]
        
        if calibration_files:
            st.subheader("Reliability Diagrams")
            
            for cal_file in sorted(calibration_files):
                signal_name = cal_file.replace('reliability_', '').replace('.png', '')
                st.write(f"**{signal_name.replace('_', ' ').title()}**")
                st.image(os.path.join(plot_dir, cal_file), use_container_width=True)
        else:
            st.info("No calibration plots found. Run evaluation to generate them.")
    else:
        st.info("Calibration plots directory not found. Run evaluation to generate plots.")


def show_safety_probes(probe_results_path: str):
    """Show safety probe results."""
    st.header("Safety Probe Results")
    
    probe_results = load_probe_results(probe_results_path)
    
    if not probe_results:
        st.warning(f"Probe results not found at {probe_results_path}")
        st.info("Run `python src/safety_probes.py` to generate probe results.")
        return
    
    # Overall risk scores
    st.subheader("Overall Risk Scores")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_risk = probe_results['summary']['base_model_overall_risk']
        st.metric("Base Model Risk", f"{base_risk:.3f}")
    
    with col2:
        rl_risk = probe_results['summary']['rl_model_overall_risk']
        st.metric("RL Model Risk", f"{rl_risk:.3f}")
    
    with col3:
        improvement = probe_results['summary']['risk_improvement']
        st.metric("Risk Improvement", f"{improvement:.3f}")
    
    # Per-probe comparison
    st.subheader("Per-Probe Risk Scores")
    
    probe_types = ['reward_hacking', 'adversarial_prompt', 'borderline_unsafe',
                   'over_optimization', 'refusal_pattern', 'semantic_jailbreak']
    
    base_risks = [probe_results['base_model'][pt]['risk_score'] for pt in probe_types]
    rl_risks = [probe_results['rl_model'][pt]['risk_score'] for pt in probe_types]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[pt.replace('_', ' ').title() for pt in probe_types],
        y=base_risks,
        name='Base Model',
        marker_color='#ff7f0e'
    ))
    fig.add_trace(go.Bar(
        x=[pt.replace('_', ' ').title() for pt in probe_types],
        y=rl_risks,
        name='RL Model',
        marker_color='#2ca02c'
    ))
    fig.update_layout(
        title='Risk Scores by Probe Type',
        xaxis_title='Probe Type',
        yaxis_title='Risk Score',
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Failure examples
    st.subheader("Failure Examples")
    
    probe_type_select = st.selectbox(
        "Select Probe Type",
        probe_types,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    model_select = st.selectbox("Select Model", ["Base Model", "RL Model"])
    
    model_key = 'base_model' if model_select == "Base Model" else 'rl_model'
    failures = probe_results[model_key][probe_type_select].get('failures', [])
    
    if failures:
        for i, failure in enumerate(failures, 1):
            with st.expander(f"Failure Example {i}"):
                st.write("**Prompt:**", failure['prompt'])
                st.write("**Response:**", failure['response'])
                st.write("**Scores:**")
                scores = failure.get('scores', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Safety", f"{scores.get('safety_score', 0):.3f}")
                with col2:
                    st.metric("Toxicity", f"{scores.get('toxicity_score', 0):.3f}")
                with col3:
                    st.metric("Hallucination", f"{scores.get('hallucination_risk', 0):.3f}")
                with col4:
                    st.metric("Compliance", f"{scores.get('compliance_score', 0):.3f}")
    else:
        st.info(f"No failures recorded for {probe_type_select} on {model_select}")


def show_interactive_tester(base_model_path: str, rl_model_path: str, reward_model_path: str):
    """Show interactive prompt tester."""
    st.header("Interactive Prompt Tester")
    
    if not TORCH_AVAILABLE:
        st.error("PyTorch is not available. Cannot load models for interactive testing.")
        return
    
    # Prompt input
    user_prompt = st.text_area(
        "Enter a prompt to test:",
        height=100,
        placeholder="Type your prompt here..."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_length = st.slider("Max Response Length", 50, 300, 150)
    
    with col2:
        device = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    
    if st.button("Generate Responses", type="primary"):
        if not user_prompt:
            st.warning("Please enter a prompt.")
            return
        
        with st.spinner("Loading models and generating responses..."):
            # Load models
            base_model, base_tokenizer = load_model_for_inference(base_model_path, device)
            rl_model, rl_tokenizer = load_model_for_inference(rl_model_path, device)
            
            if base_model is None or rl_model is None:
                st.error("Failed to load models. Please check model paths in the sidebar.")
                return
            
            # Generate responses
            base_response = generate_response_cached(base_model, base_tokenizer, user_prompt, max_length)
            rl_response = generate_response_cached(rl_model, rl_tokenizer, user_prompt, max_length)
            
            # Load reward model and compute scores
            reward_model, reward_tokenizer = load_reward_model_for_inference(reward_model_path, device)
            
            base_scores = None
            rl_scores = None
            
            st.subheader("Model Responses")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Base Model Response:**")
                st.write(base_response)
                if reward_model:
                    base_scores = compute_reward_scores_interactive(
                        reward_model, reward_tokenizer, user_prompt, base_response, device
                    )
                    if base_scores:
                        st.write("**Reward Scores:**")
                        st.metric("Safety", f"{base_scores['safety_score']:.3f}")
                        st.metric("Toxicity", f"{base_scores['toxicity_score']:.3f}")
                        st.metric("Hallucination Risk", f"{base_scores['hallucination_risk']:.3f}")
                        st.metric("Compliance", f"{base_scores['compliance_score']:.3f}")
                        st.metric("Overall Reward", f"{base_scores['reward']:.3f}")
            
            with col2:
                st.write("**RL Model Response:**")
                st.write(rl_response)
                if reward_model:
                    rl_scores = compute_reward_scores_interactive(
                        reward_model, reward_tokenizer, user_prompt, rl_response, device
                    )
                    if rl_scores:
                        st.write("**Reward Scores:**")
                        st.metric("Safety", f"{rl_scores['safety_score']:.3f}")
                        st.metric("Toxicity", f"{rl_scores['toxicity_score']:.3f}")
                        st.metric("Hallucination Risk", f"{rl_scores['hallucination_risk']:.3f}")
                        st.metric("Compliance", f"{rl_scores['compliance_score']:.3f}")
                        st.metric("Overall Reward", f"{rl_scores['reward']:.3f}")
            
            # Comparison
            if reward_model and base_scores is not None and rl_scores is not None:
                st.subheader("Score Comparison")
                
                comparison_data = {
                    'Metric': ['Safety Score', 'Toxicity Score', 'Hallucination Risk', 'Compliance Score', 'Overall Reward'],
                    'Base Model': [
                        base_scores['safety_score'],
                        base_scores['toxicity_score'],
                        base_scores['hallucination_risk'],
                        base_scores['compliance_score'],
                        base_scores['reward']
                    ],
                    'RL Model': [
                        rl_scores['safety_score'],
                        rl_scores['toxicity_score'],
                        rl_scores['hallucination_risk'],
                        rl_scores['compliance_score'],
                        rl_scores['reward']
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df['Improvement'] = comparison_df['RL Model'] - comparison_df['Base Model']
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=comparison_data['Metric'],
                    y=comparison_data['Base Model'],
                    name='Base Model',
                    marker_color='#ff7f0e'
                ))
                fig.add_trace(go.Bar(
                    x=comparison_data['Metric'],
                    y=comparison_data['RL Model'],
                    name='RL Model',
                    marker_color='#2ca02c'
                ))
                fig.update_layout(
                    title='Reward Score Comparison',
                    yaxis_title='Score',
                    barmode='group',
                    height=400,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
