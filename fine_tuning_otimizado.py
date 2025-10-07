"""
Fine-tuning Script: Mistral-7B-Instruct-v0.3 with LoRA (4-bit Quantization)
Dataset: AmazonTitles-1.3MM (trn.json)
Task: Generate product descriptions from titles
Technique: Parameter-Efficient Fine-Tuning (PEFT) with LoRA
Memory Optimization: 4-bit quantization with bitsandbytes
"""

import subprocess
import sys

BOOL_INTERACTIVE = 0 # Interactive mode flag
BATCH_SIZE_TRAIN = 6 # Batch size for training
BATCH_SIZE_EVAL = 8 # Batch size for evaluation
GRAD_ACCUMULATION = 2 # Gradient accumulation steps
SEQUENCE_LENGTH = 768 # Input sequence length
SAMPLE_SIZE = 30000 # Sample size for training
LORA_R = 64 # LoRA rank
NUM_WORKERS = 6 # Number of data loader workers

def install_dependencies():
    """Install required packages"""
    packages = [
        "torch",
        "transformers",
        "datasets", 
        "peft",
        "bitsandbytes",
        "accelerate",
        "pandas",
        "scikit-learn"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

import torch
import torch.nn as nn
import pandas as pd
import json
import os
import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainerCallback 
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import Dataset

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Paths
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
PATH_TO_TRAIN_JSONL = "D:\\FIAP_TC3\\processed_dataset\\train.jsonl"  
PATH_TO_VALID_JSONL = "D:\\FIAP_TC3\\processed_dataset\\valid.jsonl"
OUTPUT_DIR = "D:\\FIAP_TC3\\finetuned_lora"
LORA_ADAPTER_DIR = "D:\\FIAP_TC3\\finetuned_lora_adapter"
EVALUATION_DIR = "D:\\FIAP_TC3\\avaliacao"
PLOTS_DIR = "D:\\FIAP_TC3\\plots"

def load_and_prepare_dataset(train_jsonl_path, valid_jsonl_path=None, sample_size=30000):
    """
    Load dataset from processed JSONL files with instruction-input-output format
    """
    print("Loading dataset from processed JSONL files...")
    
    try:
        # Load training data
        train_data = []
        with open(train_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    train_data.append(json.loads(line))
        
        print(f"Loaded {len(train_data)} training records")
        
        if len(train_data) > sample_size:
            original_count = len(train_data)
            train_data = train_data[:sample_size]
            print(f"Trimmed training data from {original_count} to {len(train_data)} (sample_size={sample_size})")
        
        val_data = []
        if valid_jsonl_path and os.path.exists(valid_jsonl_path):
            with open(valid_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        val_data.append(json.loads(line))
            print(f"Loaded {len(val_data)} validation records")
            
            max_val = int(0.1 * len(train_data))
            if len(val_data) > max_val:
                val_original = len(val_data)
                val_data = val_data[:max_val]
                print(f"Trimmed validation data from {val_original} to {len(val_data)} (max 10% of train)")
        
        # Convert to DataFrame format expected by the rest of the pipeline
        train_records = []
        for item in train_data:
            # Map instruction-input-output to title-content format
            title = item.get('input', '')
            # Combine instruction and output for better training
            content = f"{item.get('instruction', '')} {item.get('output', '')}"
            train_records.append({'title': title, 'content': content})
        
        # Sample data if still larger than sample_size (defensive)
        if len(train_records) > sample_size:
            train_records = train_records[:sample_size]
            print(f"Sampled {sample_size} records for training")
        
        df = pd.DataFrame(train_records)
        
        # Store original data for testing and RAG
        return df, train_data, val_data
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Creating fallback sample data...")
        
        # Fallback to sample data
        sample_data = {
            'title': [
                "Wireless Bluetooth Headphones",
                "Stainless Steel Water Bottle", 
                "LED Desk Lamp with USB Charging",
                "Ergonomic Office Chair",
                "Smartphone Camera Lens Kit"
            ] * 1000,
            'content': [
                "Explain the main benefits and features of the product. High-quality wireless headphones with noise cancellation and 20-hour battery life. Perfect for music lovers and professionals.",
                "Describe the product concisely and in a human-like manner. Durable 32oz stainless steel water bottle that keeps drinks cold for 24 hours and hot for 12 hours. BPA-free and eco-friendly.",
                "Create a clear and attractive product description. Adjustable LED desk lamp with multiple brightness levels and built-in USB charging port. Eye-friendly lighting for work and study.",
                "Explain the main benefits and features of the product. Comfortable ergonomic office chair with lumbar support, adjustable height, and breathable mesh back. Ideal for long work sessions.",
                "Describe the product concisely and in a human-like manner. Professional smartphone camera lens kit with wide-angle, macro, and fisheye lenses. Enhance your mobile photography skills."
            ] * 1000
        }
        df = pd.DataFrame(sample_data)
        return df, [], []

def format_mistral_prompt_v2(instruction, input_text, output_text=None):
    """
    Format data in Mistral-Instruct style using instruction-input-output format
    """
    if output_text:
        # Training format with expected output
        return f"<s>[INST] {instruction}\n\nProduto: {input_text} [/INST] {output_text}</s>"
    else:
        # Inference format without expected output
        return f"<s>[INST] {instruction}\n\nProduto: {input_text} [/INST]"

def tokenize_function(examples, tokenizer, max_length=768):
    """
    Tokenize with shorter sequences for speed
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

def prepare_dataset_for_training_v2(df, tokenizer, original_train_data):
    """
    Prepare dataset for training with proper Mistral instruction format
    """
    print("Preparing dataset for training with instruction format...")
    
    # Use original training data for better formatting
    formatted_data = []
    
    if original_train_data:
        # Use original structured data
        for item in original_train_data:
            instruction = item.get('instruction', 'Describe the product concisely and in a human-like manner.')
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            
            if input_text and output_text:
                formatted_prompt = format_mistral_prompt_v2(instruction, input_text, output_text)
                formatted_data.append({"text": formatted_prompt})
    else:
        # Fallback to DataFrame format
        for _, row in df.iterrows():
            # Extract instruction from content if possible
            content_parts = row['content'].split('.', 1)
            instruction = content_parts[0] if content_parts else "Describe the product concisely and in a human-like manner."
            output_text = content_parts[1] if len(content_parts) > 1 else row['content']
            
            formatted_prompt = format_mistral_prompt_v2(instruction, row['title'], output_text)
            formatted_data.append({"text": formatted_prompt})
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=SEQUENCE_LENGTH),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )
    
    # Split dataset (90% train, 10% validation)
    train_size = int(0.9 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    val_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def load_model_with_quantization():
    """
    Load Mistral model with 4-bit quantization
    """
    print("Loading model with 4-bit quantization...")
    
    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model - FIXED: removed flash_attention_2 and deprecated parameters
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def create_evaluation_directory():
    """
    Create evaluation directory if it doesn't exist
    """
    if not os.path.exists(EVALUATION_DIR):
        os.makedirs(EVALUATION_DIR)
        print(f"Created evaluation directory: {EVALUATION_DIR}")

def create_plots_directory():
    """
    Create plots directory if it doesn't exist
    """
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Created plots directory: {PLOTS_DIR}")

def save_evaluation_results(results, filename):
    """
    Save evaluation results to file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(EVALUATION_DIR, f"{timestamp}_{filename}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for i, result in enumerate(results):
            f.write(f"Test {i+1}:\n")
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Generated Description: {result['response']}\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"Evaluation results saved to: {filepath}")
    return filepath

def save_training_summary(lora_config, training_args, dataset_info, output_path):
    """
    Save training summary to JSON file
    """
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": MODEL_NAME,
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": list(lora_config.target_modules) if isinstance(lora_config.target_modules, set) else lora_config.target_modules,  # Convert set to list
            "lora_dropout": lora_config.lora_dropout,
            "bias": lora_config.bias,
            "task_type": lora_config.task_type
        },
        "training_args": {
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "num_train_epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "warmup_steps": training_args.warmup_steps,
            "max_steps": training_args.max_steps,
            "lr_scheduler_type": training_args.lr_scheduler_type
        },
        "dataset_info": dataset_info,
        "output_directories": {
            "lora_adapter": LORA_ADAPTER_DIR,
            "training_output": OUTPUT_DIR,
            "evaluation": EVALUATION_DIR,
            "plots": PLOTS_DIR
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Training summary saved to: {output_path}")
    return summary

def baseline_test(model, tokenizer, validation_data):
    """
    Test model before fine-tuning using real validation data
    """
    print("\n" + "="*50)
    print("BASELINE TEST - Before Fine-tuning")
    print("="*50)
    
    # Use real validation data if available, otherwise use default prompts
    if validation_data and len(validation_data) >= 5:
        test_samples = validation_data[:5]
        test_prompts = []
        expected_outputs = []
        
        for sample in test_samples:
            instruction = sample.get('instruction', 'Describe the product concisely and in a human-like manner.')
            input_text = sample.get('input', '')
            expected_output = sample.get('output', '')
            
            prompt = format_mistral_prompt_v2(instruction, input_text)
            test_prompts.append(prompt)
            expected_outputs.append(expected_output)
    else:
        # Fallback to default prompts
        test_prompts = [
            format_mistral_prompt_v2("Explain the main benefits and features of the product.", "Smartwatch Fitness Tracker"),
            format_mistral_prompt_v2("Describe the product concisely and in a human-like manner.", "Notebook Gaming RGB"),
            format_mistral_prompt_v2("Create a clear and attractive product description.", "Câmera Digital Profissional"),
            format_mistral_prompt_v2("Explain the main benefits and features of the product.", "Fone de Ouvido Bluetooth"),
            format_mistral_prompt_v2("Describe the product concisely and in a human-like manner.", "Mouse Gamer LED"),
        ]
        expected_outputs = ["N/A"] * len(test_prompts)
    
    baseline_results = []
    
    for i, (prompt, expected) in enumerate(zip(test_prompts, expected_outputs)):
        print(f"\nBaseline Test {i+1}:")
        print(f"Prompt: {prompt}")
        if expected != "N/A":
            print(f"Expected: {expected[:100]}...")
        
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part
        generated_text = response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]
        
        baseline_results.append({
            'prompt': prompt,
            'response': generated_text.strip(),
            'expected': expected
        })
        
        print(f"Generated Description: {generated_text.strip()}")
        print("-" * 30)
    
    # Save baseline results
    save_evaluation_results(baseline_results, "baseline_results.txt")
    
    print("="*50 + "\n")
    return baseline_results

def setup_lora_config():
    lora_config = LoraConfig(
        r=64,  
        lora_alpha=64, 
        target_modules=[
            "q_proj", "v_proj", "k_proj", 
            "o_proj",  
            "gate_proj", "down_proj", "up_proj" 
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config

class TrainingMonitor:
    """
    Monitor training progress and create plots
    """
    def __init__(self):
        self.training_logs = defaultdict(list)
        self.eval_logs = defaultdict(list)
        
    def log_training_step(self, step, loss, learning_rate, epoch):
        """Log training metrics"""
        self.training_logs['step'].append(step)
        self.training_logs['loss'].append(loss)
        self.training_logs['learning_rate'].append(learning_rate)
        self.training_logs['epoch'].append(epoch)
        
    def log_eval_step(self, step, eval_loss, epoch):
        """Log evaluation metrics"""
        self.eval_logs['step'].append(step)
        self.eval_logs['eval_loss'].append(eval_loss)
        self.eval_logs['epoch'].append(epoch)
        
    def create_training_plots(self):
        """Create training loss and learning rate plots"""
        if not self.training_logs['step']:
            print("No training logs available for plotting")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Training Loss
        ax1.plot(self.training_logs['step'], self.training_logs['loss'], 
                color='blue', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Learning Rate
        ax2.plot(self.training_logs['step'], self.training_logs['learning_rate'], 
                color='red', linewidth=2, label='Learning Rate')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Learning Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_training_progress.png"
        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to: {filepath}")
        return filepath
        
    def create_eval_plot(self):
        """Create evaluation loss plot"""
        if not self.eval_logs['step']:
            print("No evaluation logs available for plotting")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot evaluation loss
        plt.plot(self.eval_logs['step'], self.eval_logs['eval_loss'], 
                color='green', linewidth=2, marker='o', markersize=4, 
                label='Validation Loss')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Over Training')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add epoch markers
        if self.eval_logs['epoch']:
            unique_epochs = sorted(set(self.eval_logs['epoch']))
            for epoch in unique_epochs:
                epoch_steps = [step for step, ep in zip(self.eval_logs['step'], self.eval_logs['epoch']) if ep == epoch]
                if epoch_steps:
                    plt.axvline(x=epoch_steps[0], color='gray', linestyle='--', alpha=0.5)
                    plt.text(epoch_steps[0], plt.ylim()[1]*0.95, f'Epoch {int(epoch)}', 
                            rotation=90, verticalalignment='top')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_validation_loss.png"
        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Validation plot saved to: {filepath}")
        return filepath
        
    def create_combined_plot(self):
        """Create combined plot with all metrics"""
        if not self.training_logs['step'] or not self.eval_logs['step']:
            print("Insufficient data for combined plot")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training Loss
        ax1.plot(self.training_logs['step'], self.training_logs['loss'], 
                color='blue', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate
        ax2.plot(self.training_logs['step'], self.training_logs['learning_rate'], 
                color='red', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Validation Loss
        ax3.plot(self.eval_logs['step'], self.eval_logs['eval_loss'], 
                color='green', linewidth=2, marker='o', markersize=3)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Validation Loss')
        ax3.set_title('Validation Loss')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Loss Comparison
        ax4.plot(self.training_logs['step'], self.training_logs['loss'], 
                color='blue', linewidth=2, alpha=0.7, label='Training Loss')
        ax4.plot(self.eval_logs['step'], self.eval_logs['eval_loss'], 
                color='green', linewidth=2, marker='o', markersize=3, 
                label='Validation Loss')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training vs Validation Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_complete_training_analysis.png"
        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Complete analysis plot saved to: {filepath}")
        return filepath

from transformers import TrainerCallback

class CustomTrainerCallback(TrainerCallback):
    """Custom callback to capture training metrics"""
    
    def __init__(self, monitor):
        self.monitor = monitor
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are available"""
        if logs is None:
            return
            
        current_step = state.global_step
        current_epoch = state.epoch
        
        # Log training metrics
        if 'loss' in logs and 'learning_rate' in logs:
            self.monitor.log_training_step(
                step=current_step,
                loss=logs['loss'],
                learning_rate=logs['learning_rate'],
                epoch=current_epoch
            )
            # Create training plots every 100 steps
            if current_step % 100 == 0:
                self.monitor.create_training_plots()
                
        # Log evaluation metrics
        if 'eval_loss' in logs:
            self.monitor.log_eval_step(
                step=current_step,
                eval_loss=logs['eval_loss'],
                epoch=current_epoch
            )
            # Create eval plot after each evaluation
            self.monitor.create_eval_plot()
            # Create combined plot
            self.monitor.create_combined_plot()

def train_model(model, tokenizer, train_dataset, val_dataset):
    """
    OPTIMIZED training for speed
    """
    print("Setting up LoRA configuration...")
    
    # Apply LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Initialize training monitor
    monitor = TrainingMonitor()
    
    # Training arguments - Optimized for 16GB VRAM
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=2,
        learning_rate=5e-5,
        fp16=False, 
        bf16=True,  
        logging_steps=20,
        eval_steps=250,
        save_steps=1000,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,  
        warmup_steps=200,
        lr_scheduler_type="cosine", 
        report_to=None,
        dataloader_pin_memory=True,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
        remove_unused_columns=True,
        optim="adamw_torch_fused",
        group_by_length=True,
        logging_first_step=True,
        gradient_checkpointing=True,
        tf32=True,
    )
    
    # Data collator - otimizado
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=16
    )
    
    # Trainer with custom callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[CustomTrainerCallback(monitor)],  # Add custom callback
    )
    
    print("Starting fine-tuning with monitoring...")
    trainer.train()
    
    # Create final plots
    print("Creating final training analysis plots...")
    monitor.create_training_plots()
    monitor.create_eval_plot() 
    monitor.create_combined_plot()
    
    # Save LoRA adapter
    print("Saving LoRA adapter...")
    model.save_pretrained(LORA_ADAPTER_DIR)
    tokenizer.save_pretrained(LORA_ADAPTER_DIR)
    
    # Save training summary
    dataset_info = {
        "train_samples": len(train_dataset),
        "eval_samples": len(val_dataset),
        "total_samples": len(train_dataset) + len(val_dataset)
    }
    summary_path = os.path.join(EVALUATION_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_training_summary.json")
    save_training_summary(lora_config, training_args, dataset_info, summary_path)
    
    return model, trainer

def build_tfidf_index_v2(original_data):
    """
    Build TF-IDF index using original structured data for better RAG
    """
    print("Building TF-IDF index for RAG from structured data...")
    
    if not original_data:
        print("No structured data available, falling back to basic index")
        return None, None, [], []
    
    # Prepare documents from structured data
    documents = []
    titles = []
    instructions = []
    
    for item in original_data:
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        instruction = item.get('instruction', '')
        
        if input_text and output_text:
            # Combine for better search
            doc = f"{input_text} {output_text}"
            documents.append(doc)
            titles.append(input_text)
            instructions.append(instruction)
    
    if not documents:
        print("No valid documents found for TF-IDF index")
        return None, None, [], []
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=2
    )
    
    # Fit and transform documents
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print(f"TF-IDF index built with {len(documents)} documents")
    
    return vectorizer, tfidf_matrix, titles, documents

def retrieve_context(query, vectorizer, tfidf_matrix, titles, documents, top_k=3):
    """
    Retrieve relevant context using TF-IDF similarity
    """
    # Transform query to TF-IDF vector
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top-k most similar documents
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Return context with sources
    context_items = []
    for idx in top_indices:
        if similarities[idx] > 0:  # Only include if there's some similarity
            context_items.append({
                'title': titles[idx],
                'content': documents[idx],
                'similarity': similarities[idx]
            })
    
    return context_items

def inference_test_with_rag_v2(base_model, tokenizer, vectorizer, tfidf_matrix, titles, documents, validation_data):
    """
    Test the fine-tuned model with RAG using real validation data
    """
    print("\n" + "="*50)
    print("INFERENCE TEST - After Fine-tuning (with RAG)")
    print("="*50)
    
    # Load and merge LoRA adapter
    print("Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    
    # Merge LoRA weights with base model
    print("Merging LoRA adapter with base model...")
    merged_model = peft_model.merge_and_unload()
    
    # Use real validation data if available
    if validation_data and len(validation_data) >= 5:
        test_samples = validation_data[:5]
        test_prompts = []
        expected_outputs = []
        
        for sample in test_samples:
            instruction = sample.get('instruction', 'Describe the product concisely and in a human-like manner.')
            input_text = sample.get('input', '')
            expected_output = sample.get('output', '')
            
            prompt = format_mistral_prompt_v2(instruction, input_text)
            test_prompts.append(prompt)
            expected_outputs.append(expected_output)
    else:
        # Fallback to default prompts
        test_prompts = [
            format_mistral_prompt_v2("Explain the main benefits and features of the product.", "Smartwatch Fitness Tracker"),
            format_mistral_prompt_v2("Describe the product concisely and in a human-like manner.", "Notebook Gaming RGB"),
            format_mistral_prompt_v2("Create a clear and attractive product description.", "Câmera Digital Profissional"),
            format_mistral_prompt_v2("Explain the main benefits and features of the product.", "Fone de Ouvido Bluetooth"),
            format_mistral_prompt_v2("Describe the product concisely and in a human-like manner.", "Mouse Gamer LED"),
        ]
        expected_outputs = ["N/A"] * len(test_prompts)
    
    finetuned_results = []
    
    for i, (prompt, expected) in enumerate(zip(test_prompts, expected_outputs)):
        print(f"\nFine-tuned Test {i+1} (with RAG):")
        print(f"Prompt: {prompt}")
        if expected != "N/A":
            print(f"Expected: {expected[:100]}...")
        
        # Retrieve relevant context if RAG is available
        sources = ["Modelo fine-tuned"]
        if vectorizer is not None and tfidf_matrix is not None:
            context_items = retrieve_context(prompt, vectorizer, tfidf_matrix, titles, documents, top_k=2)
            if context_items:
                sources = [item['title'] for item in context_items[:2]]
        
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = merged_model.generate(
                inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part
        generated_text = response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]

        # Add source citation
        source_citation = f"\n\nFonte: {', '.join(sources[:2])}"
        final_response = generated_text.strip() + source_citation
        
        finetuned_results.append({
            'prompt': prompt,
            'response': final_response,
            'expected': expected,
            'sources': sources
        })
        
        print(f"Generated Description: {generated_text.strip()}")
        print(f"Fonte: {', '.join(sources[:2])}")
        print("-" * 30)
    
    # Save fine-tuned results
    save_evaluation_results(finetuned_results, "finetuned_rag_results.txt")
    
    print("="*50 + "\n")
    return finetuned_results

def create_comparison_report(baseline_results, finetuned_results):
    """
    Create a detailed comparison report between baseline and fine-tuned results
    """
    print("Creating comparison report...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(EVALUATION_DIR, f"{timestamp}_comparison_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"MODEL COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"- Baseline tests: {len(baseline_results)}\n")
        f.write(f"- Fine-tuned tests: {len(finetuned_results)}\n")
        f.write(f"- Model: {MODEL_NAME}\n")
        f.write(f"- LoRA Configuration: r={LORA_R}, batch_size={BATCH_SIZE_TRAIN}\n\n")
        
        f.write("DETAILED COMPARISON:\n")
        f.write("-" * 50 + "\n\n")
        
        for i, (baseline, finetuned) in enumerate(zip(baseline_results, finetuned_results)):
            f.write(f"TEST {i+1}:\n")
            f.write(f"Prompt: {baseline['prompt']}\n\n")
            
            if baseline.get('expected', 'N/A') != 'N/A':
                f.write(f"Expected Output:\n{baseline['expected']}\n\n")
            
            f.write(f"BASELINE Response:\n{baseline['response']}\n\n")
            f.write(f"FINE-TUNED Response:\n{finetuned['response']}\n\n")
            
            # Simple quality metrics
            baseline_len = len(baseline['response'].split())
            finetuned_len = len(finetuned['response'].split())
            
            f.write(f"Metrics:\n")
            f.write(f"- Baseline length: {baseline_len} words\n")
            f.write(f"- Fine-tuned length: {finetuned_len} words\n")
            f.write(f"- Length improvement: {((finetuned_len - baseline_len) / baseline_len * 100):.1f}%\n")
            
            f.write("=" * 60 + "\n\n")
    
    print(f"Comparison report saved to: {report_path}")
    return report_path

def interactive_qa(model, tokenizer, vectorizer, tfidf_matrix, titles, documents):
    """
    Interactive Q&A mode with RAG
    """
    print("\n" + "="*60)
    print("MODO INTERATIVO - RAG com Mistral Fine-tuned")
    print("Digite 'sair' para encerrar")
    print("="*60)
    
    while True:
        try:
            # Get user input
            user_query = input("\nDigite sua pergunta sobre produtos: ").strip()
            
            if user_query.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando modo interativo...")
                break
            
            if not user_query:
                continue
            
            # Retrieve relevant context
            if vectorizer is not None and tfidf_matrix is not None:
                context_items = retrieve_context(user_query, vectorizer, tfidf_matrix, titles, documents, top_k=3)
                
                if context_items:
                    # Build enhanced prompt with context
                    context_text = " ".join([item['content'][:150] for item in context_items])
                    enhanced_prompt = f"Contexto: {context_text}\n\nPergunta: {user_query}\n\nResposta:"
                    sources = [item['title'] for item in context_items[:2]]
                else:
                    enhanced_prompt = f"Pergunta: {user_query}\n\nResposta:"
                    sources = ["Nenhuma fonte específica encontrada"]
            else:
                enhanced_prompt = f"Pergunta: {user_query}\n\nResposta:"
                sources = ["Modelo fine-tuned"]
            
            # Generate response
            inputs = tokenizer.encode(enhanced_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]

            # Display response with sources
            print(f"\nResposta: {generated_text.strip()}")
            print(f"Fonte: {', '.join(sources)}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\nEncerrando modo interativo...")
            break
        except Exception as e:
            print(f"Erro: {str(e)}")
            continue

def cleanup_memory():
    """
    Clean up GPU memory
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")

def main():
    """
    Main training pipeline updated for instruction-formatted dataset
    """
    print(f"Starting fine-tuning pipeline at {datetime.now()}")
    print("="*60)
    
    try:
        # Step 0: Create directories
        create_evaluation_directory()
        create_plots_directory()
        
        # Step 1: Load and prepare dataset with new format
        df, original_train_data, original_val_data = load_and_prepare_dataset(
            PATH_TO_TRAIN_JSONL, 
            PATH_TO_VALID_JSONL, 
            sample_size=SAMPLE_SIZE
        )
        
        # Step 2: Build TF-IDF index for RAG using structured data
        vectorizer, tfidf_matrix, titles, documents = build_tfidf_index_v2(original_train_data)
        
        # Step 3: Load model and tokenizer
        model, tokenizer = load_model_with_quantization()
        
        # Step 4: Baseline test using real validation data
        baseline_results = baseline_test(model, tokenizer, original_val_data)
        
        # Step 5: Prepare datasets with new format
        train_dataset, val_dataset = prepare_dataset_for_training_v2(df, tokenizer, original_train_data)
        
        # Memory cleanup
        cleanup_memory()
        
        # Step 6: Fine-tune model
        fine_tuned_model, trainer = train_model(model, tokenizer, train_dataset, val_dataset)
        
        # Memory cleanup
        cleanup_memory()
        
        # Step 7: Load base model for inference
        print("Loading base model for inference demonstration...")
        base_model, _ = load_model_with_quantization()
        
        # Step 8: Inference test with RAG using real validation data
        finetuned_results = inference_test_with_rag_v2(
            base_model, tokenizer, vectorizer, tfidf_matrix, 
            titles, documents, original_val_data
        )
        
        # Step 9: Create comparison report
        create_comparison_report(baseline_results, finetuned_results)
        
        # Step 10: Interactive mode if enabled
        if BOOL_INTERACTIVE == 1:
            print("Entrando no modo interativo...")
            peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
            merged_model = peft_model.merge_and_unload()
            interactive_qa(merged_model, tokenizer, vectorizer, tfidf_matrix, titles, documents)
        
        # Final cleanup
        cleanup_memory()
        
        print("\n" + "="*60)
        print("Fine-tuning completed successfully!")
        print(f"Dataset used: {len(original_train_data)} training samples")
        if original_val_data:
            print(f"Validation samples: {len(original_val_data)}")
        print(f"LoRA adapter saved to: {LORA_ADAPTER_DIR}")
        print(f"Training outputs saved to: {OUTPUT_DIR}")
        print(f"Evaluation results saved to: {EVALUATION_DIR}")
        print(f"Training plots saved to: {PLOTS_DIR}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        cleanup_memory()
        raise

if __name__ == "__main__":
    main()
