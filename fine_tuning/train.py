"""
Comprehensive Bill Nye Model Training and Deployment Script
This script provides multiple options for training, testing, and deploying the Bill Nye personality model.
Based on the deployed model: arberbr/bill-nye-science-guy
"""

import json
import torch
import os
import sys
import subprocess
import argparse
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import wandb
from huggingface_hub import HfApi
from dataset_creation import BillNyeDatasetGenerator
from deploy_to_huggingface import deploy_model

def check_requirements():
    """Check if all required packages are installed."""
    print("🔍 Checking requirements...")
    
    required_packages = [
        "torch", "transformers", "datasets", "accelerate", 
        "peft", "wandb", "scikit-learn", "numpy", "bitsandbytes"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", "requirements_fine_tuning.txt"
            ])
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements!")
            return False
    else:
        print("✅ All requirements satisfied!")
        return True

def create_dataset():
    """Create Bill Nye training datasets using the dynamic generator."""
    print("📊 Creating Bill Nye training datasets...")
    
    # Use the new dynamic dataset generator
    generator = BillNyeDatasetGenerator()
    combined_data = generator.create_all_datasets()
    
    print(f"✅ Created {len(combined_data)} diverse training examples")
    print("📁 Datasets saved to:")
    print("   - data/bill_nye_general.json")
    print("   - data/bill_nye_fact_checking.json")
    print("   - data/bill_nye_combined.json")
    
    return True

def load_training_data():
    """Load the Bill Nye training data."""
    data_files = [
        "data/bill_nye_combined.json",
        "data/bill_nye_fact_checking.json", 
        "data/bill_nye_general.json"
    ]
    
    all_data = []
    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
    
    # Limit dataset size for faster training (take first 100 examples)
    if len(all_data) > 100:
        all_data = all_data[:100]
        print(f"📦 Dataset reduced to {len(all_data)} examples for faster training")
    
    return all_data

def format_training_data(data):
    """Format the training data for the model."""
    formatted_data = []
    
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        # Create the full conversation format
        full_text = f"{instruction}\n\nQuery: {input_text}\n\nBill Nye: {output}"
        formatted_data.append({"text": full_text})
    
    return formatted_data

def tokenize_data(data, tokenizer, max_length=512):
    """Tokenize the training data."""
    def tokenize_function(examples):
        # Tokenize the texts with truncation only (no padding here)
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,  # Let data collator handle padding
            max_length=max_length,
            return_tensors=None
        )
        # For causal language modeling, labels are the same as input_ids
        # but we'll let the data collator handle this properly
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def train_model():
    """Train the Bill Nye model using LoRA."""
    print("🚀 Starting Bill Nye LoRA fine-tuning...")
    
    # Initialize Weights & Biases
    wandb.init(
        project="bill-nye-science-guy",
        name="lora-fine-tuning",
        config={
            "base_model": "microsoft/DialoGPT-medium",
            "method": "LoRA",
            "epochs": 3,
            "learning_rate": 2e-4,
            "batch_size": 2,
            "lora_rank": 16,
            "lora_alpha": 32,
        }
    )
    
    # Load training data
    print("📊 Loading training data...")
    raw_data = load_training_data()
    print(f"Loaded {len(raw_data)} training examples")
    
    if len(raw_data) == 0:
        print("❌ No training data found! Run dataset creation first.")
        return False
    
    # Format data
    formatted_data = format_training_data(raw_data)
    
    # Load base model and tokenizer
    model_name = "microsoft/DialoGPT-medium"  # Base model
    print(f"🤖 Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU training
        device_map=None,  # Don't use device_map for CPU
        trust_remote_code=True,
        use_cache=False,  # Disable KV cache for training
        low_cpu_mem_usage=True  # Optimize for CPU memory usage
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],  # DialoGPT attention modules
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize data with shorter sequences
    print("🔤 Tokenizing training data...")
    tokenized_data = tokenize_data(formatted_data, tokenizer, max_length=256)  # Reduce from 512 to 256
    
    # Set up training arguments (optimized for speed)
    training_args = TrainingArguments(
        output_dir="./bill_nye_simple",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Reduce from 3 to 1 epoch for speed
        per_device_train_batch_size=4,  # Increase batch size for efficiency
        gradient_accumulation_steps=1,  # Maintain effective batch size
        per_device_eval_batch_size=4,
        learning_rate=5e-4,  # Slightly higher LR for faster convergence
        warmup_steps=10,  # Reduce warmup steps
        logging_steps=5,  # More frequent logging
        save_steps=50,  # Save less frequently
        save_total_limit=2,  # Keep fewer checkpoints
        prediction_loss_only=True,
        report_to="wandb",  # Enable W&B logging
        run_name="bill-nye-lora-training-fast",
        dataloader_pin_memory=False,  # Fix Windows hanging issue
        dataloader_num_workers=0,  # Use single-threaded data loading
        dataloader_prefetch_factor=None,  # Disable prefetching
        dataloader_persistent_workers=False,  # Disable persistent workers
        remove_unused_columns=True,  # Remove unused columns since we have all we need
        group_by_length=False,  # Disable length grouping
        fp16=False,  # Disable fp16 for CPU training
        dataloader_drop_last=True,  # Drop incomplete batches
        max_steps=100,  # Limit to 100 steps maximum (about 20-25 minutes)
    )
    
    # Use proper data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked LM
        pad_to_multiple_of=8,  # Optional: pad to multiple of 8 for efficiency
    )
    
    # Create trainer
    print("🔧 Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        data_collator=data_collator,
    )
    
    # Train the model
    print("🏋️ Starting LoRA fine-tuning...")
    print("📊 Training configuration:")
    print(f"   - Dataset size: {len(tokenized_data)}")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Epochs: {training_args.num_train_epochs}")
    print(f"   - Total steps: {len(tokenized_data) // training_args.per_device_train_batch_size * training_args.num_train_epochs}")
    
    # Debug: Test data loading
    print("🔍 Testing data loader...")
    try:
        train_dataloader = trainer.get_train_dataloader()
        print(f"✅ Data loader created successfully with {len(train_dataloader)} batches")
        
        # Test first batch
        print("🧪 Testing first batch...")
        first_batch = next(iter(train_dataloader))
        
    except Exception as e:
        print(f"❌ Data loader error: {e}")
        return False
    
    print("🚀 Starting training loop...")
    print("⚡ Optimized for speed: 100 steps max, 100 examples, batch size 4")
    print("⏱️ Expected training time: 20-25 minutes on CPU")
    
    trainer.train()
    
    # Save the model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("./bill_nye_simple")
    
    # Save LoRA adapter separately
    model.save_pretrained("./bill_nye_simple")
    
    # Finish W&B run
    wandb.finish()
    
    print("✅ LoRA fine-tuning completed! Model saved to ./bill_nye_simple")
    return True

def test_model():
    """Test the trained Bill Nye model."""
    print("🧪 Testing Bill Nye Model")
    print("=" * 30)
    
    model_path = "./bill_nye_simple"
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please run training first.")
        return False
    
    try:
        # Load base model and tokenizer
        base_model = "microsoft/DialoGPT-medium"
        print("🤖 Loading base model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        
        # Load LoRA adapter
        print("🔧 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        
        print("✅ LoRA model loaded successfully!")
        
        # Test questions
        test_questions = [
            "What is climate change?",
            "How do vaccines work?",
            "What is evolution?",
            "Why is the sky blue?",
            "What is gravity?"
        ]
        
        print("\n🎯 Testing Bill Nye responses:")
        print("-" * 50)
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            
            # Create prompt
            prompt = f"""Human: You are Bill Nye, the Science Guy. Answer the following question in your characteristic enthusiastic and educational style.

Query: {question}

Bill Nye:"""
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just Bill Nye's response
            if "Bill Nye:" in response:
                bill_nye_response = response.split("Bill Nye:")[-1].strip()
            else:
                bill_nye_response = response
            
            print(f"🎭 Bill Nye: {bill_nye_response}")
            print("-" * 30)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def deploy_model_wrapper():
    """Deploy the trained model to Hugging Face Hub using the dedicated deploy script."""
    print("🚀 Deploying model to Hugging Face Hub...")
    
    # Use the dedicated deploy function from deploy_to_huggingface.py
    return deploy_model()


def full_setup():
    """Run the complete setup process."""
    print("🎯 Running Full Setup for Bill Nye Model")
    print("=" * 50)
    
    steps = [
        ("Checking Requirements", check_requirements),
        ("Creating Dataset", create_dataset),
        ("Training Model", train_model),
        ("Testing Model", test_model),
        ("Deploying Model", deploy_model_wrapper)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        print("-" * 30)
        
        try:
            success = step_func()
            if not success:
                print(f"❌ {step_name} failed!")
                return False
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            return False
    
    print("\n🎉 Full setup completed successfully!")
    return True

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Bill Nye Model Training and Deployment")
    parser.add_argument(
        "action", 
        choices=["setup", "requirements", "dataset", "train", "test", "deploy"],
        help="Action to perform"
    )
    
    args = parser.parse_args()
    
    print("🎭 Bill Nye Model Training and Deployment")
    print("=" * 40)
    
    if args.action == "setup":
        full_setup()
    elif args.action == "requirements":
        check_requirements()
    elif args.action == "dataset":
        create_dataset()
    elif args.action == "train":
        train_model()
    elif args.action == "test":
        test_model()
    elif args.action == "deploy":
        deploy_model_wrapper()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode if no arguments provided
        print("🎭 Bill Nye Model Training and Deployment")
        print("=" * 40)
        print("\nAvailable options:")
        print("1. Full Setup (requirements + dataset + train + test + deploy)")
        print("2. Check Requirements")
        print("3. Create Dataset")
        print("4. Train Model")
        print("5. Test Model")
        print("6. Deploy Model")
        print("7. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-7): ").strip()
                
                if choice == "1":
                    full_setup()
                    break
                elif choice == "2":
                    check_requirements()
                    break
                elif choice == "3":
                    create_dataset()
                    break
                elif choice == "4":
                    train_model()
                    break
                elif choice == "5":
                    test_model()
                    break
                elif choice == "6":
                    deploy_model()
                    break
                elif choice == "7":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-7.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    else:
        main()