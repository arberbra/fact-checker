"""
Setup script to pre-download the Bill Nye model.
This script downloads the model to the Hugging Face cache during installation.
"""

import os
import sys
from pathlib import Path

def download_bill_nye_model():
    """Download the Bill Nye model to cache."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch
        
        base_model_name = "microsoft/DialoGPT-medium"
        adapter_name = "arberbr/bill-nye-science-guy"
        
        print(f"Downloading Bill Nye model components...")
        print("This may take a few minutes on first run...")
        
        # Download base model tokenizer
        print("Downloading base model tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print("✅ Base tokenizer downloaded")
        
        # Download base model
        print("Downloading base model (DialoGPT-medium)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            low_cpu_mem_usage=True
        )
        print("✅ Base model downloaded")
        
        # Download LoRA adapter
        print("Downloading Bill Nye LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_name)
        print("✅ LoRA adapter downloaded")
        
        print(f"🎉 Bill Nye model successfully cached!")
        print("The app will now load much faster on first use.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install requirements first: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = download_bill_nye_model()
    sys.exit(0 if success else 1)
