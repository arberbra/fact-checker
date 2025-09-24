"""
Script to deploy the trained Bill Nye model to Hugging Face Hub.
This script uploads the LoRA fine-tuned model to the Hugging Face model repository.
"""

import os
from huggingface_hub import HfApi, Repository
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def deploy_model():
    """Deploy the trained model to Hugging Face Hub."""
    
    # Model configuration
    model_name = "arberbr/bill-nye-science-guy"
    local_model_path = "./bill_nye_simple"
    
    print(f"Deploying model to Hugging Face Hub: {model_name}")
    
    # Check if model files exist
    if not os.path.exists(local_model_path):
        print("❌ Model not found! Please run train_simple.py first.")
        return False
    
    try:
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=model_name,
                repo_type="model",
                private=False,
                exist_ok=True
            )
            print(f"✅ Repository {model_name} is ready")
        except Exception as e:
            print(f"⚠️ Repository creation issue: {e}")
        
        # Upload model files
        print("Uploading model files...")
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=model_name,
            repo_type="model",
            commit_message="Upload Bill Nye LoRA fine-tuned model"
        )
        
        # Create model card
        model_card_content = create_model_card()
        
        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card_content.encode(),
            path_in_repo="README.md",
            repo_id=model_name,
            repo_type="model",
            commit_message="Add model card"
        )
        
        print(f"🎉 Model successfully deployed to https://huggingface.co/{model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error deploying model: {e}")
        return False

def create_model_card():
    """Create a model card for the Hugging Face repository."""
    
    model_card = """---
license: apache-2.0
base_model: microsoft/DialoGPT-medium
tags:
- conversational
- science
- education
- bill-nye
- lora
- peft
---

# Bill Nye Science Guy

A fine-tuned version of DialoGPT Medium that responds in the style of Bill Nye, the Science Guy.

## Model Description

This model has been fine-tuned using LoRA (Low-Rank Adaptation) to emulate Bill Nye's characteristic enthusiastic and educational communication style. It's designed to provide scientific explanations and fact-checking responses in a way that's engaging, accessible, and true to Bill Nye's personality.

## Training Data

The model was trained on a synthetic dataset of 1,500 examples that include:

- General science Q&A responses in Bill Nye's style
- Fact-checking scenarios with scientific explanations
- Educational content covering various scientific topics

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model and tokenizer
base_model = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "arberbr/bill-nye-science-guy")

# Generate response
prompt = "Human: What is climate change?\\n\\nBill Nye:"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

- **Base Model**: microsoft/DialoGPT-medium
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Examples**: 1,500
- **Epochs**: 3
- **Learning Rate**: 2e-4
- **Batch Size**: 2
- **LoRA Rank**: 16
- **LoRA Alpha**: 32

## Limitations

- This model is designed for educational and entertainment purposes
- Responses should be fact-checked for accuracy
- The model may not always provide scientifically accurate information
- Use with caution in professional or critical applications

## License

This model is released under the Apache 2.0 license.
"""
    
    return model_card

def test_deployed_model():
    """Test the deployed model to ensure it works correctly."""
    
    model_name = "arberbr/bill-nye-science-guy"
    base_model = "microsoft/DialoGPT-medium"
    
    print("Testing deployed model...")
    
    try:
        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, model_name)
        
        # Test prompt
        test_prompt = "Human: What is science?\\n\\nBill Nye:"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Model test successful!")
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_deployed_model()
    else:
        deploy_model()
