# Bill Nye Model LoRA Fine-tuning

This directory contains the code and data for LoRA fine-tuning a Bill Nye personality model, matching the deployed model at `arberbr/bill-nye-science-guy`.

## Files

- `train_simple.py` - Main LoRA training script for the Bill Nye model
- `test_bill_nye_model.py` - Test script to evaluate the trained LoRA model
- `dataset_creation.py` - Script to create training datasets
- `deploy_to_huggingface.py` - Script to deploy model to Hugging Face Hub
- `requirements_fine_tuning.txt` - Dependencies for LoRA training
- `data/` - Training datasets in JSON format
- `bill_nye_simple/` - Trained LoRA adapter files

## Quick Start

1. Install training dependencies:
```bash
pip install -r requirements_fine_tuning.txt
```

2. Create training datasets:
```bash
python dataset_creation.py
```

3. Train the LoRA model:
```bash
python train_simple.py
```

4. Test the trained model:
```bash
python test_bill_nye_model.py
```

5. Deploy to Hugging Face Hub:
```bash
python deploy_to_huggingface.py
```

## Training Configuration

The training matches the deployed model configuration:

- **Base Model**: microsoft/DialoGPT-medium
- **Method**: LoRA (Low-Rank Adaptation)
- **Training Examples**: 1,500
- **Epochs**: 3
- **Learning Rate**: 2e-4
- **Batch Size**: 2
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Monitoring**: Weights & Biases integration

## Training Data

The training data consists of question-answer pairs where Bill Nye responds in his characteristic enthusiastic and educational style. The data includes:

- General science questions
- Fact-checking scenarios
- Educational explanations

## Model Architecture

The model uses LoRA fine-tuning on Microsoft's DialoGPT-medium to respond in Bill Nye's style while maintaining scientific accuracy. LoRA allows for efficient fine-tuning with minimal parameter updates.

## Weights & Biases Integration

Training metrics are automatically logged to Weights & Biases:
- Training loss
- Learning rate schedule
- Gradient norms
- Training progress

## Usage in Main App

The trained model can be used in the main fact-checker app by setting the model path to `./fine_tuning/bill_nye_simple` in the app's sidebar, or by using the deployed model `arberbr/bill-nye-science-guy`.
