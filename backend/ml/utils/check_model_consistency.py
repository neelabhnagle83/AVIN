#!/usr/bin/env python
"""
Model Architecture Documentation and Consistency Checker

This script documents the official model architecture and checks for consistency
across all model-related files in the application. It helps ensure that all parts
of the app use the same model structure when loading the trained weights.
"""

import os
import sys
import re
import torch
import torch.nn as nn
from torchvision import models

# Properly add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

try:
    # Import project modules
    from app.models.crop_model import create_model, PlantDiseaseModel
except ModuleNotFoundError:
    print("Error: Unable to import modules from app.models.")
    print(f"Current sys.path: {sys.path}")
    print("Creating a standalone model for consistency checking instead.")
    
    # Define the model architecture here as a fallback
    def create_model(num_classes=38):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes)
        )
        return model

def document_model_architecture():
    """
    Document the canonical model architecture used in the project.
    This provides a clear reference for training and inference.
    """
    print("="*80)
    print("CANONICAL MODEL ARCHITECTURE")
    print("="*80)
    
    # Create a standard model using the preferred approach
    model = create_model(num_classes=38)
    
    # Display model structure
    print("\nModel Structure:")
    print("-"*40)
    print(model)
    
    # Document dropout rate
    dropout_rate = None
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            dropout_rate = module.p
            break
    
    print("\nModel Configuration:")
    print("-"*40)
    print(f"Base model: ResNet50")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Number of classes: 38")
    print(f"Weights file: plant_disease_model.pth")
    
    return dropout_rate

def check_file_consistency(dropout_rate):
    """
    Check files for consistency in model architecture, especially dropout rate.
    """
    print("\n" + "="*80)
    print("CHECKING FILE CONSISTENCY")
    print("="*80)
    
    # Files to check (use absolute paths to avoid confusion)
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    files_to_check = [
        os.path.join(project_root, "app", "models", "crop_model.py"),
        os.path.join(project_root, "app", "services", "disease_service.py"),
        os.path.join(project_root, "ml", "train_model.py"),
        os.path.join(project_root, "ml", "test_model.py"),
        os.path.join(project_root, "model.py")
    ]
    
    # Check each file
    all_consistent = True
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for consistent dropout rate
            dropout_matches = re.findall(r"nn\.Dropout\(([0-9.]+)\)", content)
            if not dropout_matches:
                print(f"Warning: No dropout found in {file_path}")
                all_consistent = False
                continue
                
            file_consistent = all(float(rate) == dropout_rate for rate in dropout_matches)
            
            if file_consistent:
                print(f"✅ {file_path}: Dropout rate consistent ({dropout_rate})")
            else:
                found_rates = set(float(rate) for rate in dropout_matches)
                print(f"❌ {file_path}: Inconsistent dropout rates found: {found_rates}, should be {dropout_rate}")
                all_consistent = False
        except Exception as e:
            print(f"Error checking {file_path}: {str(e)}")
            all_consistent = False
    
    if all_consistent:
        print("\n✅ All files use consistent model architecture!")
    else:
        print("\n❌ Some files have inconsistent model architecture. See warnings above.")

if __name__ == "__main__":
    print(f"Running from: {os.path.abspath(__file__)}")
    print(f"Project root: {os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))}")
    
    # Document the canonical model architecture and get the official dropout rate
    dropout_rate = document_model_architecture()
    
    # Check all files for consistency
    check_file_consistency(dropout_rate)