#!/usr/bin/env python
"""
Quick Verification Test for Potato Early Blight Classification

This script tests the model specifically on Potato Early Blight images
to verify it's correctly identifying them after fixing the transformations.
"""

import os
import sys
import torch
from PIL import Image

# Add the backend directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# Import from project
from app.services.disease_service import predict_single_image, CLASS_LABELS

# Test an image of potato early blight
def test_potato_early_blight():
    """
    Test the model specifically on a potato early blight image.
    """
    print("="*80)
    print("POTATO EARLY BLIGHT VERIFICATION TEST")
    print("="*80)
    
    # Path to sample images
    dataset_path = os.path.join(project_root, 'dataset', 'plantvillage_dataset', 'color')
    potato_folder = os.path.join(dataset_path, 'Potato___Early_Blight')
    
    # Check if folder exists
    if not os.path.exists(potato_folder):
        print(f"Error: Dataset folder not found at {potato_folder}")
        # Look for possible alternative folder names
        print("Checking for alternative folder names...")
        potato_folders = [d for d in os.listdir(dataset_path) if 'potato' in d.lower() and 'blight' in d.lower()]
        if potato_folders:
            print(f"Found similar folders: {potato_folders}")
            potato_folder = os.path.join(dataset_path, potato_folders[0])
        else:
            print("No potato blight folders found.")
            return
    
    # Get list of images in the folder
    try:
        images = [f for f in os.listdir(potato_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            print(f"Warning: No images found in {potato_folder}")
            return
            
        print(f"Found {len(images)} potato early blight images.")
        # Use the first 3 images for testing
        test_images = images[:3]
        
        # Print predictions for each image
        for img_file in test_images:
            img_path = os.path.join(potato_folder, img_file)
            print(f"\nTesting image: {img_path}")
            
            try:
                # Load and process image
                img = Image.open(img_path).convert('RGB')
                
                # Get prediction
                result = predict_single_image(img)
                
                # Print the result
                predicted_id = result["class_id"]
                predicted_name = CLASS_LABELS.get(predicted_id, "Unknown")
                confidence = result["confidence"]
                plant_type = result["plant_type"]
                
                print(f"Prediction: {predicted_name} (ID: {predicted_id})")
                print(f"Confidence: {confidence:.2%}")
                print(f"Plant Type: {plant_type}")
                
                # Check if correct
                is_correct = predicted_id == 12  # Potato Early Blight ID
                correct_plant = plant_type == "Potato"
                
                if is_correct:
                    print("✅ CORRECT: Identified as Potato Early Blight")
                else:
                    print(f"❌ INCORRECT: Should be Potato Early Blight (ID: 12), not {predicted_name} (ID: {predicted_id})")
                
                if correct_plant:
                    print("✅ CORRECT: Identified as Potato plant")
                else:
                    print(f"❌ INCORRECT: Should be Potato plant, not {plant_type}")
                
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error listing directory: {str(e)}")

if __name__ == "__main__":
    print(f"Starting quick test from: {os.path.abspath(__file__)}")
    test_potato_early_blight()
    print("\nTest completed.")