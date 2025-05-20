#!/usr/bin/env python
"""
Model Testing Script for Plant Disease Detection

This script tests the model with sample images to verify correct classification,
especially for potato early blight which was previously misclassified.
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# Add the backend directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# Import from project
from app.services.disease_service import predict_single_image, CLASS_LABELS, PLANT_FAMILIES

def run_test_on_sample_images():
    """
    Test the model on sample images from the dataset and report accuracy.
    """
    print("="*80)
    print("PLANT DISEASE MODEL VERIFICATION TEST")
    print("="*80)
    
    # Path to sample images (use dataset path)
    dataset_path = os.path.join(project_root, 'dataset', 'plantvillage_dataset', 'color')
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Test specific diseases that were problematic
    test_cases = [
        ('Potato___Early_Blight', 12, 'Potato Early Blight'),  # ID 12: Potato Early Blight
        ('Tomato___Early_blight', 16, 'Tomato Early Blight'),  # ID 16: Tomato Early Blight
        ('Potato___Late_blight', 13, 'Potato Late Blight'),    # ID 13: Potato Late Blight
        ('Tomato___Late_blight', 17, 'Tomato Late Blight'),    # ID 17: Tomato Late Blight
        ('Potato___healthy', 14, 'Potato Healthy'),            # ID 14: Potato Healthy
        ('Tomato___healthy', 24, 'Tomato Healthy')             # ID 24: Tomato Healthy
    ]
    
    # Results tracking
    results = []
    
    for folder_name, expected_id, disease_name in test_cases:
        folder_path = os.path.join(dataset_path, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
        
        # Get list of images in the folder
        try:
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                print(f"Warning: No images found in {folder_path}")
                continue
                
            # Use a sample of images (up to 5) to test
            sample_count = min(5, len(images))
            sample_images = images[:sample_count]
            
            # Process each sample image
            correct_count = 0
            plant_family_correct = 0
            
            print(f"\nTesting {sample_count} images for: {disease_name}")
            print("-" * 50)
            
            for img_file in sample_images:
                img_path = os.path.join(folder_path, img_file)
                try:
                    # Load and process image
                    img = Image.open(img_path).convert('RGB')
                    
                    # Get prediction
                    result = predict_single_image(img)
                    
                    # Extract result information
                    predicted_id = result["class_id"]
                    predicted_name = CLASS_LABELS.get(predicted_id, "Unknown")
                    confidence = result["confidence"]
                    
                    # Get plant family information
                    expected_plant = "Potato" if "Potato" in disease_name else "Tomato"
                    predicted_plant = result["plant_type"]
                    
                    # Check correctness
                    is_correct = predicted_id == expected_id
                    plant_correct = predicted_plant == expected_plant
                    
                    if is_correct:
                        correct_count += 1
                    if plant_correct:
                        plant_family_correct += 1
                    
                    # Print result
                    print(f"Image: {img_file}")
                    print(f"  Expected: {disease_name} (ID: {expected_id})")
                    print(f"  Predicted: {predicted_name} (ID: {predicted_id}) with {confidence:.2%} confidence")
                    print(f"  Plant Type: Expected {expected_plant}, Got {predicted_plant}")
                    print(f"  Correct: {is_correct}, Plant Family Correct: {plant_correct}")
                    print()
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                
            # Calculate accuracy for this disease
            accuracy = correct_count / sample_count if sample_count > 0 else 0
            plant_accuracy = plant_family_correct / sample_count if sample_count > 0 else 0
            
            print(f"Accuracy for {disease_name}: {accuracy:.2%} ({correct_count}/{sample_count})")
            print(f"Plant family accuracy: {plant_accuracy:.2%} ({plant_family_correct}/{sample_count})")
            
            # Store result
            results.append({
                "disease": disease_name,
                "expected_id": expected_id,
                "sample_count": sample_count,
                "correct_count": correct_count,
                "accuracy": accuracy,
                "plant_accuracy": plant_accuracy
            })
            
        except Exception as e:
            print(f"Error processing folder {folder_name}: {str(e)}")
    
    # Overall results
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    
    total_images = sum(r["sample_count"] for r in results)
    total_correct = sum(r["correct_count"] for r in results)
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    
    print(f"Overall accuracy: {overall_accuracy:.2%} ({total_correct}/{total_images})")
    print("\nDetailed results by disease:")
    
    for r in results:
        print(f"- {r['disease']}: {r['accuracy']:.2%}")
    
    print("\nConclusion:")
    if overall_accuracy >= 0.9:
        print("✅ The model is performing well with the updated transformations!")
    elif overall_accuracy >= 0.7:
        print("⚠️ The model is performing adequately, but could be improved.")
    else:
        print("❌ The model is not performing well. Further investigation needed.")

if __name__ == "__main__":
    print(f"Starting test from: {os.path.abspath(__file__)}")
    start_time = time.time()
    run_test_on_sample_images()
    print(f"\nTest completed in {time.time() - start_time:.2f} seconds.")