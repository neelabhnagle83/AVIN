import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFile
import random
from collections import Counter
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report

# Allow for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Add the root directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_model():
    print("Testing the plant disease model with dataset images...")

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    else:
        print("Using CPU. Testing will be slower.")
    
    # Path to the dataset
    DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'plantvillage_dataset', 'color')
    print(f"Dataset path: {DATASET_PATH}")

    # Path to the model
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plant_disease_model.pth')
    print(f"Model path: {MODEL_PATH}")

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    # First, list all potential folder classes from the dataset
    def discover_classes():
        if not os.path.exists(DATASET_PATH):
            print(f"Error: Dataset directory not found at {DATASET_PATH}")
            return []
        
        try:
            class_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
            print(f"Found {len(class_folders)} class folders in dataset")
            for i, folder in enumerate(sorted(class_folders)):
                print(f"{i}: {folder}")
            return sorted(class_folders)
        except Exception as e:
            print(f"Error listing directory {DATASET_PATH}: {str(e)}")
            return []

    dataset_folders = discover_classes()

    # Define image transformations - match exactly with what was used in training
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize to same size used in training
        transforms.CenterCrop(224),  # Center crop to same size used in training
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset to get number of classes
    try:
        full_dataset = datasets.ImageFolder(DATASET_PATH)
        num_classes = len(full_dataset.classes)
        class_names = full_dataset.classes
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        sys.exit(1)

    # Create model with the same structure as used in training
    try:
        # Direct ResNet50 model - same structure as in training
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),  # Match existing model dropout of 0.4
            nn.Linear(num_ftrs, num_classes)  # Final classification layer
        )
        
        # Load the saved weights
        print("Loading model weights...")
        start_time = time.time()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)  # Move model to GPU if available
        model.eval()  # Set to evaluation mode - important for dropout layers
        print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

    print(f"Model configured for {num_classes} classes")

    # Test images from each class with detailed analytics
    def test_dataset(samples_per_class=10):  # Test with more samples per class for better statistics
        print(f"\nTesting model with {samples_per_class} samples per class...")
        results = []
        all_true_labels = []
        all_pred_labels = []
        correct_count = 0
        total_count = 0
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}
        
        if not os.path.exists(DATASET_PATH):
            print(f"Error: Dataset directory not found at {DATASET_PATH}")
            print(f"Current working directory: {os.getcwd()}")
            return
            
        # Get all class folders
        try:
            class_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
            print(f"Found {len(class_folders)} class folders")
        except Exception as e:
            print(f"Error listing directory {DATASET_PATH}: {str(e)}")
            return
        
        # Process each class folder
        for folder in class_folders:
            folder_path = os.path.join(DATASET_PATH, folder)
            try:
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Skip if no images found
                if not image_files:
                    print(f"No images found in {folder}")
                    continue
                    
                # Sample random images from this class
                if len(image_files) > samples_per_class:
                    sampled_images = random.sample(image_files, samples_per_class)
                else:
                    sampled_images = image_files
                    
                print(f"Testing {len(sampled_images)} images from {folder}")
                
                # Process each image
                for img_name in sampled_images:
                    img_path = os.path.join(folder_path, img_name)
                    result = test_image(img_path, folder)
                    results.append(result)
                    
                    if "error" not in result:
                        total_count += 1
                        true_class_id = result['true_class'][0]
                        pred_class_id = result['predictions'][0][0]
                        
                        all_true_labels.append(true_class_id)
                        all_pred_labels.append(pred_class_id)
                        
                        # Track per-class accuracy
                        class_total[true_class_id] = class_total.get(true_class_id, 0) + 1
                        
                        if result["correct"]:
                            correct_count += 1
                            class_correct[true_class_id] = class_correct.get(true_class_id, 0) + 1
                        
                        print(f"Image: {img_name}")
                        print(f"True class: {result['true_class'][1]} (ID: {result['true_class'][0]})")
                        print(f"Top 5 predictions:")
                        for i, (class_id, class_name, prob) in enumerate(result['predictions']):
                            print(f"  {i+1}. {class_name} (ID: {class_id}) ({prob:.2%})")
                        print(f"Correct: {result['correct']}")
                        print("---")
            except Exception as e:
                print(f"Error processing folder {folder}: {str(e)}")
                continue
        
        # Calculate overall accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"\nOverall accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        
        # Print per-class accuracy
        print("\nPer-class accuracy:")
        class_accuracies = {}
        for class_id in range(num_classes):
            if class_total.get(class_id, 0) > 0:
                class_acc = class_correct.get(class_id, 0) / class_total[class_id]
                class_accuracies[class_id] = class_acc
                print(f"{class_names[class_id]}: {class_acc:.2%} ({class_correct.get(class_id, 0)}/{class_total[class_id]})")
        
        # Report most common misclassifications
        if total_count - correct_count > 0:
            print("\nMost common misclassifications:")
            misclassifications = []
            for result in results:
                if "error" not in result and not result["correct"]:
                    true_class = result["true_class"][1]
                    pred_class = result["predictions"][0][1]
                    misclassifications.append((true_class, pred_class))
            
            misclass_counter = Counter(misclassifications)
            for (true, pred), count in misclass_counter.most_common(10):
                print(f"{true} → {pred}: {count} times")
        
        # Generate confusion matrix
        if len(all_true_labels) > 0:
            print("\nGenerating confusion matrix...")
            cm = confusion_matrix(all_true_labels, all_pred_labels)
            
            # Plot confusion matrix if matplotlib available
            try:
                plt.figure(figsize=(20, 16))
                
                # Use a more readable format for large class counts
                if num_classes > 20:
                    # Plot a heatmap without text
                    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                                xticklabels=range(num_classes),
                                yticklabels=range(num_classes))
                    plt.title('Confusion Matrix')
                else:
                    # Plot with class names
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=class_names,
                                yticklabels=class_names)
                    plt.title('Confusion Matrix')
                
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                
                # Save the confusion matrix
                confusion_matrix_path = os.path.join(os.path.dirname(MODEL_PATH), 'confusion_matrix.png')
                plt.savefig(confusion_matrix_path)
                print(f"Confusion matrix saved to {confusion_matrix_path}")
            except Exception as e:
                print(f"Error generating confusion matrix plot: {str(e)}")
            
            # Generate classification report
            print("\nClassification Report:")
            report = classification_report(all_true_labels, all_pred_labels, 
                                           target_names=[class_names[i] for i in range(num_classes)],
                                           zero_division=0)
            print(report)
            
            # Save classification report to file
            try:
                report_path = os.path.join(os.path.dirname(MODEL_PATH), 'classification_report.txt')
                with open(report_path, 'w') as f:
                    f.write("PLANT DISEASE MODEL EVALUATION\n")
                    f.write("============================\n\n")
                    f.write(f"Overall Accuracy: {accuracy:.2%}\n\n")
                    f.write("Per-Class Accuracy:\n")
                    for class_id in range(num_classes):
                        if class_id in class_accuracies:
                            f.write(f"{class_names[class_id]}: {class_accuracies[class_id]:.2%}\n")
                    f.write("\n\nClassification Report:\n")
                    f.write(report)
                print(f"Classification report saved to {report_path}")
            except Exception as e:
                print(f"Error saving classification report: {str(e)}")

    # Function to test the model on a single image
    def test_image(image_path, folder_name):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
                # Get top predictions
                top_probs, top_classes = torch.topk(probabilities, 5)
                predictions = []
                for i in range(5):
                    class_id = top_classes[i].item()
                    prob = top_probs[i].item()
                    class_name = class_id_to_name(class_id)
                    predictions.append((class_id, class_name, prob))
                
                # Get true class from folder name
                true_class_id = folder_to_class_id(folder_name)
                true_class_name = class_id_to_name(true_class_id)
                
                return {
                    "image_path": image_path,
                    "true_class": (true_class_id, true_class_name),
                    "predictions": predictions,
                    "correct": predictions[0][0] == true_class_id if true_class_id is not None else False
                }
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return {
                "image_path": image_path,
                "error": str(e)
            }

    # Helper function to get class name from id
    def class_id_to_name(class_id):
        if class_id is None:
            return "Unknown"
        
        # Get class name from the dataset
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        return f"Unknown (ID: {class_id})"

    # Helper function to get class id from folder name
    def folder_to_class_id(folder_name):
        # Get the index from the dataset's class_to_idx mapping
        try:
            return full_dataset.class_to_idx.get(folder_name)
        except:
            return None

    # Run test with 15 samples
    test_dataset(samples_per_class=15)

if __name__ == "__main__":
    test_model()