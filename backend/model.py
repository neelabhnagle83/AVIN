import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from app.models.crop_model import create_model
import os
import sys
import numpy as np
import cv2

# Enhanced error reporting
print("Loading plant disease model...")

# Use absolute path for the model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml', 'plant_disease_model.pth')
print(f"Model path: {MODEL_PATH}")

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)

try:
    # Create model directly using ResNet50 without the wrapper class 
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),  # Add dropout with 0.4 probability
        nn.Linear(num_ftrs, 38)  # Final classification layer
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

# Define complete class labels for all 38 classes
CLASS_LABELS = {
    0: "Apple Scab",
    1: "Apple Black Rot",
    2: "Apple Cedar Rust",
    3: "Apple Healthy",
    4: "Corn Gray Leaf Spot",
    5: "Corn Common Rust",
    6: "Corn Northern Leaf Blight",
    7: "Corn Healthy",
    8: "Grape Black Rot",
    9: "Grape Esca (Black Measles)",
    10: "Grape Leaf Blight (Isariopsis Leaf Spot)",
    11: "Grape Healthy",
    12: "Potato Early Blight",
    13: "Potato Late Blight",
    14: "Potato Healthy",
    15: "Tomato Bacterial Spot",
    16: "Tomato Early Blight",
    17: "Tomato Late Blight",
    18: "Tomato Leaf Mold",
    19: "Tomato Septoria Leaf Spot",
    20: "Tomato Spider Mites",
    21: "Tomato Target Spot",
    22: "Tomato Yellow Leaf Curl Virus",
    23: "Tomato Mosaic Virus",
    24: "Tomato Healthy",
    25: "Strawberry Healthy",
    26: "Strawberry Leaf Scorch",
    27: "Background No Plant",
    28: "Blueberry Healthy",
    29: "Cherry Healthy",
    30: "Cherry Powdery Mildew",
    31: "Citrus Greening",
    32: "Peach Bacterial Spot",
    33: "Peach Healthy",
    34: "Pepper Bell Bacterial Spot",
    35: "Pepper Bell Healthy",
    36: "Raspberry Healthy",
    37: "Soybean Healthy"
}

# Group diseases by plant types
PLANT_FAMILIES = {
    "Apple": [0, 1, 2, 3],  # Apple Scab, Black Rot, Cedar Rust, Healthy
    "Corn": [4, 5, 6, 7],   # Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
    "Grape": [8, 9, 10, 11], # Black Rot, Esca, Leaf Blight, Healthy
    "Potato": [12, 13, 14],  # Early Blight, Late Blight, Healthy
    "Tomato": list(range(15, 25)), # All tomato diseases (15-24)
    "Strawberry": [25, 26],  # Healthy, Leaf Scorch
    "Pepper": [34, 35],      # Bell Bacterial Spot, Bell Healthy
    "Other": [27, 28, 29, 30, 31, 32, 33, 36, 37]  # Other plants/classes
}

# Map each class ID to its plant family
CLASS_TO_PLANT = {}
for plant_type, class_ids in PLANT_FAMILIES.items():
    for class_id in class_ids:
        CLASS_TO_PLANT[class_id] = plant_type

# Enhanced image preprocessing for real-world images
def preprocess_image(image):
    """Apply multiple preprocessing techniques to improve real-world image compatibility"""
    # Convert PIL Image to OpenCV format (RGB to BGR)
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    # Apply leaf enhancement techniques
    try:
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(open_cv_image, 9, 75, 75)
        
        # Convert to HSV to enhance green channel (typical for plant leaves)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        
        # Enhance the saturation for better disease spot visibility
        # Adjust H, S, V channels based on typical plant disease appearances
        h, s, v = cv2.split(hsv)
        s = cv2.convertScaleAbs(s, alpha=1.3, beta=0)  # Enhance saturation
        v = cv2.convertScaleAbs(v, alpha=1.1, beta=10)  # Slight brightness boost
        
        # Merge channels back
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Convert back to PIL image (BGR to RGB)
        enhanced_pil = Image.fromarray(enhanced_image[:, :, ::-1].copy())
        
        # Apply additional PIL enhancements
        enhanced_pil = ImageEnhance.Contrast(enhanced_pil).enhance(1.2)
        enhanced_pil = ImageEnhance.Sharpness(enhanced_pil).enhance(1.5)
        
        return enhanced_pil
    except Exception as e:
        print(f"Warning: Advanced preprocessing failed, using original image: {e}")
        return image

# Multiple transforms for ensemble prediction
base_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create alternative transforms for ensemble approach
transforms_list = [
    # Standard transform
    base_transform,
    
    # Transform with slight rotation
    transforms.Compose([
        transforms.Resize((170, 170)),  # Slightly larger resize
        transforms.CenterCrop(150),     # Then crop to standard size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    # Transform with color jittering
    transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    # Transform with slight blur to reduce noise
    transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
]

def predict_disease(image_path):
    try:
        # Load and preprocess the image
        print(f"Processing image: {image_path}")
        original_image = Image.open(image_path).convert('RGB')
        
        # Apply enhanced preprocessing
        enhanced_image = preprocess_image(original_image)
        
        # Use ensemble approach with multiple transforms
        all_outputs = []
        
        # Process both original and enhanced images through different transforms
        images_to_process = [original_image, enhanced_image]
        
        for img in images_to_process:
            for transform_fn in transforms_list:
                try:
                    # Apply the transformation and get prediction
                    image_tensor = transform_fn(img).unsqueeze(0)  # Add batch dimension
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        all_outputs.append(outputs)
                except Exception as e:
                    print(f"Warning: A transform failed, skipping: {e}")
        
        # Average the predictions
        if all_outputs:
            stacked_outputs = torch.stack(all_outputs).mean(dim=0)
            probabilities = torch.nn.functional.softmax(stacked_outputs, dim=1)[0]
        else:
            print("Warning: All ensemble transforms failed. Using base transform on original image.")
            image_tensor = base_transform(original_image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Stage 1: Raw Predictions - Get top 5 predictions
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        print("\nTop 5 Raw Predictions:")
        for i in range(5):
            class_id = top_indices[i].item()
            prob = top_probs[i].item()
            plant_type = CLASS_TO_PLANT.get(class_id, "Unknown")
            print(f"{i+1}. {CLASS_LABELS.get(class_id, f'Unknown (ID: {class_id})')} ({plant_type}) - {prob:.2%}")
        
        # Stage 2: Plant Family Analysis - Determine likely plant families
        plant_family_scores = {}
        for plant_type, class_ids in PLANT_FAMILIES.items():
            # Sum probabilities for this plant type
            score = sum(probabilities[idx].item() for idx in class_ids)
            plant_family_scores[plant_type] = score
        
        # Get the most likely plant family
        sorted_families = sorted(plant_family_scores.items(), key=lambda x: x[1], reverse=True)
        most_likely_plant = sorted_families[0][0]
        plant_confidence = sorted_families[0][1]
        
        print(f"\nPlant family analysis:")
        for plant, score in sorted_families:
            print(f"- {plant}: {score:.2%}")
        print(f"Most likely plant type: {most_likely_plant} ({plant_confidence:.2%})")
        
        # Stage 3: Intelligent Prediction - Filter by plant family
        if plant_confidence > 0.05:  # If we have reasonable confidence in plant type
            # Filter predictions to only this plant family
            filtered_probs = torch.zeros_like(probabilities)
            for idx in PLANT_FAMILIES[most_likely_plant]:
                filtered_probs[idx] = probabilities[idx]
            
            if filtered_probs.sum().item() > 0:
                # Get top prediction within this plant family
                confidence, predicted_class = torch.max(filtered_probs, 0)
                disease_name = CLASS_LABELS.get(predicted_class.item(), "Unknown Disease")
                confidence_value = confidence.item()
                
                print(f"\nFiltered prediction within {most_likely_plant} family:")
                print(f"- {disease_name}, Confidence: {confidence_value:.2%}")
            else:
                # Fallback to raw prediction if filtering yields no results
                predicted_class = top_indices[0].item()
                confidence_value = top_probs[0].item()
                disease_name = CLASS_LABELS.get(predicted_class.item(), "Unknown Disease")
                
                print("\nNo strong predictions within plant family, using top raw prediction:")
                print(f"- {disease_name}, Confidence: {confidence_value:.2%}")
        else:
            # Use raw prediction if plant family confidence is low
            predicted_class = top_indices[0].item()
            confidence_value = top_probs[0].item()
            disease_name = CLASS_LABELS.get(predicted_class.item(), "Unknown Disease")
            
            print("\nLow plant family confidence, using top raw prediction:")
            print(f"- {disease_name}, Confidence: {confidence_value:.2%}")
        
        # Add plant type to disease name if it's not already included
        plant_prefix = most_likely_plant
        if plant_prefix not in disease_name:
            disease_name = f"{plant_prefix} - {disease_name}"
        
        # Only return uncertain if confidence is very low
        if confidence_value < 0.3:
            return {
                "disease": f"Uncertain - Possibly {disease_name}",
                "confidence": f"{confidence_value:.2%}",
                "plant_type": most_likely_plant
            }
        
        return {
            "disease": disease_name,
            "confidence": f"{confidence_value:.2%}",
            "plant_type": most_likely_plant
        }
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {
            "disease": "Error in processing image",
            "confidence": "0%"
        }