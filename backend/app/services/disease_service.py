import sys
import os
# Add the root directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import the required modules
import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
from torchvision import models
import numpy as np
from collections import Counter

# Enhanced error reporting
print("Initializing plant disease service...")

# Define the ResNet-50 model architecture that directly matches the saved weights
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        # Direct ResNet50 model without nesting it under 'self.model'
        resnet = models.resnet50(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        num_ftrs = resnet.fc.in_features
        self.fc = nn.Sequential(
            nn.Dropout(0.4),  # Add dropout with 0.4 probability
            nn.Linear(num_ftrs, num_classes)  # Final classification layer
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Use absolute path for the model file
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../ml/plant_disease_model.pth'))
print(f"Loading model from: {MODEL_PATH}")

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: GPU not available. Using CPU for inference (this will be slower).")

try:
    # Load the trained model
    model = PlantDiseaseModel(num_classes=38)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)  # Move model to GPU if available
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

# Define preprocessing for images - match the same transformations used during training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define complete class labels for all 38 classes
CLASS_LABELS = {
    0: "Apple Scab",
    1: "Apple Black Rot",
    2: "Apple Cedar Rust",
    3: "Apple Healthy",
    4: "Blueberry Healthy",
    5: "Cherry Healthy",
    6: "Cherry Powdery Mildew",
    7: "Corn Gray Leaf Spot",
    8: "Corn Common Rust",
    9: "Corn Healthy",
    10: "Corn Northern Leaf Blight",
    11: "Grape Black Rot",
    12: "Grape Esca (Black Measles)",
    13: "Grape Healthy",
    14: "Grape Leaf Blight (Isariopsis Leaf Spot)",
    15: "Orange Haunglongbing (Citrus Greening)",
    16: "Peach Bacterial Spot",
    17: "Peach Healthy",
    18: "Pepper Bell Bacterial Spot",
    19: "Pepper Bell Healthy",
    20: "Potato Early Blight",
    21: "Potato Healthy",
    22: "Potato Late Blight",
    23: "Raspberry Healthy",
    24: "Soybean Healthy",
    25: "Squash Powdery Mildew",
    26: "Strawberry Healthy",
    27: "Strawberry Leaf Scorch",
    28: "Tomato Bacterial Spot",
    29: "Tomato Early Blight",
    30: "Tomato Healthy",
    31: "Tomato Late Blight",
    32: "Tomato Leaf Mold",
    33: "Tomato Septoria Leaf Spot",
    34: "Tomato Spider Mites Two-spotted Spider Mite",
    35: "Tomato Target Spot",
    36: "Tomato Mosaic Virus",
    37: "Tomato Yellow Leaf Curl Virus"
}

# Group diseases by plant types
PLANT_FAMILIES = {
    "Apple": [0, 1, 2, 3],  # Apple Scab, Black Rot, Cedar Rust, Healthy
    "Blueberry": [4],       # Healthy
    "Cherry": [5, 6],       # Healthy, Powdery Mildew
    "Corn": [7, 8, 9, 10],  # Gray Leaf Spot, Common Rust, Healthy, Northern Leaf Blight
    "Grape": [11, 12, 13, 14], # Black Rot, Esca, Healthy, Leaf Blight
    "Orange": [15],         # Citrus Greening
    "Peach": [16, 17],      # Bacterial Spot, Healthy
    "Pepper": [18, 19],     # Bell Bacterial Spot, Bell Healthy
    "Potato": [20, 21, 22], # Early Blight, Healthy, Late Blight
    "Raspberry": [23],      # Healthy
    "Soybean": [24],        # Healthy
    "Squash": [25],         # Powdery Mildew
    "Strawberry": [26, 27], # Healthy, Leaf Scorch
    "Tomato": list(range(28, 38)) # All tomato diseases (28-37)
}

# Map each class ID to its plant family
CLASS_TO_PLANT = {}
for plant_type, class_ids in PLANT_FAMILIES.items():
    for class_id in class_ids:
        CLASS_TO_PLANT[class_id] = plant_type

def predict_single_image(image):
    try:
        # Process image
        img_tensor = transform(image).unsqueeze(0)
        img_tensor = img_tensor.to(device)  # Move tensor to the same device as model
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
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
            
            # Add plant type to disease name for clarity
            if most_likely_plant not in disease_name:
                disease_name = f"{most_likely_plant} - {disease_name}"
                
            # Return results
            return {
                "disease": disease_name,
                "confidence": confidence_value,
                "plant_type": most_likely_plant,
                "class_id": predicted_class.item()
            }
    except Exception as e:
        print(f"Error predicting single image: {str(e)}")
        return {
            "disease": "Error in processing",
            "confidence": 0,
            "plant_type": "Unknown",
            "class_id": -1
        }

def process_images(images):
    try:
        results = []
        confidences = []
        plant_types = []
        debug_info = []
        
        # Process each image
        for image in images:
            try:
                print(f"Processing image: {image.filename}")
                img = Image.open(image).convert('RGB')
                
                # Get prediction for this image
                prediction = predict_single_image(img)
                
                # Store results
                results.append(prediction["disease"])
                confidences.append(prediction["confidence"])
                plant_types.append(prediction["plant_type"])
                
                # Store debug info
                image_debug = {}
                for class_id, class_name in CLASS_LABELS.items():
                    image_debug[class_name] = 1.0 if class_id == prediction["class_id"] else 0.0
                debug_info.append(image_debug)
                
            except Exception as e:
                print(f"Error processing image {image.filename}: {str(e)}")
                results.append("Error processing image")
                confidences.append(0)
                plant_types.append("Unknown")

        # Aggregate results
        if not results:
            return {
                "disease": "No valid images processed",
                "confidence": "0%",
                "plant_type": "Unknown"
            }
            
        # Get most common plant type and disease
        plant_counter = Counter(plant_types)
        most_common_plant = plant_counter.most_common(1)[0][0] if plant_counter else "Unknown"
        
        disease_counter = Counter(results)
        aggregated_result = disease_counter.most_common(1)[0][0] if disease_counter else "Unknown Disease"
        
        # Calculate average confidence for the predicted disease
        matching_confidences = [conf for res, conf in zip(results, confidences) if aggregated_result in res]
        avg_confidence = sum(matching_confidences) / len(matching_confidences) if matching_confidences else 0
        
        print(f"Final result: {aggregated_result} with {avg_confidence:.2%} confidence")
        print(f"Plant type: {most_common_plant}")
        
        # Format confidence as percentage string
        confidence_str = f"{avg_confidence:.2%}"
        
        return {
            "disease": aggregated_result,
            "confidence": confidence_str,
            "plant_type": most_common_plant,
            "debug_info": debug_info
        }
        
    except Exception as e:
        print(f"Error in process_images: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "disease": "Error processing images",
            "confidence": "0%",
            "plant_type": "Unknown",
            "error": str(e)
        }
