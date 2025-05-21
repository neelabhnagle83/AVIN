import sys
import os
# Add the root directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import the required modules
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
import torch.nn as nn
from torchvision import models
import numpy as np
from collections import Counter
import cv2  # Add OpenCV for enhanced image preprocessing

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

# Enhanced image preprocessing for real-world images
def preprocess_image(image):
    """Apply multiple preprocessing techniques to improve real-world image compatibility"""
    # Convert PIL Image to OpenCV format (RGB to BGR)
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    # Store original dimensions for debugging
    original_h, original_w = open_cv_image.shape[:2]
    print(f"Original image dimensions: {original_w}x{original_h}")
    
    # Create multiple enhanced versions for ensemble prediction
    enhanced_images = []
    
    try:
        # Version 1: Standard enhancement
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(open_cv_image, 9, 75, 75)
        
        # Convert to HSV to enhance green channel (typical for plant leaves)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        
        # Enhance the saturation for better disease spot visibility
        h, s, v = cv2.split(hsv)
        s = cv2.convertScaleAbs(s, alpha=1.3, beta=0)  # Enhance saturation
        v = cv2.convertScaleAbs(v, alpha=1.1, beta=10) # Slight brightness boost
        
        # Merge channels back
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced_image1 = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast
        lab = cv2.cvtColor(enhanced_image1, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_image1 = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Add sharpening to enhance disease boundaries
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        enhanced_image1 = cv2.filter2D(enhanced_image1, -1, kernel)
        
        # Convert back to PIL image (BGR to RGB)
        enhanced_pil1 = Image.fromarray(enhanced_image1[:, :, ::-1].copy())
        
        # Apply additional PIL enhancements
        enhanced_pil1 = ImageEnhance.Contrast(enhanced_pil1).enhance(1.2)
        enhanced_pil1 = ImageEnhance.Sharpness(enhanced_pil1).enhance(1.5)
        
        enhanced_images.append(enhanced_pil1)
        
        # Version 2: Stronger enhancement for difficult cases
        # Apply more aggressive noise reduction
        filtered2 = cv2.fastNlMeansDenoisingColored(open_cv_image, None, 10, 10, 7, 21)
        
        # More aggressive contrast normalization using CLAHE
        lab2 = cv2.cvtColor(filtered2, cv2.COLOR_BGR2LAB)
        l2, a2, b2 = cv2.split(lab2)
        clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        cl2 = clahe2.apply(l2)
        enhanced_lab2 = cv2.merge((cl2, a2, b2))
        enhanced_image2 = cv2.cvtColor(enhanced_lab2, cv2.COLOR_LAB2BGR)
        
        # Enhance color and contrast for better disease visibility
        enhanced_image2 = cv2.convertScaleAbs(enhanced_image2, alpha=1.3, beta=15)
        
        # Convert back to PIL image
        enhanced_pil2 = Image.fromarray(enhanced_image2[:, :, ::-1].copy())
        enhanced_pil2 = ImageEnhance.Color(enhanced_pil2).enhance(1.4)
        
        enhanced_images.append(enhanced_pil2)
        
        # Version 3: Focus on edge detection for disease boundary enhancement
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(open_cv_image, (5, 5), 0)
        
        # Create edge-enhanced version using Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine the edge map with original image to enhance boundaries
        edge_enhanced = cv2.addWeighted(open_cv_image, 0.7, edges_3channel, 0.3, 0)
        
        # Apply additional contrast and sharpness
        edge_enhanced = cv2.convertScaleAbs(edge_enhanced, alpha=1.2, beta=10)
        
        # Convert back to PIL image
        enhanced_pil3 = Image.fromarray(edge_enhanced[:, :, ::-1].copy())
        enhanced_pil3 = ImageEnhance.Sharpness(enhanced_pil3).enhance(1.7)
        
        enhanced_images.append(enhanced_pil3)
        
        # Version 4: Handle screenshots and digital images with color correction
        # Remove potential screenshot artifacts by applying median blur
        median_blur = cv2.medianBlur(open_cv_image, 5)
        
        # Color correction for digital images that might have different color balance
        # Convert to LAB for better color adjustment
        lab4 = cv2.cvtColor(median_blur, cv2.COLOR_BGR2LAB)
        l4, a4, b4 = cv2.split(lab4)
        
        # Adjust a and b channels to normalize color balance
        a4 = cv2.convertScaleAbs(a4, alpha=1.1, beta=-5)  # Adjust green-red balance
        b4 = cv2.convertScaleAbs(b4, alpha=1.1, beta=5)   # Adjust blue-yellow balance
        
        corrected_lab = cv2.merge((l4, a4, b4))
        corrected_image = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        # Add final sharpening
        sharpening_kernel = np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]])
        corrected_image = cv2.filter2D(corrected_image, -1, sharpening_kernel)
        
        # Convert back to PIL image
        enhanced_pil4 = Image.fromarray(corrected_image[:, :, ::-1].copy())
        
        enhanced_images.append(enhanced_pil4)
        
        print(f"Successfully created {len(enhanced_images)} enhanced versions of the image")
        return enhanced_images
    except Exception as e:
        print(f"Warning: Advanced preprocessing failed: {e}")
        # If enhancement fails, return the original image
        return [image]
        
def predict_single_image(image):
    try:
        # Apply enhanced image preprocessing for real-world images
        enhanced_images = preprocess_image(image)
        
        # Process both original and all enhanced images
        images_to_process = [image] + enhanced_images
        all_outputs = []
        all_probabilities = []
        
        print(f"Running predictions on {len(images_to_process)} image variations")
        
        for idx, img in enumerate(images_to_process):
            try:
                # Process image
                img_tensor = transform(img).unsqueeze(0)
                img_tensor = img_tensor.to(device)  # Move tensor to the same device as model
                
                # Predict
                with torch.no_grad():
                    outputs = model(img_tensor)
                    all_outputs.append(outputs)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    all_probabilities.append(probs)
                    
                    # Print top prediction for this variation
                    confidence, predicted_class = torch.max(probs, 0)
                    disease_name = CLASS_LABELS.get(predicted_class.item(), "Unknown Disease")
                    print(f"Variation {idx}: {disease_name} - {confidence.item():.2%}")
                    
            except Exception as e:
                print(f"Error processing image variation {idx}: {e}")
                # Continue with other variations if one fails
                continue
        
        # If no predictions worked, return error
        if not all_outputs:
            print("All image variations failed prediction")
            return {
                "disease": "Error in processing",
                "confidence": 0,
                "plant_type": "Unknown",
                "class_id": -1
            }
        
        # Average the predictions to get more robust results
        stacked_outputs = torch.stack(all_outputs).mean(dim=0)
        probabilities = torch.nn.functional.softmax(stacked_outputs, dim=1)[0]
        
        # Also calculate a weighted average based on confidence
        if len(all_probabilities) > 1:
            # Get max confidence for each variation
            max_confidences = [torch.max(p).item() for p in all_probabilities]
            # Normalize confidences to use as weights
            weights = torch.tensor(max_confidences) / sum(max_confidences)
            # Weight predictions by confidence
            weighted_probs = torch.zeros_like(all_probabilities[0])
            for i, p in enumerate(all_probabilities):
                weighted_probs += p * weights[i]
            
            # Use weighted probabilities if confidence variation is significant
            confidence_std = np.std(max_confidences)
            if confidence_std > 0.05:  # If there's significant variation in confidence
                print(f"Using weighted predictions (confidence STD: {confidence_std:.4f})")
                probabilities = weighted_probs
        
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
        for plant, score in sorted_families[:3]:  # Show top 3 plant families
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
            
        # Return results with additional metadata
        return {
            "disease": disease_name,
            "confidence": confidence_value,
            "plant_type": most_likely_plant,
            "class_id": predicted_class.item(),
            "top_candidates": [
                {
                    "disease": CLASS_LABELS.get(top_indices[i].item()),
                    "confidence": top_probs[i].item(),
                    "plant_type": CLASS_TO_PLANT.get(top_indices[i].item(), "Unknown")
                } for i in range(min(3, len(top_indices)))  # Return top 3 candidates
            ]
        }
    except Exception as e:
        print(f"Error predicting single image: {str(e)}")
        import traceback
        traceback.print_exc()
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
