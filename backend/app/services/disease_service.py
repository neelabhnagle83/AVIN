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
import json
import base64
from datetime import datetime

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

# Disease-specific visual characteristics and symptoms for improved detection
DISEASE_CHARACTERISTICS = {
    # Apple Cedar Rust characteristics - focus on distinctive orange spots
    2: {
        "name": "Apple Cedar Rust",
        "color_ranges": [
            # Orange-yellow spots (in HSV) - expanded range
            {"lower": np.array([10, 80, 100]), "upper": np.array([35, 255, 255])},
            # Reddish-orange areas (in HSV)
            {"lower": np.array([0, 120, 100]), "upper": np.array([20, 255, 255])},
            # Yellowish-brown spots - common in Cedar Rust
            {"lower": np.array([20, 60, 80]), "upper": np.array([40, 255, 255])}
        ],
        "texture_patterns": "circular_spots",
        "key_symptoms": [
            "Bright orange or yellow spots on leaves",
            "Circular lesions with orange centers",
            "Black spots in center of lesions (later stages)",
            "Swollen or deformed leaf tissue"
        ],
        "confidence_boost": 1.5,  # Increased boost
        "common_confusions": [7, 8]  # Corn Gray Leaf Spot, Corn Common Rust
    },
    # Squash Powdery Mildew characteristics - focus on white powdery patches
    25: {
        "name": "Squash Powdery Mildew",
        "color_ranges": [
            # White-grayish patches (in HSV) - expanded and more inclusive
            {"lower": np.array([0, 0, 160]), "upper": np.array([180, 40, 255])},
            # Yellowish affected areas (in HSV)
            {"lower": np.array([25, 50, 180]), "upper": np.array([35, 160, 255])}
        ],
        "texture_patterns": "powdery_patches",
        "key_symptoms": [
            "White powdery spots on leaves and stems",
            "Fuzzy or talcum-powder-like appearance",
            "Spots that merge into larger patches",
            "Yellowing of affected tissue",
            "Leaf curling or distortion"
        ],
        "lookalike_diseases": [11, 12, 14],  # Grape Black Rot, Esca, Leaf Blight often confused
        "confidence_boost": 1.5  # Increased confidence boost for this disease class
    }
}

# Segmentation utilities for isolating plant leaf material
def segment_leaf(image):
    """
    Segment the leaf from the background using multiple techniques.
    Returns the segmented leaf mask and the masked image.
    """
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
    
    # Store original for comparison
    original = open_cv_image.copy()
    h, w = original.shape[:2]
    
    # Create an empty mask to store our result
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    try:
        # Method 1: Color-based segmentation (focus on green plant material)
        # Convert to HSV color space which is better for color segmentation
        hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
        
        # Define range of green color in HSV
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        # Create a mask for green areas
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Also consider yellowish areas for diseased plants
        lower_yellow = np.array([15, 40, 80])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Also consider brownish areas for diseased plants
        lower_brown = np.array([5, 50, 50])
        upper_brown = np.array([25, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Consider whitish areas for powdery mildew
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Consider reddish areas for cedar apple rust
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combine masks for comprehensive leaf coverage
        color_mask = cv2.bitwise_or(green_mask, yellow_mask)
        color_mask = cv2.bitwise_or(color_mask, brown_mask)
        color_mask = cv2.bitwise_or(color_mask, white_mask)
        color_mask = cv2.bitwise_or(color_mask, red_mask1)
        color_mask = cv2.bitwise_or(color_mask, red_mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        # Method 2: Otsu's thresholding for grayscale-based segmentation
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Method 3: Grab Cut for more advanced segmentation
        try:
            # Initialize mask for GrabCut
            grabcut_mask = np.zeros((h, w), np.uint8)
            
            # Set a rectangle covering most of the image as probable foreground
            margin = min(h, w) // 8
            rect = (margin, margin, w - 2*margin, h - 2*margin)
            
            # Temporary arrays used by grabCut
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut with our rectangle as initial guess
            cv2.grabCut(open_cv_image, grabcut_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
            
            # Create a mask where sure background is 0, everything else is 1
            grabcut_mask2 = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')
            grabcut_mask = grabcut_mask2 * 255
        except Exception as e:
            print(f"GrabCut segmentation failed: {e}")
            grabcut_mask = np.ones((h, w), np.uint8) * 255  # Default to all foreground
        
        # Method 4: Edge-based segmentation
        # Apply Canny edge detection
        edges = cv2.Canny(open_cv_image, 100, 200)
        
        # Dilate edges to create connected regions
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours in the edge map
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask from contours - focus on larger contours
        edge_mask = np.zeros((h, w), np.uint8)
        if contours:
            # Sort contours by area (descending)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            # Keep only the largest contours that are likely to be leaves
            significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
            # If no significant contours, keep the largest one
            if not significant_contours and contours:
                significant_contours = [contours[0]]
            # Draw filled contours on the mask
            cv2.drawContours(edge_mask, significant_contours, -1, 255, -1)
        
        # Combine all masks - weight different methods
        final_mask = cv2.bitwise_or(final_mask, color_mask)  # Start with color mask
        
        # Combine with GrabCut mask if it's not all zeros or all ones
        if np.mean(grabcut_mask) > 5 and np.mean(grabcut_mask) < 250:
            final_mask = cv2.bitwise_and(final_mask, grabcut_mask)
        
        # Use edge mask to refine boundaries
        if np.sum(edge_mask) > 0:
            final_mask = cv2.bitwise_and(final_mask, edge_mask)
        
        # If final mask is too small or empty, fallback to color mask
        if np.sum(final_mask) < (h * w * 0.05):  # Less than 5% of image
            final_mask = color_mask
        
        # Final cleaning and smoothing
        kernel_clean = np.ones((7, 7), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_clean)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_clean)
        
        # Apply slight Gaussian blur to smooth the mask edges
        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        
    except Exception as e:
        print(f"Error in segmentation: {e}")
        # Fallback to simple color-based mask if segmentation fails
        hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([80, 255, 255])
        final_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply the mask to the original image
    mask_3channel = cv2.merge([final_mask, final_mask, final_mask])
    masked_image = cv2.bitwise_and(original, mask_3channel)
    
    # Convert masked image back to PIL
    masked_pil = Image.fromarray(masked_image[:, :, ::-1])  # BGR to RGB
    
    # Also convert the mask to PIL for potential return
    mask_pil = Image.fromarray(final_mask)
    
    return mask_pil, masked_pil

# Function to detect disease-specific features
def detect_disease_features(image, disease_id):
    """
    Detect visual features specific to a particular disease.
    Returns a confidence score based on feature detection.
    """
    if disease_id not in DISEASE_CHARACTERISTICS:
        return 0.0
    
    disease_info = DISEASE_CHARACTERISTICS[disease_id]
    
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
    
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
    
    feature_confidence = 0.0
    
    try:
        # Check for disease-specific color ranges
        color_match_percent = 0
        for color_range in disease_info["color_ranges"]:
            mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
            color_pixels = cv2.countNonZero(mask)
            total_pixels = hsv.shape[0] * hsv.shape[1]
            color_match_percent += (color_pixels / total_pixels) * 100
        
        # Normalize color match percent (cap at 30% for full confidence)
        color_confidence = min(color_match_percent / 30.0, 1.0)
        
        # Analyze texture patterns based on disease type
        texture_confidence = 0.0
        
        if disease_info["texture_patterns"] == "circular_spots":
            # For Cedar Apple Rust - look for circular patterns
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use Hough Circle Transform to detect circular patterns
            circles = cv2.HoughCircles(
                blurred, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=20, 
                param1=50, 
                param2=30, 
                minRadius=5, 
                maxRadius=30
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Calculate confidence based on number of detected circles
                num_circles = len(circles[0])
                texture_confidence = min(num_circles / 10.0, 1.0)  # Cap at 10 circles
                
                # Check if circles align with the color masks
                for color_range in disease_info["color_ranges"]:
                    color_mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
                    circle_mask = np.zeros_like(gray)
                    
                    for i in circles[0, :]:
                        # Draw filled circle
                        cv2.circle(circle_mask, (i[0], i[1]), i[2], 255, -1)
                    
                    # Count pixels where both circle and color mask are positive
                    overlap = cv2.bitwise_and(color_mask, circle_mask)
                    overlap_pixels = cv2.countNonZero(overlap)
                    circle_pixels = cv2.countNonZero(circle_mask)
                    
                    if circle_pixels > 0:
                        overlap_ratio = overlap_pixels / circle_pixels
                        # Boost texture confidence if circles align with disease colors
                        texture_confidence *= (1.0 + overlap_ratio)
            
        elif disease_info["texture_patterns"] == "powdery_patches":
            # For Powdery Mildew - look for textured white patches
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            
            # Use local standard deviation to detect texture
            mean, stddev = cv2.meanStdDev(gray)
            
            # High contrast areas in grayscale often indicate powdery patches
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Threshold to find bright areas
            _, thresh = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to identify clusters
            kernel = np.ones((5, 5), np.uint8)
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Calculate texture metrics
            texture_pixels = cv2.countNonZero(opened)
            total_pixels = gray.size
            texture_ratio = texture_pixels / total_pixels
            
            # Standard deviation in grayscale indicates texture
            stddev_norm = float(stddev) / 255.0
            
            # Combine for texture confidence
            texture_confidence = (texture_ratio * 0.7) + (stddev_norm * 0.3)
            texture_confidence = min(texture_confidence * 3.0, 1.0)  # Scale up and cap
        
        # Combine color and texture confidences
        feature_confidence = (color_confidence * 0.6) + (texture_confidence * 0.4)
        
        # Scale by the confidence boost factor for this disease
        feature_confidence *= disease_info.get("confidence_boost", 1.0)
        
        # Ensure confidence is within [0, 1] range
        feature_confidence = min(max(feature_confidence, 0.0), 1.0)
        
        print(f"Disease {disease_info['name']} feature confidence: {feature_confidence:.2f} "
              f"(color: {color_confidence:.2f}, texture: {texture_confidence:.2f})")
        
        return feature_confidence
        
    except Exception as e:
        print(f"Error detecting disease features: {e}")
        return 0.0

# Enhanced image preprocessing for real-world images
def preprocess_image(image):
    """Apply multiple preprocessing techniques to improve real-world image compatibility"""
    try:
        # First segment the leaf (new step)
        mask, segmented_image = segment_leaf(image)
        
        # Convert PIL Image to OpenCV format (RGB to BGR)
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        # Use the segmented image for processing when possible
        segmented_cv = np.array(segmented_image)
        segmented_cv = segmented_cv[:, :, ::-1].copy()
        
        # Store original dimensions for debugging
        original_h, original_w = open_cv_image.shape[:2]
        print(f"Original image dimensions: {original_w}x{original_h}")
        
        # Create multiple enhanced versions for ensemble prediction
        enhanced_images = []
        
        # Check if segmented image has content
        has_segmented_content = np.mean(segmented_cv) > 5 
        
        # Start with the segmented image as our base if it has content
        base_image = segmented_cv if has_segmented_content else open_cv_image
            
        # Version 1: Standard enhancement
        try:
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(base_image, 9, 75, 75)
            
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
        except Exception as e:
            print(f"Warning: Standard enhancement failed: {e}")
        
        # Version 2: Stronger enhancement for difficult cases
        try:
            # Apply more aggressive noise reduction
            filtered2 = cv2.fastNlMeansDenoisingColored(base_image, None, 10, 10, 7, 21)
            
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
        except Exception as e:
            print(f"Warning: Strong enhancement failed: {e}")
        
        # Version 3: Focus on edge detection for disease boundary enhancement
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(base_image, (5, 5), 0)
            
            # Create edge-enhanced version using Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Combine the edge map with original image to enhance boundaries
            edge_enhanced = cv2.addWeighted(base_image, 0.7, edges_3channel, 0.3, 0)
            
            # Apply additional contrast and sharpness
            edge_enhanced = cv2.convertScaleAbs(edge_enhanced, alpha=1.2, beta=10)
            
            # Convert back to PIL image
            enhanced_pil3 = Image.fromarray(edge_enhanced[:, :, ::-1].copy())
            enhanced_pil3 = ImageEnhance.Sharpness(enhanced_pil3).enhance(1.7)
            
            enhanced_images.append(enhanced_pil3)
        except Exception as e:
            print(f"Warning: Edge enhancement failed: {e}")
        
        # Version 4: Apple Cedar Rust Enhancement (NEW)
        try:
            if base_image.size > 0:
                # Convert to HSV for better color manipulation
                hsv_rust = cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)
                
                # Split channels
                h_rust, s_rust, v_rust = cv2.split(hsv_rust)
                
                # Enhance orange-yellow range (hue ~ 20-40)
                # Create a mask for orange-yellow colors
                lower_orange = np.array([15, 100, 100])
                upper_orange = np.array([40, 255, 255])
                orange_mask = cv2.inRange(hsv_rust, lower_orange, upper_orange)
                
                # Dilate the mask slightly to enhance spotted areas
                kernel_dilate = np.ones((3, 3), np.uint8)
                orange_mask_dilated = cv2.dilate(orange_mask, kernel_dilate, iterations=1)
                
                # Create enhanced versions of s and v channels
                s_enhanced = s_rust.copy()
                v_enhanced = v_rust.copy()
                
                mask_idx = np.where(orange_mask_dilated > 0)
                if len(mask_idx[0]) > 0:  # Check if mask found any pixels
                    try:
                        # Get values at mask positions
                        s_values = s_rust[mask_idx]
                        v_values = v_rust[mask_idx]
                        
                        # Manual enhancement with direct calculations
                        s_values_enhanced = (s_values.astype(float) * 1.5).astype(np.uint8)
                        s_values_enhanced = np.clip(s_values_enhanced + 50, 0, 255).astype(np.uint8)
                        
                        v_values_enhanced = (v_values.astype(float) * 1.3).astype(np.uint8)
                        v_values_enhanced = np.clip(v_values_enhanced + 30, 0, 255).astype(np.uint8)
                        
                        # Put enhanced values back
                        s_enhanced[mask_idx] = s_values_enhanced
                        v_enhanced[mask_idx] = v_values_enhanced
                    except Exception as e:
                        print(f"Warning: Error enhancing apple cedar rust color values: {e}")
                
                # Merge channels back
                hsv_rust_enhanced = cv2.merge([h_rust, s_enhanced, v_enhanced])
                rust_enhanced = cv2.cvtColor(hsv_rust_enhanced, cv2.COLOR_HSV2BGR)
                
                # Apply sharpening to enhance rust spots
                rust_enhanced = cv2.filter2D(rust_enhanced, -1, kernel)
                
                # Convert back to PIL
                enhanced_pil4 = Image.fromarray(rust_enhanced[:, :, ::-1].copy())
                enhanced_pil4 = ImageEnhance.Contrast(enhanced_pil4).enhance(1.3)
                
                enhanced_images.append(enhanced_pil4)
        except Exception as e:
            print(f"Warning: Apple Cedar Rust enhancement failed: {e}")
        
        # Version 5: Squash Powdery Mildew Enhancement (NEW)
        try:
            if base_image.size > 0:
                # Start with bilateral filtering to preserve edges
                bilateral = cv2.bilateralFilter(base_image, 9, 75, 75)
                
                # Convert to LAB for better color manipulation
                lab_mildew = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
                l_mildew, a_mildew, b_mildew = cv2.split(lab_mildew)
                
                # Enhance brightness for white powdery areas
                clahe_mildew = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_enhanced = clahe_mildew.apply(l_mildew)
                
                # Create a mask for white areas (high L, low a, low b)
                _, white_thresh = cv2.threshold(l_enhanced, 180, 255, cv2.THRESH_BINARY)
                
                mask_idx = np.where(white_thresh > 0)
                if len(mask_idx[0]) > 0:  # Check if mask found any pixels
                    try:
                        # Get L channel values at mask positions
                        l_values = l_mildew[mask_idx]
                        
                        # Manual enhancement with direct calculations
                        l_enhanced_values = (l_values.astype(float) * 1.2).astype(np.uint8)
                        l_enhanced_values = np.clip(l_enhanced_values + 15, 0, 255).astype(np.uint8)
                        
                        # Put enhanced values back
                        l_enhanced = l_mildew.copy()
                        l_enhanced[mask_idx] = l_enhanced_values
                    except Exception as e:
                        print(f"Warning: Error enhancing squash powdery mildew white areas: {e}")
                        l_enhanced = clahe_mildew.apply(l_mildew)  # Fallback
                else:
                    # No white areas found, just use CLAHE enhanced version
                    l_enhanced = clahe_mildew.apply(l_mildew)
                
                # Merge channels back
                lab_enhanced = cv2.merge([l_enhanced, a_mildew, b_mildew])
                mildew_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
                
                # Apply local contrast enhancement for powdery texture
                gray_mildew = cv2.cvtColor(mildew_enhanced, cv2.COLOR_BGR2GRAY)
                clahe_texture = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
                texture_enhanced = clahe_texture.apply(gray_mildew)
                
                # Use the texture-enhanced image to modify the color image
                texture_3channel = cv2.cvtColor(texture_enhanced, cv2.COLOR_GRAY2BGR)
                mildew_enhanced = cv2.addWeighted(mildew_enhanced, 0.7, texture_3channel, 0.3, 0)
                
                # Convert back to PIL
                enhanced_pil5 = Image.fromarray(mildew_enhanced[:, :, ::-1].copy())
                enhanced_pil5 = ImageEnhance.Sharpness(enhanced_pil5).enhance(1.5)
                
                enhanced_images.append(enhanced_pil5)
        except Exception as e:
            print(f"Warning: Squash Powdery Mildew enhancement failed: {e}")
        
        # Include the original segmented image as one of the variations if it has content
        if has_segmented_content:
            enhanced_images.append(segmented_image)
        
        # If all enhancements failed, return the original image
        if not enhanced_images:
            enhanced_images = [image]
            
        print(f"Successfully created {len(enhanced_images)} enhanced versions of the image")
        return enhanced_images
    except Exception as e:
        print(f"Warning: Advanced preprocessing failed completely: {e}")
        import traceback
        traceback.print_exc()
        # If everything fails, just return the original image
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
        filtered_probs = torch.zeros_like(probabilities)
        
        if plant_confidence > 0.05:  # If we have reasonable confidence in plant type
            # Filter predictions to only this plant family
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
        
        # Stage 4: Disease-specific Feature Detection (NEW)
        # Check for disease-specific visual features in the top candidates
        feature_scores = {}
        
        # Check for Squash Powdery Mildew specifically
        # Enhanced detection for cases where the model confuses it with grape diseases
        squash_mildew_feature_score = 0
        if 25 in [top_indices[i].item() for i in range(min(5, len(top_indices)))] or any("Grape" in CLASS_LABELS.get(top_indices[i].item(), "") for i in range(3)):
            # If Squash Powdery Mildew is in top 5, or there are grape diseases in top 3
            # Perform targeted feature detection for white powdery patches
            squash_mildew_feature_score = detect_disease_features(image, 25)
            feature_scores[25] = squash_mildew_feature_score
            print(f"Special check for Squash Powdery Mildew: {squash_mildew_feature_score:.2f}")
            
            # If we have significant white powdery patches, boost Squash Powdery Mildew probability
            if squash_mildew_feature_score > 0.35:
                # Check for confusion with grape diseases
                grape_disease_detected = False
                for i in range(min(2, len(top_indices))):
                    if top_indices[i].item() in [11, 12, 14]:  # Grape diseases
                        grape_disease_detected = True
                        break
                
                if grape_disease_detected:
                    # If top prediction is a grape disease but we detect powdery mildew features
                    print("Potential Squash Powdery Mildew misclassified as Grape disease")
                    # Boost squash powdery mildew score and reduce grape disease score
                    probabilities[25] = max(probabilities[25] * 2.5, 0.6)  # Significant boost
                    
                    # Update prediction if feature detection is confident
                    if squash_mildew_feature_score > 0.4:
                        predicted_class = 25
                        confidence_value = probabilities[25].item()
                        disease_name = CLASS_LABELS[25]
                        print(f"Overriding with Squash Powdery Mildew, new confidence: {confidence_value:.2%}")
        
        # Check each top candidate for specific disease features
        for i in range(min(3, len(top_indices))):
            class_id = top_indices[i].item()
            if class_id in DISEASE_CHARACTERISTICS and class_id not in feature_scores:
                # Detect disease-specific features
                feature_score = detect_disease_features(image, class_id)
                feature_scores[class_id] = feature_score
                
                print(f"Feature detection for {CLASS_LABELS[class_id]}: {feature_score:.2f}")
                
                # If this is Apple Cedar Rust or Squash Powdery Mildew, give special attention
                if class_id in [2, 25] and feature_score > 0.3:
                    # Boost the probability if disease-specific features are detected
                    boost_factor = 1.0 + (feature_score * 0.5)  # Up to 50% boost
                    probabilities[class_id] *= boost_factor
                    
                    # Update predictions with boosted probabilities
                    if probabilities[class_id] > probabilities[predicted_class]:
                        predicted_class = class_id
                        confidence_value = probabilities[class_id].item()
                        disease_name = CLASS_LABELS.get(predicted_class, "Unknown Disease")
                        
                        print(f"Updated prediction based on disease-specific features:")
                        print(f"- {disease_name}, Confidence: {confidence_value:.2%}")
        
        # Special checks for confusion between Apple Cedar Rust and Corn diseases
        # Specifically address the case where Apple Cedar Rust is misidentified as Corn Gray Leaf Spot
        if "Corn" in most_likely_plant and predicted_class in [7, 8]:  # Corn Gray Leaf Spot or Corn Common Rust
            # Check if Apple Cedar Rust has strong feature detection
            cedar_rust_feature_score = detect_disease_features(image, 2)  # Apple Cedar Rust
            feature_scores[2] = cedar_rust_feature_score
            print(f"Special confusion check: Apple Cedar Rust vs Corn disease - Feature score: {cedar_rust_feature_score:.2f}")
            
            # If Apple Cedar Rust features are strong (>=0.4) and it's in top candidates, override
            apple_cedar_rust_prob = probabilities[2].item()
            if cedar_rust_feature_score >= 0.4 and apple_cedar_rust_prob > 0.05:
                print(f"Detected potential confusion between Apple Cedar Rust and Corn disease")
                # Calculate a new confidence based on feature score and model probability
                new_confidence = max(apple_cedar_rust_prob * 1.5, cedar_rust_feature_score * 0.7)
                
                # Override only if we're reasonably confident
                if new_confidence > 0.4:
                    predicted_class = 2  # Apple Cedar Rust
                    confidence_value = new_confidence
                    disease_name = CLASS_LABELS[2]
                    most_likely_plant = "Apple"  # Override the plant family too
                    print(f"Overriding with Apple Cedar Rust, new confidence: {confidence_value:.2%}")
        
        # Special handling for Apple Cedar Rust (class_id 2)
        if 2 in feature_scores and feature_scores[2] > 0.4:
            # If we have strong Apple Cedar Rust features, override prediction regardless of plant family
            if predicted_class != 2:
                print("High confidence Apple Cedar Rust features detected, overriding prediction")
                predicted_class = 2
                confidence_value = max(confidence_value, 0.7)  # Minimum confidence of 70%
                disease_name = CLASS_LABELS[2]
                most_likely_plant = "Apple"  # Set the plant type to Apple
        
        # Add plant type to disease name for clarity
        if most_likely_plant not in disease_name:
            disease_name = f"{most_likely_plant} - {disease_name}"
            
        # Return results with additional metadata
        return {
            "disease": disease_name,
            "confidence": confidence_value,
            "plant_type": most_likely_plant,
            "class_id": predicted_class if isinstance(predicted_class, int) else predicted_class.item(),
            "top_candidates": [
                {
                    "disease": CLASS_LABELS.get(top_indices[i].item()),
                    "confidence": top_probs[i].item(),
                    "plant_type": CLASS_TO_PLANT.get(top_indices[i].item(), "Unknown"),
                    "feature_score": feature_scores.get(top_indices[i].item(), 0.0)
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
        segmented_images = []
        
        # Process each image
        for image in images:
            try:
                print(f"Processing image: {image.filename}")
                img = Image.open(image).convert('RGB')
                
                # Segment the image (save for potential output)
                mask, segmented = segment_leaf(img)
                
                # Save segmented image in temporary folder if available
                try:
                    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../static/temp')
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Generate unique filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    seg_filename = f"segmented_{timestamp}_{image.filename}"
                    seg_path = os.path.join(temp_dir, seg_filename)
                    
                    # Save segmented image
                    segmented.save(seg_path)
                    segmented_images.append(seg_path)
                except Exception as e:
                    print(f"Warning: Could not save segmented image: {e}")
                
                # Get prediction for this image
                prediction = predict_single_image(img)
                
                # Store results
                results.append(prediction["disease"])
                confidences.append(prediction["confidence"])
                plant_types.append(prediction["plant_type"])
                
                # Store debug info
                image_debug = {}
                for class_id, class_name in CLASS_LABELS.items():
                    # Check if this class is in top candidates
                    is_top_candidate = False
                    feature_score = 0.0
                    for candidate in prediction.get("top_candidates", []):
                        if candidate.get("disease") == class_name:
                            is_top_candidate = True
                            feature_score = candidate.get("feature_score", 0.0)
                            break
                    
                    # Store score (1.0 if top match, between 0-1 if feature detected)
                    image_debug[class_name] = 1.0 if class_id == prediction["class_id"] else (
                        feature_score if is_top_candidate else 0.0
                    )
                
                # Add segmentation info
                image_debug["segmented_image"] = seg_path if segmented_images else None
                
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
        
        # Additional disease-specific information
        disease_info = {}
        
        # For Apple Cedar Rust
        if "Apple Cedar Rust" in aggregated_result:
            disease_info = {
                "common_name": "Cedar Apple Rust",
                "scientific_name": "Gymnosporangium juniperi-virginianae",
                "description": "A fungal disease affecting apple trees that requires cedar trees to complete its life cycle.",
                "symptoms": [
                    "Bright orange or yellow spots on leaves",
                    "Circular lesions with orange centers",
                    "Black spots in center of lesions (later stages)",
                    "Swollen or deformed leaf tissue"
                ],
                "treatment": [
                    "Apply fungicides labeled for rust diseases early in the growing season",
                    "Remove nearby cedar trees if possible (source of spores)",
                    "Improve air circulation by pruning",
                    "Clean up fallen leaves to reduce overwinter survival",
                    "Use resistant apple varieties for new plantings"
                ],
                "severity_level": "Moderate to severe if left untreated"
            }
        
        # For Squash Powdery Mildew
        elif "Squash Powdery Mildew" in aggregated_result:
            disease_info = {
                "common_name": "Powdery Mildew",
                "scientific_name": "Erysiphe cichoracearum or Sphaerotheca fuliginea",
                "description": "A fungal disease that appears as a white powdery substance on leaves, stems and sometimes fruit of squash plants.",
                "symptoms": [
                    "White powdery spots on upper leaf surfaces",
                    "Spots that merge to cover larger areas",
                    "Yellowing of infected leaves",
                    "Premature leaf drop",
                    "Stunted growth of new foliage",
                    "Reduced yield and quality of fruit"
                ],
                "treatment": [
                    "Apply fungicides such as sulfur, neem oil, or potassium bicarbonate",
                    "Ensure good air circulation by proper plant spacing",
                    "Use drip irrigation instead of overhead watering to keep foliage dry",
                    "Remove and destroy infected plant parts",
                    "Plant resistant varieties in future seasons",
                    "Apply organic options like diluted milk spray (1:10 ratio of milk to water)"
                ],
                "severity_level": "Moderate, can reduce yield significantly if untreated"
            }
        
        return {
            "disease": aggregated_result,
            "confidence": confidence_str,
            "plant_type": most_common_plant,
            "debug_info": debug_info,
            "segmented_images": segmented_images,
            "disease_info": disease_info
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
