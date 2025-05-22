#!/usr/bin/env python
# model_validator.py - Comprehensive model validation through internet image scraping
# Usage: python model_validator.py --diseases "apple cedar rust,apple black rot" --images 20

import os
import sys
import time
import json
import random
import logging
import requests
import numpy as np
import torch
import cv2
import re
from PIL import Image
from io import BytesIO
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.disease_service import predict_single_image, CLASS_LABELS, PLANT_FAMILIES, CLASS_TO_PLANT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories
VALIDATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation')
IMAGES_DIR = os.path.join(VALIDATION_DIR, 'images')
RESULTS_DIR = os.path.join(VALIDATION_DIR, 'results')

os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Constants
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0'
}
DELAY_RANGE = (1, 3)  # Random delay between requests in seconds

# Disease mapping: Maps search terms to expected class indices
DISEASE_MAPPING = {
    'apple scab': 0,
    'apple black rot': 1,
    'apple cedar rust': 2,
    'healthy apple': 3,
    'blueberry healthy': 4,
    'cherry healthy': 5,
    'cherry powdery mildew': 6,
    'corn gray leaf spot': 7,
    'corn common rust': 8,
    'corn healthy': 9,
    'corn northern leaf blight': 10,
    'grape black rot': 11,
    'grape esca black measles': 12,
    'grape healthy': 13,
    'grape leaf blight': 14,
    'orange citrus greening': 15,
    'peach bacterial spot': 16,
    'peach healthy': 17,
    'pepper bacterial spot': 18,
    'pepper healthy': 19,
    'potato early blight': 20,
    'potato healthy': 21,
    'potato late blight': 22,
    'raspberry healthy': 23,
    'soybean healthy': 24,
    'squash powdery mildew': 25,
    'strawberry healthy': 26,
    'strawberry leaf scorch': 27,
    'tomato bacterial spot': 28,
    'tomato early blight': 29,
    'tomato healthy': 30,
    'tomato late blight': 31,
    'tomato leaf mold': 32,
    'tomato septoria leaf spot': 33,
    'tomato spider mites': 34,
    'tomato target spot': 35,
    'tomato mosaic virus': 36,
    'tomato yellow leaf curl virus': 37
}

# Add sample image source from project
SAMPLE_IMAGES = {
    'apple cedar rust': [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'sample_images', 'cedar_rust_1.jpg'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'sample_images', 'cedar_rust_2.jpg'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'sample_images', 'cedar_rust_3.jpg')
    ],
    'apple black rot': [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'sample_images', 'black_rot_1.jpg'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'sample_images', 'black_rot_2.jpg'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'sample_images', 'black_rot_3.jpg')
    ]
}

def create_synthetic_cedar_rust_image(output_path):
    """Create a synthetic image that resembles cedar apple rust disease with improved realism"""
    # Create a base green leaf
    img = np.ones((256, 256, 3), dtype=np.uint8) * np.array([30, 120, 40], dtype=np.uint8)  # Dark green base
    
    # Add leaf texture with more natural veining
    # Main vein
    cv2.line(img, (128, 0), (128, 256), (20, 90, 30), 2)
    
    # Secondary veins
    for i in range(5, 10):
        angle = random.uniform(20, 70)
        length = random.uniform(0.5, 0.9) * 128
        x1 = 128
        y1 = i * 25
        x2 = int(x1 + length * np.cos(np.radians(angle)))
        y2 = y1
        cv2.line(img, (x1, y1), (x2, y2), (20, 100, 30), 1)
        
        # Mirror to other side
        x2_mirror = int(x1 - length * np.cos(np.radians(angle)))
        cv2.line(img, (x1, y1), (x2_mirror, y2), (20, 100, 30), 1)
    
    # Add leaf border irregularity
    leaf_contour = np.array([[10, 128], [30, 30], [128, 10], [226, 30], [246, 128], 
                             [226, 226], [128, 246], [30, 226]], dtype=np.int32)
    # Add randomness to leaf shape
    for i in range(len(leaf_contour)):
        leaf_contour[i] = leaf_contour[i] + np.random.randint(-15, 15, 2)
    
    # Create leaf mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.fillPoly(mask, [leaf_contour], 255)
    
    # Add cedar rust spots with gradual blending (orange-yellow circular lesions)
    for _ in range(random.randint(15, 25)):
        center = (random.randint(40, 216), random.randint(40, 216))
        radius = random.randint(8, 20)
        
        # Create more realistic rust spots with gradient
        for r in range(radius, 0, -1):
            intensity = 1 - (r / radius)
            # Blend rust color: orange-yellow with darker border
            color = (
                int(40 + 30 * intensity),  # B
                int(120 + 80 * intensity),  # G
                int(220 - 40 * intensity)   # R
            )
            cv2.circle(img, center, r, color, 1)
        
        # Sometimes add the typical black dots in the center of the lesion
        if random.random() > 0.4:
            for _ in range(random.randint(5, 12)):
                offset_x = random.randint(-3, 3)
                offset_y = random.randint(-3, 3)
                cv2.circle(img, (center[0] + offset_x, center[1] + offset_y), 1, (0, 0, 0), -1)
    
    # Apply leaf mask
    img_masked = img.copy()
    for c in range(3):
        img_masked[:, :, c] = cv2.bitwise_and(img[:, :, c], mask)
    
    # Add a subtle synthetic image marker in bottom right (small "SYN" text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_masked, f"SYN {datetime.now().strftime('%m%d')}", (175, 245), font, 0.4, (200, 200, 200), 1)
    
    # Save the image
    cv2.imwrite(output_path, img_masked)
    logger.info(f"Created synthetic Cedar Rust image: {output_path}")
    return output_path

def create_synthetic_black_rot_image(output_path):
    """Create a synthetic image that resembles black rot disease with improved realism"""
    # Create a base green leaf
    img = np.ones((256, 256, 3), dtype=np.uint8) * np.array([30, 120, 40], dtype=np.uint8)  # Dark green base
    
    # Add leaf texture with more natural veining
    # Main vein
    cv2.line(img, (128, 0), (128, 256), (20, 90, 30), 2)
    
    # Secondary veins
    for i in range(5, 10):
        angle = random.uniform(20, 70)
        length = random.uniform(0.5, 0.9) * 128
        x1 = 128
        y1 = i * 25
        x2 = int(x1 + length * np.cos(np.radians(angle)))
        y2 = y1
        cv2.line(img, (x1, y1), (x2, y2), (20, 100, 30), 1)
        
        # Mirror to other side
        x2_mirror = int(x1 - length * np.cos(np.radians(angle)))
        cv2.line(img, (x1, y1), (x2_mirror, y2), (20, 100, 30), 1)
    
    # Add leaf border irregularity
    leaf_contour = np.array([[10, 128], [30, 30], [128, 10], [226, 30], [246, 128], 
                             [226, 226], [128, 246], [30, 226]], dtype=np.int32)
    # Add randomness to leaf shape
    for i in range(len(leaf_contour)):
        leaf_contour[i] = leaf_contour[i] + np.random.randint(-15, 15, 2)
    
    # Create leaf mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.fillPoly(mask, [leaf_contour], 255)
    
    # Add black rot spots (brown-black circular lesions with concentric rings)
    for _ in range(random.randint(5, 9)):
        center = (random.randint(40, 216), random.randint(40, 216))
        max_radius = random.randint(15, 40)
        
        # Create the typical "frog-eye" pattern with gradient edges
        for r in range(max_radius, 0, -1):
            # Calculate intensity based on radius position
            intensity = r / max_radius
            cycle_position = (r % 10) / 10  # Create bands
            
            # Use the cycle position to create concentric rings
            if cycle_position < 0.5:
                # Darker rings
                color = (
                    int(30 * intensity),  # B
                    int(30 * intensity),  # G
                    int(30 * intensity)   # R
                )
            else:
                # Lighter rings
                color = (
                    int(40 * intensity),  # B
                    int(50 * intensity),  # G
                    int(80 * intensity)   # R
                )
            
            cv2.circle(img, center, r, color, 1)
        
        # Add the tiny black fruiting bodies that are characteristic of black rot
        for _ in range(random.randint(15, 30)):
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(5, max_radius - 5)
            x = int(center[0] + distance * np.cos(angle))
            y = int(center[1] + distance * np.sin(angle))
            cv2.circle(img, (x, y), 1, (10, 10, 10), -1)
    
    # Apply leaf mask
    img_masked = img.copy()
    for c in range(3):
        img_masked[:, :, c] = cv2.bitwise_and(img[:, :, c], mask)
    
    # Add a subtle synthetic image marker in bottom right (small "SYN" text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_masked, f"SYN {datetime.now().strftime('%m%d')}", (175, 245), font, 0.4, (200, 200, 200), 1)
    
    # Save the image
    cv2.imwrite(output_path, img_masked)
    logger.info(f"Created synthetic Black Rot image: {output_path}")
    return output_path

def create_synthetic_disease_image(output_path, disease_name="generic"):
    """Create a generic synthetic disease image as fallback with improved realism"""
    # Create a base green leaf
    img = np.ones((256, 256, 3), dtype=np.uint8) * np.array([30, 120, 40], dtype=np.uint8)
    
    # Add leaf texture with more natural veining
    # Main vein
    cv2.line(img, (128, 0), (128, 256), (20, 90, 30), 2)
    
    # Secondary veins
    for i in range(5, 10):
        angle = random.uniform(20, 70)
        length = random.uniform(0.5, 0.9) * 128
        x1 = 128
        y1 = i * 25
        x2 = int(x1 + length * np.cos(np.radians(angle)))
        y2 = y1
        cv2.line(img, (x1, y1), (x2, y2), (20, 100, 30), 1)
        
        # Mirror to other side
        x2_mirror = int(x1 - length * np.cos(np.radians(angle)))
        cv2.line(img, (x1, y1), (x2_mirror, y2), (20, 100, 30), 1)
    
    # Add leaf border irregularity
    leaf_contour = np.array([[10, 128], [30, 30], [128, 10], [226, 30], [246, 128], 
                             [226, 226], [128, 246], [30, 226]], dtype=np.int32)
    # Add randomness to leaf shape
    for i in range(len(leaf_contour)):
        leaf_contour[i] = leaf_contour[i] + np.random.randint(-15, 15, 2)
    
    # Create leaf mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.fillPoly(mask, [leaf_contour], 255)
    
    # Adjust coloration based on disease name
    if "healthy" in disease_name.lower():
        # Create a healthy-looking leaf with minimal spots
        base_color = np.array([30, 150, 50], dtype=np.uint8)  # Bright green
        spot_count = random.randint(0, 5)  # Few or no spots
        spot_color_range = [(25, 145, 45), (35, 155, 55)]  # Similar to leaf color
    elif any(x in disease_name.lower() for x in ["blight", "late blight", "early blight"]):
        # Create blight-like symptoms (dark brown spots with yellow halos)
        base_color = np.array([30, 130, 40], dtype=np.uint8)  # Slightly pale green
        spot_count = random.randint(15, 25)
        spot_color_range = [(20, 20, 20), (50, 50, 80)]  # Dark spots
    elif "rust" in disease_name.lower():
        # Create rust-like symptoms (orange/rust colored spots)
        base_color = np.array([30, 120, 40], dtype=np.uint8)
        spot_count = random.randint(20, 30)
        spot_color_range = [(30, 100, 180), (50, 140, 220)]  # Orange/rust colors
    elif "mold" in disease_name.lower() or "mildew" in disease_name.lower():
        # Create mold/mildew-like symptoms (white/gray powdery spots)
        base_color = np.array([30, 120, 40], dtype=np.uint8)
        spot_count = random.randint(20, 40)
        spot_color_range = [(150, 150, 150), (200, 200, 200)]  # White/gray colors
    else:
        # Generic disease symptoms
        base_color = np.array([30, 110, 35], dtype=np.uint8)
        spot_count = random.randint(10, 20)
        spot_color_range = [(20, 50, 70), (60, 90, 110)]  # Brown spots
    
    # Apply base color
    img = np.ones((256, 256, 3), dtype=np.uint8) * base_color
    
    # Add disease spots with varied appearance
    for _ in range(spot_count):
        center = (random.randint(40, 216), random.randint(40, 216))
        radius = random.randint(5, 20)
        
        # Get a random color from the range
        color_b = random.randint(spot_color_range[0][0], spot_color_range[1][0])
        color_g = random.randint(spot_color_range[0][1], spot_color_range[1][1])
        color_r = random.randint(spot_color_range[0][2], spot_color_range[1][2])
        color = (color_b, color_g, color_r)
        
        # Random shape: circle, ellipse, or irregular
        shape_type = random.choice(["circle", "ellipse", "irregular"])
        
        if shape_type == "circle":
            cv2.circle(img, center, radius, color, -1)
            # Add feathered edge for realism
            for r in range(1, 5):
                edge_color = (
                    int((color_b + base_color[0])/2),
                    int((color_g + base_color[1])/2),
                    int((color_r + base_color[2])/2)
                )
                cv2.circle(img, center, radius + r, edge_color, 1)
        
        elif shape_type == "ellipse":
            axes = (radius, int(radius * random.uniform(0.5, 1.5)))
            angle = random.randint(0, 180)
            cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
        
        else:  # irregular
            points = []
            for i in range(8):
                angle = i * 45 + random.randint(-20, 20)
                r = radius * random.uniform(0.7, 1.3)
                x = int(center[0] + r * np.cos(np.radians(angle)))
                y = int(center[1] + r * np.sin(np.radians(angle)))
                points.append((x, y))
            
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(img, [points], color)
    
    # Apply leaf mask
    img_masked = img.copy()
    for c in range(3):
        img_masked[:, :, c] = cv2.bitwise_and(img[:, :, c], mask)
    
    # Add a subtle synthetic image marker in bottom right (small "SYN" text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_masked, f"SYN {datetime.now().strftime('%m%d')}", (175, 245), font, 0.4, (200, 200, 200), 1)
    
    # Save the image
    cv2.imwrite(output_path, img_masked)
    logger.info(f"Created synthetic disease image for {disease_name}: {output_path}")
    return output_path

def search_images(query, num_images=20):
    """Search for images using DuckDuckGo as primary source with fallbacks"""
    logger.info(f"Searching for '{query}' images")
    image_urls = []
    sanitized_query = query.replace(' ', '_').lower()
    target_dir = os.path.join(IMAGES_DIR, sanitized_query)
    os.makedirs(target_dir, exist_ok=True)

    # 1. DuckDuckGo image search as exclusive primary source
    try:
        logger.info(f"Searching DuckDuckGo for '{query}' images")
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # Enhanced search terms for more specific results
            search_terms = [
                f"{query} plant disease leaf high resolution photo",
                f"{query} plant disease symptoms close-up",
                f"{query} plant disease identification guide",
                f"{query} agriculture plant disease diagnostic",
                f"{query} plant pathology specimen",
                f"{query} {query.split()[0]} leaf disease field photo" # Specific plant term
            ]
            
            for search_term in search_terms:
                if len(image_urls) >= num_images * 1.5:  # Get extra images to filter later
                    break
                    
                ddg_results = list(ddgs.images(
                    search_term, 
                    max_results=min(30, num_images),  # Request more per term
                    safesearch='off',
                    size=None,  # Any size
                    color=None, # Any color
                    type_image=None,  # Any type
                    layout=None,
                    license_image=None
                ))
                
                for item in ddg_results:
                    url = item.get('image')
                    if url and url not in image_urls:
                        # Basic URL validation
                        if url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                            image_urls.append(url)
                
                # Pause between searches
                time.sleep(random.uniform(*DELAY_RANGE))
                
            logger.info(f"Found {len(image_urls)} images via DuckDuckGo")
    except ImportError:
        logger.warning("duckduckgo_search module not found. Installing...")
        try:
            import pip
            pip.main(['install', 'duckduckgo_search'])
            # Try again after installation
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                ddg_results = list(ddgs.images(
                    f"{query} plant disease", 
                    max_results=num_images,
                    safesearch='off'
                ))
                
                for item in ddg_results:
                    url = item.get('image')
                    if url and url not in image_urls:
                        image_urls.append(url)
                        
                logger.info(f"Found {len(image_urls)} images via DuckDuckGo after installation")
        except Exception as e:
            logger.error(f"Failed to install duckduckgo_search: {e}")
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
    
    # 2. Google Custom Search API fallback (only if DuckDuckGo failed completely)
    if len(image_urls) == 0:
        logger.warning(f"DuckDuckGo found no images. Trying Google Custom Search API for '{query}' images")
        api_key = os.getenv('GOOGLE_API_KEY')
        cse_id = os.getenv('GOOGLE_CSE_ID')
        if api_key and cse_id:
            try:
                from googleapiclient.discovery import build
                service = build('customsearch', 'v1', developerKey=api_key)
                res = service.cse().list(
                    q=f"{query} plant disease leaf", searchType='image', num=num_images, cx=cse_id
                ).execute()
                for item in res.get('items', []):
                    url = item.get('link')
                    if url and url not in image_urls:
                        image_urls.append(url)
                logger.info(f"Found {len(image_urls)} total URLs after Google API")
            except Exception as e:
                logger.warning(f"Google Custom Search API failed: {e}")

    # 3. Local sample images fallback
    if len(image_urls) < num_images and query.lower() in SAMPLE_IMAGES:
        logger.info(f"Using local sample images for '{query}'")
        for path in SAMPLE_IMAGES[query.lower()][: num_images - len(image_urls)]:
            if os.path.exists(path):
                dest = os.path.join(target_dir, os.path.basename(path))
                open(dest, 'wb').write(open(path, 'rb').read())
                image_urls.append(dest)

    # 4. Synthetic fallback
    if len(image_urls) < num_images:
        need = num_images - len(image_urls)
        logger.warning(f"Generating {need} synthetic images for '{query}'")
        for i in range(need):
            p = os.path.join(target_dir, f"{sanitized_query}_synthetic_{i+1}.jpg")
            if 'cedar rust' in query.lower(): 
                create_synthetic_cedar_rust_image(p)
            elif 'black rot' in query.lower(): 
                create_synthetic_black_rot_image(p)
            else: 
                create_synthetic_disease_image(p, query)
            image_urls.append(p)

    logger.info(f"Total images collected for '{query}': {len(image_urls)}")
    return image_urls[:num_images]

def process_image(image_path_or_url, disease_name=None, expected_class=None, apply_segmentation=False):
    """Process an image from a URL or local path, applying leaf segmentation if required and make predictions."""
    try:
        if isinstance(image_path_or_url, str) and image_path_or_url.startswith(('http://', 'https://')):
            # It's a URL, use requests to download
            logger.info(f"Downloading image from {image_path_or_url[:50]}...")
            try:
                response = requests.get(image_path_or_url, headers=HEADERS, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Failed to download image, status code: {response.status_code}")
                    return None
                img_data = BytesIO(response.content)
                image = Image.open(img_data)
            except Exception as e:
                logger.error(f"Error downloading image from URL: {e}")
                return None
            
            # Save the image locally for future reference
            if disease_name:
                sanitized_query = disease_name.replace(' ', '_').lower()
                os.makedirs(os.path.join(IMAGES_DIR, sanitized_query), exist_ok=True)
                local_path = os.path.join(IMAGES_DIR, sanitized_query, f"{int(time.time())}_{random.randint(1000, 9999)}.jpg")
                image.save(local_path)
                image_path = local_path
            else:
                image_path = image_path_or_url
        else:
            # It's a local file path, open directly
            logger.info(f"Opening local image from {image_path_or_url}")
            try:
                image = Image.open(image_path_or_url)
                image_path = image_path_or_url
            except Exception as e:
                logger.error(f"Error opening local image file: {e}")
                return None
            
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply leaf segmentation if requested
        if apply_segmentation:
            try:
                # Convert to numpy array for OpenCV processing
                img_array = np.array(image)
                segmented = segment_leaf(img_array)
                if segmented is not None:
                    image = Image.fromarray(segmented)
            except Exception as e:
                logger.warning(f"Segmentation failed: {e} - using original image")
        
        # Resize to a standard size expected by the model
        image = image.resize((224, 224))
        
        # Only run prediction if expected_class is provided (we're evaluating, not just processing)
        if expected_class is not None:
            # Use the disease_service's predict function
            try:
                prediction_result = predict_single_image(image)
                
                predicted_class = prediction_result['predicted_class']
                confidence = prediction_result['confidence']
                
                # Check if the prediction is correct
                is_correct = (predicted_class == expected_class)
                
                # Determine if this is a synthetic image (look for "SYN" or "synthetic" marker in filename or path)
                is_synthetic = "synthetic" in str(image_path).lower() or "_syn_" in str(image_path).lower()
                
                logger.info(f"Image {os.path.basename(str(image_path))} - Expected: {expected_class}, Predicted: {predicted_class}, Correct: {is_correct}, Confidence: {confidence:.2f}")
                
                return {
                    'image_path': image_path,
                    'expected_class': expected_class,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'is_correct': is_correct,
                    'is_synthetic': is_synthetic,
                    'disease_name': disease_name
                }
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return None
        
        return image
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def scrape_images(queries, num_images=20, use_segmentation=True):
    """
    Scrape images for given queries and evaluate them
    
    Args:
        queries: Dict mapping search terms to expected class indices
        num_images: Number of images to scrape per query
        use_segmentation: Whether to use leaf segmentation
        
    Returns:
        Dictionary with results and evaluation metrics
    """
    all_results = {}
    
    for query, expected_class in queries.items():
        logger.info(f"Processing query: '{query}' (Expected class: {expected_class})")
        
        # Create folder for this disease
        sanitized_query = query.replace(' ', '_').lower()
        disease_dir = os.path.join(IMAGES_DIR, sanitized_query)
        os.makedirs(disease_dir, exist_ok=True)
        
        # Search for images using enhanced search function
        image_paths = search_images(query, num_images)
        
        logger.info(f"Evaluating {len(image_paths)} images for '{query}'")
        
        # Process each image
        disease_results = []
        for path in image_paths:
            result = process_image(path, query, expected_class, use_segmentation)
            if result:
                disease_results.append(result)
        
        all_results[query] = disease_results
    
    return all_results

def generate_visualization(results, output_dir):
    """Generate visualizations for the validation results"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all predictions for confusion matrix
    y_true = []
    y_pred = []
    classes = []
    
    # Also split by synthetic vs real
    y_true_real = []
    y_pred_real = []
    y_true_syn = []
    y_pred_syn = []
    
    synthetic_counts = {"total": 0, "correct": 0}
    real_counts = {"total": 0, "correct": 0}
    
    for disease_name, disease_results in results.items():
        for result in disease_results:
            y_true.append(result['expected_class'])
            y_pred.append(result['predicted_class'])
            
            # Track synthetic vs real separately
            is_synthetic = result.get('is_synthetic', False)
            
            if is_synthetic:
                y_true_syn.append(result['expected_class'])
                y_pred_syn.append(result['predicted_class'])
                synthetic_counts["total"] += 1
                if result['is_correct']:
                    synthetic_counts["correct"] += 1
            else:
                y_true_real.append(result['expected_class'])
                y_pred_real.append(result['predicted_class'])
                real_counts["total"] += 1
                if result['is_correct']:
                    real_counts["correct"] += 1
            
            # Keep track of class names for plotting
            if result['expected_class'] not in classes:
                classes.append(result['expected_class'])
            if result['predicted_class'] not in classes:
                classes.append(result['predicted_class'])
    
    # Sort classes for consistency
    classes = sorted(classes)
    
    # If no results, return early
    if not y_true:
        logger.warning("No results to visualize")
        return
    
    # 1. Confusion Matrix - All Images
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Create a mapping from class index to class name for better readability
    class_names = [CLASS_LABELS[c] for c in classes]
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - All Images')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_all.png"))
    
    # 2. Confusion Matrix - Real Images Only
    if len(y_true_real) > 0:
        plt.figure(figsize=(14, 12))
        cm_real = confusion_matrix(y_true_real, y_pred_real, labels=classes)
        sns.heatmap(cm_real, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Real Images Only')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_real.png"))
    
    # 3. Confusion Matrix - Synthetic Images Only
    if len(y_true_syn) > 0:
        plt.figure(figsize=(14, 12))
        cm_syn = confusion_matrix(y_true_syn, y_pred_syn, labels=classes)
        sns.heatmap(cm_syn, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Synthetic Images Only')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_synthetic.png"))
    
    # 4. Bar chart comparing real vs synthetic accuracy
    plt.figure(figsize=(10, 6))
    real_acc = real_counts["correct"] / real_counts["total"] if real_counts["total"] > 0 else 0
    syn_acc = synthetic_counts["correct"] / synthetic_counts["total"] if synthetic_counts["total"] > 0 else 0
    
    accuracies = [real_acc, syn_acc]
    labels = ['Real Images', 'Synthetic Images']
    colors = ['#3498db', '#e74c3c']
    
    plt.bar(labels, accuracies, color=colors)
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy: Real vs Synthetic Images')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.05, f"{v:.2%}", ha='center')
    plt.savefig(os.path.join(output_dir, "real_vs_synthetic_accuracy.png"))
    
    # 5. Generate summary report
    with open(os.path.join(output_dir, "validation_summary.txt"), "w") as f:
        f.write("Plant Disease Model Validation Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Overall Results:\n")
        total_accuracy = (real_counts["correct"] + synthetic_counts["correct"]) / (real_counts["total"] + synthetic_counts["total"]) if (real_counts["total"] + synthetic_counts["total"]) > 0 else 0
        f.write(f"Total Images: {real_counts['total'] + synthetic_counts['total']}\n")
        f.write(f"Correct Predictions: {real_counts['correct'] + synthetic_counts['correct']}\n")
        f.write(f"Overall Accuracy: {total_accuracy:.2%}\n\n")
        
        f.write("Real Images:\n")
        f.write(f"Total: {real_counts['total']}\n")
        f.write(f"Correct: {real_counts['correct']}\n")
        f.write(f"Accuracy: {real_acc:.2%}\n\n")
        
        f.write("Synthetic Images:\n")
        f.write(f"Total: {synthetic_counts['total']}\n")
        f.write(f"Correct: {synthetic_counts['correct']}\n")
        f.write(f"Accuracy: {syn_acc:.2%}\n\n")
        
        f.write("Difference: Real accuracy is {:.2%} ".format(real_acc - syn_acc))
        f.write("higher than synthetic\n" if real_acc > syn_acc else "lower than synthetic\n")
    
    logger.info(f"Validation visualizations saved to {output_dir}")
    return

def statistical_significance_test(results):
    """Perform statistical significance testing between real and synthetic image results"""
    # Collect accuracy data
    real_accuracies = []
    synthetic_accuracies = []
    
    for disease_name, disease_results in results.items():
        real_correct = sum(1 for r in disease_results if r['is_correct'] and not r['is_synthetic'])
        real_total = sum(1 for r in disease_results if not r['is_synthetic'])
        if real_total > 0:
            real_accuracies.append(real_correct / real_total)
        
        synthetic_correct = sum(1 for r in disease_results if r['is_correct'] and r['is_synthetic'])
        synthetic_total = sum(1 for r in disease_results if r['is_synthetic'])
        if synthetic_total > 0:
            synthetic_accuracies.append(synthetic_correct / synthetic_total)
    
    # Calculate means
    real_mean = np.mean(real_accuracies) if real_accuracies else 0
    synthetic_mean = np.mean(synthetic_accuracies) if synthetic_accuracies else 0
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(real_accuracies, synthetic_accuracies, equal_var=False)
    
    logger.info(f"Statistical Significance Test Results")
    logger.info(f"Real Mean Accuracy: {real_mean:.2%}")
    logger.info(f"Synthetic Mean Accuracy: {synthetic_mean:.2%}")
    logger.info(f"T-statistic: {t_stat:.3f}")
    logger.info(f"P-value: {p_value:.3f}")
    
    # Interpret p-value
    alpha = 0.05
    if p_value < alpha:
        logger.info("Reject the null hypothesis - significant difference between real and synthetic image performance")
    else:
        logger.info("Fail to reject the null hypothesis - no significant difference between real and synthetic image performance")
    
    return {
        "real_mean": real_mean,
        "synthetic_mean": synthetic_mean,
        "t_stat": t_stat,
        "p_value": p_value,
        "significant_difference": p_value < alpha
    }

def generate_side_by_side_comparison(results, output_dir):
    """Generate a side-by-side comparison of real and synthetic images with predictions"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare figure for displaying multiple example predictions
    plt.figure(figsize=(18, 12))
    
    # Count how many diseases we have
    disease_count = len(results)
    rows = max(1, min(disease_count, 3))  # Up to 3 rows
    
    # Current position in the grid
    plot_idx = 1
    
    for disease_name, disease_results in results.items():
        # Filter for real and synthetic images
        real_images = [r for r in disease_results if not r.get('is_synthetic', False)]
        synthetic_images = [r for r in disease_results if r.get('is_synthetic', False)]
        
        # Take a sample of images (up to 3 of each type)
        real_sample = random.sample(real_images, min(3, len(real_images)))
        synthetic_sample = random.sample(synthetic_images, min(3, len(synthetic_images)))
        
        # Create a row for this disease
        plt.subplot(rows, 1, plot_idx)
        plt.title(f"Disease: {disease_name}", fontsize=14)
        
        # Display images with predictions
        for i, (real_result, synth_result) in enumerate(zip(real_sample, synthetic_sample)):
            # Load and display real image
            real_img = plt.imread(real_result['image_path']) if isinstance(real_result['image_path'], str) and os.path.exists(real_result['image_path']) else np.ones((256, 256, 3))
            ax1 = plt.subplot(rows, 6, plot_idx*6 - 5 + i*2)
            plt.imshow(real_img)
            plt.title(f"Real: {real_result['is_correct']}", color='green' if real_result['is_correct'] else 'red')
            plt.axis('off')
            
            # Load and display synthetic image
            synth_img = plt.imread(synth_result['image_path']) if isinstance(synth_result['image_path'], str) and os.path.exists(synth_result['image_path']) else np.ones((256, 256, 3))
            ax2 = plt.subplot(rows, 6, plot_idx*6 - 4 + i*2)
            plt.imshow(synth_img)
            plt.title(f"Synthetic: {synth_result['is_correct']}", color='green' if synth_result['is_correct'] else 'red')
            plt.axis('off')
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "real_vs_synthetic_examples.png"))
    plt.close()
    
    logger.info(f"Side-by-side comparison saved to {output_dir}")

def generate_detailed_analysis_report(results, output_dir, stats_results):
    """Generate a detailed analysis report comparing real vs synthetic performance"""
    # Format to show percentages with 2 decimal places
    def pct(val):
        return f"{val:.2%}"
    
    # Collect metrics by disease and image type
    metrics_by_disease = {}
    
    for disease_name, disease_results in results.items():
        real_results = [r for r in disease_results if not r.get('is_synthetic', False)]
        synth_results = [r for r in disease_results if r.get('is_synthetic', False)]
        
        # Skip if no results for either category
        if not real_results or not synth_results:
            continue
            
        # Get real and synthetic ground truth and predictions
        real_true = [r['expected_class'] for r in real_results]
        real_pred = [r['predicted_class'] for r in real_results]
        synth_true = [r['expected_class'] for r in synth_results]
        synth_pred = [r['predicted_class'] for r in synth_results]
        
        # Calculate metrics
        if real_true and real_pred:
            real_acc = accuracy_score(real_true, real_pred)
            real_prec, real_rec, real_f1, _ = precision_recall_fscore_support(
                real_true, real_pred, average='weighted', zero_division=0
            )
        else:
            real_acc = real_prec = real_rec = real_f1 = 0
            
        if synth_true and synth_pred:
            synth_acc = accuracy_score(synth_true, synth_pred)
            synth_prec, synth_rec, synth_f1, _ = precision_recall_fscore_support(
                synth_true, synth_pred, average='weighted', zero_division=0
            )
        else:
            synth_acc = synth_prec = synth_rec = synth_f1 = 0
        
        # Store metrics
        metrics_by_disease[disease_name] = {
            'real': {
                'count': len(real_results),
                'accuracy': real_acc,
                'precision': real_prec,
                'recall': real_rec,
                'f1': real_f1
            },
            'synthetic': {
                'count': len(synth_results),
                'accuracy': synth_acc,
                'precision': synth_prec,
                'recall': synth_rec,
                'f1': synth_f1
            },
            'difference': {
                'accuracy': real_acc - synth_acc,
                'precision': real_prec - synth_prec,
                'recall': real_rec - synth_rec,
                'f1': real_f1 - synth_f1
            }
        }
    
    # Generate report file
    report_path = os.path.join(output_dir, "real_vs_synthetic_analysis.txt")
    with open(report_path, 'w') as f:
        f.write("DETAILED ANALYSIS: REAL VS SYNTHETIC IMAGES\n")
        f.write("==========================================\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        f.write("STATISTICAL SIGNIFICANCE\n")
        f.write("-----------------------\n")
        f.write(f"Real images mean accuracy: {pct(stats_results['real_mean'])}\n")
        f.write(f"Synthetic images mean accuracy: {pct(stats_results['synthetic_mean'])}\n")
        f.write(f"Absolute difference: {pct(abs(stats_results['real_mean'] - stats_results['synthetic_mean']))}\n")
        f.write(f"T-statistic: {stats_results['t_stat']:.4f}\n")
        f.write(f"P-value: {stats_results['p_value']:.4f}\n")
        f.write(f"Statistically significant difference: {'Yes' if stats_results['significant_difference'] else 'No'}\n\n")
        
        # Per-disease breakdown
        f.write("PERFORMANCE BY DISEASE\n")
        f.write("---------------------\n\n")
        
        # Table header
        f.write(f"{'Disease':<25} | {'Type':<10} | {'Count':<6} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}\n")
        f.write("-" * 90 + "\n")
        
        # Table rows
        for disease, metrics in metrics_by_disease.items():
            disease_name = disease[:24]  # Truncate if too long
            
            # Real image metrics
            f.write(f"{disease_name:<25} | {'Real':<10} | {metrics['real']['count']:<6} | {pct(metrics['real']['accuracy']):<10} | {pct(metrics['real']['precision']):<10} | {pct(metrics['real']['recall']):<10} | {pct(metrics['real']['f1']):<10}\n")
            
            # Synthetic image metrics
            f.write(f"{'':<25} | {'Synthetic':<10} | {metrics['synthetic']['count']:<6} | {pct(metrics['synthetic']['accuracy']):<10} | {pct(metrics['synthetic']['precision']):<10} | {pct(metrics['synthetic']['recall']):<10} | {pct(metrics['synthetic']['f1']):<10}\n")
            
            # Difference row
            diff_acc = metrics['difference']['accuracy']
            diff_prec = metrics['difference']['precision']
            diff_rec = metrics['difference']['recall']
            diff_f1 = metrics['difference']['f1']
            
            f.write(f"{'':<25} | {'Diff (R-S)':<10} | {'':<6} | {diff_acc:+.2%} | {diff_prec:+.2%} | {diff_rec:+.2%} | {diff_f1:+.2%}\n")
            f.write("-" * 90 + "\n")
        
        # Summary and recommendations
        f.write("\nSUMMARY AND RECOMMENDATIONS\n")
        f.write("---------------------------\n")
        
        # Overall difference analysis
        avg_acc_diff = np.mean([m['difference']['accuracy'] for m in metrics_by_disease.values()])
        avg_f1_diff = np.mean([m['difference']['f1'] for m in metrics_by_disease.values()])
        
        if avg_acc_diff > 0.15:  # Arbitrary threshold for concerning difference
            f.write("WARNING: Substantial accuracy gap between real and synthetic images detected.\n")
            f.write("This suggests the synthetic images may not be representative of real-world conditions.\n")
            
            # Find worst performing diseases for synthetic images
            worst_diseases = sorted(metrics_by_disease.items(), 
                                  key=lambda x: x[1]['difference']['accuracy'], 
                                  reverse=True)[:3]
            
            f.write("\nLargest discrepancies in performance:\n")
            for disease, metrics in worst_diseases:
                if metrics['difference']['accuracy'] > 0.1:  # Only report significant differences
                    f.write(f"- {disease}: Real accuracy {pct(metrics['real']['accuracy'])} vs. Synthetic accuracy {pct(metrics['synthetic']['accuracy'])}\n")
                    f.write(f"  (Difference: {metrics['difference']['accuracy']:+.2%})\n")
            
            f.write("\nRecommendations:\n")
            f.write("1. Improve synthetic image generation for the diseases listed above\n")
            f.write("2. Consider augmenting the training dataset with more real images\n")
            f.write("3. Re-train model with a mix of real and improved synthetic images\n")
        elif avg_acc_diff < -0.15:  # Synthetic performs better than real
            f.write("OBSERVATION: Synthetic images show better results than real images.\n")
            f.write("This could indicate that real test images are more challenging or less representative than training data.\n")
            f.write("\nRecommendations:\n")
            f.write("1. Verify real image quality and diversity\n")
            f.write("2. Consider adding more challenging synthetic images to the training set\n")
            f.write("3. Review image segmentation performance on real images\n")
        else:
            f.write("Model performs similarly on real and synthetic images, indicating good generalization.\n")
            f.write(f"Average accuracy difference: {avg_acc_diff:+.2%}\n")
            f.write(f"Average F1 score difference: {avg_f1_diff:+.2%}\n")
            
            if abs(avg_acc_diff) < 0.05:
                f.write("\nThe minimal performance gap suggests synthetic images are effective proxies for real-world data.\n")
                f.write("Recommendation: Continue using the current synthetic image generation approach.\n")
    
    # Also create a visual chart comparing diseases
    plt.figure(figsize=(12, 8))
    
    # Prepare data for chart
    diseases = list(metrics_by_disease.keys())
    real_acc = [metrics_by_disease[d]['real']['accuracy'] for d in diseases]
    synth_acc = [metrics_by_disease[d]['synthetic']['accuracy'] for d in diseases]
    
    # Create grouped bar chart
    x = np.arange(len(diseases))
    width = 0.35
    
    plt.bar(x - width/2, real_acc, width, label='Real Images', color='#3498db')
    plt.bar(x + width/2, synth_acc, width, label='Synthetic Images', color='#e74c3c')
    
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Disease: Real vs Synthetic Images')
    plt.xticks(x, [d[:15] + '...' if len(d) > 15 else d for d in diseases], rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "disease_accuracy_comparison.png"))
    plt.close()

def segment_leaf(image):
    """
    Segment the leaf from the background in the image.
    Args:
        image: NumPy array of image in BGR format (OpenCV default)
    Returns:
        Segmented image as NumPy array
    """
    try:
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a broad mask for green and brown colors (common in plant leaves)
        # Green mask
        lower_green = np.array([25, 40, 50])
        upper_green = np.array([95, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Brown/yellow mask (for diseased or autumn leaves)
        lower_brown = np.array([10, 40, 50])
        upper_brown = np.array([30, 255, 255])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_green, mask_brown)
        
        # Optional: perform morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return original image
        if not contours:
            logger.warning("No leaf contours found in the image")
            return image
        
        # Find the largest contour by area (assumed to be the leaf)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask for the largest contour
        leaf_mask = np.zeros_like(mask)
        cv2.drawContours(leaf_mask, [largest_contour], 0, 255, -1)
        
        # Apply the mask to the original image
        segmented = cv2.bitwise_and(image, image, mask=leaf_mask)
        
        # Save the segmented image if in a specific disease directory
        try:
            # Check if we're in a known disease directory
            for disease_name in DISEASE_MAPPING.keys():
                sanitized = disease_name.replace(' ', '_').lower()
                if sanitized in str(image.filename):
                    seg_dir = os.path.join(VALIDATION_DIR, 'segmented_images', sanitized)
                    os.makedirs(seg_dir, exist_ok=True)
                    # Save with timestamp to avoid overwriting
                    seg_path = os.path.join(seg_dir, f"seg_{int(time.time())}_{random.randint(1000,9999)}.jpg")
                    cv2.imwrite(seg_path, segmented)
                    break
        except (AttributeError, Exception) as e:
            # Not critical if this fails, just log it
            logger.debug(f"Could not save segmented image: {e}")
        
        return segmented
    
    except Exception as e:
        logger.error(f"Error in leaf segmentation: {e}")
        # Return original image if segmentation fails
        return image
