import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
import sys
import multiprocessing
import random
import time
from datetime import datetime

# Allow for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Add the root directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Class to handle combining dataset types with different transforms - moved outside function for pickling
class MultiModalDataset(Dataset):
    def __init__(self, color_samples=None, grayscale_samples=None, segmented_samples=None, 
                 color_transform=None, grayscale_transform=None, segmented_transform=None,
                 class_names=None, class_to_idx=None):
        self.samples = []
        self.transforms = {}
        self.classes = class_names
        self.class_to_idx = class_to_idx
        
        # Add color samples
        if color_samples:
            for i, (path, label) in enumerate(color_samples):
                self.samples.append((path, label, 'color'))
                self.transforms['color'] = color_transform
        
        # Add grayscale samples
        if grayscale_samples:
            for i, (path, label) in enumerate(grayscale_samples):
                self.samples.append((path, label, 'grayscale'))
                self.transforms['grayscale'] = grayscale_transform
        
        # Add segmented samples
        if segmented_samples:
            for i, (path, label) in enumerate(segmented_samples):
                self.samples.append((path, label, 'segmented'))
                self.transforms['segmented'] = segmented_transform or color_transform  # Fall back to color transform
                
        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label, modality = self.samples[idx]
        image = datasets.folder.default_loader(path)
        
        # Apply the appropriate transform based on modality
        if self.transforms[modality]:
            image = self.transforms[modality](image)
            
        return image, label

def train_model():
    print("Starting the enhanced training script...")
    
    # Create directory for saving models
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    MODEL_PREFIX = os.path.join(MODELS_DIR, f"plant_disease_model_{TIMESTAMP}")

    # Check for GPU availability first - exit if not available
    if not torch.cuda.is_available():
        print("\n" + "="*80)
        print("ERROR: GPU not available. This training script requires a GPU.")
        print("Training a plant disease model on CPU would be extremely slow.")
        print("Please run this script on a machine with a CUDA-compatible GPU.")
        print("="*80 + "\n")
        return
    
    # Define paths to all dataset folders
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    COLOR_DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'plantvillage_dataset', 'color')
    GRAYSCALE_DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'plantvillage_dataset', 'grayscale')
    SEGMENTED_DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'plantvillage_dataset', 'segmented')
    
    # Final model save path
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'plant_disease_model.pth')
    
    print(f"Color dataset path: {COLOR_DATASET_PATH}")
    print(f"Grayscale dataset path: {GRAYSCALE_DATASET_PATH}")
    print(f"Segmented dataset path: {SEGMENTED_DATASET_PATH}")
    print(f"Model save path: {MODEL_SAVE_PATH}")
    
    # Check if dataset paths exist
    if not os.path.exists(COLOR_DATASET_PATH):
        print(f"Warning: Color dataset path not found at {COLOR_DATASET_PATH}")
    
    if not os.path.exists(GRAYSCALE_DATASET_PATH):
        print(f"Warning: Grayscale dataset path not found at {GRAYSCALE_DATASET_PATH}")
    
    if not os.path.exists(SEGMENTED_DATASET_PATH):
        print(f"Warning: Segmented dataset path not found at {SEGMENTED_DATASET_PATH}")
    
    # Image transformations with strong data augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'grayscale_train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Grayscale normalization
        ]),
        'grayscale_val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Grayscale normalization
        ])
    }

    # Load all available datasets
    dataset_info = {}
    datasets_loaded = 0
    
    # Load color dataset
    try:
        color_dataset = datasets.ImageFolder(COLOR_DATASET_PATH)
        datasets_loaded += 1
        print(f"Loaded color dataset with {len(color_dataset)} images and {len(color_dataset.classes)} classes")
        class_names = color_dataset.classes
        class_to_idx = color_dataset.class_to_idx
        dataset_info['color'] = color_dataset
    except Exception as e:
        print(f"Error loading color dataset: {str(e)}")
        color_dataset = None
    
    # Load grayscale dataset
    try:
        grayscale_dataset = datasets.ImageFolder(GRAYSCALE_DATASET_PATH)
        datasets_loaded += 1
        print(f"Loaded grayscale dataset with {len(grayscale_dataset)} images and {len(grayscale_dataset.classes)} classes")
        if not color_dataset:
            class_names = grayscale_dataset.classes
            class_to_idx = grayscale_dataset.class_to_idx
        dataset_info['grayscale'] = grayscale_dataset
    except Exception as e:
        print(f"Error loading grayscale dataset: {str(e)}")
        grayscale_dataset = None
    
    # Load segmented dataset
    try:
        segmented_dataset = datasets.ImageFolder(SEGMENTED_DATASET_PATH)
        datasets_loaded += 1
        print(f"Loaded segmented dataset with {len(segmented_dataset)} images and {len(segmented_dataset.classes)} classes")
        if not color_dataset and not grayscale_dataset:
            class_names = segmented_dataset.classes
            class_to_idx = segmented_dataset.class_to_idx
        dataset_info['segmented'] = segmented_dataset
    except Exception as e:
        print(f"Error loading segmented dataset: {str(e)}")
        segmented_dataset = None
    
    # Ensure we have at least one dataset
    if datasets_loaded == 0:
        print("Error: No datasets could be loaded. Please check your dataset paths.")
        return
    
    # Get class names and number of classes
    num_classes = len(class_names)
    print(f"Using a total of {num_classes} classes: {', '.join(class_names)}")
    
    # Create a more balanced dataset by sampling classes
    # Using all images but limiting to a maximum per class
    MAX_IMAGES_PER_CLASS_PER_TYPE = 500  # Adjust based on memory constraints
    
    def get_balanced_indices(dataset):
        class_count = {}
        indices = []
        
        for idx, (_, class_idx) in enumerate(dataset.samples):
            if class_idx not in class_count:
                class_count[class_idx] = 0
            
            if class_count[class_idx] < MAX_IMAGES_PER_CLASS_PER_TYPE:
                indices.append(idx)
                class_count[class_idx] += 1
        
        return indices
    
    # Get balanced indices for each dataset
    color_indices = get_balanced_indices(color_dataset) if color_dataset else []
    grayscale_indices = get_balanced_indices(grayscale_dataset) if grayscale_dataset else []
    segmented_indices = get_balanced_indices(segmented_dataset) if segmented_dataset else []
    
    print(f"Using {len(color_indices)} color images")
    print(f"Using {len(grayscale_indices)} grayscale images")
    print(f"Using {len(segmented_indices)} segmented images")
    print(f"Total training images: {len(color_indices) + len(grayscale_indices) + len(segmented_indices)}")
    
    # Create lists of samples for each type
    color_samples = [color_dataset.samples[i] for i in color_indices] if color_dataset else []
    grayscale_samples = [grayscale_dataset.samples[i] for i in grayscale_indices] if grayscale_dataset else []
    segmented_samples = [segmented_dataset.samples[i] for i in segmented_indices] if segmented_dataset else []
    
    # Split into training and validation sets
    train_color, val_color = train_test_split(
        color_samples, test_size=0.15, random_state=42,
        stratify=[s[1] for s in color_samples] if color_samples else None
    ) if color_samples else ([], [])
    
    train_grayscale, val_grayscale = train_test_split(
        grayscale_samples, test_size=0.15, random_state=42,
        stratify=[s[1] for s in grayscale_samples] if grayscale_samples else None
    ) if grayscale_samples else ([], [])
    
    train_segmented, val_segmented = train_test_split(
        segmented_samples, test_size=0.15, random_state=42,
        stratify=[s[1] for s in segmented_samples] if segmented_samples else None
    ) if segmented_samples else ([], [])
    
    # Create the combined datasets - passing class info explicitly
    train_dataset = MultiModalDataset(
        color_samples=train_color, 
        grayscale_samples=train_grayscale,
        segmented_samples=train_segmented,
        color_transform=data_transforms['train'],
        grayscale_transform=data_transforms['grayscale_train'],
        segmented_transform=data_transforms['train'],
        class_names=class_names,
        class_to_idx=class_to_idx
    )
    
    val_dataset = MultiModalDataset(
        color_samples=val_color,
        grayscale_samples=val_grayscale,
        segmented_samples=val_segmented,
        color_transform=data_transforms['val'],
        grayscale_transform=data_transforms['grayscale_val'],
        segmented_transform=data_transforms['val'],
        class_names=class_names,
        class_to_idx=class_to_idx
    )
    
    # Implement class weighting to address imbalance
    class_counts = np.zeros(num_classes)
    for _, label, _ in train_dataset.samples:
        class_counts[label] += 1
    
    class_weights = 1.0 / np.array(class_counts)
    class_weights = class_weights / np.sum(class_weights) * num_classes
    
    # Create sample weights based on class weights
    sample_weights = [class_weights[label] for _, label, _ in train_dataset.samples]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    
    # Data loaders with safer settings for Windows
    dataloaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=24,  # Adjusted to prevent CUDA out of memory
            sampler=sampler,  # Use weighted sampler
            num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset, 
            batch_size=24, 
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
            pin_memory=True
        )
    }
    
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    print(f"Training dataset size: {dataset_sizes['train']}")
    print(f"Validation dataset size: {dataset_sizes['val']}")
    
    # Device configuration - force CUDA
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpu_name}")
    
    # Define the model - use ResNet50 with proper weights initialization
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Add dropout to match existing architecture
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),  # Using dropout 0.4 to match existing model files
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(device)
    
    # Loss function with class weighting
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    # Learning rate scheduler - removed verbose parameter which is not supported in this PyTorch version
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)
    
    # Number of training epochs
    num_epochs = 30
    
    # For early stopping
    patience = 7
    best_acc = 0.0
    best_model_wts = model.state_dict()
    best_epoch = 0
    no_improve_epochs = 0
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\nStarting training...\n")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode
            
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass - track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
            
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save checkpoint every 5 epochs
            if phase == 'val' and (epoch + 1) % 5 == 0:
                checkpoint_path = f"{MODEL_PREFIX}_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
            # If best performing model, save it
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                best_epoch = epoch
                no_improve_epochs = 0
                
                # Save the best model so far
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"New best model saved with accuracy: {best_acc:.4f}")
            elif phase == 'val':
                no_improve_epochs += 1
                
            # Update learning rate
            if phase == 'val':
                scheduler.step(epoch_acc)
        
        # Print current learning rate
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"Time elapsed: {(time.time() - start_time) / 60:.2f} minutes")
        
        # Early stopping
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
            break
            
        print()
    
    total_time = time.time() - start_time
    print(f'\nTraining complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save final model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")
    
    return model, history

if __name__ == "__main__":
    # Fix for multiprocessing issues in Windows
    multiprocessing.freeze_support()
    model, history = train_model()
