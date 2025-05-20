# Plant Disease Model Architecture

## Overview
This document describes the official model architecture used for plant disease classification throughout the application. All model implementations should follow this specification to ensure compatibility with trained weights.

## Model Structure
The application uses a ResNet50-based model with the following specifications:

- **Base architecture**: ResNet50
- **Input size**: 224x224 RGB images (3 channels)
- **Output size**: 38 classes (various plant diseases)
- **Dropout rate**: 0.4 (consistent across all implementations)
- **Weights file**: `ml/plant_disease_model.pth`

## Standardized Implementation

```python
def create_model(num_classes=38):
    """
    Create and return a ResNet50 model without wrapper class structure
    to match the format of the saved weights.
    """
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),  # Must use 0.4 to match trained weights
        nn.Linear(num_ftrs, num_classes)
    )
    return model
```

## Loading the Model

When loading the model, the following pattern should be used:

```python
# Create the model with the same architecture
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, num_classes)
)

# Load the weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Set to evaluation mode for inference
```

## Data Preprocessing

For inference, images should be preprocessed using the following transforms:

```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## Extending the Model

When training on more data or improving the model:

1. Use the enhanced training script in `ml/train_model.py`
2. Keep the dropout rate at 0.4 for consistency
3. Ensure the output weights are saved to the standard path
4. Run the consistency checker to verify alignment: `python ml/utils/check_model_consistency.py`

## Available Datasets

Multiple dataset formats can be used for training:

- **Color**: Standard RGB images (`dataset/plantvillage_dataset/color/`)
- **Grayscale**: Grayscale versions (`dataset/plantvillage_dataset/grayscale/`)
- **Segmented**: Preprocessed segmented images (`dataset/plantvillage_dataset/segmented/`)

The enhanced training script can utilize all datasets simultaneously for better robustness.