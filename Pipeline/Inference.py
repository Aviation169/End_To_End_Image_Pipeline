import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import logging
import argparse
import json
from typing import List, Tuple
from tqdm import tqdm
import os 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CONFIGURATION ===
class_names_file = "class_names.json"

# === FEATURE ENHANCEMENT LAYER (FEL) ===
class FEL(nn.Module):
    def __init__(self, in_channels: int = 512, out_channels: int = 128):
        super(FEL, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        attn = self.attention(x)
        x = x * attn
        x = self.relu(x)
        if residual.shape[1] != x.shape[1]:
            residual = residual[:, :x.shape[1], :, :]
        return x + residual

# === SEAL-CNN MODEL ===
class SEAL_CNN(nn.Module):
    def __init__(self, num_classes: int, class_names: List[str] = None):
        super(SEAL_CNN, self).__init__()
        self.resnet = models.resnet18(weights=None)
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        self.fel = FEL(in_channels=512, out_channels=128)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        for param in self.fel.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        self.class_names = class_names if class_names else [f"Class_{i}" for i in range(num_classes)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.fel(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

    def get_class_names(self) -> List[str]:
        """Retrieve stored class names."""
        try:
            with open(class_names_file, 'r') as f:
                class_map = json.load(f)
            return [class_map[str(i)] for i in range(len(class_map))]
        except FileNotFoundError:
            logger.warning(f"Class names file {class_names_file} not found, returning current class names")
            return self.class_names

# === SEAL-CNN WITH EWC ===
class SEAL_CNN_EWC(nn.Module):
    def __init__(self, num_classes: int, pretrained_model_path: str, original_num_classes: int = None, class_names: List[str] = None):
        super(SEAL_CNN_EWC, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.fel = FEL(in_channels=512, out_channels=128)
        
        # Load pre-trained weights
        state_dict = torch.load(pretrained_model_path, map_location=device)
        if original_num_classes and original_num_classes != num_classes:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('resnet.fc')}
        self.load_state_dict(state_dict, strict=False)
        
        # Freeze all parameters except FEL and final FC layer
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.fel.parameters():
            param.requires_grad = True
        
        # Replace the fully connected layer
        in_features = 128
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        
        self.class_names = class_names if class_names else [f"Class_{i}" for i in range(num_classes)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.fel(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

    def get_class_names(self) -> List[str]:
        """Retrieve stored class names."""
        try:
            with open(class_names_file, 'r') as f:
                class_map = json.load(f)
            return [class_map[str(i)] for i in range(len(class_map))]
        except FileNotFoundError:
            logger.warning(f"Class names file {class_names_file} not found, returning current class names")
            return self.class_names

# === GET NUMBER OF CLASSES ===
def get_num_classes(model_path: str) -> int:
    try:
        state_dict = torch.load(model_path, map_location=device)
        for key in state_dict:
            if "resnet.fc.1.weight" in key:
                num_classes = state_dict[key].shape[0]
                print(f"✅ Number of classes in the model: {num_classes}")
                return num_classes
        raise ValueError("Could not find 'resnet.fc.1.weight' in state_dict")
    except Exception as e:
        logger.error(f"Error determining number of classes: {str(e)}")
        raise

# === INFER SINGLE IMAGE ===
def infer_single_image(model_path: str, image_path: str, is_finetuned: bool = False) -> Tuple[str, int, np.ndarray]:
    try:
        # Load class names
        try:
            with open(class_names_file, 'r') as f:
                class_map = json.load(f)
            num_classes = len(class_map)
            class_names = [class_map[str(i)] for i in range(num_classes)]
        except FileNotFoundError:
            logger.error(f"Class names file {class_names_file} not found.")
            raise FileNotFoundError(f"Class names file {class_names_file} not found.")

        # Initialize model
        if is_finetuned:
            num_classes_model = get_num_classes(model_path)
            model = SEAL_CNN_EWC(
                num_classes=num_classes_model,
                pretrained_model_path=model_path,
                class_names=class_names
            ).to(device)
        else:
            model = SEAL_CNN(num_classes=num_classes, class_names=class_names).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

        model.eval()

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if not os.path.exists(image_path):
            logger.error(f"Image {image_path} not found.")
            raise FileNotFoundError(f"Image {image_path} not found.")
        
        image = np.array(Image.open(image_path).convert('RGB'))
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            logger.error(f"Invalid image {image_path}.")
            raise ValueError(f"Invalid image {image_path}.")
        
        img_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in tqdm(range(1), desc="Inferring"):
                out = model(img_tensor)
                probabilities = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred = out.argmax(dim=1).item()

        class_name = class_names[pred] if pred < len(class_names) else f"Class_{pred}"
        return class_name, pred, probabilities

    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

# === MAIN FUNCTION ===
def main():
    parser = argparse.ArgumentParser(description="Infer a single image using SEAL-CNN or SEAL-CNN-EWC")
    parser.add_argument('--model_path', type=str, required=True, 
                       help="Path to the trained model (e.g., adapted_model.pth or finetuned_model_ewc.pth)")
    parser.add_argument('--image_path', type=str, required=True, 
                       help="Path to the input image for inference")
    parser.add_argument('--is_finetuned', action='store_true', 
                       help="Specify if the model is fine-tuned with EWC (use for finetuned_model_ewc.pth)")
    args = parser.parse_args()

    try:
        class_name, class_id, probabilities = infer_single_image(
            model_path=args.model_path,
            image_path=args.image_path,
            is_finetuned=args.is_finetuned
        )
        print(f"Predicted class for {args.image_path}: {class_name} (ID: {class_id})")
        print("Confidence scores:")
        try:
            with open(class_names_file, 'r') as f:
                class_map = json.load(f)
            class_names = [class_map[str(i)] for i in range(len(class_map))]
        except FileNotFoundError:
            class_names = [f"Class_{i}" for i in range(len(probabilities))]
        for idx, prob in enumerate(probabilities):
            class_label = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            print(f"{class_label}: {prob:.4f}")

    except Exception as e:
        logger.error(f"Failed to run inference: {str(e)}")

# === ENTRY POINT ===
if __name__ == "__main__":
    main()