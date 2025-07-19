import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

import nest_asyncio
nest_asyncio.apply()

import asyncio
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import logging
import pandas as pd
from typing import List, Tuple, Dict
import clip
import argparse
import json
from icrawler.builtin import GoogleImageCrawler
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CONFIGURATION ===
required_good_images = 200
threshold_similarity = 0.85  # Increased from 0.75
output_dir = "augmented_dataset_finetune"
csv_file = "image_labels_finetune.csv"
class_names_file = "class_names.json"
max_threads = min(32, os.cpu_count() or 4)
max_images_per_class = 1000  # Cap for class balancing

# === AUGMENTATION PIPELINES ===
pil_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=(0, 30)),  # Reduced rotation range
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.2)),  # Less aggressive crop
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),  # Softer jitter
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # Reduced probability
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),  # Tighter sigma
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),  # Softer affine
    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # Softer perspective
])

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),  # Softer erasing
    transforms.ToPILImage(),
])

# === DOWNLOAD FUNCTION ===
def download_images(keyword, limit=50, storage_dir='dataset/raw_finetune'):
    crawler = GoogleImageCrawler(storage={'root_dir': f'{storage_dir}/{keyword}'})
    crawler.crawl(keyword=keyword, max_num=limit)

async def async_download(keyword, limit=50, storage_dir='dataset/raw_finetune'):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, partial(download_images, keyword, limit, storage_dir))

# === PROCESSING FUNCTION ===
def process_image(original_image, model, preprocess, original_embedding, label, index, save_dir):
    img = pil_transforms(original_image)
    img = tensor_transform(img)
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_tensor).cpu().numpy()
        similarity = cosine_similarity(original_embedding, emb)[0][0]

    if similarity >= threshold_similarity:
        filename = f"{label}_{index:04}.png"
        img.save(os.path.join(save_dir, filename))
        return (filename, similarity)
    return None

# === ASYNC PROCESSING OF SINGLE IMAGE ===
async def process_single_image(image_path, class_name, filename_base, model, preprocess, log_writer):
    try:
        original_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Skipped invalid image: {image_path} ({e})")
        return

    save_dir = os.path.join(output_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)

    original_tensor = preprocess(original_image).unsqueeze(0).to(device)
    with torch.no_grad():
        original_embedding = model.encode_image(original_tensor).cpu().numpy()

    count = 0
    index = 0
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        while count < required_good_images:
            future = loop.run_in_executor(
                executor,
                process_image,
                original_image,
                model,
                preprocess,
                original_embedding,
                filename_base,
                index,
                save_dir
            )
            futures.append(future)
            index += 1

            if len(futures) >= max_threads * 2:
                done, _ = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    result = task.result()
                    if result:
                        filename, similarity = result
                        log_writer.write(f"{filename},{class_name},{similarity:.4f}\n")
                        count += 1
                        print(f"[{count}/{required_good_images}] {class_name}/{filename} ✓")
                futures = [f for f in futures if not f.done()]

# === EWC IMPLEMENTATION ===
class EWC:
    def __init__(self, model: nn.Module, dataset: List[Tuple[np.ndarray, int]], ewc_lambda: float = 100.0):
        self.model = model
        self.dataset = dataset
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self._compute_fisher()

    def _compute_fisher(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.fisher[n] = torch.zeros_like(p).to(device)

        for img, lbl in tqdm(self.dataset, desc="Computing Fisher Information"):
            img = preprocess(img).unsqueeze(0).to(device)
            lbl = torch.tensor([lbl], dtype=torch.long).to(device)
            self.model.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                out = self.model(img)
                loss = criterion(out, lbl)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += (p.grad ** 2)

        for n in self.fisher:
            self.fisher[n] /= len(self.dataset) if len(self.dataset) > 0 else 1
        self.model.train()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = 0
        for n, p in model.named_parameters():
            if n in self.params and p.requires_grad:
                penalty = (self.fisher[n] * (p - self.params[n]) ** 2).sum()
                loss += penalty
        return self.ewc_lambda * loss

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
        
        # Store class names
        self.class_names = class_names if class_names else [f"Class_{i}" for i in range(num_classes)]
        self.save_class_names()

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

    def save_class_names(self):
        """Save class names to a JSON file."""
        class_map = {str(i): name for i, name in enumerate(self.class_names)}
        with open(class_names_file, 'w') as f:
            json.dump(class_map, f, indent=4)
        logger.info(f"Saved class names to {class_names_file}")

    def get_class_names(self) -> List[str]:
        """Retrieve stored class names."""
        try:
            with open(class_names_file, 'r') as f:
                class_map = json.load(f)
            return [class_map[str(i)] for i in range(len(class_map))]
        except FileNotFoundError:
            logger.warning(f"Class names file {class_names_file} not found, returning current class names")
            return self.class_names

# === GET NUMBER OF ORIGINAL CLASSES ===
def get_original_num_classes(model_path: str) -> int:
    try:
        state_dict = torch.load(model_path, map_location=device)
        for key in state_dict:
            if "resnet.fc.1.weight" in key:
                num_classes = state_dict[key].shape[0]
                print(f"✅ Number of classes in the pre-trained model: {num_classes}")
                return num_classes
        raise ValueError("Could not find 'resnet.fc.1.weight' in state_dict")
    except Exception as e:
        logger.error(f"Error determining original number of classes: {str(e)}")
        raise

# === CREATE CLASS MAP ===
def create_class_map(new_keywords: List[str], original_num_classes: int, original_class_names: List[str] = None) -> Dict:
    class_map = {}
    original_names = original_class_names if original_class_names else [f"OriginalClass_{i}" for i in range(original_num_classes)]
    for idx, name in enumerate(original_names):
        class_map[idx] = name
        class_map[name] = idx
    for idx, keyword in enumerate(new_keywords, start=original_num_classes):
        class_map[idx] = keyword
        class_map[keyword] = idx
    return class_map

# === LOAD DATASET ===
def load_dataset(class_dirs: List[Tuple[str, str, int]]):
    try:
        dfs = []
        for dir_path, class_name, label in class_dirs:
            if not os.path.exists(dir_path):
                logger.warning(f"{class_name} directory {dir_path} not found, skipping")
                continue
            files = [
                os.path.normpath(os.path.join(dir_path, f))
                for f in os.listdir(dir_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(dir_path, f))
            ]
            valid_files = []
            for file_path in files:
                try:
                    img = Image.open(file_path).convert('RGB')
                    img.verify()
                    img.close()
                    valid_files.append(file_path)
                except Exception as e:
                    logger.warning(f"Invalid image {file_path}: {str(e)}")
            # Cap number of images per class
            if len(valid_files) > max_images_per_class:
                valid_files = valid_files[:max_images_per_class]
            logger.info(f"Found {len(valid_files)} valid {class_name} images")
            df = pd.DataFrame({
                'filename': valid_files,
                'label': [label] * len(valid_files)
            })
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        if df.empty:
            raise ValueError("No valid images found in directories")

        unique_labels = df['label'].unique()
        logger.info(f"Unique labels: {unique_labels}")
        class_counts = df['label'].value_counts()
        print(f"Label counts: {class_counts.to_dict()}")

        train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

        logger.info(f"Training set size: {len(train_df)} images")
        logger.info(f"Validation set size: {len(val_df)} images")
        logger.info(f"Test set size: {len(test_df)} images")

        return train_df, val_df, test_df
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

# === BALANCED SAMPLER ===
def create_balanced_sampler(labels: List[int], num_samples_per_class: int = 100):
    class_counts = {}
    for lbl in labels:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
    
    indices = []
    for lbl in set(labels):
        lbl_indices = [i for i, l in enumerate(labels) if l == lbl]
        np.random.shuffle(lbl_indices)
        indices.extend(lbl_indices[:min(num_samples_per_class, len(lbl_indices))])
    
    np.random.shuffle(indices)
    return indices

# === FINETUNE WITH EWC ===
def finetune_ewc(
    model: nn.Module,
    train_images: List[np.ndarray],
    train_labels: List[int],
    val_images: List[np.ndarray],
    val_labels: List[int],
    original_dataset: List[Tuple[np.ndarray, int]],
    lr: float = 0.0005,  # Reduced from 0.001
    epochs: int = 20,    # Increased from 10
    patience: int = 5,   # Increased from 3
    ewc_lambda: float = 100.0  # Reduced from 1000.0
) -> float:
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    
    ewc = EWC(model, original_dataset, ewc_lambda=ewc_lambda)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    total_loss = 0

    for epoch in tqdm(range(epochs), desc="Epochs"):
        indices = create_balanced_sampler(train_labels, num_samples_per_class=100)

        epoch_loss = 0
        batch_size = 8
        num_batches = (len(indices) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(indices), batch_size), total=num_batches, desc=f"Epoch {epoch+1} Batches"):
            batch_indices = indices[i:i + batch_size]
            batch_images = [train_images[idx] for idx in batch_indices]
            batch_labels = [train_labels[idx] for idx in batch_indices]
            batch_tensors = [preprocess(img).unsqueeze(0).to(device) for img in batch_images]
            batch_tensors = torch.cat(batch_tensors, dim=0)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                out = model(batch_tensors)
                ce_loss = criterion(out, batch_labels)
                ewc_loss = ewc.penalty(model)
                loss = ce_loss + ewc_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_images)
        avg_loss = epoch_loss / len(indices) if len(indices) > 0 else 0.0
        total_loss += avg_loss
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, CE Loss: {ce_loss:.4f}, EWC Loss: {ewc_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        model.eval()
        val_loss = 0
        total_correct = 0
        with torch.no_grad():
            for img, lbl in zip(val_images, val_labels):
                img = preprocess(img).unsqueeze(0).to(device)
                lbl = torch.tensor([lbl], dtype=torch.long).to(device)
                out = model(img)
                loss = criterion(out, lbl)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                total_correct += (pred == lbl).float().mean().item()
        val_loss = val_loss / len(val_images) if len(val_images) > 0 else 0.0
        val_accuracy = total_correct / len(val_images) if len(val_images) > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break

        scheduler.step(val_loss)
        model.train()

    return total_loss / (epoch + 1) if (epoch + 1) > 0 else 0.0

# === EVALUATE ACCURACY AND LOSS ===
def evaluate_accuracy_and_loss(
    model: nn.Module,
    val_images: List[np.ndarray],
    val_labels: List[int]
) -> Tuple[float, float]:
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    criterion = nn.CrossEntropyLoss()

    total_correct = 0
    total_loss = 0
    total = len(val_images)
    try:
        with torch.no_grad():
            for img, lbl in tqdm(zip(val_images, val_labels), total=total, desc="Validating"):
                img = preprocess(img).unsqueeze(0).to(device)
                lbl = torch.tensor([lbl], dtype=torch.long).to(device)
                out = model(img)
                pred = out.argmax(dim=1)
                total_correct += (pred == lbl).float().mean().item()
                loss = criterion(out, lbl)
                total_loss += loss.item()
        accuracy = total_correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        print(f"Validation accuracy: {accuracy:.4f}, Validation loss: {avg_loss:.4f}")
        return accuracy, avg_loss
    except Exception as e:
        logger.error(f"Error computing validation metrics: {str(e)}")
        return 0.5, 0.0

# === MAIN COROUTINE ===
async def main():
    parser = argparse.ArgumentParser(description="Fine-tune SEAL-CNN with EWC on new classes")
    parser.add_argument('--model_path', type=str, required=True, help="Path to pre-trained model (adapted_model.pth)")
    parser.add_argument('--new_keywords', type=str, nargs='+', required=True, 
                       help="Keywords for new classes to download and fine-tune (e.g., Man Apple_Watch)")
    parser.add_argument('--limit', type=int, default=50, 
                       help="Number of images to download per keyword")
    parser.add_argument('--storage_dir', type=str, default='dataset/raw_finetune', 
                       help="Directory to store downloaded images")
    parser.add_argument('--ewc_lambda', type=float, default=100.0, 
                       help="EWC regularization strength")
    parser.add_argument('--use_old_data', action='store_true', 
                       help="Include original class data during fine-tuning")
    args = parser.parse_args()

    # Load existing class names from previous training (if available)
    original_class_names = []
    try:
        with open(class_names_file, 'r') as f:
            class_map = json.load(f)
            original_class_names = [class_map[str(i)] for i in range(len(class_map))]
        logger.info(f"Loaded original class names: {original_class_names}")
    except FileNotFoundError:
        logger.warning(f"Class names file {class_names_file} not found, assuming no prior class names")

    # Determine original number of classes
    original_num_classes = get_original_num_classes(args.model_path)

    # Create class map
    class_map = create_class_map(args.new_keywords, original_num_classes, original_class_names)
    print(f"Class mapping: {class_map}")

    # Download images for new keywords and optionally for original classes
    print("Starting image download...")
    download_tasks = [async_download(keyword, args.limit, args.storage_dir) for keyword in args.new_keywords]
    if args.use_old_data:
        old_class_limit = args.limit  # Use full limit for original classes
        for i in range(original_num_classes):
            class_name = class_map[i]
            download_tasks.append(async_download(class_name, old_class_limit, args.storage_dir))
        print(f"Downloading {old_class_limit} images per original class: {original_class_names}")
    await asyncio.gather(*download_tasks)
    print("Image download complete.")

    # Augment images for all relevant classes
    os.makedirs(output_dir, exist_ok=True)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    image_tasks = []
    input_dir = args.storage_dir
    keywords = args.new_keywords if not args.use_old_data else args.new_keywords + original_class_names
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, input_dir)
                class_name = os.path.dirname(rel_path).replace("\\", "/")
                if class_name not in keywords:
                    continue
                filename_base = os.path.splitext(os.path.basename(full_path))[0]
                image_tasks.append((full_path, class_name, filename_base))

    with open(csv_file, "w") as log:
        log.write("filename,class,similarity\n")
        for img_path, class_name, filename_base in image_tasks:
            print(f"\n🔄 Processing: {class_name}/{filename_base}")
            try:
                await process_single_image(img_path, class_name, filename_base, clip_model, preprocess, log)
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
    print("\n✅ Augmentation complete.")

    # Prepare class directories for fine-tuning
    class_dirs = [
        (os.path.normpath(os.path.join(output_dir, keyword)), keyword, class_map[keyword])
        for keyword in args.new_keywords
    ]
    if args.use_old_data:
        class_dirs.extend([
            (os.path.normpath(os.path.join(output_dir, class_map[i])), class_map[i], i)
            for i in range(original_num_classes)
        ])

    # Load dataset
    try:
        train_df, val_df, test_df = load_dataset(class_dirs)
        
        # Load images and labels
        train_images = []
        train_labels = []
        skipped_images = 0
        for _, row in train_df.iterrows():
            image_path = row['filename']
            if not os.path.exists(image_path):
                logger.warning(f"Image {image_path} not found, skipping.")
                skipped_images += 1
                continue
            img = np.array(Image.open(image_path).convert('RGB'))
            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                logger.warning(f"Invalid image {image_path}, skipping.")
                skipped_images += 1
                continue
            train_images.append(img)
            train_labels.append(row['label'])
        
        if skipped_images > 0:
            logger.info(f"Skipped {skipped_images} images due to invalid paths or content")
        
        if not train_images:
            raise ValueError("No valid images loaded for training")

        val_images = []
        val_labels = []
        skipped_val_images = 0
        for _, row in val_df.iterrows():
            image_path = row['filename']
            if not os.path.exists(image_path):
                logger.warning(f"Image {image_path} not found, skipping.")
                skipped_val_images += 1
                continue
            img = np.array(Image.open(image_path).convert('RGB'))
            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                logger.warning(f"Invalid image {image_path}, skipping.")
                skipped_val_images += 1
                continue
            val_images.append(img)
            val_labels.append(row['label'])
        
        if skipped_val_images > 0:
            logger.info(f"Skipped {skipped_val_images} validation images due to invalid paths or content")

        # Load original dataset for EWC
        original_dataset = []
        if args.use_old_data:
            for i in range(original_num_classes):
                class_dir = os.path.normpath(os.path.join(output_dir, class_map[i]))
                if os.path.exists(class_dir):
                    files = [
                        os.path.join(class_dir, f)
                        for f in os.listdir(class_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ]
                    for file_path in files:
                        try:
                            img = np.array(Image.open(file_path).convert('RGB'))
                            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                                continue
                            original_dataset.append((img, i))
                        except Exception as e:
                            logger.warning(f"Invalid image {file_path} for EWC: {str(e)}")
        else:
            original_dataset = list(zip(val_images, val_labels))  # Fallback to validation set

        if not original_dataset:
            logger.warning("No original dataset loaded for EWC, using empty dataset")
            original_dataset = []

        # Initialize model with EWC
        total_num_classes = original_num_classes + len(args.new_keywords)
        class_names = original_class_names + args.new_keywords if original_class_names else args.new_keywords
        model = SEAL_CNN_EWC(
            num_classes=total_num_classes,
            pretrained_model_path=args.model_path,
            original_num_classes=original_num_classes,
            class_names=class_names
        ).to(device)
        print(f"Initialized SEAL-CNN with EWC for {total_num_classes} classes: {class_names}")

        # Fine-tune with EWC
        finetune_loss = finetune_ewc(
            model,
            train_images,
            train_labels,
            val_images,
            val_labels,
            original_dataset,
            lr=0.0005,
            epochs=20,
            patience=5,
            ewc_lambda=args.ewc_lambda
        )

        # Evaluate
        accuracy, val_loss = evaluate_accuracy_and_loss(model, val_images, val_labels)

        # Save fine-tuned model
        fine_tuned_model_path = "finetuned_model_ewc.pth"
        torch.save(model.state_dict(), fine_tuned_model_path)
        print(f"Fine-tuned model saved to {fine_tuned_model_path}")

        print(f"EWC fine-tuning complete with validation accuracy: {accuracy:.4f}, validation loss: {val_loss:.4f}")

        # Print stored class names
        print(f"Stored class names: {model.get_class_names()}")

    except Exception as e:
        logger.error(f"Failed to run EWC fine-tuning: {str(e)}")

# === ENTRY POINT ===
if __name__ == "__main__":
    asyncio.run(main())