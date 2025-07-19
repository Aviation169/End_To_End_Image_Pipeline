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
import torch.amp
import pandas as pd
from typing import List, Tuple
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
threshold_similarity = 0.75
output_dir = "augmented_dataset"
csv_file = "image_labels_dataset.csv"
class_names_file = "class_names.json"
max_threads = min(32, os.cpu_count() or 4)

# === AUGMENTATION PIPELINES ===
pil_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=(0, 45)),
    transforms.RandomResizedCrop(224, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.3),
])

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
    transforms.ToPILImage(),
])

# === DOWNLOAD FUNCTION ===
def download_images(keyword, limit=20, storage_dir='dataset/raw'):
    crawler = GoogleImageCrawler(storage={'root_dir': f'{storage_dir}/{keyword}'})
    crawler.crawl(keyword=keyword, max_num=limit)

async def async_download(keyword, limit=20, storage_dir='dataset/raw'):
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

# === CLASS MAPPING ===
def create_class_map(keywords):
    class_map = {}
    for idx, keyword in enumerate(keywords):
        class_map[idx] = keyword
        class_map[keyword] = idx
    return class_map

# === MODEL DEFINITION ===
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

class SEAL_CNN(nn.Module):
    def __init__(self, num_classes: int, class_names: List[str] = None):
        super(SEAL_CNN, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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

# === FINETUNING ===
def finetune_cnn(
    model: nn.Module,
    images: List[np.ndarray],
    labels: List[int],
    val_images: List[np.ndarray],
    val_labels: List[int],
    lr: float = 0.001,
    epochs: int = 10,
    patience: int = 3
) -> float:
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

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
        indices = []
        for lbl in range(len(set(labels))):
            lbl_indices = [i for i, l in enumerate(labels) if l == lbl]
            indices.extend(lbl_indices)
        np.random.shuffle(indices)

        epoch_loss = 0
        batch_size = 8
        num_batches = (len(indices) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(indices), batch_size), total=num_batches, desc=f"Epoch {epoch+1} Batches"):
            batch_indices = indices[i:i + batch_size]
            batch_images = [images[idx] for idx in batch_indices]
            batch_labels = [labels[idx] for idx in batch_indices]
            batch_tensors = [preprocess(img).unsqueeze(0).to(device) for img in batch_images]
            batch_tensors = torch.cat(batch_tensors, dim=0)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(batch_tensors)
                loss = criterion(out, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * len(batch_images)
        avg_loss = epoch_loss / len(indices) if len(indices) > 0 else 0.0
        total_loss += avg_loss
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

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

# === TEST SAVED MODEL ===
def test_saved_model(
    model_path: str,
    test_image_paths: List[str],
    test_labels: List[int],
    class_map: dict
) -> None:
    try:
        model = SEAL_CNN(num_classes=len(class_map) // 2, class_names=list(class_map.values())[::2]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        total_correct = 0
        total = len(test_image_paths)
        for test_image_path, test_label in tqdm(zip(test_image_paths, test_labels), total=total, desc="Testing"):
            if not os.path.exists(test_image_path):
                logger.warning(f"Image {test_image_path} not found, skipping.")
                continue
            test_image = np.array(Image.open(test_image_path).convert('RGB'))
            img = preprocess(test_image).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(img)
                pred = out.argmax(dim=1).item()

            class_name = class_map.get(pred, f"Class_{pred}")
            print(f"Predicted class for test image {test_image_path}: {class_name} (ID: {pred})")
            test_label_name = class_map.get(test_label, f"Class_{test_label}")
            print(f"True label: {test_label_name} (ID: {test_label}), Correct: {pred == test_label}")
            total_correct += (pred == test_label)
        accuracy = total_correct / total if total > 0 else 0.0
        print(f"Test accuracy: {accuracy:.4f}")

    except FileNotFoundError:
        logger.error(f"Model file {model_path} or image not found.")
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")

# === LOAD DATASET ===
def load_dataset(class_dirs: List[Tuple[str, str, int]]):
    try:
        dfs = []
        for dir_path, class_name, label in class_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"{class_name} directory {dir_path} not found")
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

        if not all(lbl in class_counts for lbl in range(len(class_dirs))):
            raise ValueError("No images found for one or more classes")

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

# === SEAL ADAPTATION ===
def seal_adapt(
    class_dirs: List[Tuple[str, str, int]],
    lr: float = 0.001,
    epochs: int = 10,
    patience: int = 3
) -> Tuple[nn.Module, pd.DataFrame]:
    try:
        train_df, val_df, test_df = load_dataset(class_dirs)
        model = SEAL_CNN(num_classes=len(class_dirs), class_names=[class_dir[1] for class_dir in class_dirs]).to(device)
        print(f"Initialized ResNet-18 with FEL for {len(class_dirs)} classes: {[class_dir[1] for class_dir in class_dirs]}")

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

        finetune_loss = finetune_cnn(model, train_images, train_labels, val_images, val_labels, lr=lr, epochs=epochs, patience=patience)

        accuracy, val_loss = evaluate_accuracy_and_loss(model, val_images, val_labels)

        print(f"SEAL adaptation complete with validation accuracy: {accuracy:.4f}, validation loss: {val_loss:.4f}")
        return model, test_df

    except Exception as e:
        logger.error(f"Error during adaptation: {str(e)}")
        raise

# === INFER SINGLE IMAGE ===
def infer_single_image(model_path: str, image_path: str, class_map: dict) -> Tuple[str, int, np.ndarray]:
    try:
        model = SEAL_CNN(num_classes=len(class_map) // 2, class_names=list(class_map.values())[::2]).to(device)
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

        class_name = class_map.get(pred, f"Class_{pred}")
        return class_name, pred, probabilities

    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

# === MAIN COROUTINE ===
async def main():
    parser = argparse.ArgumentParser(description="Download, augment, and train SEAL-CNN on images")
    parser.add_argument('--keywords', type=str, nargs='+', required=True, 
                       help="Keywords for image download (e.g., Iphone12 Iphone16)")
    parser.add_argument('--limit', type=int, default=20, 
                       help="Number of images to download per keyword")
    parser.add_argument('--storage_dir', type=str, default='dataset/raw', 
                       help="Directory to store downloaded images")
    parser.add_argument('--test_image', type=str, default='data/Test_image.jpg',
                       help="Path to a single test image for inference")
    args = parser.parse_args()

    # Create class map based on keywords
    class_map = create_class_map(args.keywords)
    print(f"Class mapping: {class_map}")

    # Download images
    print("Starting image download...")
    download_tasks = [async_download(keyword, args.limit, args.storage_dir) for keyword in args.keywords]
    await asyncio.gather(*download_tasks)
    print("Image download complete.")

    # Augment images
    os.makedirs(output_dir, exist_ok=True)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    image_tasks = []
    input_dir = args.storage_dir
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, input_dir)
                class_name = os.path.dirname(rel_path).replace("\\", "/")
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

    # Prepare class directories for training
    class_dirs = [
        (os.path.normpath(os.path.join(output_dir, keyword)), keyword, class_map[keyword])
        for keyword in args.keywords
    ]

    # Run SEAL adaptation
    model_path = "adapted_model.pth"
    try:
        adapted_model, test_df = seal_adapt(
            class_dirs=class_dirs,
            lr=0.001,
            epochs=10,
            patience=3
        )
        torch.save(adapted_model.state_dict(), model_path)
        print(f"Adapted model saved to {model_path}")

        # Print stored class names
        print(f"Stored class names: {adapted_model.get_class_names()}")

        # Test the saved model
        print("Testing the saved model...")
        test_image_paths = test_df['filename'].tolist()
        test_labels = test_df['label'].tolist()
        test_saved_model(model_path, test_image_paths, test_labels, class_map)

        # Test on single unseen image
        print(f"\nTesting on unseen image: {args.test_image}")
        class_name, class_id, probabilities = infer_single_image(model_path, args.test_image, class_map)
        print(f"Predicted class for {args.test_image}: {class_name} (ID: {class_id})")
        print("Confidence scores:")
        for idx, prob in enumerate(probabilities):
            class_label = class_map.get(idx, f"Class_{idx}")
            print(f"{class_label}: {prob:.4f}")

    except FileNotFoundError:
        logger.error("Directories or images not found. Please provide valid paths.")
    except Exception as e:
        logger.error(f"Failed to run adaptation or inference: {str(e)}")

# === ENTRY POINT ===
if __name__ == "__main__":
    asyncio.run(main())
