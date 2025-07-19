# 🤖End_To_End_Image_Pipeline🪟

This repository presents a novel approach that combines *adaptive learning*, *semantic augmentation*, and *resilient continual learning* to build an intelligent system capable of learning incrementally with minimal human supervision—mimicking a key property of **Artificial General Intelligence (AGI)**.

---

## 🚀 Key Components

### 1. **Data Acquisition & Semantic Augmentation** (`augment_generator.py`)
- Uses `icrawler` to **automatically download images** based on keyword concepts.
- Employs **CLIP (Contrastive Language-Image Pretraining)** to **filter and augment images** using semantic similarity, ensuring contextual relevance.
- Applies advanced **data augmentation** techniques (rotation, jitter, erasing, etc.) to mimic real-world conditions.

### 2. **Base Learning via SEAL-CNN**
- Builds upon **ResNet18**, integrating a **custom Feature Enhancement Layer (FEL)** with attention mechanisms.
- Only trains deeper layers (`layer4`, `fc`) and FEL for **parameter efficiency**.
- Ensures generalization on small datasets through **transfer learning and feature boosting**.

### 3. **Continual Learning with EWC** (`Fine_tuning.py`)
- Introduces **Elastic Weight Consolidation (EWC)** to preserve knowledge of previous tasks/classes.
- Enables **safe addition of new concepts** without catastrophic forgetting.
- Provides a controlled framework to scale learning towards **lifelong memory**, a key trait of AGI systems.

### 4. **Inference Utility** (`Inference.py`)
- Seamless testing of single unseen images.
- Handles both standard and fine-tuned models with EWC.

---

## 🧠 How This Moves Towards AGI

AGI is not just about large models but systems that can **learn continuously**, **adapt**, and **understand semantically**. This project introduces several AGI-aligned advancements:

- **Self-supervised semantic filtering**: Using CLIP to evaluate image "meaning" reduces human labeling effort.
- **Adaptive continual learning**: The EWC mechanism retains older knowledge, akin to human long-term memory.
- **Efficient plasticity**: FEL + ResNet allows learning without retraining entire models—an energy-efficient AGI trait.
- **Real-world robustness**: The augmentation strategies simulate diverse environments, helping the model adapt better to variability.
- **Open-world extensibility**: You can add new categories or fine-tune with minimal data and computation.

---

## 🔧 Usage

### 1. Train Base Model
```bash
python augment_generator.py --keywords "Cat" "Dog" --limit 30 --test_image path/to/test.jpg
```

### 2. Fine-tune with New Classes

```bash
python Fine_tuning.py --model_path adapted_model.pth --new_keywords "Car" "Bike" --use_old_data
```

### 3. Run Inference

```bash
python Inference.py --model_path finetuned_model_ewc.pth --image_path path/to/image.jpg --is_finetuned
```

## 📁 Project Structure

* augment_generator.py: End-to-end training pipeline with augmentation & SEAL-CNN training.

* Fine_tuning.py: Incremental learning with EWC support.

* Inference.py: Predict new inputs using trained models.

* adapted_model.pth / finetuned_model_ewc.pth: Saved models from different stages.

### ✍️ Author
M. Ajay Sivakumar

Independent **AI & AGI Researcher,** Chennai.


## 📝License
MIT License
