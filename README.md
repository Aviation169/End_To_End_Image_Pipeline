# 🤖End_To_End_Image_Pipeline🪟

A Python package for training, fine-tuning, and inferring with CNN models using Elastic Weight Consolidation (EWC) for image classification.

## ⬇️Installation

```
pip install end_to_end_image_pipeline
```

## 📒Usage

* Train and Adapt Model

```
python -m pipeline.augment_generator --keywords Monkey Iphone12 --limit 20 --test_image data/Test_image.jpg
```

* Fine-Tune with EWC

```
python -m pipeline.fine_tuning --model_path adapted_model.pth --new_keywords Man Apple_Watch --limit 50 --use_old_data --ewc_lambda 100
```

* Inference

```
python -m pipeline.inference --model_path finetuned_model_ewc.pth --image_path data/Apple_Watch.jpg --is_finetuned
```

## 🔽Requirements

```
Python >= 3.7
torch
torchvision
numpy
pandas
pillow
clip
icrawler
scikit-learn
tqdm
```

## 📝License
MIT License
