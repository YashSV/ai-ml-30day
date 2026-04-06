# Day 13: Transfer Learning with ResNet18

## What I Learned

**Transfer Learning:** Use pre-trained models (trained on millions of images) and adapt them to new tasks.

**Fixed Features:** Freeze all pre-trained weights, only train the new final layer. Fast, requires less data.

**Fine-tuning:** Unfreeze some layers and train them on new data. Slower but more adaptable.

## What I Built

Ants vs Bees classifier using ResNet18:
- Downloaded ant/bee images
- Loaded pre-trained ResNet18 (ImageNet weights)
- Replaced final layer (1000 classes → 2 classes)
- Trained with fixed features (frozen pre-trained layers)
- **Result: 92.81% accuracy**

## Key Insight

Transfer learning is powerful because:
- Pre-trained models already know how to detect edges, shapes, textures
- We only teach it to distinguish ants vs bees
- Much faster and better accuracy than training from scratch
- Works well with small datasets

## Code Pattern
```python
model = models.resnet18(pretrained=True)
# Freeze layers
for param in model.parameters():
    param.requires_grad = False
# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)
# Train
```

## Results

**Fixed Features:** 95.42% accuracy ✓ (Better)
**Fine-tuning:** 92.81% accuracy

Fixed features outperformed fine-tuning on this small dataset. Pre-trained features were already optimal for ants vs bees.