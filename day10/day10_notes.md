# Day 10: Convolutional Neural Networks (CIFAR-10)

## What I Built
A CNN with 3 convolutional layers, pooling, and fully connected layers. Achieved 65.47% accuracy on CIFAR-10 (object classification).

## Architecture
Input → Conv1(16 filters) → ReLU → Pool → Conv2(32 filters) → ReLU → Pool → Conv3(64 filters) → ReLU → Pool → FC1(256) → FC2(10 output)

## What I Learned

**Convolution:** Sliding a filter over an image to detect patterns (edges, textures, shapes).

**ReLU:** Activation function that introduces non-linearity (allows network to learn complex patterns).

**Pooling:** Reduces image size while keeping important features (max pooling takes max value).

**CNN vs Dense Networks:** CNNs are better for images because they learn spatial patterns through convolutions, not just dense connections.

## Results
- Started at 62.43% accuracy with 2 conv layers
- Added 3rd conv layer → 65.47% accuracy
- Experimented with epochs/learning rate (made it worse, reverted)

## Key Insight
Not all tweaks help. Sometimes the simple approach works best. More layers ≠ always better—need the right balance.

## Next Steps
Learn transfer learning (use pre-trained models) to push accuracy higher.