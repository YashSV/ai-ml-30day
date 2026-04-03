# Day 9: First Neural Network (MNIST)

## What I Built
A 3-layer neural network that classifies handwritten digits with 97.38% accuracy.

## What I Understand
- Tensors are like NumPy arrays but optimized for neural networks
- A neural network has layers that transform data
- Forward pass: input → layers → output
- Loss function: measures how wrong the prediction is
- Backward pass: computes gradients to improve weights
- Training loop: repeat forward/backward to reduce loss

## What Confused Me
- `torch.no_grad()` and why we need it
- How `optimizer.zero_grad()` works exactly
- The details of backpropagation (gradients flowing backward)
- Why we use specific activations (ReLU, etc.)

## Key Insight
The model works. I trained it. Loss dropped. Accuracy is 97%. But I don't fully *understand* every line yet. That's OK—understanding comes with repetition and deeper study later.

## Next Steps
Build more models. Experiment. Understanding deepens with practice, not memorization.