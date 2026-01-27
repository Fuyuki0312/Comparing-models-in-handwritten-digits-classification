## Single Handwritten Numerical Digit Classification
Image classification using Convolutional Neural Network, demonstrating inference pipeline with PyTorch.


## Project Overview
- Task: Numerical Image Classification (from 0 to 9)
- Model: Convolutional Neural Network (CNN)
- Framework: PyTorch


## Dataset
- The dataset is not included due to size limitations.

- Structure expected:
- numbers/
- ├─ 0/
- ├─ 1/
- ├─ 2/
- ├─ ...
- ├─ 9/


## Inference
- You may use: images from Kaggle Single Handwritten Number Digit dataset or any custom single-handwritten-number images.
- Note: Input images are converted to grayscale and resized before inference and their background color should be white.


## Limitation
- Model usually gives right predictions only when the background color of input images is white because this model was trained primarily on numerical images with white backgrounds.


## Possible Improvements
- Expanding the dataset to include numerical images with diverse backgrounds (dark, textured, etc).
- Applying background-related data augmentation techniques during training.


## Others
- This project was built in 1/2026 as part of my journey to become an AI Engineer.
