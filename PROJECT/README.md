#  People Counting using Radar Delay-Doppler Maps

##  Overview
This repository contains the final project for the Machine Learning Frameworks (MPA-MLF) course. The objective of this project is to develop a Convolutional Neural Network (CNN) capable of accurately counting the number of persons (ranging from 0 to 3) in an enclosed space using **radar Delay-Doppler maps**.

##  Key Insights & Methodology
This project serves as a strong practical demonstration of the **"Garbage In, Garbage Out"** principle in Data Science. Initial attempts using complex transfer learning models and standard data augmentation plateaued at ~41% accuracy. 

The breakthrough to **97%+ accuracy** was achieved not by increasing computational power, but by deep domain analysis and data integrity fixes:
1. **Data Alignment:** Discovered and resolved a critical indexing mismatch in the provided dataset (CSV labels started at `0`, while images started at `img_1.png`).
2. **Grayscale Conversion:** Removed the artificial RGB colors from the radar data to force the model to focus purely on physical signal intensity.
3. **Preserving Physics:** Removed all spatial data augmentation (shifting, rotating), as moving pixels in a Delay-Doppler map fundamentally alters the physical velocity and distance represented in the radar signal.
4. **Native Resolution:** Trained the network on the original `51x45` images to preserve micro-features instead of upscaling and blurring the input.

##  Model Architecture
A lightweight, custom CNN was designed to respect the simplicity and physics of the input data. 
* **Feature Extraction:** Two Convolutional Blocks (`3x3` filters, ReLU) with `MaxPooling2D` and `Dropout`.
* **Classification Head:** Dense layer with 128 units, utilizing **L2 Regularization** instead of Batch Normalization (which was found to destroy relative signal intensity differences crucial for counting).
* **Output:** 4-class Softmax layer.

##  Kaggle Results
The model demonstrated exceptional generalization capabilities, scoring *higher* on the unseen private dataset than on the public test set (zero overfitting):
* **Validation Accuracy:** ~97%
* **Kaggle Public Score:** `0.96855`
* **Kaggle Private Score:** `0.97173`
* **Final Placement:** **6th out of 22 participants**

##  Technologies Used
* **Python 3**
* **TensorFlow / Keras** (Model creation and training)
* **Pandas & NumPy** (Data manipulation)
* **Scikit-learn** (Evaluation metrics, Data splitting)
* **Matplotlib & Seaborn** (Data visualization and Confusion Matrices)
* **PIL (Pillow)** (Image processing)

##  Repository Structure
* `main.ipynb`: The main Google Colab notebook containing the data loading, model architecture, training loop, and evaluation.
* `final_submission.csv`: The final predictions submitted to the Kaggle competition.
* `model_architecture.png`: Visual representation of the CNN topology.
* *(Note: Datasets are not included in this repository due to size limits and course policies).*
