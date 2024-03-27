# Breast Cancer Detection using Machine Learning

## Overview
This project focuses on detecting breast cancer using machine learning algorithms. The dataset used in this project contains various features extracted from breast cancer cell images. We will explore different machine learning models to classify the diagnosis of breast cancer as malignant or benign.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras

## Installation
You can install the required libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
```

## Usage
1. Clone the repository or download the project files.
2. Make sure you have installed all the required libraries.
3. Open the Jupyter Notebook or Python environment where you saved the project files.
4. Run the code cells in the provided notebook or script.

## Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset. It contains features computed from digitized images of fine needle aspirates (FNA) of breast masses. The goal is to classify the diagnosis as malignant or benign based on these features.

## Exploratory Data Analysis (EDA)
The project begins with exploratory data analysis to understand the dataset:
- Loading and inspecting the data
- Handling missing values (if any)
- Exploring data types
- Data preprocessing (removing unnecessary columns, encoding categorical data)

## Data Visualization
We visualize the data to gain insights:
- Distribution of diagnosis (malignant or benign)
- Correlation matrix to understand feature relationships

## Machine Learning Models
We build and evaluate several machine learning models:
1. **K-Nearest Neighbors (KNN) Classifier**
2. **Support Vector Machine (SVM) Classifier**
3. **Decision Tree Classifier**
4. **Sequential Neural Network (NN) Model**

For each model, we perform the following steps:
- Preprocess the data (split into training and testing sets, scale if necessary)
- Train the model on the training data
- Evaluate the model's performance using accuracy score

## Results
The accuracy of each model is evaluated on the test set:
- K-Nearest Neighbors (KNN): *Accuracy: 0.93*
- Support Vector Machine (SVM): *Accuracy: 0.96*
- Decision Tree: *Accuracy: 0.95*
- Sequential Neural Network (NN): *Test Accuracy: 0.96*

The Sequential Neural Network (NN) model achieved the highest accuracy on the test set among the evaluated models.

## Conclusion
Machine learning models can effectively classify breast cancer based on features extracted from cell images. Further optimization and fine-tuning of models can improve accuracy and reliability in real-world applications.

## Author
Abdelrahman Mohamed 
