# Cirrhosis Prediction Analysis

**Machine Learning Project | Patient Clustering Analysis**

This project analyzes liver cirrhosis patient data using unsupervised machine learning techniques to identify patterns and patient subgroups that may inform clinical decision-making.

## 📋 Project Overview

This analysis applies clustering algorithms (K-Means and DBSCAN) to a cirrhosis patient dataset to uncover natural groupings based on medical, demographic, and laboratory features. The goal is to identify distinct patient subgroups that may have different disease progression patterns or treatment responses.

## 🏥 Dataset

- **Source**: UCI Machine Learning Repository - Cirrhosis Patient Survival Prediction Dataset
- **Samples**: 418 patients with liver cirrhosis
- **Features**: 20 clinical and demographic variables including bilirubin, albumin, prothrombin time, age, and treatment information
- **Target**: Patient survival status and disease progression

## 🧠 Methodology

### Data Preprocessing
- Handling missing values with mean imputation
- One-hot encoding for categorical variables
- Feature standardization using StandardScaler

### Clustering Techniques
- **K-Means**: Partition-based clustering with k=3
- **DBSCAN**: Density-based clustering with noise detection
- **Evaluation**: Silhouette score for cluster quality assessment

### Visualization
- PCA for dimensionality reduction
- 2D scatter plots for cluster visualization

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt