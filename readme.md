# 🤖 Google Machine Learning Crash Course — Programming Exercises

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.23+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

---

## 📌 Overview

This repository contains my completed programming exercises from the **[Google Machine Learning Crash Course (MLCC)](https://developers.google.com/machine-learning/crash-course)** — Google's fast-paced, practical introduction to machine learning. Each notebook demonstrates hands-on application of core ML concepts, from data exploration and preprocessing to model training, evaluation, and fairness analysis.

The exercises span five key domains of applied ML and are implemented using **TensorFlow**, **Keras**, **YDF**, and the broader Python data science ecosystem in **Google Colaboratory**.

---

## 📁 Repository Structure

```
google-mlcc-exercises/
│
├── 01_linear_regression_taxi.ipynb       # Predicting taxi fares with linear regression
├── 02_classification.ipynb               # Binary classification of rice species
├── 03_Statistics_on_Dataset.ipynb        # Outlier detection & statistical analysis
├── 04_Fairness_income.ipynb              # Bias detection & fairness remediation in ML
├── 05_Decision_Tree_Forest.ipynb         # Decision tree classification with YDF
│
└── README.md
```

---

## 🧪 Exercises

### 01 — Linear Regression: Chicago Taxi Fare Prediction

> **Notebook:** `01_linear_regression_taxi.ipynb`

**Objective:** Train a linear regression model to predict taxi ride fares in Chicago, Illinois using real-world trip data from the City of Chicago Taxi Trips dataset (May 2022 subset).

**Key Skills Demonstrated:**
- Loading and exploring a real-world `.csv` dataset with **pandas**
- Performing **exploratory data analysis (EDA)** using descriptive statistics and correlation matrices
- Building and visualizing feature relationships using **pair plots**
- Training models using a **single feature** (`TRIP_MILES`) and extending to **two features** (`TRIP_MILES` + `TRIP_MINUTES`)
- Evaluating model performance using **Root Mean Squared Error (RMSE)**
- Tuning **hyperparameters** — learning rate, batch size, and epochs — and analyzing their impact on loss curves
- Comparing experiments and making predictions on new data

**Dataset:** [City of Chicago Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) · **Tech Stack:** TensorFlow · Keras · Pandas · Matplotlib

---

### 02 — Binary Classification: Turkish Rice Species

> **Notebook:** `02_classification.ipynb`

**Objective:** Build a binary classifier to distinguish between two species of Turkish rice — *Osmancik* and *Cammeo* — based on morphological grain measurements derived from image analysis.

**Key Skills Demonstrated:**
- Performing **descriptive statistical analysis** to understand feature ranges and distributions
- **Feature visualization** with 2D and 3D scatter plots using **Plotly**
- **Z-score normalization** of multi-feature datasets to prevent scale bias
- Stratified data splitting into **train / validation / test** sets (80/10/10)
- Implementing **logistic regression** via a Keras model with a sigmoid output layer
- Evaluating models using **Accuracy**, **Precision**, **Recall**, and **AUC-ROC**
- Comparing a 3-feature model against a full-feature model using **validation metrics**
- Computing **final test set metrics** after model selection to avoid data leakage

**Dataset:** [Cinar & Koklu (2019) Rice Dataset via Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) · **Tech Stack:** TensorFlow · Keras · Pandas · Plotly · NumPy

---

### 03 — Numerical Data: Statistical Analysis & Outlier Detection

> **Notebook:** `03_Statistics_on_Dataset.ipynb`

**Objective:** Apply statistical methods and visualization techniques to find hidden *bad* values and outliers within structured datasets — a critical skill in real-world data preparation pipelines.

This exercise covers two distinct sub-tasks:

**Sub-task A — Get Statistics on a Dataset:**
- Using `DataFrame.describe()` to identify columns with anomalous `min`, `max`, and `std` values
- Identifying features with **blatant outliers** by comparing mean vs. median
- Developing intuition for data quality red flags purely from summary statistics

**Sub-task B — Find the Bad Part of the Dataset:**
- Working with a fictitious dataset of 50 students evaluated over 28 days (1,400 rows)
- Using **scatter plots** to visually detect data encoding anomalies
- Applying **per-week** and **per-day-of-week** statistical breakdowns to isolate the source of corruption
- Writing custom code to **statistically confirm** day-specific encoding inconsistencies (Day 4 — Thursday)

**Key Skills Demonstrated:**
- Mastery of `pandas` statistics methods (`describe`, `groupby`, slicing)
- Visual outlier detection with **matplotlib** scatter plots
- Hypothesis-driven EDA and data validation

**Tech Stack:** Pandas · Matplotlib · NumPy

---

### 04 — Fairness & Bias: Income Prediction with MinDiff Remediation

> **Notebook:** `04_Fairness_income.ipynb`

**Objective:** Train a model to predict whether a US working adult earns above $50,000/year, then rigorously evaluate the model for fairness across demographic groups and apply a remediation technique to reduce bias.

**Key Skills Demonstrated:**
- Working with the **ACSIncome dataset** — a modern, ethically reconstructed alternative to UCI Adult, drawn from the 2018 US Census PUMS data (1.6M+ records)
- Building a **Keras functional API** model with a normalization preprocessing layer
- Converting pandas DataFrames to **`tf.data.Dataset`** format for compatibility with MinDiff
- Evaluating fairness using **TensorFlow Model Analysis (TFMA)** and **Fairness Indicators**
- Identifying **False Negative Rate (FNR) disparity** between Male and Female demographic groups (~16% gap in base model)
- Understanding **equality of opportunity** as a fairness criterion
- Implementing **MinDiff** (from `tensorflow_model_remediation`) to penalize distributional differences between sensitive and non-sensitive group subsets during training
- Comparing base model vs. MinDiff model: FNR for female group reduced from ~35% → ~30% with preserved overall AUC (~0.88)

**Dataset:** [ACSIncome — Ding et al., 2021](https://arxiv.org/abs/2108.04884) · **Tech Stack:** TensorFlow · Keras · TFMA · Fairness Indicators · TensorFlow Model Remediation · Pandas

---

### 05 — Decision Trees & Decision Forests: Palmer Penguins

> **Notebook:** `05_Decision_Tree_Forest.ipynb`

**Objective:** Use the **YDF (Yggdrasil Decision Forests)** library to train, visualize, and interpret a decision tree for multi-class species classification of the Palmer Penguins dataset.

**Key Skills Demonstrated:**
- Installing and using **YDF** — Google's modern decision forest library
- Loading and splitting mixed-type data (numerical + categorical + missing values) without requiring normalization or one-hot encoding
- Training a **CART (Classification and Regression Trees)** decision tree using `ydf.CartLearner`
- **Visualizing the decision tree** with `model.plot_tree()` and interpreting node-level class distributions
- Understanding how the tree splits on `bill_depth_mm ≥ 42.3` and `flipper_length_mm ≥ 207.5` to separate Adelie, Gentoo, and Chinstrap species
- Evaluating **training accuracy (93.4%)** and **test accuracy (91.7%)**
- Understanding the role of automatic validation set reservation for **tree pruning**
- Exploring pathways to improvement via **Random Forests** and **Gradient Boosted Trees**

**Dataset:** [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) · **Tech Stack:** YDF · Pandas · NumPy

---

## 🧠 Concepts Covered

| Topic | Exercise |
|---|---|
| Linear Regression & RMSE | Ex. 01 |
| Feature Engineering & Correlation Analysis | Ex. 01, 02 |
| Z-Score Normalization | Ex. 02 |
| Binary Classification & Logistic Regression | Ex. 02 |
| AUC-ROC, Precision, Recall | Ex. 02 |
| Descriptive Statistics & Outlier Detection | Ex. 03 |
| EDA & Visual Data Validation | Ex. 03 |
| Fairness Metrics & Bias Detection | Ex. 04 |
| MinDiff Model Remediation | Ex. 04 |
| TensorFlow Model Analysis (TFMA) | Ex. 04 |
| Decision Trees (CART) | Ex. 05 |
| Decision Forests (RF, GBT) | Ex. 05 |
| Hyperparameter Tuning | Ex. 01, 02, 05 |
| Train / Validation / Test Splits | Ex. 01–05 |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow pandas numpy matplotlib plotly scikit-learn ydf \
            tensorflow-model-analysis fairness-indicators \
            tensorflow-model-remediation tensorflow-data-validation
```

### Running the Notebooks

Each notebook is self-contained and can be run independently in **Google Colab** or a local **Jupyter** environment. Simply open the desired `.ipynb` file and execute the cells sequentially.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📚 Course Reference

- 🔗 [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- 🔗 [YDF — Yggdrasil Decision Forests](https://ydf.readthedocs.io/)
- 🔗 [TensorFlow Fairness Indicators](https://www.tensorflow.org/responsible_ai/fairness_indicators/guide)
- 🔗 [TensorFlow Model Remediation](https://www.tensorflow.org/responsible_ai/model_remediation)

---

## 👤 Author

**Syed Rafay Ali**  
*Machine Learning Enthusiast | Actively seeking ML/Data Science roles*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/syed-rafay-ali-9bb24817a/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/syedrafayme143)

---

> *These exercises were completed as part of self-directed study through Google's Machine Learning Crash Course. All datasets and course materials are credited to Google LLC and their respective original sources.*
