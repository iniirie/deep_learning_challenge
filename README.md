# AlphabetSoup Charity Deep Learning Model  
Creators: Luke Roberts  
Date: May 2025  

## Table of Contents
- [AlphabetSoup Charity Deep Learning Model](#alphabetsoup-charity-deep-learning-model)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
  - [Research Questions to Answer](#research-questions-to-answer)
  - [Features](#features)
  - [Deployment](#deployment)
  - [Key Findings](#key-findings)
  - [Recommendation](#recommendation)
  - [Methodology](#methodology)
  - [Ethical Considerations](#ethical-considerations)
  - [Opportunities for Further Analysis](#opportunities-for-further-analysis)
  - [Resources](#resources)

## Project Description  
This project applies deep learning to predict whether a nonprofit organization will be successful in securing funding. We cleaned and preprocessed historical charity application data and trained a binary classification model using TensorFlow and Keras.

## Research Questions to Answer  
1. Can we predict whether a nonprofit will be successful in receiving funding?
2. Which features have the greatest predictive power?
3. How well does a basic deep learning model perform on this type of structured data?

## Features  
- **Neural Network Classifier:** Built using TensorFlow/Keras with two hidden layers.
- **Categorical Feature Consolidation:** Grouped rare application types and classifications into "Other" to reduce noise.
- **Standardization:** Used `StandardScaler` to normalize numerical feature values.
- **Model Export:** Trained model saved in Keras native format (`.keras`).

## Deployment  
To run the model locally:

**1. Clone the Repository:**
```bash
git clone <your-repo-url>
```

**2. Open the Project:**
Open the `Completed_Charity_Notebook.ipynb` file in Jupyter Notebook.

**3. Install Required Libraries:**
```bash
pip install pandas scikit-learn tensorflow
```

**4. Run the notebook and review results.**

## Key Findings  
- The model achieved an accuracy of approximately **X%** on the test data.
- Many categorical features (e.g., `APPLICATION_TYPE`, `CLASSIFICATION`) had high cardinality but low frequency. Grouping these improved model performance.
- Removing non-informative features like `EIN` and `NAME` was crucial to reduce noise.

## Recommendation  
The current model demonstrates moderate accuracy and is a good baseline. For better performance:
- Add more hidden layers or neurons
- Experiment with different activation functions or optimizers
- Perform feature engineering or dimensionality reduction

## Methodology  
- **Pandas:** Used to clean and encode categorical variables.
- **TensorFlow/Keras:** Used to define and train a feedforward neural network.
- **Scikit-learn:** Used for scaling and splitting the data.

## Ethical Considerations  
- Model predictions should not be the sole factor in deciding funding eligibility.
- Ensure fairness by monitoring for potential bias in features such as organization classification or application type.

## Opportunities for Further Analysis  
- Add dropout layers to prevent overfitting.
- Use automated hyperparameter tuning (e.g., Keras Tuner).
- Try ensemble methods like Random Forest or XGBoost for comparison.
- Apply SHAP or LIME for model interpretability.

## Resources  
- **TensorFlow Documentation:** https://www.tensorflow.org/  
- **Sklearn Documentation:** https://scikit-learn.org/  
- **Bootcamp Module 19:** Supervised Learning Neural Networks  
- **CSV Source:** [Charity Dataset from Bootcamp](https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv)
