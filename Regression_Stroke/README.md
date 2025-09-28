<h1>Predictive Modeling for Stroke Risk (Regression)</h1> 

1. Project Outline
This project focuses on developing and optimizing a regression model to predict the quantitative Stroke Risk Percentage for individuals based on their health profiles and reported symptoms. This model aims to serve as a proactive tool for medical professionals to assess risk, enabling targeted preventative care.

Problem Definition

The core challenge is a regression task: accurately estimating a continuous numerical value (stroke risk percentage) to provide a precise, quantitative risk assessment.

Expected Social Impact (SDG 3)

The project aligns with SDG 3: Good Health and Well-being by supporting:

Preventative Healthcare: Facilitating early identification and intervention for high-risk individuals.

Resource Optimization: Helping healthcare systems allocate resources efficiently by focusing on high-risk populations.

2. Dataset and Preprocessing
Dataset Details

The analysis utilizes the Stroke Risk Prediction Dataset Based on Symptoms, a synthetic dataset available on Kaggle (Kaggle Link).

Target: Stroke Risk (%) (continuous value).

Feature Types: 1 numerical feature (Age) and 15 binary (0/1) symptom/condition features.

Data Preparation Notes

The column "At Risk (Binary)" was dropped.

A total of 2,033 duplicate rows were removed.

3. Instructions to Run the Code
Required Project Files

To successfully run the project's Jupyter Notebook, you must ensure the following files are saved in the same directory as your notebook file:

The dataset file downloaded from the provided Kaggle link.

The external Python functions file: Functions_EAD.py.

Required Libraries

The project requires the following Python libraries. They can be installed via a package manager like pip:

Library	Description
pandas, numpy	Data manipulation and numerical operations.
scikit-learn	Machine learning models (Random Forest, Decision Tree, Linear Regression), cross-validation, and metrics.
seaborn, matplotlib.pyplot	Data visualization and plotting.
Execution

Once all required files and libraries are in place, open the Jupyter Notebook. All preprocessing steps, model training (including hyperparameter tuning), and cross-validation evaluations are contained within the notebook and can be executed sequentially.
