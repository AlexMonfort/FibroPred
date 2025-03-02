---
title: "FibroPred Predictive System"
emoji: "🩺"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
sdk_version: "5.9.1"
app_file: "app.py"
datasets:
  - "clinical_data"
license: "apache-2.0"
tags:
  - "machine-learning"
  - "healthcare"
  - "predictive-model"
  - "fibrosis"
  - "random-forest"
inference: false
---


### README: **FibroPred Predictive System**

---

#### **Description**
FibroPred is a predictive system designed to analyze clinical data and provide predictions for specific medical outcomes related to fibrosis. The tool employs machine learning models, primarily Random Forest classifiers, to forecast the likelihood of various events or conditions, such as mortality, necessity for transplantation, and progressive disease.

The system provides an easy-to-use interface built with Gradio, allowing users to input feature values and obtain predictions. Models and their configurations are stored and loaded dynamically, ensuring modularity and adaptability.

---

#### **Features**
1. **Prediction Targets**:
   - **Death**: Likelihood of patient mortality based on clinical and diagnostic features.
   - **Binary Diagnosis**: Classification of patients into specific diagnostic categories.
   - **Necessity of Transplantation**: Assessment of whether a patient is likely to need a transplant.
   - **Progressive Disease**: Prediction of disease progression based on longitudinal data.

2. **Dynamic Model Loading**: Models and feature sets are loaded dynamically from a pre-configured directory (`models`).

3. **Gradio Interface**: A tabbed interface for each prediction target, where users can input values interactively and receive predictions in real-time.

---

#### **Predictive Features**
Below are the features used for prediction across all targets:

1. **Pedigree** (0 - 67):
   Represents the familial history related to fibrotic conditions.

2. **Age at diagnosis** (36.0 - 92.0):
   Age of the patient at the time of diagnosis. A critical factor as progression and treatment response vary with age.

3. **FVC (L) at diagnosis** (0.0 - 5.0):
   Forced vital capacity in liters at the time of diagnosis, reflecting lung function.

4. **FVC (%) at diagnosis** (0.0 - 200.0):
   Forced vital capacity as a percentage of the expected value for the patient’s age and sex.

5. **DLCO (%) at diagnosis** (0.0 - 200.0):
   Diffusion capacity for carbon monoxide as a percentage, measuring gas exchange efficiency in the lungs.

6. **RadioWorsening2y** (0 - 3):
   Radiological assessment of lung deterioration over two years. Higher values indicate significant progression.

7. **Severity of telomere shortening - Transform 4** (1 - 6):
   Indicates the degree of telomere shortening.

8. **Progressive disease** (0 - 1):
   Binary variable indicating whether the disease is progressive (1) or stable (0).

9. **Antifibrotic Drug** (0 - 1):
   Binary variable representing the use of antifibrotic drugs. 1 indicates use; 0 indicates none.

10. **Prednisone** (0 - 1):
    Binary variable reflecting prednisone usage. 1 indicates use; 0 indicates none.

11. **Mycophenolate** (0 - 1):
    Binary variable indicating mycophenolate usage. 1 indicates use; 0 indicates none.

12. **FVC (L) 1 year after diagnosis** (0.0 - 5.0):
    Forced vital capacity in liters one year after diagnosis, used to evaluate changes in lung function.

13. **FVC (%) 1 year after diagnosis** (0.0 - 200.0):
    Forced vital capacity as a percentage one year after diagnosis.

14. **DLCO (%) 1 year after diagnosis** (0.0 - 200.0):
    Diffusion capacity for carbon monoxide as a percentage one year after diagnosis.

15. **Genetic mutation studied in patient** (0 - 1):
    Binary variable indicating the presence of specific genetic mutations. 1 indicates mutation found; 0 indicates none.

16. **Comorbidities** (0 - 1):
    Binary variable representing the presence of relevant comorbidities. 1 indicates presence; 0 indicates absence.

---

#### **Setup Instructions**
1. Clone or download the repository.
2. Ensure Python 3.8+ is installed.
3. Install dependencies using the command:
   ```bash
   pip install -r requirements.txt
   ```
4. Place trained models in the `models` directory. Models should be `.pkl` files containing:
   - `model`: Trained Random Forest model.
   - `features`: Feature list used during model training.

---

#### **Usage**
1. Run the application:
   ```bash
   python app.py
   ```
2. Access the Gradio interface through the displayed local or public URL.
3. Select a prediction tab, input feature values, and click "Submit" to get predictions.

---

#### **Key Scripts**
1. **app.py**:
   - Implements the Gradio interface.
   - Maps user-friendly model names to actual models and their features.

2. **fibropred_model.py**:
   - Contains the preprocessing pipeline, including imputation and feature selection.
   - Includes functions for training, evaluation, and visualization.

3. **model_utils.py**:
   - Functions to load models and their features dynamically.
   - Handles predictions using preloaded models.

4. **requirements.txt**:
   - Lists the Python dependencies required for the system.

---

#### **Technical Highlights**
- **Machine Learning Framework**: Models are built using `scikit-learn`, leveraging Random Forest classifiers for robust predictions.
- **Visualization**: The script includes utilities for plotting feature importance, ROC-AUC curves, and overfitting diagnostics.
- **Dynamic Handling**: Feature lists and models are dynamically linked, ensuring flexibility when adding new prediction targets.

---

#### **Future Improvements**

- **Optimizing Variable Names**:
  Review and refine the naming conventions for variables to improve clarity and consistency, facilitating better understanding for medical practitioners and data scientists.

- **Improving Model Precision**:
  Retrain the model with a larger and more diverse dataset, incorporating data from additional patients to enhance accuracy and generalization.

- **Identifying Optimal Medical Variables**:
  Conduct a detailed analysis to identify which medical variables contribute most significantly to prediction accuracy and consider eliminating less relevant ones to simplify the model.

- **Testing Model Performance with Reduced Variables**:
  Assess whether the model maintains strong predictive performance with a reduced set of optimized medical variables, which could enhance interpretability and efficiency.

- **Expanding Dataset Diversity**:
  Incorporate data from different demographics, regions, and clinical conditions to ensure the model performs well across diverse patient groups.

- **Adding Longitudinal Data Analysis**:
  Integrate longitudinal data to capture temporal patterns in disease progression, which could significantly enhance prediction capabilities.

- **Real-time Model Retraining**:
  Develop an interface or mechanism for users to upload new patient data and retrain the model seamlessly, keeping it up-to-date with the latest insights.

### Model

This Space uses the model available at: [FibroPred Model Repository](https://huggingface.co/amonfortc/FibroPred).

---

This README provides a comprehensive guide to understanding and using the **FibroPred** predictive system effectively.