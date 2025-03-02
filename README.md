# Predicting Patient Readmission Risks Using Machine Learning

## Project Overview
This project aims to predict patient readmission risks using machine learning models based on hospital data. By analyzing patient demographics, admission details, medical history, and treatment details, I identify key factors contributing to hospital readmissions and develop predictive models to assist healthcare providers in reducing avoidable readmissions.

## Dataset
- **Source**: The dataset  was source from **https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008** and contains **101,766 rows and 50 features**.
- **Key Features**:
  - **Demographics**: Age, Gender, Race
  - **Admission & Discharge Details**: Admission type, Discharge disposition
  - **Medical History**: Diagnoses, Comorbidities, Number of inpatient visits
  - **Treatment Details**: Medications, Lab test results, Insulin use
- **Target Variable**: `Readmitted` (Indicates whether a patient was readmitted)

## Data Preprocessing
- **Handling Missing Values**: Removed columns with over **85% missing values**.
- **Feature Selection**: Used correlation analysis, `SelectKBest`, and **PCA** to identify the most relevant features.
- **Addressing Class Imbalance**: Applied **SMOTE** and **Cost-Sensitive Learning** to handle the imbalance in the target variable.

## Exploratory Data Analysis (EDA)
- **Patients aged 70-80 are the most readmitted.**
- **Higher hospital stays correlate with increased readmission likelihood.**
- **Certain discharge conditions have a 40%+ readmission rate.**
- **Patients with 10-15 diagnoses have a 10%+ risk of readmission.**

## Model Training & Evaluation
Four machine learning models were trained:

| Model               | Accuracy | F1 Score | Precision | Recall | ROC-AUC |
|---------------------|----------|----------|-----------|--------|---------|
| **LightGBM**       | **0.8884** | 0.0009   | 0.5000    | 0.0004 | 0.5890  |
| **Random Forest**  | 0.8172   | 0.1610   | 0.1650    | 0.1572 | 0.5979  |
| **Gradient Boosting** | 0.7385 | 0.1975   | 0.1502    | 0.2884 | 0.5701  |
| **XGBoost**        | 0.3968   | **0.2187** | 0.1278  | **0.7565** | 0.5888  |

- **Best Accuracy**: LightGBM
- **Best Recall (Identifying Readmitted Patients)**: XGBoost
- **Most Influential Features** (via SHAP Analysis):
  - **Number of inpatient visits**
  - **Number of emergency visits**
  - **Discharge disposition**
  - **Time in hospital**
  - **Age**

### Conclusion
- **Best performing model**: **LightGBM** (highest accuracy but poor recall for readmitted cases).
- **XGBoost** performed **best for recall** (detecting readmitted patients) but at the cost of low accuracy.
- **SMOTE & Cost-Sensitive Learning** were essential in handling class imbalance.
- **SHAP analysis provided interpretability**, highlighting key risk factors for readmission.
- **Future Improvements**:
  - Fine-tuning hyperparameters
  - Exploring deep learning models
  - Incorporating external datasets for better predictions.

## Recommendations to Reduce Readmissions
Based on the analysis, hospitals should:
1. **Monitor High-Risk Patients**: Implement close monitoring for patients with **high inpatient/emergency visits** and **long hospital stays**.
2. **Improve Discharge Planning**: Ensure that patients discharged under high-risk conditions receive **proper follow-up care** and **outpatient services**.
3. **Enhance Medication Management**: Track and optimize medication plans for patients with **multiple prescriptions**.
4. **Personalized Care for Elderly Patients**: Since readmission risk increases with age, **targeted care plans** should be implemented for elderly patients.
5. **Optimize Resource Allocation**: Use predictive insights to **prioritize resources** for high-risk patients.

## How to Use
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/patient-readmission-prediction.git
   cd patient-readmission-prediction
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Model Training Script**:
   ```bash
   python train_model.py
   ```
4. **Evaluate Model Performance**:
   ```bash
   python evaluate_model.py
   ```
5. **Generate Readmission Predictions**:
   ```bash
   python predict.py --input patient_data.csv
   ```

## Future Improvements
- **Hyperparameter tuning** for improved model performance.
- **Deep learning models** for better accuracy.
- **Integration with real-time hospital systems** for automated risk detection.

---

### Author: Kehinde Ogundana  
📧 Contact: [ogundanakehinde2022@gmail.com](mailto:ogundanakehinde2022@gmail.com)  
🔗 GitHub: [github.com/KennyOgun](https://github.com/KennyOgun)

