# Diabetes-Hospital-Readmission-Predictor
## Overview
This project aims to create a binary classification model to predict whether or not a diabetes patient who was admitted to the hospital will be readmitted. This predictive model can help healthcare providers identify high-risk patients early, enabling targeted interventions and improving care management. This analysis can also help identify key risk factors that contribute to hospital readmissions among diabetes patients. Ultimately, the goal is to reduce avoidable readmissions, optimize hospital resources, and support data-driven decision-making in healthcare settings.

## Data Sources
This data is from the UC Irvine Machine Learning Repository, and can be accessed [here](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008). It contains data from 130 different US hospitals and integrated delivery networks for over ten years (1999-2008). The data is an extract from a national data warehouse that collects clinical records in US hospitals - the Health Facts database (Cerner Corporation, Kansas City, MO). The data includes patient demographics, ICM-9 codes, in-hospital procedures, encounters (inpatient, outpatient, emergency), medical usage, lab test results, and hospital visit details. There are 101,766 encounters included, and 75,518 from unique patients.

## Machine Learning Models
I implemented three different classification models: logistic regression, random forest, and XGBoost. These models were chosen to compare a baseline linear approach (logistic regression) with more advanced ensemble methods (random forest and XGBoost) capable of capturing complex, non-linear relationships in the data.

## Outcome
