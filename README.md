# 🕵 Crime Data Analysis and Case Closure Prediction

##  Overview
This mini-project focuses on analyzing crime data and predicting the likelihood of a criminal case being **closed** or **remaining open**.  
By applying **data preprocessing, exploratory data analysis (EDA), and machine learning models**, the project uncovers key factors influencing case outcomes and provides valuable insights for law enforcement and policy planning.

---

##  Project Objectives
- Perform **data cleaning and preprocessing** to handle missing, duplicate, and inconsistent records.  
- Conduct **exploratory data analysis (EDA)** to identify crime trends across cities, genders, and domains.  
- Build **classification models** to predict case closure status.  
- Apply **regression models** to predict operational parameters like *Police Deployed*.  
- Use **clustering algorithms** to identify hidden crime patterns.  
- Evaluate and compare model performance using standard metrics.

---

##  Dataset Description
**Dataset Name:** `crime_dataset_india.csv`  
**Target Variable:** `Case Closed` (Yes/No)

### 🔹 Features Overview
| Feature | Type | Description |
|----------|------|-------------|
| City | Categorical | City where the crime occurred |
| Crime Code | Numerical | Unique identifier for crime type |
| Crime Description | Categorical | Description of the crime |
| Victim Age | Numerical | Age of the victim |
| Victim Gender | Categorical | Gender of the victim |
| Weapon Used | Categorical | Type of weapon involved |
| Crime Domain | Categorical | Category/domain of crime |
| Police Deployed | Numerical | Number of police officers assigned |

###  Dropped Columns
- Report Number  
- Time of Occurrence  
- Date Reported  
- Date of Occurrence  
- Date Case Closed  

These were removed to avoid redundancy and prevent data leakage.

---

##  Data Preprocessing
- **Missing Values:** Filled using mode or constant values (`Unknown` for categorical data).  
- **Duplicates:** Removed duplicate records.  
- **Encoding:** Used **One-Hot Encoding** for categorical features.  
- **Scaling:** Applied **StandardScaler** for numerical features.  
- **Pipeline:** Combined all preprocessing steps using a `ColumnTransformer` and `Pipeline` for efficient model training.  

---

##  Exploratory Data Analysis (EDA)
EDA was performed to identify relationships and patterns within the data:
- Distribution of crimes across cities and domains.  
- Victim demographics (age and gender).  
- Relationship between police deployment and closure rate.  
- Trends over time (by week and month of reporting).  
- Correlation heatmaps and visualization of inter-feature relationships.  

Key visualizations:
- Countplots for categorical variables  
- Boxplots and histograms for numerical distributions  
- Pie charts for city-wise case closures  
- Scatter plots for relationships (e.g., Victim Age vs Police Deployed)  
- Heatmap for correlation analysis  

---

##  Machine Learning Models Implemented
| Model | Type | Algorithm | Key Characteristics |
|--------|------|------------|----------------------|
| Logistic Regression | Classification | Linear model | Baseline performance |
| Decision Tree | Classification | Tree-based | Interpretable rules |
| Random Forest | Classification | Ensemble | Robust, stable model |
| K-Nearest Neighbors (KNN) | Classification | Distance-based | Simple and intuitive |
| Support Vector Machine (SVM) | Classification | Kernel-based | **Best performer (F1 ≈ 0.57)** |
| Naive Bayes | Classification | Probabilistic | Fast, low-complexity |
| XGBoost | Classification | Gradient Boosting | High accuracy, strong generalization |
| Linear Regression | Regression | Linear | For continuous prediction (e.g., Police Deployed) |
| K-Means Clustering | Unsupervised | Centroid-based | Crime pattern grouping |

---

##  Predictions Made

###  **Classification Prediction**
**Goal:** Predict whether a case will be *Closed* or *Open*.  
**Best Model:** Support Vector Machine (SVM)  
**Result:**  
- F1-score: **0.57**
- Accuracy: **~0.78**
- Best Predictors: `Crime Domain`, `City`, `Weapon Used`, `Police Deployed`, and `Victim Age`.

**Sample Output:**
| Actual | Predicted |
|--------|------------|
| Yes | Yes |
| No | No |
| Yes | Yes |
| No | Yes |
| No | No |

---

###  **Regression Prediction**
**Goal:** Predict continuous values such as `Police Deployed` and `Victim Age`.  
**Best Model:** Linear Regression  
**Performance:**
| Target | R² Score | MAE | MSE |
|---------|-----------|-----|-----|
| Police Deployed | **0.80** | 1.22 | 2.85 |
| Crime Code | 0.70 | 1.51 | 3.24 |
| Victim Age | 0.60 | 3.02 | 8.15 |

---

###  **Clustering Prediction**
**Goal:** Identify groups of crimes with similar characteristics using **K-Means (k=4)**.  
**Cluster Insights:**
| Cluster | Characteristics | Crime Type |
|----------|------------------|-------------|
| Cluster 0 | Urban, weapon-related, high closure rate | Violent crimes |
| Cluster 1 | Rural, low deployment, low closure | Petty crimes |
| Cluster 2 | Cyber or financial crimes | Delayed closure |
| Cluster 3 | Balanced profile | Mixed-category crimes |

---

##  Model Evaluation Metrics
All classification models were evaluated using:
- Accuracy Score  
- Precision Score  
- Recall Score  
- F1 Score  
- Confusion Matrix  

The **Support Vector Machine (SVM)** model achieved the best overall F1-score, making it the most suitable for predicting case closure outcomes.

---

##  Key Insights

### 📍 Data Insights
- **Urban cities** show higher crime counts but also greater police presence.  
- **Weapon-involved crimes** are more likely to be solved due to stronger evidence.  
- **Police deployment** directly affects the likelihood of a case being closed.  
- **Middle-aged victims** (20–40 years) form the majority of crime victims.  
- Case closure trends vary seasonally, showing operational peaks in specific months.

### 📍 Model Insights
- **SVM achieved the highest F1-score (0.57)** and consistent predictions.  
- **Random Forest and XGBoost** showed strong, stable results and interpretability.  
- **Key Predictive Features:** Crime Domain, City, Weapon Used, Police Deployed.  
- Feature scaling and encoding improved performance significantly.  

### 📍 Overall Insights
- Machine learning can successfully predict **case closure likelihood** with reasonable accuracy.  
- The project demonstrates how **data-driven policing** can optimize investigations.  
- Clustering and regression provided deeper operational insights into **crime behavior and police workload**.

---

##  How to Run This Project

### Step 1️: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2️: Load Dataset
```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/ColabData/crime_dataset_india.csv')
```

### Step 3️: Run the Notebook
Open `crime_criem.ipynb` and execute all cells in order.

### Step 4️: Save Cleaned Data and Models
```python
import joblib
df.to_csv('/content/drive/MyDrive/ColabData/cleaned_crime_data.csv', index=False)
joblib.dump(svm_model, '/content/drive/MyDrive/ColabData/best_model.pkl')
```

---

## 📂 Folder Structure
```
crime-case-closure/
│
├── crime_criem.ipynb              # Main project notebook
├── crime_dataset_india.csv         # Raw dataset (not uploaded to GitHub)
├── cleaned_crime_data.csv          # Processed dataset
├── models/
│   ├── svm_model.pkl
│   ├── random_forest.pkl
│   └── xgboost_model.pkl
└── README.md                       # Project documentation
```

---

## 🔮 Future Scope
- Integrate **real-time crime data APIs** for live predictions.  
- Deploy model using **Flask/Streamlit** for public dashboards.  
- Implement **Geo-mapping (Folium/Mapbox)** for spatial visualization.  
- Train advanced **deep learning models (LSTM, ANN)** for temporal crime forecasting.  
- Develop a **smart resource allocation system** for predictive policing.

---

##Author
**Sanskruti Nerkar**  
B.Tech Computer Science and Engineering  
Email: [sanskruti.nerkar@gmail.com]  
Project Title: *Crime Data Analysis and Case Closure Prediction*  
Institute: [Symbiosis Institute of Technology, Nagpur]  

---

> “Crimes involving weapons, occurring in urban areas with higher police deployment, have a significantly greater chance of being closed.  
> Machine learning can effectively predict closure likelihood, supporting data-driven decisions in law enforcement.”
