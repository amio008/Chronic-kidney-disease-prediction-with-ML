# CKD Prediction Model - Machine Learning Project

## Overview
This project implements a Chronic Kidney Disease (CKD) prediction system using multiple machine learning algorithms. The system processes clinical data to classify patients as having CKD or not, utilizing a comprehensive dataset of medical attributes.

## Dataset Information
The model uses a kidney disease dataset containing various clinical parameters including:
- **Demographics**: Age
- **Clinical Measurements**: Blood pressure, specific gravity, albumin, sugar
- **Blood Tests**: Blood glucose, blood urea, serum creatinine, sodium, potassium, hemoglobin
- **Physical Attributes**: Red blood cells, pus cells, white blood cells, red blood cell count
- **Medical History**: Hypertension, diabetes mellitus, coronary artery disease
- **Symptoms**: Appetite, pedal edema, anemia

## Features
- Data preprocessing and cleaning
- Handling of missing values using appropriate imputation strategies
- Feature correlation analysis
- Data normalization using MinMaxScaler
- Implementation of three classification algorithms:
  - K-Nearest Neighbors (KNN)
  - Gaussian Naive Bayes
  - Logistic Regression

## Technology Stack
- Python 3.x
- Google Colab
- Libraries:
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - Imbalanced-learn

## Installation

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

### Setup
1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Place the dataset `kidney_disease.csv` in your Google Drive at:
```
/content/drive/MyDrive/Colab Notebooks/422/Project/
```

## Data Preprocessing Pipeline

### 1. Data Cleaning
- Convert categorical variables to numerical values
- Handle corrupted data entries
- Remove rows with excessive missing values (>10 missing attributes)

### 2. Missing Value Imputation
- **Numerical features**: Mean imputation for:
  - Float columns: specific gravity, serum creatinine, hemoglobin, red blood cell count
  - Integer columns: age, blood pressure, albumin, sugar, blood glucose, blood urea, sodium, potassium, packed cell volume, white blood cell count
- **Categorical features**: Most frequent imputation for:
  - Red blood cells, pus cells, pus cell clumps, bacteria, hypertension, diabetes, coronary artery disease, appetite, pedal edema, anemia

### 3. Feature Selection
- Correlation analysis with classification target
- Features with correlation coefficient between -0.25 and 0.25 are dropped

## Model Implementation

### Data Split
- Train/Test split: 70%/30%
- Stratified sampling to maintain class distribution

### Data Scaling
- MinMaxScaler applied to training and testing data

### Algorithms Used

#### 1. K-Nearest Neighbors
```python
knn_model = KNeighborsClassifier(n_neighbors=3)
```

#### 2. Gaussian Naive Bayes
```python
nb_model = GaussianNB()
```

#### 3. Logistic Regression
```python
lg_model = LogisticRegression()
```

## Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## Results
The model outputs performance metrics for each algorithm:
- **Accuracy**: Overall prediction accuracy
- **Confusion Matrix**: True/False Positives and Negatives
- **Classification Report**: Detailed per-class performance metrics

## File Structure
```
CKD-Prediction/
│
├── ckd_code_script.py          # Main implementation file
├── kidney_disease.csv          # Dataset (not included)
├── new1.csv                    # Processed dataset
└── README.md                   # Project documentation
```

## Usage

1. Mount Google Drive and ensure dataset is in correct path
2. Run the preprocessing pipeline
3. Train models and evaluate performance
4. Export processed dataset for further analysis

## Key Functions

### Data Transformation
```python
# Example: Converting categorical to numerical
kidneycsv["rbc"] = kidneycsv["rbc"].apply(
    lambda x: 1 if x == "normal" else (0 if x == "abnormal" else np.nan)
)
```

### Missing Value Imputation
```python
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
```

### Model Training
```python
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```

## Contributing
Feel free to fork this repository and submit pull requests for any improvements or bug fixes.

## License
This project is open-source and available for educational and research purposes.

## Acknowledgments
- Dataset source: Kidney Disease dataset for CKD prediction
- Project developed as part of CSE422 coursework

## Contact
For questions or issues, please open an issue in the repository.
