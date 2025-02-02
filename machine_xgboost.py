# heart_risk_xgb_v2.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.preprocessing import OrdinalEncoder
warnings.filterwarnings('ignore')

print("1. Loading Data...")
datas = pd.read_csv("CVD_cleaned.csv")
print("2. Data Preprocessing...")
datas = datas.dropna(subset=['Heart_Disease'])
datas['Heart_Disease'] = datas['Heart_Disease'].apply(lambda x: 1 if x == 'Yes' else 0)

datas['BMI_Risk'] = pd.cut(
    datas['BMI'], 
    bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese_1', 'Obese_2', 'Obese_3'],
    ordered=False
)

bmi_risk_map = {
    'Underweight': 2,
    'Normal': 0,
    'Overweight': 1,
    'Obese_1': 2,
    'Obese_2': 3,
    'Obese_3': 4
}
datas['BMI_Risk_Score'] = datas['BMI_Risk'].map(bmi_risk_map)

age_risk_map = {
    '18-24': 0, '25-29': 1, '30-34': 1, '35-39': 2,
    '40-44': 2, '45-49': 3, '50-54': 3, '55-59': 4,
    '60-64': 4, '65-69': 5, '70-74': 5, '75-79': 6, '80+': 6
}
datas['Age_Risk'] = datas['Age_Category'].map(age_risk_map)

datas['Combined_Risk'] = datas['Age_Risk'] + datas['BMI_Risk_Score']

ordinal_mappings = {
    'General_Health': ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
    'Checkup': ['Never', 'Within the past year', 'Within the past 2 years', 'Within the past 5 years', '5 or more years ago'],
    'Exercise': ['No', 'Yes'],
    'Skin_Cancer': ['No', 'Yes'],
    'Other_Cancer': ['No', 'Yes'],
    'Depression': ['No', 'Yes'],
    'Diabetes': ['No', 'Yes'],
    'Arthritis': ['No', 'Yes'],
    'Sex': ['Female', 'Male'],
    'Smoking_History': ['No', 'Yes'],
    'BMI_Risk': ['Normal', 'Underweight', 'Overweight', 'Obese_1', 'Obese_2', 'Obese_3']
}

# Apply Ordinal encoding instead of One-hot encoding
ordinal_encoder = OrdinalEncoder()
categorikvalues = ['General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 
            'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 
            'Sex', 'Smoking_History', 'BMI_Risk', 'Age_Category']

datas_encoded = datas.copy()
datas_encoded[categorikvalues] = ordinal_encoder.fit_transform(datas[categorikvalues])

# Separate numerical and categorical columns
numericcols = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'BMI_Risk_Score', 
                'Age_Risk', 'Combined_Risk', 'Alcohol_Consumption', 'Fruit_Consumption',
                'Green_Vegetables_Consumption', 'FriedPotato_Consumption']

X = datas_encoded.drop(columns=['Heart_Disease'])
y = datas_encoded['Heart_Disease']

print("\nOriginal class distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numericcols] = scaler.fit_transform(X_train[numericcols])
X_test_scaled[numericcols] = scaler.transform(X_test[numericcols])

print("\n3. Applying SMOTE...")
smote = SMOTE(random_state=42, sampling_strategy=0.4)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

print("4. Training Model...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    learning_rate=0.01,  # Lower learning rate
    max_depth=6,         # Reduced depth from 8 to 6
    n_estimators=500,    # Increased number of trees
    min_child_weight=1,  # Reduced value from 2 to 1
    gamma=0.1,           # Adjusted gamma value
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10, # Increased positive class weight from 8 to 10
    eval_metric='auc'    # Added AUC as performance metric
)
xgb_model.fit(X_train_balanced, y_train_balanced)
print("\n5. Evaluating Model...")
y_pred = xgb_model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))