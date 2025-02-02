import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE 
from lightgbm import LGBMClassifier  

print("1. Loading Data...")
df = pd.read_csv("CVD_cleaned.csv")

print("2. Transforming Categorical Variables...")
from sklearn.preprocessing import LabelEncoder
categorical_columns = ['General_Health', 'Checkup', 'Exercise', 'Heart_Disease', 
                       'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 
                       'Arthritis', 'Sex', 'Age_Category', 'Smoking_History']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
print("3. Splitting Data into Training and Test Sets...")
X = df.drop(columns=['Heart_Disease'])
y = df['Heart_Disease']

# Splitting data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balancing data using SMOTE
print("4. Balancing Data with SMOTE...")
smote = SMOTE(random_state=42)
X_trainsmote, y_trainsmote = smote.fit_resample(X_train, y_train)
print("Class Distribution After SMOTE:")
print(y_trainsmote.value_counts())

print("5. Training LightGBM Model...")
# Defining the LightGBM model
lgb_model = LGBMClassifier(
    random_state=42,
    class_weight={0: 1, 1: 10},  # Balancing minority class using class weights
    n_estimators=200,            # Number of trees
    max_depth=7,                 # Maximum tree depth
    learning_rate=0.1            # Learning rate
)
lgb_model.fit(X_trainsmote, y_trainsmote)

print("6. Making Predictions...")
y_pred = lgb_model.predict(X_test)

print("7. Evaluating Model Performance...")
# Accuracy Score
print("Accuracy Score:", accuracy_score(y_test, y_pred))
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Extract feature importances
feature_importances = lgb_model.feature_importances_
features = X.columns

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importance Levels (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
