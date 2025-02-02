import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_validate

# Veri setini yükle
df = pd.read_csv('CVD_cleaned.csv')

# NaN değerleri temizle
df = df.dropna()

# Heart_Disease sütununu dönüştür
df['Heart_Disease'] = df['Heart_Disease'].map({'No': 0, 'Yes': 1})

# Veri setini eğitim ve test olarak böl
train, test = train_test_split(df, test_size=0.2, random_state=22, 
                              stratify=df['Heart_Disease'])

# X ve y değişkenlerini ayır
X_train = train.drop("Heart_Disease", axis=1)
y_train = train["Heart_Disease"].copy()
X_test = test.drop("Heart_Disease", axis=1)
y_test = test["Heart_Disease"].copy()

# Pipeline'ları oluştur
cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown='ignore', drop='first'))

num_pipeline = make_pipeline(
    FunctionTransformer(np.log1p, feature_names_out='one-to-one'),
    StandardScaler()
)

agecat_pipeline = make_pipeline(OrdinalEncoder())

genhealth_pipeline = make_pipeline(
    OrdinalEncoder(categories=[['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']])
)

checkup_pipeline = make_pipeline(
    OrdinalEncoder(categories=[['Within the past year', 'Within the past 2 years',
                                'Within the past 5 years', '5 or more years ago', 'Never']])
)

# Kolonları kategorilere ayır
numerical = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
             'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']

categorical = ['Arthritis', 'Depression', 'Diabetes', 'Exercise', 'Other_Cancer',
               'Sex', 'Skin_Cancer', 'Smoking_History']

# Preprocessing pipeline'ı oluştur
preprocessing = ColumnTransformer([
    ('Categorical', cat_pipeline, categorical),
    ('Age_Category', agecat_pipeline, ['Age_Category']),
    ('Checkup', checkup_pipeline, ['Checkup']),
    ('Gen_health', genhealth_pipeline, ['General_Health']),
    ('Numerical', num_pipeline, numerical)
], remainder='drop')  # 'passthrough' yerine 'drop' kullanıyoruz

# Mode
# Pipeline'ı güncelleyelim
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessing),
    ('sampler', SMOTE(
        sampling_strategy=0.8,  # Tam dengeleme yerine 0.8 oranında dengeleme
        random_state=42,
        k_neighbors=3  # Daha az komşu
    )),
    ('classifier', RandomForestClassifier(
        n_estimators=500,  # Ağaç sayısını azalttık
        max_depth=15,      # Maksimum derinliği sınırladık
        min_samples_split=8,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight={0: 1, 1: 10},
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
        criterion='entropy'  # 'gini' yerine 'entropy' kullanıyoruz
    ))
])
# Dengesizlik durumunu kontrol etmek için bir fonksiyon ekleyelim
def print_class_distribution(y, title):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\n{title}")
    print("-" * 50)
    for val, count in zip(unique, counts):
        print(f"Sınıf {val}: {count} ({count/total*100:.2f}%)")

# Train_and_evalua
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    print("Model eğitiliyor...")
    
    # Eğitim seti için SMOTE uygula
    preprocessor_train = preprocessing.fit_transform(X_train)
    smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(preprocessor_train, y_train)
    
    print("\nEğitim Seti - SMOTE Sonrası Dağılım:")
    print_class_distribution(y_train_resampled, "Eğitim Seti")
    
    # Test seti için SMOTE uygula
    preprocessor_test = preprocessing.transform(X_test)
    X_test_resampled, y_test_resampled = smote.fit_resample(preprocessor_test, y_test)
    
    print("\nTest Seti - SMOTE Sonrası Dağılım:")
    print_class_distribution(y_test_resampled, "Test Seti")
    
    # Model eğitimi
    classifier = RandomForestClassifier(
        n_estimators=1000,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight={0: 1, 1: 15},
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
        criterion='entropy'
    )
    def evaluate_model(model, X, y, title):
        y_pred = model.predict(X)
        print(f"\n{title}:")
        print("-" * 50)
        print(classification_report(y, y_pred, 
                                  target_names=['Risk Yok', 'Risk Var'], 
                                  digits=4))
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('Gerçek Değerler')
        plt.xlabel('Tahmin Edilen Değerler')
        plt.show()
    
    # Modeli eğit
    classifier.fit(X_train_resampled, y_train_resampled)
    
    # Orijinal veriler üzerinde değerlendir
    evaluate_model(classifier, preprocessor_test, y_test, 
                  "Orijinal Test Seti Sonuçları")
    
    # SMOTE uygulanmış veriler üzerinde değerlendir
    evaluate_model(classifier, X_test_resampled, y_test_resampled, 
                  "SMOTE Uygulanmış Test Seti Sonuçları")
    
    return classifier
    # Modeli eğit
    classifier.fit(X_train_resampled, y_train_resampled)
    
    # Tahminler
    y_pred_train = classifier.predict(X_train_resampled)
    y_pred_test = classifier.predict(X_test_resampled)
    
    # Sonuçları yazdır
    print("\nEğitim Seti Sınıflandırma Raporu:")
    print("-----------------------------------")
    print(classification_report(y_train_resampled, y_pred_train, 
                              target_names=['Risk Yok', 'Risk Var'], 
                              digits=4))
    
    print("\nTest Seti Sınıflandırma Raporu:")
    print("--------------------------------")
    print(classification_report(y_test_resampled, y_pred_test, 
                              target_names=['Risk Yok', 'Risk Var'], 
                              digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_resampled, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Gerçek Değerler')
    plt.xlabel('Tahmin Edilen Değerler')
    plt.show()
    
    return classifier



def predict_heart_risk(height, weight, age_category, exercise, smoking, 
                      alcohol, fruits, vegetables, fried_potatoes,
                      general_health='Good', diabetes='No', depression='No',
                      sex='Female', arthritis='No', other_cancer='No', 
                      skin_cancer='No', checkup='Within the past year'):
    
    # BMI hesaplama
    bmi = weight / ((height/100) ** 2)
    
    # Test verisi oluştur
    test_data = pd.DataFrame({
        'Arthritis': [arthritis],
        'Depression': [depression],
        'Diabetes': [diabetes],
        'Exercise': [exercise],
        'Other_Cancer': [other_cancer],
        'Sex': [sex],
        'Skin_Cancer': [skin_cancer],
        'Smoking_History': [smoking],
        'Age_Category': [age_category],
        'Checkup': [checkup],
        'General_Health': [general_health],
        'Alcohol_Consumption': [alcohol],
        'BMI': [bmi],
        'FriedPotato_Consumption': [fried_potatoes],
        'Fruit_Consumption': [fruits],
        'Green_Vegetables_Consumption': [vegetables],
        'Height_(cm)': [height],
        'Weight_(kg)': [weight]
    })

    # Tahmin yap
    probabilities = model_pipeline.predict_proba(test_data)[0]
    prediction = 1 if probabilities[1] >= 0.3 else 0
    
    return {
        'Risk Yok': f"{probabilities[0]:.2%}",
        'Risk Var': f"{probabilities[1]:.2%}",
        'BMI': f"{bmi:.1f}",
        'Tahmin': 'Risk Var' if prediction == 1 else 'Risk Yok'
    }

def predict_heart_risk_user_input():
    try:
        print("\n Kalp Hastalığı Risk Değerlendirmesi")

        height = float(input("Boy (cm): "))
        weight = float(input("Kilo (kg): "))

        alcohol = float(input("Haftalık alkol tüketimi (gün sayısı): "))
        fruits = float(input("Günlük meyve tüketimi (porsiyon): "))
        vegetables = float(input("Günlük yeşil sebze tüketimi (porsiyon): "))
        fried_potatoes = float(input("Aylık kızartma tüketimi (porsiyon): "))
        bmi = weight / ((height / 100) ** 2)

        arthritis = input("Artrit (eklem iltihabı) var mı? (Yes/No): ").capitalize()
        depression = input("Depresyon tanısı aldınız mı? (Yes/No): ").capitalize()
        diabetes = input("Diyabet hastası mısınız? (Yes/No): ").capitalize()
        exercise = input("Düzenli egzersiz yapıyor musunuz? (Yes/No): ").capitalize()
        other_cancer = input("Başka kanser türü geçmişi var mı? (Yes/No): ").capitalize()
        sex = input("Cinsiyet (Male/Female): ").capitalize()
        skin_cancer = input("Cilt kanseri geçmişi var mı? (Yes/No): ").capitalize()
        smoking = input("Sigara içme geçmişiniz var mı? (Yes/No): ").capitalize()

        
        age_category = input("Yaş Kategorisi (örn. '25-29'): ")
        general_health = input("Genel Sağlık Durumu (Poor/Fair/Good/Very Good/Excellent): ").capitalize()
        checkup_input = input("Son Check-up Zamanı (örn. 'Within the past year'): ").strip().lower()
        checkup_mapping = {
            "within the past year": "Within the past year",
            "within the past 2 years": "Within the past 2 years",
            "within the past 5 years": "Within the past 5 years",
            "5 or more years ago": "5 or more years ago",
            "never": "Never"
        }
        checkup = checkup_mapping.get(checkup_input, None)

        if checkup is None:
            raise ValueError("Geçersiz bir 'Check-up' değeri girdiniz. Lütfen uygun bir seçenek girin.")


        test_data = pd.DataFrame({
            'Arthritis': [arthritis],
            'Depression': [depression],
            'Diabetes': [diabetes],
            'Exercise': [exercise],
            'Other_Cancer': [other_cancer],
            'Sex': [sex],
            'Skin_Cancer': [skin_cancer],
            'Smoking_History': [smoking],
            'Age_Category': [age_category],
            'Checkup': [checkup],
            'General_Health': [general_health],
            'Alcohol_Consumption': [alcohol],
            'BMI': [bmi],
            'FriedPotato_Consumption': [fried_potatoes],
            'Fruit_Consumption': [fruits],
            'Green_Vegetables_Consumption': [vegetables],
            'Height_(cm)': [height],
            'Weight_(kg)': [weight]
        })

        probabilities = model_pipeline.predict_proba(test_data)[0]
        prediction = 1 if probabilities[1] >= 0.3 else 0

        print("\n Risk Değerlendirme Sonuçları:")
        print("----------------------------------------")
        print(f"Risk Yok: {probabilities[0]:.2%}")
        print(f"Risk Var: {probabilities[1]:.2%}")
        print(f"BMI: {bmi:.1f}")
        print(f"Tahmin: {'Risk Var' if prediction == 1 else 'Risk Yok'}")

    except Exception as e:
        print(f"\n Hata: {str(e)}")

if __name__ == "__main__":
    print("Model eğitimi başlıyor...")
    model_pipeline.fit(X_train, y_train) 
    

    print("\nOrijinal Veri Dağılımı:")
    print_class_distribution(df['Heart_Disease'], "Tüm Veri Seti")
    
   
    model = train_and_evaluate_model(X_train, y_train, X_test, y_test)

    predict_heart_risk_user_input()