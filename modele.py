#--------------------- IMPORTATION DE BIBLIOTHEQUES -------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#---------------------- PREPARATION DU DATASET ------------------------

data = pd.read_csv("StudentPerformanceFactors.csv")

print("Lignes, Colonnes:", data.shape)
print(data.head(2))

print("\nInfo:")
print(data.info())

print("\nVariables manquantes:")
print(data.isnull().sum())

#---------------------- FIXER LES VALEURS MANQUANTES ------------------

cat_missing = ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]
for col in cat_missing:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\nValeurs manquantes après nettoyage:")
print(data.isnull().sum())

#---------------------- VARIABLE TARGET -------------------------------

data["Success"] = (data["Exam_Score"] >= 70).astype(int)
print(f"\nSuccess rate (>= 70): {data['Success'].mean():.2%}")

#---------------------- PREPARATION FEATURES --------------------------

df_features = data.drop(columns=["Exam_Score"])

categorical_cols = df_features.select_dtypes(include=["object"]).columns
numeric_cols = df_features.select_dtypes(include=["int64", "float64"]).columns.drop("Success")

print("\nColonnes catégorielles:", list(categorical_cols))
print("Colonnes numériques:", list(numeric_cols))

#---------------------- SCALING DES NUMERIQUES -------------------------

scaler = StandardScaler()
df_features[numeric_cols] = scaler.fit_transform(df_features[numeric_cols])

#---------------------- ONE HOT ENCODING ------------------------------

encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded = encoder.fit_transform(df_features[categorical_cols])

encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
encoded_df.index = df_features.index

#---------------------- DATASET FINAL ---------------------------------

dcopyfinal = pd.concat([df_features[numeric_cols], encoded_df, df_features["Success"]], axis=1)
print("\nShape Final:", dcopyfinal.shape)
print(dcopyfinal.head(2))

#---------------------- TRAIN / TEST SPLIT ----------------------------

X = dcopyfinal.drop(columns=["Success"])
y = dcopyfinal["Success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#---------------------- LOGISTIC REGRESSION --------------------------

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1] 

#---------------------- EVALUATION -----------------------------------

print("\nAccuracy:", accuracy_score(y_test, pred))
print("Cross-validation accuracy:", cross_val_score(model, X, y, cv=5).mean())
print("\nClassification Report:\n", classification_report(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

#---------------------- PLOTS -----------------------------------------

# 1) Histogram of Exam Scores
plt.figure(figsize=(8,5))
sns.histplot(data=data, x="Exam_Score", bins=20, kde=True, color="skyblue")
plt.title("Distribution des Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Nombre d'étudiants")
plt.tight_layout()
plt.show()

# 2) Success vs Failure
plt.figure(figsize=(6,5))
counts = data["Success"].value_counts().sort_index()
df_counts = pd.DataFrame({
    "Class": ["Fail","Success"],
    "Count": counts.values
})
sns.barplot(
    data=df_counts,
    x="Class",
    y="Count",
    hue="Class",
    palette={"Fail":"red","Success":"green"},
    dodge=False,
    legend=False
)
plt.title("Répartition: Success vs Failure")
plt.xlabel("Classe")
plt.ylabel("Nombre d'étudiants")
plt.tight_layout()
plt.show()

# 3) Scatter: Hours Studied vs Exam Score
if "Hours_Studied" in data.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(
        x=data["Hours_Studied"], 
        y=data["Exam_Score"], 
        hue=data["Success"], 
        palette={0:"red",1:"blue"}, 
        alpha=0.6
    )
    plt.title("Hours Studied vs Exam Score")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.tight_layout()
    plt.show()

#------------- FEATURE IMPORTANCE ----------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})
importance["Abs_Coefficient"] = importance["Coefficient"].abs()
top_features = importance.sort_values("Abs_Coefficient", ascending=False).head(20)

plt.figure(figsize=(10,12))
sns.barplot(
    data=top_features, 
    x="Coefficient", 
    y="Feature", 
    palette="coolwarm"
)
plt.title("Top 20 Feature Importances (Logistic Regression)")
plt.tight_layout()
plt.show()

#---------------------- NEW STUDENT PREDICTIONS -----------------------

def predict_student(student_dict, threshold=0.5):
    new_df = pd.DataFrame([student_dict])
    # scale numeric
    new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])
    # one-hot encode categorical
    encoded_new = encoder.transform(new_df[categorical_cols])
    encoded_new_df = pd.DataFrame(encoded_new, columns=encoder.get_feature_names_out(categorical_cols))
    encoded_new_df.index = new_df.index
    X_new = pd.concat([new_df[numeric_cols], encoded_new_df], axis=1)
    # predict
    prob = model.predict_proba(X_new)[:,1][0]
    pred = (prob >= threshold).astype(int)
    print(f"Predicted probability of Success: {prob:.2f}")
    print("Predicted class:", "Success" if pred==1 else "Fail")

print("# *****************************************************************************************************")

# Example low-performing student
new_student1 = {
    "Hours_Studied": 18,
    "Attendance": 65,
    "Parental_Involvement": "Low",
    "Access_to_Resources": "Medium",
    "Extracurricular_Activities": "No",
    "Sleep_Hours": 7,
    "Previous_Scores": 70,
    "Motivation_Level": "Medium",
    "Internet_Access": "Yes",
    "Tutoring_Sessions": 1,
    "Family_Income": "Medium",
    "Teacher_Quality": "High",
    "School_Type": "Public",
    "Peer_Influence": "Positive",
    "Physical_Activity": 3,
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "College",
    "Distance_from_Home": "Near",
    "Gender": "Male"
}
predict_student(new_student1)

print("# *****************************************************************************************************")

# Example high-performing student
new_student2 = {
    "Hours_Studied": 30,
    "Attendance": 95,     
    "Parental_Involvement": "High",     
    "Access_to_Resources": "High",
    "Extracurricular_Activities": "Yes",
    "Sleep_Hours": 8,               
    "Previous_Scores": 90,       
    "Motivation_Level": "High",
    "Internet_Access": "Yes",
    "Tutoring_Sessions": 3,        
    "Family_Income": "High",
    "Teacher_Quality": "High",
    "School_Type": "Private",
    "Peer_Influence": "Positive",
    "Physical_Activity": 4,
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "Postgraduate",
    "Distance_from_Home": "Near",
    "Gender": "Female"
}
predict_student(new_student2)

print("# *****************************************************************************************************")
