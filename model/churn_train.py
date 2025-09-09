
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('C:/ml-ci-cd-pipeline/data/Churn_Modelling.csv')

# Drop irrelevant columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encode categorical variables
le_gender = LabelEncoder()
data["Gender"] = le_gender.fit_transform(data["Gender"])

# One-hot encode Geography
data = pd.get_dummies(data, columns=["Geography"], drop_first=True)

# Features and target
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'C:/ml-ci-cd-pipeline/model/churn_model.pkl')
joblib.dump(scaler, 'C:/ml-ci-cd-pipeline/model/scaler.pkl')
