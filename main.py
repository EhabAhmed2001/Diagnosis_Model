import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Dataset/Training.csv")

# Remove duplicate rows based on all columns (default)
df_unique = data.drop_duplicates()
df_unique.to_csv("unique_data.csv", index=False)

uniqueData = pd.read_csv("unique_data.csv")

X = uniqueData.drop("prognosis", axis=1)  # Features (symptoms)
y = uniqueData["prognosis"]  # Target variable (prognosis)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the target variable
Y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=45)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")

joblib.dump(model, 'models/model_trained.joblib')

