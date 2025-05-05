import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
print("📋 Available columns:", df.columns.tolist())

# Drop missing values
df.dropna(inplace=True)

# ✅ Fix: Convert 'origin' to numeric
if df["origin"].dtype == object:
    df = pd.get_dummies(df, columns=["origin"], drop_first=True)

# Label cars as Good based on fuel efficiency and weight
df["GoodCar"] = df.apply(lambda x: 1 if x["mpg"] >= 25 and x["weight"] < 3000 else 0, axis=1)

# Features and labels
X = df.drop(columns=["name", "GoodCar"], errors='ignore')
y = df["GoodCar"]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
