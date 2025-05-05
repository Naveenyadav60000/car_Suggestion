import streamlit as st
st.set_page_config(page_title="Smart Car Recommender")

import pandas as pd
import joblib

# Load model and data
model = joblib.load("model.pkl")
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

# Encode 'origin' column like in training
df = pd.get_dummies(df, columns=["origin"], drop_first=True)

st.title("🚗 Smart Car Recommender App")

# Show available columns for debug
st.write("📋 Columns:", df.columns.tolist())

car1 = st.selectbox("Choose Car A", df["name"].unique())
car2 = st.selectbox("Choose Car B", df["name"].unique())

if st.button("Compare"):
    # Drop label and identifier
    feature_columns = [col for col in df.columns if col not in ["name", "GoodCar"]]

    input1 = df[df["name"] == car1][feature_columns]
    input2 = df[df["name"] == car2][feature_columns]

    # Convert to NumPy arrays (model expects no headers)
    score1 = model.predict_proba(input1.values)[0][1]
    score2 = model.predict_proba(input2.values)[0][1]

    better = car1 if score1 > score2 else car2

    st.metric(label=f"{car1} Score", value=round(score1, 2))
    st.metric(label=f"{car2} Score", value=round(score2, 2))
    st.success(f"✅ Recommended: {better}")

    # Save for Power BI
    pd.DataFrame({
        "Car A": [car1],
        "Car A Score": [score1],
        "Car B": [car2],
        "Car B Score": [score2],
        "Recommended": [better]
    }).to_csv("comparison_results.csv", index=False)
    st.info("📤 Exported to comparison_results.csv")
