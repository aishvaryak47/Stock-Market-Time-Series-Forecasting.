# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv("model_results.csv")

st.title("📊 Stock Price Prediction - Model Comparison Dashboard")

st.write("This dashboard compares different models (ARIMA, LSTM, Random Forest, etc.) on performance metrics.")

# Show dataframe
st.subheader("📌 Model Results")
st.dataframe(df)

# Best model
best_model = df.loc[df['RMSE'].idxmin(), "Model"]
st.success(f"🏆 Best Model: **{best_model}** (lowest RMSE)")

# Visualization
st.subheader("📉 Metrics Comparison")

metrics = ["MAE", "MSE", "RMSE", "R2"]
for metric in metrics:
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y=metric, data=df, ax=ax)
    ax.set_title(f"{metric} Comparison")
    st.pyplot(fig)
