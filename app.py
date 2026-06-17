import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import predict_category   # your custom ML function

st.title("💰 AI Personal Finance Advisor")
st.info(
    "AI Personal Finance Advisor helps users track expenses, "
    "predict future spending, and evaluate financial health using machine learning."
)

# -------------------------------
# Load & Clean Data
# -------------------------------
df = pd.read_csv("data/Personal_Finance_Dataset.csv")

df.rename(columns={
    "Date": "date",
    "Transaction Description": "description",
    "Category": "category",
    "Amount": "amount",
    "Type": "type"
}, inplace=True)

# Keep only expenses
df = df[df["type"] == "Expense"]

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["category"] = df["category"].astype(str).str.strip()

df = df.dropna(subset=["date", "amount"])
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["amount"] = df["amount"].abs()   # ensure positive values only

# -------------------------------
# Add New Expense
st.header("Add Expense")

desc = st.text_input("Description")
amount = st.number_input("Amount", min_value=0.0)

expense_date = st.date_input("Expense Date")

if st.button("Add"):
    category = predict_category(desc)

    new_data = pd.DataFrame(
    [[pd.to_datetime(expense_date), desc, category, amount, "Expense"]],
    columns=["date", "description", "category", "amount", "type"])

    df = pd.concat([df, new_data], ignore_index=True)

    # force datetime after concat
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.to_csv("data/Personal_Finance_Dataset.csv", index=False)
    st.success(f"Added under category: {category}")
# -------------------------------
# Show Data
# -------------------------------
st.header("Your Expenses")
display_df = df.copy()
display_df["date"] = display_df["date"].astype(str)

st.dataframe(display_df.tail(20))
# -------------------------------
# Spending Breakdown
# -------------------------------
st.header("Spending Breakdown")
category_sum = df.groupby("category")["amount"].sum()
st.bar_chart(category_sum)

# -------------------------------
# Monthly Trend
# -------------------------------
monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()
monthly.index = monthly.index.astype(str)  # nicer labels
st.line_chart(monthly)

# -------------------------------
# Future Spending Prediction
# -------------------------------
st.header("🤖 Future Spending Prediction")

monthly_spend = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()

monthly_spend = monthly_spend.reset_index()
monthly_spend["month_num"] = range(len(monthly_spend))

X = monthly_spend[["month_num"]]
y = monthly_spend["amount"]

if len(monthly_spend) > 1:
    model = LinearRegression()
    model.fit(X, y)

    next_month_num = monthly_spend["month_num"].max() + 1
    prediction = model.predict([[next_month_num]])

    st.write(f"📊 Predicted spending next month: ₹{int(prediction[0])}")
else:
    st.warning("⚠️ Not enough data for prediction")

# -------------------------------
# Financial Health Score
# -------------------------------
st.header("Financial Health Score")

total_income = st.number_input(
    "Enter Monthly Income",
    min_value=0,
    value=100000,
    step=1000
)

df["year_month"] = df["date"].dt.to_period("M")

# Latest month's spending
monthly_expenses = df.groupby("year_month")["amount"].sum()

if len(monthly_expenses) == 0:
    st.warning("⚠️ No expenses recorded")
else:
    # Current score based on actual latest month
    total_spent = monthly_expenses.iloc[-1]

    current_savings = total_income - total_spent

    current_score = max(
        0,
        min(100, (current_savings / total_income) * 100)
    )

    # Projected score based on prediction
    predicted_spent = float(prediction[0])

    projected_savings = total_income - predicted_spent

    projected_score = max(
        0,
        min(100, (projected_savings / total_income) * 100)
    )

    st.subheader("📈 Financial Health Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Current Financial Score",
            f"{int(current_score)}/100"
        )

    with col2:
        st.metric(
            "Projected Financial Score",
            f"{int(projected_score)}/100"
        )

    st.info(
        f"Current score is based on your latest month's actual spending of ₹{int(total_spent):,}."
    )

    st.info(
        f"Projected score is based on predicted future spending of ₹{int(predicted_spent):,}."
    )

    # Current financial health message
    current_ratio = total_spent / total_income if total_income > 0 else 0

    if current_ratio > 1:
        st.error("⚠️ You are spending more than your income!")
    elif current_score < 30:
        st.error("⚠️ Current financial health is poor.")
    elif current_score < 60:
        st.warning("⚠️ Current financial health needs improvement.")
    else:
        st.success("✅ Current financial health is good.")

    # Projected financial health message
    if projected_score < 30:
        st.warning("📉 Predicted spending may negatively affect future financial health.")
    elif projected_score < 60:
        st.info("📊 Future financial health appears moderate.")
    else:
        st.success("🚀 Future financial health looks strong.")