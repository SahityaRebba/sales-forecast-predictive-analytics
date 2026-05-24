import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page setup
st.set_page_config(page_title="Predictive Analytics", layout="wide")

st.title("🔮 Predictive Analytics Dashboard")
st.markdown("*Forecast Future Sales Using Machine Learning*")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("historical_sales_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Revenue"] = df["Revenue"].astype(float)
    df["Units_Sold"] = df["Units_Sold"].astype(float)
    df["Traffic_Score"] = df["Traffic_Score"].astype(float)
    
    # Add features for better prediction
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear
    
    return df

df = load_data()

st.subheader("📊 Historical Data Overview")
st.dataframe(df.head(20), use_container_width=True)

# Display data statistics
col1, col2, col3 = st.columns(3)
col1.metric("📅 Data Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
col2.metric("📊 Total Records", len(df))
col3.metric("💰 Total Revenue", f"₹{df['Revenue'].sum():,.0f}")

st.markdown("---")

# Sidebar for model selection
st.sidebar.header("⚙️ Model Settings")

model_type = st.sidebar.selectbox(
    "Select Prediction Model",
    ["Linear Regression", "Random Forest"]
)

test_size = st.sidebar.slider("Test Size (%)", 20, 40, 25) / 100

predict_target = st.sidebar.selectbox(
    "What to Predict?",
    ["Revenue", "Units_Sold"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Features used for prediction:**\n"
    "- Day of Year\n"
    "- Month\n"
    "- Day\n"
    "- Is Weekend\n"
    "- Is Holiday\n"
    "- Promotion\n"
    "- Traffic Score"
)

# Prepare data for modeling
st.subheader("🤖 Model Training & Prediction")

# Features for prediction
feature_columns = ["DayOfYear", "Month", "Day", "Is_Weekend", "Is_Holiday", "Promotion", "Traffic_Score"]
target_column = "Revenue" if predict_target == "Revenue" else "Units_Sold"

X = df[feature_columns]
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train model
if model_type == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

# Display metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Model Type", model_type)
col2.metric("✅ Train MAE", f"{train_mae:,.0f}")
col3.metric("🎯 Test MAE", f"{test_mae:,.0f}")
col4.metric("📊 R² Score", f"{test_r2:.2%}")

st.markdown("---")

# Actual vs Predicted Chart
st.subheader("📈 Actual vs Predicted Values")

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_test,
    "Index": range(len(y_test))
})

fig_compare = px.scatter(
    comparison_df,
    x="Actual",
    y="Predicted",
    title=f"Actual vs Predicted {predict_target}",
    labels={"Actual": f"Actual {predict_target}", "Predicted": f"Predicted {predict_target}"},
    trendline="ols",
    color_discrete_sequence=["blue"]
)
fig_compare.add_shape(
    type="line",
    x0=comparison_df["Actual"].min(),
    y0=comparison_df["Actual"].min(),
    x1=comparison_df["Actual"].max(),
    y1=comparison_df["Actual"].max(),
    line=dict(color="red", dash="dash")
)
st.plotly_chart(fig_compare, use_container_width=True)

# Feature Importance (for Random Forest)
if model_type == "Random Forest":
    st.subheader("🔑 Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    
    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Which Factors Most Influence Sales?",
        color="Importance",
        color_continuous_scale="Viridis",
        text="Importance"
    )
    fig_importance.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    st.plotly_chart(fig_importance, use_container_width=True)

st.markdown("---")

# Future Predictions
st.subheader("🔮 Predict Future Sales")

col1, col2 = st.columns(2)

with col1:
    future_date = st.date_input("Select Future Date", datetime.now() + timedelta(days=7))
    is_weekend = st.selectbox("Is Weekend?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_holiday = st.selectbox("Is Holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    promotion = st.selectbox("Has Promotion?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    traffic_score = st.slider("Expected Traffic Score", 0, 100, 75)

# Create future features
future_features = pd.DataFrame({
    "DayOfYear": [future_date.timetuple().tm_yday],
    "Month": [future_date.month],
    "Day": [future_date.day],
    "Is_Weekend": [is_weekend],
    "Is_Holiday": [is_holiday],
    "Promotion": [promotion],
    "Traffic_Score": [traffic_score]
})

# Make prediction
future_prediction = model.predict(future_features)[0]

# Display prediction
st.subheader("📊 Predicted Result")

if predict_target == "Revenue":
    st.metric("💰 Predicted Revenue", f"₹{future_prediction:,.0f}")
    st.info(f"📅 For Date: **{future_date.strftime('%B %d, %Y')}**")
else:
    st.metric("📦 Predicted Units Sold", f"{future_prediction:,.0f}")
    st.info(f"📅 For Date: **{future_date.strftime('%B %d, %Y')}**")

# Model explanation
with st.expander("📖 How Does This Work?"):
    st.markdown("""
    **Model Explanation:**
    
    - **Linear Regression**: Finds linear relationship between features and target
    - **Random Forest**: Uses multiple decision trees for better accuracy
    
    **Features Used:**
    - Day of Year, Month, Day (Time-based patterns)
    - Is Weekend, Is Holiday (Calendar effects)
    - Promotion (Marketing impact)
    - Traffic Score (Customer interest)
    
    **Model Performance:**
    - R² Score: How well the model explains variance (closer to 100% = better)
    - MAE: Average prediction error (lower = better)
    """)

# Download predictions
st.markdown("---")
st.subheader("📥 Download Model Results")

# Create results dataframe
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_test,
    "Error": y_test.values - y_pred_test,
    "Error_Percentage": abs((y_test.values - y_pred_test) / y_test.values) * 100
})

csv = results_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Prediction Results (CSV)",
    data=csv,
    file_name="prediction_results.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption("✅ Predictive Analytics Dashboard | Machine Learning | Built with Python, Scikit-learn, Streamlit & Plotly")