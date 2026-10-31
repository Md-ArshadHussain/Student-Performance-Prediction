import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Prediction")
st.write("Predict final examination marks using attendance, internal test scores, assignment score and daily study hours.")

FEATURES = [
    "Attendance (%)",
    "Internal Test 1 (out of 40)",
    "Internal Test 2 (out of 40)",
    "Assignment Score (out of 10)",
    "Daily Study Hours",
]
TARGET = "Final Exam Marks (out of 100)"
REQUIRED_COLUMNS = FEATURES + [TARGET]

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def train_models(data):
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            criterion="squared_error", splitter="best", max_depth=5, random_state=42
        ),
        "KNN": KNeighborsRegressor(n_neighbors=15, weights="uniform", leaf_size=4),
        "SVR": SVR(kernel="rbf", degree=3),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, criterion="squared_error", max_depth=3,
            min_samples_split=2, min_samples_leaf=1, random_state=42
        ),
    }

    fitted = {}
    rows = []
    for name, estimator in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        fitted[name] = pipe
        rows.append({
            "Model": name,
            "Train R² (%)": pipe.score(X_train, y_train) * 100,
            "Test R² (%)": r2_score(y_test, predictions) * 100,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        })

    voting = VotingRegressor([
        ("lr", LinearRegression()),
        ("dtr", DecisionTreeRegressor(random_state=42)),
        ("svr", SVR()),
        ("knr", KNeighborsRegressor()),
    ])
    voting_pipe = Pipeline([("scaler", StandardScaler()), ("model", voting)])
    voting_pipe.fit(X_train, y_train)
    fitted["Voting Regressor"] = voting_pipe
    voting_pred = voting_pipe.predict(X_test)
    rows.append({
        "Model": "Voting Regressor",
        "Train R² (%)": voting_pipe.score(X_train, y_train) * 100,
        "Test R² (%)": r2_score(y_test, voting_pred) * 100,
        "MAE": mean_absolute_error(y_test, voting_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, voting_pred)),
    })

    return fitted, pd.DataFrame(rows).sort_values("Test R² (%)", ascending=False), X_test, y_test

st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Final_Marks_Data.csv", type=["csv"])
st.sidebar.caption("Expected columns are listed below. Student_ID is optional and is not used for prediction.")

if uploaded_file is None:
    st.info("Upload your **Final_Marks_Data.csv** file in the sidebar to start. The app uses the same five input features and final-marks target as your notebook.")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
if missing:
    st.error("Your CSV is missing these required columns: " + ", ".join(missing))
    st.stop()

data = df[REQUIRED_COLUMNS].copy()
for column in REQUIRED_COLUMNS:
    data[column] = pd.to_numeric(data[column], errors="coerce")
data = data.dropna()

if len(data) < 10:
    st.error("At least 10 complete numeric rows are required to train and evaluate the models.")
    st.stop()

if (data[FEATURES].nunique() < 2).any():
    st.error("Each input feature must contain at least two distinct values.")
    st.stop()

st.success(f"Loaded {len(data):,} complete student records.")
tab_overview, tab_eda, tab_models, tab_predict = st.tabs(
    ["Dataset Overview", "Correlation & EDA", "Model Evaluation", "Predict Marks"]
)

with tab_overview:
    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Complete records", f"{len(data):,}")
    c2.metric("Input features", len(FEATURES))
    c3.metric("Average final marks", f"{data[TARGET].mean():.2f}")
    st.subheader("Descriptive statistics")
    st.dataframe(data.describe().T, use_container_width=True)

with tab_eda:
    st.subheader("Correlation heatmap")
    correlation = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Student Performance Features")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature correlation with final exam marks")
    target_corr = correlation[TARGET].drop(TARGET).sort_values()
    st.dataframe(target_corr.rename("Pearson Correlation").to_frame(), use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    target_corr.plot(kind="barh", ax=ax)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Correlation with Final Exam Marks")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Final marks distribution")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(data[TARGET], kde=True, ax=ax)
    ax.set_xlabel(TARGET)
    st.pyplot(fig)
    plt.close(fig)

with st.spinner("Training the machine learning models..."):
    models, comparison, X_test, y_test = train_models(data)

with tab_models:
    st.subheader("Model comparison")
    st.caption("All scores below are recalculated from the uploaded dataset using an 80/20 train-test split and random_state=42. Scaling is fitted on training data only.")
    st.dataframe(
        comparison.style.format({
            "Train R² (%)": "{:.2f}",
            "Test R² (%)": "{:.2f}",
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
        }),
        use_container_width=True
    )
    best_model_name = comparison.iloc[0]["Model"]
    st.metric("Best model on this test split", best_model_name,
              f"Test R²: {comparison.iloc[0]['Test R² (%)']:.2f}%")
    selected_for_plot = st.selectbox("Choose a model for actual vs predicted plot", list(models.keys()))
    predictions = models[selected_for_plot].predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, predictions, alpha=0.65)
    lower = min(float(y_test.min()), float(predictions.min()))
    upper = max(float(y_test.max()), float(predictions.max()))
    ax.plot([lower, upper], [lower, upper], "r--")
    ax.set_xlabel("Actual final marks")
    ax.set_ylabel("Predicted final marks")
    ax.set_title(f"Actual vs Predicted: {selected_for_plot}")
    st.pyplot(fig)
    plt.close(fig)

with tab_predict:
    st.subheader("Enter student details")
    st.write("Use the sliders to enter the student's academic information.")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        inputs = {}
        for index, feature in enumerate(FEATURES):
            series = data[feature]
            minimum = float(series.min())
            maximum = float(series.max())
            default = float(series.median())
            step = 1.0 if pd.api.types.is_integer_dtype(series) else 0.1
            target_col = col1 if index % 2 == 0 else col2
            inputs[feature] = target_col.slider(
                feature, min_value=minimum, max_value=maximum,
                value=min(max(default, minimum), maximum), step=step
            )
        model_choice = st.selectbox("Prediction model", list(models.keys()), index=0)
        submitted = st.form_submit_button("Predict Final Exam Marks", type="primary")

    if submitted:
        input_df = pd.DataFrame([inputs], columns=FEATURES)
        prediction = float(models[model_choice].predict(input_df)[0])
        prediction = float(np.clip(prediction, 0, 100))
        st.success(f"Predicted Final Exam Marks: {prediction:.2f} / 100")
        st.caption(f"Model used: {model_choice}. This is an estimate, not a guaranteed exam result.")

st.divider()
st.caption("Student Performance Prediction | Built with Python, Pandas, Scikit-learn, Matplotlib, Seaborn and Streamlit.")
