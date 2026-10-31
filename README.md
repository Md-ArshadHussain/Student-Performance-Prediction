# Student Performance Prediction - Streamlit App

## Dataset
Upload your `Final_Marks_Data.csv` file in the sidebar.

The app expects these exact columns from your notebook:
- Attendance (%)
- Internal Test 1 (out of 40)
- Internal Test 2 (out of 40)
- Assignment Score (out of 10)
- Daily Study Hours
- Final Exam Marks (out of 100)

`Student_ID` is optional and is not used for model training.

## Setup (Windows / VS Code)

1. Keep `app.py`, `requirements.txt`, and `Final_Marks_Data.csv` in one folder.
2. Open that folder in VS Code.
3. Open the terminal and run:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. Upload `Final_Marks_Data.csv` in the app's sidebar if it is not already selected.

## Included
- Dataset preview and descriptive statistics
- Correlation heatmap and target-correlation chart
- Linear Regression, Decision Tree, KNN, SVR, Random Forest and Voting Regressor
- Recalculated test metrics (R², MAE and RMSE)
- Actual-vs-predicted plot
- Interactive final-marks prediction form

The app uses an 80/20 split with `random_state=42`. StandardScaler is fitted only on training data via a scikit-learn Pipeline to avoid test-data leakage. Model scores may differ slightly from values manually recorded in the notebook because they are recalculated from the CSV at runtime.
