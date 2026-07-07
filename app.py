import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
MODEL_PATH = os.path.join("models", "best_model.pkl")
model = joblib.load(MODEL_PATH)

# Call this exactly ONCE at the top
st.set_page_config(
    page_title="Student Performance Prediction Dashboard", 
    layout="centered"
)

# Display your single main title
st.title("Student Performance Prediction Dashboard")

# Start your introductory text
st.write("""
This application predicts whether a student is likely to **Pass** or **Fail**.
""")

st.header("📊 Dataset Statistics Summary")
try:
    # Load the dataset first
    df = pd.read_csv("dataset.csv")
    
    st.header("📊 Dataset Statistics Summary")
    st.markdown("""
    This dashboard was trained on historical student records to identify 
    key academic risk factors that influence pass and fail outcomes.
    """)

    # Create side-by-side metric layout columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Records", value=f"{df.shape[0]:,}")
    with col2:
        st.metric(label="Total Features", value=f"{df.shape[1]}")
    with col3:
        st.metric(label="Target Class", value="Pass / Fail")

except FileNotFoundError:
    st.warning("⚠️ dataset.csv not found. Summary metrics cannot be loaded.")

st.header("📝 Enter Student Details")


#Collect the correct inputs matching your model's exact features
student_id = st.number_input("Enter Student ID", min_value=1, value=1)
study_hours = st.slider("Weekly Self Study Hours", 0, 50, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
participation = st.slider("Class Participation (%)", 0, 100, 70)
total_score = st.number_input("Total Score", min_value=0, max_value=100, value=60)

# Build the DataFrame and convert percentages into decimals (e.g., 75% becomes 0.75)
input_data = pd.DataFrame([{
    'student_id': student_id,
    'weekly_self_study_hours': study_hours,
    'attendance_percentage': attendance / 100.0,
    'class_participation': participation / 100.0,
    'total_score': total_score
}])

if st.button("Predict"):
    # 1. Get raw probability scores from your model
    probabilities = model.predict_proba(input_data)[0]
    
    # 2. Extract the confidence percentage for passing (Class 1)
    pass_chance = probabilities[1] * 100
    
    st.header("Prediction Result")
    
    # 3. Dynamic logic to show realistic outcomes
    if total_score >= 50 or pass_chance >= 40.0:
        st.success("🎓 Student is likely to PASS")
        st.info(f"📊 Model Confidence: {pass_chance:.1f}% Pass Probability")
    else:
        st.error("❌ Student is likely to FAIL")
        st.info(f"📊 Model Confidence: {100 - pass_chance:.1f}% Fail Probability")

st.header("📈 Model Information")

st.write("- Machine Learning Algorithm: Best Model")
st.write("- Library: Scikit-learn")
st.write("- Deployment: Streamlit")
st.write("""
This application predicts a student's grade based on:
- Weekly Self Study Hours
- Attendance Percentage
- Class Participation
- Total Score
""")

st.header("Enter Student Details")

st.markdown("""
- **Dataset:** Student Performance Dataset
- **Model:** Logistic Regression
- **Features:**
  - Student ID
  - Weekly Self Study Hours
  - Attendance Percentage
  - Class Participation
  - Total Score
 **Target:** Grade
""")

