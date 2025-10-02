import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Reading dataset
df = pd.read_csv('student_depression_dataset.csv')
df = df.drop_duplicates()

# Replacing string to numeric value in sleep duration column
df['Sleep Duration'] = df['Sleep Duration'].str.replace("'", "").str.strip()
sleep_map = {
    'Less than 5 hours': 0,
    '5-6 hours': 1,
    '7-8 hours': 2,
    'More than 8 hours': 3,
    'Others': 4
}
df['Sleep Duration Category'] = df['Sleep Duration'].map(sleep_map)

df['Financial Stress'] = df['Financial Stress'].replace('?', np.nan)
df['Financial Stress'] = df['Financial Stress'].astype(float)
df['Financial Stress'] = df['Financial Stress'].fillna(df['Financial Stress'].median())

# Mapping columns
df['Suicidal Thoughts'] = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
df['Family History Mental Illness'] = df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
df['Gender Code'] = df['Gender'].map(gender_map)
df['Age'] = df['Age'].astype(int)
diatery_map = {'Unhealthy': 0, 'Healthy': 2, 'Moderate': 1}
df['Diet'] = df['Dietary Habits'].map(diatery_map)
df['Diet'].fillna(df['Diet'].median(), inplace=True)

# Dropping unneeded columns
df = df.drop(columns=[
    'id', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness',
    'Sleep Duration', 'Gender', 'Dietary Habits', 'City', 'Profession',
    'Work Pressure', 'Degree', 'Job Satisfaction'
])

# Splitting features and target
X = df.drop(columns=['Depression'])
Y = df['Depression']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Training models
model_LR = LogisticRegression(max_iter=500)
model_DT = DecisionTreeClassifier(random_state=42)
model_NB = GaussianNB()
model_RFC = RandomForestClassifier(n_estimators=200, random_state=42)

model_LR.fit(X_train, Y_train)
model_DT.fit(X_train, Y_train)
model_NB.fit(X_train, Y_train)
model_RFC.fit(X_train, Y_train)


# Calculating accuracies
acc_LR = accuracy_score(Y_test, model_LR.predict(X_test))
acc_DT = accuracy_score(Y_test, model_DT.predict(X_test))
acc_NB = accuracy_score(Y_test, model_NB.predict(X_test))
acc_RFC = accuracy_score(Y_test, model_RFC.predict(X_test))

# Determine best model
accuracy_dict = {
    "Logistic Regression": acc_LR,
    "Decision Tree": acc_DT,
    "Naive Bayes": acc_NB,
    "Random Forest": acc_RFC
}
best_model_name = max(accuracy_dict, key=accuracy_dict.get)
model_dict = {
    "Logistic Regression": model_LR,
    "Decision Tree": model_DT,
    "Naive Bayes": model_NB,
    "Random Forest": model_RFC
}
best_model = model_dict[best_model_name]

# ----------------- Streamlit App -----------------
st.set_page_config(page_title="Depression Assessment", layout="centered")
st.title("Student Depression Risk Assessment")

st.write("Welcome! This tool helps identify potential signs of depression based on lifestyle and mental health indicators.")

# Patient Info
name = st.text_input("What should we call you?", value="")

gender = st.selectbox("Select your gender", options=['Male', 'Female', 'Other'], index=None)

age = st.number_input("How old are you?", min_value=16, max_value=100, value=18)

sleep = st.selectbox("How much do you sleep?", ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'], index=None)

diet = st.selectbox("How would you describe your diet?", ['Healthy', 'Moderate', 'Unhealthy'], index=None)

finance = st.slider("How stressed do you feel about finances?", min_value=1, max_value=5, value=3)

fam_his = st.selectbox("Any family history of mental illness?", ['Yes', 'No'], index=None)

study_hrs = st.slider("How many hours do you study daily?", min_value=0, max_value=24, value=4)

acad_press = st.slider("Academic pressure level", min_value=0, max_value=5, value=2)

study_satisfy = st.slider("Study satisfaction level", min_value=1, max_value=5, value=3)

cgpa = st.number_input("Your current CGPA (on a 10-point scale)", min_value=5.0, max_value=10.0, value=7.5)

suicide = st.selectbox("Have you ever had suicidal thoughts?", ['Yes', 'No'], index=None)
st.write(print(model_LR.coef_))
# Check if all fields are filled
if None in (sleep, diet, gender, fam_his, suicide) or name.strip() == "":
    st.warning("Please fill in all the fields above.")
else:
    # Mapping inputs
    gender_val = gender_map[gender]
    sleep_val = sleep_map[sleep]
    diet_val = diatery_map[diet]
    fam_his_val = {'Yes': 1, 'No': 0}[fam_his]
    suicide_val = {'Yes': 1, 'No': 0}[suicide]

    # Button to Analyze
    if st.button("Analyze"):
        input_features = [[
            age, acad_press, cgpa, study_satisfy, study_hrs,
            finance, sleep_val, suicide_val, fam_his_val, gender_val, diet_val
        ]]
        prediction = best_model.predict(input_features)[0]

        st.subheader("Model Accuracies on Dataset:")
        st.write(f" Logistic Regression: **{acc_LR*100:.2f}%**")
        st.write(f" Decision Tree: **{acc_DT*100:.2f}%**")
        st.write(f" Naive Bayes: **{acc_NB*100:.2f}%**")
        st.write(f" Random Forest: **{acc_RFC*100:.2f}%**")

        st.info(f"Using the best-performing model: **{best_model_name}**")

        if prediction == 1:
            st.error(f"{name}, you may need to consult a doctor. Please take care of your mental health.")
        else:
            st.success(f"{name}, you appear to be doing okay. Keep taking care of yourself.")
