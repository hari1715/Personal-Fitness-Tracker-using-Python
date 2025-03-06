import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

import warnings
warnings.filterwarnings('ignore')
st.markdown("""
    <style>
        /* Background color */
        .stApp {
            background-color: #1E1E1E;
        }

        /* Sidebar styling */
        .stSidebar {
            background-color: #292929 !important;
            color: white;
        }

        /* Text color */
        h1, h2, h3, h4, h5, h6, p, div {
            color: #E0E0E0;
        }

        /* Buttons */
        .stButton > button {
            background-color: #FF5733;
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: #FF5733;'>🏋️‍♂️ Personal Fitness Tracker</h1>", unsafe_allow_html=True)

st.write("### 🔥 Track Your Fitness Progress Effortlessly!")
st.info("Enter your details in the sidebar to get an accurate prediction of your **calories burned** during exercise.", icon="ℹ️")

st.sidebar.markdown("<h2 style='color: #FF5733;'>⚙️ User Information</h2>", unsafe_allow_html=True)

col1, col2 = st.sidebar.columns(2)
age = col1.slider("Age: ", 10, 100, 30)
bmi = col2.slider("BMI: ", 15, 40, 20)

col1, col2 = st.sidebar.columns(2)
duration = col1.slider("Duration (min): ", 0, 35, 15)
heart_rate = col2.slider("Heart Rate: ", 60, 130, 80)

body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

gender = 1 if gender_button == "Male" else 0



def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30, key="slider_age")
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20, key="slider_bmi")
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15, key="slider_duration")
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80, key="slider_heart_rate")
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38, key="slider_body_temp")

    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"), key="gender_radio")


    gender = 1 if gender_button == "Male" else 0

    # BMI Calculation (Indented Correctly)
    bmi_value = round(bmi, 2)
    if bmi_value < 18.5:
        bmi_category = "🔵 Underweight"
    elif 18.5 <= bmi_value <= 24.9:
        bmi_category = "🟢 Healthy Weight"
    elif 25 <= bmi_value <= 29.9:
        bmi_category = "🟠 Overweight"
    else:
        bmi_category = "🔴 Obese"

    st.sidebar.markdown(f"### 📊 Your BMI: **{bmi_value}** ({bmi_category})")

    # Use column names to match the training data
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Gender is encoded as 1 for male, 0 for female
    }

    features = pd.DataFrame([data_model])  # Fixed DataFrame Creation
    return features

# Call function and store user input in df
df = user_input_features()


st.write("---")
st.header("Your Parameters: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Add BMI column to both training and test sets
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

# Prepare the training and testing sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.markdown("""
    <div style='background-color: #333; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2 style='color: #FF5733;'>🔥 Calories Burned:</h2>
        <h1 style='color: white;'>"""+str(round(prediction[0], 2))+""" kcal</h1>
    </div>
""", unsafe_allow_html=True)

st.write("---")
st.subheader("🏃 Exercise Recommendation")

st.write("---")
st.header("🏃 Suggested Activity to Reach Your Goal")

activity_options = {
    "Running": 10,  # Burns 10 kcal per minute
    "Cycling": 8,
    "Jump Rope": 12,
    "Yoga": 4,
}

if 'goal' not in locals():
    goal = 500  # Set a default goal value to prevent errors

if prediction[0] < goal:
    deficit = goal - prediction[0]
    st.write(f"To reach your goal, consider doing one of these activities:")
    for activity, burn_rate in activity_options.items():
        time_needed = deficit / burn_rate
        st.write(f"- {activity}: **{time_needed:.1f} minutes**")
else:
    st.success("🔥 You've met your goal! Keep up the great work!")


if prediction[0] < 150:
    st.warning("⚡ Your calories burned is **low**. Try increasing your **exercise duration** or **intensity**!")
elif prediction[0] > 400:
    st.success("🔥 Great job! You're burning a high number of calories. Keep up the good work!")
else:
    st.info("💪 Your calorie burn is within a normal range. Maintain a balanced workout routine.")

st.write("---")




st.write("---")
st.header("Similar Results: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

# Find similar results based on predicted calories
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

st.write("---")
st.header("📊 Weekly Progress Chart")

import plotly.express as px

# Simulated weekly data (Replace with real data if available)
weekly_data = pd.DataFrame({
    "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "Calories": np.random.randint(200, 600, size=7)
})

fig = px.line(weekly_data, x="Day", y="Calories", markers=True, title="Calories Burned This Week")
st.plotly_chart(fig)


st.write("---")
st.subheader("📈 Calories Burned: You vs. Average")

# Compare user's predicted calories with dataset average
average_calories = exercise_df["Calories"].mean()

fig, ax = plt.subplots(figsize=(6, 4))
bars = sn.barplot(x=["You", "Dataset Average"], y=[prediction[0], average_calories], palette=["#FF5733", "#1E90FF"], ax=ax)

# Add value labels on top of bars
for bar in bars.patches:
    ax.annotate(f"{bar.get_height():.2f}", 
                (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                ha='center', va='bottom', fontsize=12, color="white", fontweight="bold")

ax.set_ylabel("Calories Burned", fontsize=12, color="white")
ax.set_title("🔥 Calories Burned: You vs. Dataset Average", fontsize=14, fontweight='bold', color="white")

# Customize chart appearance
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("white")
ax.spines["bottom"].set_color("white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
ax.tick_params(colors="white")

st.pyplot(fig)


st.write("---")
st.header("General Information: ")

# Boolean logic for age, duration, etc., compared to the user's input
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.markdown("## 📊 **General Insights About Your Fitness**")

col1, col2 = st.columns(2)
col1.metric("👴 Age Comparison", f"{round(sum(boolean_age) / len(boolean_age), 2) * 100}% older")
col2.metric("🏋️ Duration Comparison", f"{round(sum(boolean_duration) / len(boolean_duration), 2) * 100}% longer")

col1.metric("❤️ Heart Rate Comparison", f"{round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}% higher")
col2.metric("🌡️ Body Temperature", f"{round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}% higher")
