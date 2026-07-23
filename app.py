import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered"
)

# 2. App Headers
st.title("🌸 Iris Flower Species Predictor")
st.write("Adjust the features in the sidebar to predict the species of the flower in real-time.")

# 3. Load & Cache Dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# 4. Sidebar Input Sliders
st.sidebar.header("Input Flower Features")

def get_user_inputs():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df.iloc[:, 0].min()), float(df.iloc[:, 0].max()), float(df.iloc[:, 0].mean()))
    sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df.iloc[:, 1].min()), float(df.iloc[:, 1].max()), float(df.iloc[:, 1].mean()))
    petal_length = st.sidebar.slider("Petal Length (cm)", float(df.iloc[:, 2].min()), float(df.iloc[:, 2].max()), float(df.iloc[:, 2].mean()))
    petal_width = st.sidebar.slider("Petal Width (cm)", float(df.iloc[:, 3].min()), float(df.iloc[:, 3].max()), float(df.iloc[:, 3].mean()))
    
    # Store data in a dictionary and format as a DataFrame
    input_data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame([input_data])

user_df = get_user_inputs()

# 5. Display Selected Parameters
st.subheader("Selected Features")
st.write(user_df)

# 6. Train the Machine Learning Model
X = df.iloc[:, :-1]
y = df['species']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 7. Model Prediction and Probabilities
prediction = model.predict(user_df)[0]
prediction_proba = model.predict_proba(user_df)[0]

# 8. Display Results
st.subheader("Prediction")
predicted_species = target_names[prediction]
st.success(f"The predicted species is **{predicted_species.capitalize()}**")

st.subheader("Prediction Probability Distribution")
proba_df = pd.DataFrame(prediction_proba, index=target_names, columns=["Probability"])
st.bar_chart(proba_df)