import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from data_prep import load_data
from eda import perform_eda
from visualization import data_visualization, visualize_body
from modeling import data_preprocessing, model_building, predict_diabetes, predict_diabetes_custom


def main():
    # Set page title and icon
    st.set_page_config(page_title="Diabetes Prediction App", page_icon=":hospital:")
    st.markdown('<p style="text-align:center;"><img src="https://assets.newatlas.com/dims4/default/3461759/2147483647/strip/true/crop/7360x4907+0+3/resize/840x560!/quality/90/?url=http%3A%2F%2Fnewatlas-brightspot.s3.amazonaws.com%2F10%2F7f%2F5e48f79245c0b831a58d7cf8fb1d%2Fdepositphotos-228244172-xl.jpg" width="400"/></p>', unsafe_allow_html=True)

    # Load data
    data = load_data("C:\\Users\\shyam\\Downloads\\diabetes.csv")

    # Sidebar title
    st.sidebar.title("Diabetes Prediction")

    # Sidebar options
    option = st.sidebar.selectbox("Select Option", ["Exploratory Data Analysis", "Data Visualization", "Prediction"])

    # Perform actions based on selected option
    if option == "Exploratory Data Analysis":
        perform_eda(data)
    elif option == "Data Visualization":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        data_visualization(data)
    elif option == "Prediction":
        X_train, X_test, y_train, y_test, scaler = data_preprocessing(data)
        model = model_building(X_train, X_test, y_train, y_test)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Making prediction
        prediction = predict_diabetes(model, X_test)
        
        # Displaying results
        st.subheader("Prediction Results")
        st.write("Accuracy: ", accuracy_score(y_test, prediction))
        st.write("Classification Report:\n")
        st.code(classification_report(y_test, prediction))

        # Simulate diabetes risk factors for visualization
        diabetes_risk_factors = ['Obesity', 'High Blood Pressure', 'High Glucose', 'High Cholesterol']
        st.subheader("Body Visualization for Diabetes Risk Factors")
        st.write("Visualizing diabetes risk factors on a body diagram.")
        body_fig = visualize_body(diabetes_risk_factors)
        st.pyplot(body_fig)

        # Prediction with custom features
        st.subheader("Predict Diabetes with Custom Features")
        feature_values = {}
        for feature in data.columns:
            if feature != 'Outcome':
                feature_values[feature] = st.number_input(f"Enter {feature}", value=data[feature].mean())
        
        if st.button("Predict"):
            prediction_custom = predict_diabetes_custom(model, scaler, list(feature_values.values()))
            st.write("Prediction:", "Diabetic" if prediction_custom[0] == 1 else "Non-Diabetic")

if __name__ == "__main__":
    main()

