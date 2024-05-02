import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Perform exploratory data analysis
def perform_eda(data):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Exploratory Data Analysis")
    st.write("### Data Overview")
    st.write(data.head())

    st.write("### Data Statistics")
    st.write(data.describe())

    st.write("### Data Distribution by Outcome")
    st.write(data['Outcome'].value_counts())

    st.write("### Correlation Heatmap")
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot()
