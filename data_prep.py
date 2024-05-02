import streamlit as st
import pandas as pd

# Load the diabetes dataset
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)
