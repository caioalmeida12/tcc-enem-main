import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io


# Define COLUNAS_NOTAS if it's not already defined globally in your utils.py
COLUNAS_NOTAS = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]


def init_session_state(cat_features, num_features):
    """Initializes session state variables for feature selections."""
    if "selected_categorical_features_selected" not in st.session_state:
        st.session_state.selected_categorical_features_selected = []
    if "selected_numerical_features_selected" not in st.session_state:
        st.session_state.selected_numerical_features_selected = []
