"""
Diamond Price Prediction – Streamlit web app
Serves the model trained in diamond_price_kfold_full_dataset_10fold.ipynb.

Run locally : streamlit run app.py
Deploy      : push this folder to GitHub, then create an app on https://share.streamlit.io
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).parent / "diamond_price_model.pkl"

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")


@st.cache_resource
def load_bundle():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


bundle = load_bundle()
model = bundle["model"]

st.title("💎 Diamond Price Predictor")
st.write(
    f"Enter the characteristics of a diamond to estimate its price. "
    f"Model: **{bundle['algorithm']}**, selected using 10-fold cross validation on "
    f"53,000+ diamonds (typical error ≈ **\\${bundle['cv_rmse']:,.0f}** RMSE)."
)

with st.form("diamond_form"):
    st.subheader("Quality (the 4 Cs)")
    carat = st.number_input("Carat (weight)", min_value=0.2, max_value=5.0, value=1.0, step=0.01)
    col1, col2, col3 = st.columns(3)
    cut = col1.selectbox("Cut", bundle["cut_order"], index=len(bundle["cut_order"]) - 1)
    color = col2.selectbox("Color (J worst → D best)", bundle["color_order"], index=3)
    clarity = col3.selectbox("Clarity (I1 worst → IF best)", bundle["clarity_order"], index=4)

    st.subheader("Proportions and size")
    col4, col5 = st.columns(2)
    depth = col4.number_input("Depth %", min_value=40.0, max_value=80.0, value=61.8, step=0.1)
    table = col5.number_input("Table %", min_value=40.0, max_value=95.0, value=57.0, step=0.5)
    col6, col7, col8 = st.columns(3)
    x = col6.number_input("Length x (mm)", min_value=0.1, max_value=11.0, value=6.40, step=0.01)
    y = col7.number_input("Width y (mm)", min_value=0.1, max_value=11.0, value=6.43, step=0.01)
    z = col8.number_input("Depth z (mm)", min_value=0.1, max_value=7.0, value=3.97, step=0.01)

    submitted = st.form_submit_button("Predict Price", type="primary")

if submitted:
    diamond = pd.DataFrame([{
        "carat": carat, "depth": depth, "table": table, "x": x, "y": y, "z": z,
        "cut": cut, "color": color, "clarity": clarity,
    }])[bundle["numeric_features"] + bundle["categorical_features"]]

    price = float(model.predict(diamond)[0])
    # "\$" stops Streamlit's markdown from reading a pair of dollar signs as LaTeX math
    st.success(f"### Estimated price: \\${price:,.0f}")
    low, high = max(price - bundle["cv_rmse"], 0), price + bundle["cv_rmse"]
    st.caption(f"Likely range (± one RMSE): \\${low:,.0f} – \\${high:,.0f}")
    with st.expander("Input sent to the model"):
        st.dataframe(diamond, hide_index=True)

st.divider()
st.caption("Machine Learning Fundamentals (AIC354) – Assignment 3 · Dataset: Diamonds (53,940 records)")
