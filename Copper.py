import os
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# ---------------------- CONFIG ----------------------
st.set_page_config(
    page_title="Industrial Copper Modeling",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath("S:\VS CODE\models"))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------------------- MODEL LOADERS ----------------------
@st.cache_resource
def load_classification_pipeline():
    with open(os.path.join(MODEL_DIR, "Classification_model.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_regression_pipeline():
    with open(os.path.join(MODEL_DIR, "Regression_Model.pkl"), "rb") as f:
        return pickle.load(f)

clf_model = load_classification_pipeline()
reg_model = load_regression_pipeline()

# ---------------------- UTILS ----------------------
def extract_date_features(date):
    return date.day, date.month, date.year

# ---------------------- UI ----------------------
st.title("🔩 INDUSTRIAL COPPER MODELING")

with st.sidebar:
    option = option_menu(
        "Venkatesan M",
        ["Predict Status", "Predict Selling Price"],
        icons=["check-circle", "currency-dollar"]
    )

# ==========================================================
#                  PREDICT STATUS
# ==========================================================
if option == "Predict Status":

    st.subheader("Predict Deal Status (Won / Lost)")

    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country", [25, 26, 27, 28, 30, 32, 38, 39, 40, 77, 78, 79, 80, 84, 89, 107, 113])
        item_type = st.selectbox("Item Type", list(range(0, 7)))
        application = st.number_input("Application", min_value=1.0)
        width = st.number_input("Width", min_value=1.0)
        product_ref = st.number_input("Product Ref", min_value=1)
        quantity_log = st.number_input("Quantity (log)", format="%.6f")
    with col2:
        customer_log = st.number_input("Customer (log)", format="%.6f")
        thickness_log = st.number_input("Thickness (log)", format="%.6f")
        selling_price_log = st.number_input("Selling Price (log)", format="%.6f")
        item_date = st.date_input("Item Date")
        delivery_date = st.date_input("Delivery Date")

    if delivery_date <= item_date:
        st.error("Delivery date must be after item date.")
        st.stop()

    if st.button("Predict Status", use_container_width=True):

        it_d, it_m, it_y = extract_date_features(item_date)
        del_d, del_m, del_y = extract_date_features(delivery_date)

        X = np.array([[country, item_type, application, width, product_ref,
                       quantity_log, customer_log, thickness_log, selling_price_log,
                       it_d, it_m, it_y, del_d, del_m, del_y]])

        try:
            pred = clf_model.predict(X)[0]
            st.success("WON ✅" if pred == 1 else "LOST ❌")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ==========================================================
#              PREDICT SELLING PRICE
# ==========================================================
if option == "Predict Selling Price":

    st.subheader("Predict Selling Price")

    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country", [25, 26, 27, 28, 30, 32, 38, 39, 40, 77, 78, 79, 80, 84, 89, 107, 113])
        status = st.selectbox("Status", [0, 1])
        item_type = st.selectbox("Item Type", list(range(0, 7)))
        application = st.number_input("Application", min_value=1.0)
        width = st.number_input("Width", min_value=1.0)
        product_ref = st.number_input("Product Ref", min_value=1)
    with col2:
        quantity_log = st.number_input("Quantity (log)", format="%.6f")
        customer_log = st.number_input("Customer (log)", format="%.6f")
        thickness_log = st.number_input("Thickness (log)", format="%.6f")
        item_date = st.date_input("Item Date")
        delivery_date = st.date_input("Delivery Date")

    if delivery_date <= item_date:
        st.error("Delivery date must be after item date.")
        st.stop()

    if st.button("Predict Selling Price", use_container_width=True):

        it_d, it_m, it_y = extract_date_features(item_date)
        del_d, del_m, del_y = extract_date_features(delivery_date)

        X = np.array([[country, status, item_type, application, width, product_ref,
                       quantity_log, customer_log, thickness_log,
                       it_d, it_m, it_y, del_d, del_m, del_y]])

        try:
            log_price = reg_model.predict(X)[0]
            price = np.exp(log_price)
            st.success(f"Predicted Selling Price: {price:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
