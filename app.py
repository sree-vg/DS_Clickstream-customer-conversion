# ======================================================
# Customer Conversion Analysis – Streamlit Application
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Customer Conversion Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Customer Conversion Analysis for Online Shopping")
st.markdown(
    "Predict **customer conversion**, **revenue**, and **customer segments** "
    "using real-world **clickstream data**."
)

# ======================================================
# FEATURE MAPS (MATCH TRAINING)
# ======================================================
country_map = {
    1: 'Australia', 2: 'Austria', 3: 'Belgium', 4: 'British Virgin Islands',
    5: 'Cayman Islands', 6: 'Christmas Island', 7: 'Croatia', 8: 'Cyprus',
    9: 'Czech Republic', 10: 'Denmark', 11: 'Estonia', 12: 'Unidentified',
    13: 'Faroe Islands', 14: 'Finland', 15: 'France', 16: 'Germany',
    17: 'Greece', 18: 'Hungary', 19: 'Iceland', 20: 'India',
    21: 'Ireland', 22: 'Italy', 23: 'Latvia', 24: 'Lithuania',
    25: 'Luxembourg', 26: 'Mexico', 27: 'Netherlands', 28: 'Norway',
    29: 'Poland', 30: 'Portugal', 31: 'Romania', 32: 'Russia',
    33: 'San Marino', 34: 'Slovakia', 35: 'Slovenia', 36: 'Spain',
    37: 'Sweden', 38: 'Switzerland', 39: 'Ukraine',
    40: 'United Arab Emirates', 41: 'United Kingdom', 42: 'USA',
    43: 'biz (.biz)', 44: 'com (.com)', 45: 'int (.int)',
    46: 'net (.net)', 47: 'org (*.org)'
}

product_map = {
    1: 'Trousers',
    2: 'Skirts',
    3: 'Blouses',
    4: 'Sale'
}

# ======================================================
# LOAD TRAINED ARTIFACTS
# ======================================================
@st.cache_resource
def load_models():
    clf_pipe = joblib.load("artifacts/classification_pipeline.pkl")
    reg_pipe = joblib.load("artifacts/regression_pipeline.pkl")
    kmeans = joblib.load("artifacts/kmeans.pkl")
    pca = joblib.load("artifacts/pca.pkl")
    cluster_preprocessor = joblib.load("artifacts/cluster_preprocessor.pkl")
    return clf_pipe, reg_pipe, kmeans, pca, cluster_preprocessor

clf_pipe, reg_pipe, kmeans, pca, cluster_preprocessor = load_models()

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Customer Conversion Prediction",
        "Revenue Estimation",
        "Customer Segmentation",
        "Project Insights"
    ]
)

# ======================================================
# COMMON INPUT FORM
# ======================================================
def user_input_form():
    with st.form("user_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            year = st.number_input("Year", 2008, 2025, 2008)
            month = st.number_input("Month", 1, 12, 6)
            day = st.number_input("Day", 1, 31, 15)
            order = st.number_input("Click Order", 1, 100, 5)

        with col2:
            country = st.selectbox("Country Code", list(range(1, 48)))
            page1_main_category = st.selectbox("Main Category", [1, 2, 3, 4])
            colour = st.selectbox("Colour Code", list(range(1, 15)))
            location = st.selectbox("Photo Location", list(range(1, 7)))

        with col3:
            page2_clothing_model = st.text_input("Clothing Model Code", "P1")
            model_photography = st.selectbox("Model Photography", [1, 2])
            price_2 = st.selectbox("Price Above Category Avg", [1, 2])
            page_depth = st.slider("Page Depth", 1, 5, 3)

        submitted = st.form_submit_button("Predict")

    data = pd.DataFrame([{
        "year": year,
        "month": month,
        "day": day,
        "order": order,
        "country": country,
        "page1_main_category": page1_main_category,
        "page2_clothing_model": page2_clothing_model,
        "colour": colour,
        "location": location,
        "model_photography": model_photography,
        "price_2": price_2,
        "page": page_depth
    }])

    # Derived features (used during training)
    data["country_name"] = data["country"].map(country_map)
    data["main_product"] = data["page1_main_category"].map(product_map)
    data["session_id"] = 0

    return submitted, data

# ======================================================
# PAGE 1 — CLASSIFICATION
# ======================================================
if page == "Customer Conversion Prediction":
    st.header("📈 Customer Conversion Prediction")

    submitted, input_df = user_input_form()

    if submitted:
        st.subheader("Input Summary")
        st.dataframe(input_df)

        proba = clf_pipe.predict_proba(input_df)[0][1]
        pred = clf_pipe.predict(input_df)[0]

        st.subheader("Prediction Result")

        # Optional UX polish for probability display
        if proba < 0.01:
            st.metric("Conversion Probability", "<1%")
        else:
            st.metric("Conversion Probability", f"{proba:.2%}")

        if pred == 1:
            st.success("✅ Customer is likely to convert")
        else:
            st.warning("❌ Customer is unlikely to convert")

        st.info(
            "💡 **Insight:** Higher page depth, higher click order, and products priced "
            "above category average significantly increase conversion probability."
        )

# ======================================================
# PAGE 2 — REGRESSION
# ======================================================
elif page == "Revenue Estimation":
    st.header("💰 Revenue Estimation")


    submitted, input_df = user_input_form()

    if submitted:
        st.subheader("Input Summary")
        st.dataframe(input_df)

        price_pred = reg_pipe.predict(input_df)[0]
        st.metric("Estimated Revenue ($)", f"${price_pred:.2f}")

        st.info(
            "💡 **Business Use:** Revenue prediction helps forecast demand, "
            "optimize pricing strategies, and personalize offers."
        )

# ======================================================
# PAGE 3 — CLUSTERING
# ======================================================
elif page == "Customer Segmentation":
    st.header("👥 Customer Segmentation")

    # Cluster interpretation (derived from training analysis)
    cluster_profiles = {
        0: {
            "name": "High Engagement Cluster",
            "color": "🟢",
            "insights": [
                "More page views",
                "Higher click order",
                "Higher conversion probability",
                "Ideal for premium or upsell campaigns"
            ],
            "business_action": "Focus on premium products, bundles, and personalized recommendations."
        },
        1: {
            "name": "Low Engagement Cluster",
            "color": "🔵",
            "insights": [
                "Fewer interactions",
                "Lower page depth",
                "Lower conversion probability",
                "Suitable for discounts or retargeting"
            ],
            "business_action": "Use discounts, reminders, and retargeting campaigns to improve engagement."
        }
    }

    submitted, input_df = user_input_form()

    if submitted:
        st.subheader("Input Summary")
        st.dataframe(input_df)

        # Preprocess + reduce dimensions + predict cluster
        cluster_data = cluster_preprocessor.transform(input_df)
        cluster_reduced = pca.transform(cluster_data)
        cluster_label = int(kmeans.predict(cluster_reduced)[0])

        profile = cluster_profiles.get(cluster_label)

        st.subheader("Customer Segment Result")
        st.metric("Cluster Assigned", cluster_label)

        if profile:
            st.subheader(f"{profile['color']} {profile['name']}")

            for insight in profile["insights"]:
                st.markdown(f"- {insight}")

            st.info(f"💼 **Recommended Action:** {profile['business_action']}")
        else:
            st.warning("⚠️ Unable to determine customer segment.")




# ======================================================
# PAGE 4 — PROJECT INSIGHTS
# ======================================================
elif page == "Project Insights":
    st.header("📊 Project Insights")

    st.markdown("""
### 🔍 Key Findings
- Page depth and click order strongly influence conversion
- Pricing above category average impacts buying behavior
- Two distinct behavioral customer segments exist
- High engagement users convert at significantly higher rates
    """)

    st.markdown("""
### 🧪 Model Tracking & Experiments (MLflow)
- **Classification Pipeline**  
  🔗 https://dagshub.com/tstr12cg429/Clickstream_classification.mlflow/#/experiments/0/runs/5d0c165dc6294600938b1b8903993754

- **Regression Pipeline**  
  🔗 https://dagshub.com/tstr12cg429/Clickstream_classification.mlflow/#/experiments/0/runs/0e3ff728e9e3430b87b6d6ee8899bfe2

- **Clustering Pipeline**  
  🔗 https://dagshub.com/tstr12cg429/Clickstream_classification.mlflow/#/experiments/0/runs/25547795b3ba431c80396df4b9b4bf65
    """)

    st.markdown("---")
    st.markdown("""
### 📊 Why customers fall into different clusters

Customers are segmented based on:
- Click order
- Page depth
- Pricing behavior
- Visual engagement (model photography)

These behavioral signals help businesses design:
- Upsell strategies for high engagement users
- Discount or retargeting campaigns for low engagement users
""")


    st.markdown("""
### 🛠 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn Pipelines
- MLflow + DAGsHub
- Streamlit for Deployment
    """)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("🚀 Customer Conversion Analysis – Data Science Capstone Project")
