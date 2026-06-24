import streamlit as st
import pandas as pd
import joblib

# ==========================
# LOAD MODELS
# ==========================

product_similarity_df = joblib.load(
    "product_similarity.pkl"
)

kmeans = joblib.load(
    "kmeans_customer_segmentation.pkl"
)

scaler = joblib.load(
    "rfm_scaler.pkl"
)

segment_map = joblib.load(
    "segment_mapping.pkl"
)

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Customer Segmentation & Product Recommendation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Customer Segmentation & Product Recommendation System")

# ==========================
# PRODUCT RECOMMENDATION
# ==========================

st.header("🎯 Product Recommendation Module")

product_name = st.text_input(
    "Enter Product Name"
)

def recommend_products(product_name, top_n=5):

    product_name = product_name.upper()

    products_upper = {
        p.upper(): p
        for p in product_similarity_df.index
    }

    if product_name not in products_upper:
        return None

    actual_product = products_upper[product_name]

    recommendations = (
        product_similarity_df[actual_product]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )

    return recommendations.index.tolist()

if st.button("Get Recommendations"):

    recommendations = recommend_products(
        product_name
    )

    if recommendations is None:

        st.error(
            "Product not found."
        )

    else:

        st.success(
            "Top 5 Recommended Products"
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        cols = [
            col1,
            col2,
            col3,
            col4,
            col5
        ]

        for col, product in zip(
            cols,
            recommendations
        ):

            with col:

                st.markdown(
                    f"""
                    <div style="
                    padding:15px;
                    border-radius:10px;
                    border:1px solid #ddd;
                    text-align:center;
                    min-height:120px;">
                    <h5>{product}</h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.divider()

# ==========================
# CUSTOMER SEGMENTATION
# ==========================

st.header("🎯 Customer Segmentation Module")

col1, col2, col3 = st.columns(3)

with col1:

    recency = st.number_input(
        "Recency (days)",
        min_value=0.0,
        value=30.0
    )

with col2:

    frequency = st.number_input(
        "Frequency",
        min_value=0.0,
        value=5.0
    )

with col3:

    monetary = st.number_input(
        "Monetary",
        min_value=0.0,
        value=1000.0
    )

if st.button("Predict Cluster"):

    customer = pd.DataFrame(
        [[
            recency,
            frequency,
            monetary
        ]],
        columns=[
            "Recency",
            "Frequency",
            "Monetary"
        ]
    )

    customer_scaled = scaler.transform(
        customer
    )

    cluster = kmeans.predict(
        customer_scaled
    )[0]

    segment = segment_map[
        cluster
    ]

    st.success(
        f"Customer Segment: {segment}"
    )

    if segment == "High-Value":

        st.balloons()

        st.markdown(
            """
            ### 🌟 High-Value Customer
            Recent, frequent buyer with high spending.
            """
        )

    elif segment == "Regular":

        st.markdown(
            """
            ### 👍 Regular Customer
            Consistent purchasing behaviour.
            """
        )

    elif segment == "Occasional":

        st.markdown(
            """
            ### 🛒 Occasional Customer
            Purchases infrequently.
            """
        )

    elif segment == "At-Risk":

        st.markdown(
            """
            ### ⚠️ At-Risk Customer
            Long time since last purchase.
            """
        )