# Shopper Spectrum: Customer Segmentation and Product Recommendation System

## Overview

Shopper Spectrum is a machine learning-based e-commerce analytics project that combines customer segmentation and product recommendation techniques to improve customer engagement and business decision-making. The project analyzes customer transaction data, identifies purchasing patterns using RFM analysis, and segments customers into meaningful groups using clustering algorithms. It also provides personalized product recommendations using item-based collaborative filtering.

## Features

### Customer Segmentation

* RFM (Recency, Frequency, Monetary) Analysis
* Customer clustering using K-Means Clustering
* Segment prediction for new customers
* Customer categories:

  * High-Value
  * Regular
  * Occasional
  * At-Risk

### Product Recommendation System

* Item-Based Collaborative Filtering
* Cosine Similarity-based recommendations
* Top 5 similar product suggestions
* Product similarity matrix visualization

## Dataset

The project uses an Online Retail transaction dataset containing:

* Invoice Number
* Product Information
* Quantity Purchased
* Unit Price
* Customer ID
* Invoice Date
* Country

## Data Processing

* Removed duplicate records
* Handled missing values
* Removed cancelled transactions
* Removed invalid quantities and prices
* Created Revenue feature
* Generated RFM metrics
* Standardized features using StandardScaler

## Machine Learning Models

The following clustering algorithms were evaluated:

* K-Means Clustering
* Hierarchical Clustering
* DBSCAN

### Final Model Selected: K-Means Clustering

K-Means was selected because it provided:

* Better cluster separation
* Higher Silhouette Score
* Clear business interpretation
* Improved customer segmentation performance

## Evaluation Metrics

* Elbow Method (WCSS)
* Silhouette Score
* Cluster Profile Analysis

## Visualizations

* Transaction Volume by Country
* Revenue Distribution by Country
* Top Selling Products
* Monthly Purchase Trends
* Transaction Amount Distribution
* Customer Spending Distribution
* RFM Distributions
* Elbow Curve
* Customer Cluster Profiles
* Product Similarity Heatmap

## Business Impact

* Enables targeted marketing campaigns
* Improves customer retention strategies
* Identifies high-value customers
* Supports inventory optimization
* Enhances product recommendation quality
* Helps improve customer lifetime value

## Streamlit Application

### Product Recommendation Module

Input a product name to receive:

* Top 5 similar product recommendations

### Customer Segmentation Module

Input:

* Recency
* Frequency
* Monetary

Output:

* Predicted customer segment

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Plotly
* Streamlit
* Joblib

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Project Files

* `app.py` — Streamlit application
* `kmeans_customer_segmentation.pkl` — Trained K-Means model
* `rfm_scaler.pkl` — Feature scaler
* `segment_mapping.pkl` — Cluster label mapping

## Author

Akhila Rao
