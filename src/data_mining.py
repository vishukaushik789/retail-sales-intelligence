import sqlite3
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def perform_customer_segmentation():
    """Calculates RFM metrics and applies K-Means clustering for customer segmentation."""
    print("Starting Customer Segmentation (RFM + K-Means)...")
    
    db_path = os.path.join("sql", "retail_warehouse.db")
    conn = sqlite3.connect(db_path)
    
    # 1. Fetch data required for RFM
    # We exclude CustomerID = -1 (Unknown customers) and Cancelled orders
    query = """
    SELECT 
        CustomerID,
        MAX(DateKey) as LastPurchaseDate,
        COUNT(DISTINCT InvoiceNo) as Frequency,
        SUM(SalesAmount) as Monetary
    FROM fact_sales
    WHERE CustomerID != -1 AND IsCancelled = 0
    GROUP BY CustomerID
    """
    df_rfm = pd.read_sql_query(query, conn)
    
    # 2. Calculate Recency
    # Convert string DateKey (YYYYMMDDHHMM) back to datetime to calculate days
    df_rfm['LastPurchaseDate'] = pd.to_datetime(df_rfm['LastPurchaseDate'], format='%Y%m%d%H%M')
    current_date = df_rfm['LastPurchaseDate'].max()
    df_rfm['Recency'] = (current_date - df_rfm['LastPurchaseDate']).dt.days
    
    # Keep only R, F, M columns for clustering
    rfm_features = df_rfm[['Recency', 'Frequency', 'Monetary']]
    
    # 3. Data Scaling (Crucial for K-Means)
    print("Scaling data for machine learning...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    
    # 4. Apply K-Means Clustering (k=4 segments)
    print("Applying K-Means Clustering to create 4 customer segments...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 5. Assign Business Labels to Clusters
    # (Simplified labeling logic based on general RFM patterns)
    def assign_label(cluster):
        if cluster == 0: return "Loyal Customers"
        elif cluster == 1: return "At Risk"
        elif cluster == 2: return "Champions"
        else: return "Recent/Low-Spenders"
        
    df_rfm['SegmentLabel'] = df_rfm['Cluster'].apply(assign_label)
    
    # 6. Save segments back to the Data Warehouse
    print("Saving customer segments to the database...")
    final_segments = df_rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'SegmentLabel']]
    final_segments.to_sql('customer_segments', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print("\n✅ Customer Segmentation completed successfully!")
    print("-" * 50)
    print(final_segments['SegmentLabel'].value_counts())
    print("-" * 50)
    print(final_segments.head())

if __name__ == "__main__":
    perform_customer_segmentation()