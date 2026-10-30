import pandas as pd
import sqlite3
import os

def setup_data_warehouse():
    """Reads cleaned CSV and builds a Star Schema SQLite Data Warehouse."""
    csv_path = os.path.join("data", "cleaned_retail_data.csv")
    db_path = os.path.join("sql", "retail_warehouse.db")
    
    print("Reading cleaned data...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("❌ Error: Cleaned CSV not found. Run data_cleaning.py first.")
        return
    
    # Convert InvoiceDate to datetime format for proper date extraction
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    print("Connecting to local SQLite database...")
    # This will create the database file if it does not exist
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create dim_customer (Dimension Table)
    print("Building dim_customer...")
    dim_customer = df[['CustomerID', 'Country']].drop_duplicates().dropna(subset=['CustomerID'])
    dim_customer.to_sql('dim_customer', conn, if_exists='replace', index=False)
    
    # 2. Create dim_product (Dimension Table)
    print("Building dim_product...")
    dim_product = df[['StockCode', 'Description']].drop_duplicates()
    # Keep only the most recent description for a StockCode if multiple exist
    dim_product = dim_product.drop_duplicates(subset=['StockCode'], keep='last')
    dim_product.to_sql('dim_product', conn, if_exists='replace', index=False)
    
    # 3. Create dim_date (Dimension Table)
    print("Building dim_date...")
    dim_date = pd.DataFrame({'InvoiceDate': df['InvoiceDate'].unique()})
    # Create a unique DateKey string (YYYYMMDDHHMM) for joining with fact table
    dim_date['DateKey'] = dim_date['InvoiceDate'].dt.strftime('%Y%m%d%H%M')
    dim_date['Year'] = dim_date['InvoiceDate'].dt.year
    dim_date['Month'] = dim_date['InvoiceDate'].dt.month
    dim_date['Day'] = dim_date['InvoiceDate'].dt.day
    dim_date['Hour'] = dim_date['InvoiceDate'].dt.hour
    dim_date.to_sql('dim_date', conn, if_exists='replace', index=False)
    
    # 4. Create fact_sales (Fact Table)
    print("Building fact_sales...")
    df['DateKey'] = df['InvoiceDate'].dt.strftime('%Y%m%d%H%M')
    fact_sales = df[['InvoiceNo', 'StockCode', 'CustomerID', 'DateKey', 'Quantity', 'UnitPrice', 'SalesAmount', 'IsCancelled']]
    fact_sales.to_sql('fact_sales', conn, if_exists='replace', index=False)
    
    print("\n✅ Data Warehouse (Star Schema) built successfully!")
    print(f"Database saved at: {db_path}")
    
    # Verify the insertion
    cursor.execute("SELECT COUNT(*) FROM fact_sales")
    total_records = cursor.fetchone()[0]
    print(f"Total records in fact_sales: {total_records}")
    
    conn.close()

if __name__ == "__main__":
    setup_data_warehouse()