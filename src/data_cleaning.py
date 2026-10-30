import pandas as pd
import os

def clean_data(df):
    """
    Cleans the raw retail dataset by handling missing values, duplicates,
    and separating valid sales from cancelled transactions.
    """
    print("Starting data cleaning process...")
    
    # 1. Drop completely duplicate rows
    initial_rows = df.shape[0]
    df_cleaned = df.drop_duplicates().copy()
    print(f"Dropped {initial_rows - df_cleaned.shape[0]} duplicate rows.")
    
    # 2. Handle missing CustomerIDs
    # We will not drop them. Instead, we assign a placeholder 'Unknown' (or -1) 
    # so we don't lose the revenue data for unassigned customers.
    missing_customers = df_cleaned['CustomerID'].isnull().sum()
    df_cleaned['CustomerID'] = df_cleaned['CustomerID'].fillna(-1).astype(int)
    print(f"Handled {missing_customers} missing CustomerIDs by assigning ID -1.")
    
    # 3. Handle Missing Descriptions
    df_cleaned['Description'] = df_cleaned['Description'].fillna("UNKNOWN ITEM")
    
    # 4. Identify Cancellations (InvoiceNo starts with 'C' or Quantity < 0)
    # We create a new column to flag cancellations instead of deleting them.
    df_cleaned['IsCancelled'] = df_cleaned['InvoiceNo'].astype(str).str.startswith('C')
    
    # 5. Calculate Gross Sales Amount
    df_cleaned['SalesAmount'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']
    
    print("\n✅ Data cleaning complete!")
    print("-" * 40)
    print(f"Final Cleaned Rows: {df_cleaned.shape[0]}")
    print(f"Total Cancelled Transactions Flagged: {df_cleaned['IsCancelled'].sum()}")
    print("-" * 40)
    
    return df_cleaned

if __name__ == "__main__":
    from data_ingestion import load_data
    
    DATA_PATH = os.path.join("data", "Online Retail.xlsx")
    raw_data = load_data(DATA_PATH)
    
    if raw_data is not None:
        cleaned_data = clean_data(raw_data)
        
        # Save the cleaned dataset to a CSV file for the Data Warehouse module
        output_path = os.path.join("data", "cleaned_retail_data.csv")
        cleaned_data.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to {output_path}")