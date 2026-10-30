import pandas as pd
import os

def load_data(file_path):
    """Loads data from the Excel file and prints basic information."""
    print(f"Loading data from {file_path} (This might take 10-20 seconds)...")
    
    try:
        # Read the Excel file using Pandas
        df = pd.read_excel(file_path)
        
        print("\n✅ Data loaded successfully!")
        print("-" * 40)
        print(f"Total Rows (Records): {df.shape[0]}")
        print(f"Total Columns (Features): {df.shape[1]}")
        print("-" * 40)
        print("First 5 rows of the dataset:\n")
        print(df.head())
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

if __name__ == "__main__":
    # Use os module to ensure cross-platform compatibility for file paths
    DATA_PATH = os.path.join("data", "Online Retail.xlsx")
    
    # Call the function to load data
    raw_data = load_data(DATA_PATH)