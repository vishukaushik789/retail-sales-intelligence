import sqlite3
import pandas as pd
import os
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def perform_market_basket_analysis():
    """Applies the Apriori algorithm to find frequently bought together products."""
    print("Starting Market-Basket Analysis (Apriori)...")
    
    db_path = os.path.join("sql", "retail_warehouse.db")
    conn = sqlite3.connect(db_path)
    
    # We filter data for 'France' to prevent MemoryErrors. 
    # Apriori requires a huge amount of memory for 500k rows (entire UK dataset).
    query = """
    SELECT f.InvoiceNo, d.Description, f.Quantity
    FROM fact_sales f
    JOIN dim_product d ON f.StockCode = d.StockCode
    JOIN dim_customer c ON f.CustomerID = c.CustomerID
    WHERE c.Country = 'France' AND f.IsCancelled = 0
    """
    df = pd.read_sql_query(query, conn)
    
    print("Building the basket matrix (Invoice x Product)...")
    # Group data: Rows = InvoiceNo, Columns = Product Descriptions, Values = Quantity
    basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
    
    # Convert quantities to 1 (Bought) and 0 (Not Bought)
    basket_sets = (basket > 0).astype(int)
    
    print("Applying Apriori algorithm (Finding frequent itemsets)...")
    # min_support=0.05 means the items must appear together in at least 5% of all transactions
    frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
    
    print("Generating Association Rules (Calculating Lift and Confidence)...")
    # Generate rules with a minimum Lift of 1 (positive association)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # Formatting columns to convert frozensets to normal strings for database storage
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Sort rules by highest Lift
    rules = rules.sort_values('lift', ascending=False)
    
    print("Saving rules to database...")
    # Keep important columns
    final_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    final_rules.to_sql('market_basket_rules', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print("\n✅ Market-Basket Analysis completed successfully!")
    print("-" * 70)
    print("Top 5 Product Combinations (Highest Lift):")
    print(final_rules.head())
    print("-" * 70)

if __name__ == "__main__":
    perform_market_basket_analysis()