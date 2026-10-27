import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP: Creating a "database-like" DataFrame from scratch
# In a real scenario, you'd use: df = pd.read_csv('data.csv') or pd.read_sql(query, engine)
data = {
    'Transaction_ID': [101, 102, 103, 104, 105, 106],
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03'],
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing', 'Home'],
    # Notice the 'nan' (missing value)
    'Price': [1200, 50, 300, np.nan, 45, 150],
    'Quantity': [1, 2, 1, 3, 4, 2]
}

df = pd.DataFrame(data)

# 2. DATA CLEANING
# Convert 'Date' column to actual datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Fill the missing Price (NaN) with the average price of the dataset
avg_price = df['Price'].mean()
df['Price'] = df['Price'].fillna(avg_price)

# 3. FEATURE ENGINEERING (Adding new columns)
# Calculate total revenue per row
df['Revenue'] = df['Price'] * df['Quantity']

# 4. DATA ANALYSIS (The "Specific View")
# Let's see the total revenue and average quantity per Category
summary = df.groupby('Category').agg({
    'Revenue': 'sum',
    'Quantity': 'mean'
}).rename(columns={'Quantity': 'Avg_Qty'})

# 5. FILTERING
# Find all high-value transactions (Revenue > 100)
high_value_sales = df[df['Revenue'] > 100]

# 6. OUTPUT
print("--- Full Cleaned DataFrame ---")
print(df)
print("\n--- Category Summary ---")
print(summary)

# To save this to a "database" or file:
# df.to_csv('cleaned_sales.csv', index=False)
# df.to_sql('sales_table', con=your_engine)

# 7. CHART VISUALIZATION
df.set_index('price').plot(marker='o')
plt.ylabel('quantity')
