import os
import sys
import pandas as pd
import numpy as np

# Reproducible anonymization
np.random.seed(42)

# Load cleaned data
INPUT_PATH = "data/client_clean_data.csv"
if not os.path.exists(INPUT_PATH):
    print("Error: input file not found:", INPUT_PATH, file=sys.stderr)
    sys.exit(1)
df = pd.read_csv(INPUT_PATH)

# Anonymize Product Names
df['PRODUCT'] = [f"Wine_SKU_{i+1:03d}" for i in range(len(df))]

# --- Anonymize Winery Names ---
# Fill missing wineries with NA
df['Winery'] = df['Winery'].fillna('NA').astype(str)

# Get unique wineries
unique_wineries = sorted([w for w in df['Winery'].unique() if w != 'NA'])

# Mapping for wineries to anonymized names
winery_map = {name: f"Winery_{i+1:02d}" for i, name in enumerate(unique_wineries)}

# Map the wineries to anonymized names, keeping 'NA' as is
df['Winery'] = df['Winery'].apply(lambda x: winery_map.get(x, 'NA'))

# --- Anonymize Numerical Columns ---
# Perturb Sales, Costs, reorder points and max levels
multipliers = np.random.uniform(0.85, 1.15, size=len(df))

# Columns to obscure
sales_cols = ['Dec-24', 'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 
              'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25', 'Oct-25', 
              'TotalSales_11Months', 'AdjustedAnnualSales']
cost_cols = ['COST']
logic_cols = ['Reorder Point', 'Max Level']

# Apply perturbation to numeric columns
for col in logic_cols:
    df[col] = (df[col] * multipliers).round(0)

for col in sales_cols:
    df[col] = (df[col] * multipliers).round(1)

for col in cost_cols:
    df[col] = (df[col] * multipliers).round(2)

#Format cost column with two decimal places
df['COST'] = df['COST'].map('{:,.2f}'.format)

# Use discrete integer shift to mask lead times
df['Lead Time (Days)'] = (df['Lead Time (Days)'] + np.random.choice([-1, 0, 1], size=len(df))).clip(lower=1)

# Shuffle 'Wholesale vs Direct' column if it exists to break link between wine and distribution channel
# Helper function
def stratified_shuffle(
        df: pd.DataFrame, 
        group_col: str, 
        target_col: str) -> pd.DataFrame:
    """Shuffles the target_col within groups defined by group_col."""
    
    # Create a copy to avoid modifying original data
    df_shuffled = df.copy()
    
    # Define groups based on volume (3 tiers: Low, Med, High)
    df_shuffled['Vol_Tier'] = pd.qcut(df[group_col], q=3, labels=['Low', 'Med', 'High'])
    
    # Shuffle target_col within each tier
    df_shuffled[target_col] = df_shuffled.groupby('Vol_Tier', observed=False)[target_col].transform(
        lambda x: np.random.permutation(x.values)
    )
    
    # Drop the temporary tier column
    return df_shuffled.drop(columns=['Vol_Tier'])

# Apply stratified shuffle to 'Wholesale vs Direct' column
df = stratified_shuffle(df, 'AdjustedAnnualSales', 'Wholesale vs Direct')

# Save the anonymized data
df.to_csv('data/dummy_data.csv', index=False)
