import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Tuple, Any
import pulp

def generate_order_menu(
        row: pd.Series, 
        admin_cost: float, 
        holding_cost_pct: float) -> List[int]:
    """
    Helper function to generate a valid menu of order quantities (Q) for a specific wine.
    """
    demand = row['AdjustedAnnualSales']
    h_unit = row['COST'] * holding_cost_pct
    moq = row['MOQ']
    
    # Calculate Economic Order Quantity (EOQ) to help set upper bound for options
    if h_unit > 0 and demand > 0:
        eoq = np.sqrt((2 * demand * admin_cost) / h_unit)
    else:
        eoq = moq
    
    options = []

    # If wine is anchor and can be ordered by the bottle, order size options should be 1-12
    if row['Is_Anchor'] == 1 and row['Is_By_the_Bottle'] == 1:
        options = list(range(1, 13))

    # If wine cannot be ordered by the bottle, must order in cases of 12
    # Set upper bound at 3 times EOQ or MOQ, whichever is greater
    else: 
        max_qty = max(moq, int(np.ceil(3 * eoq)))
        qty = moq
        while qty <= max_qty:
            options.append(int(qty))
            qty += 12

    # If wine is ordered direct from winery, provide a 60-bottle (5 cases) option
    # A constraint will enforce MOQ of 60 if wine comes from winery
    if row['Wholesale vs Direct'] == 'Direct' and 60 not in options:
        options.append(60)
    
    return sorted(list(set(options)))

def prepare_optimization_data(
    df: pd.DataFrame, 
    service_level: float, 
    admin_cost: float, 
    holding_cost_pct: float
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Pipeline to transform clean data into optimization parameters.
    
    Steps:
    1. Calculate Monthly DDLT and Standard Deviation.
    2. Calculate Safety Stock and Reorder Points.
    3. Generate possible order quantities (Q_options).
    4. Pre-calculate total annual costs (C_ik) for every option.
    """
    df_prep = df.copy()
    
    # ---Step 1: Average demand during lead time and standard deviation calculation
    month_order = ['Dec-24', 'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 
                   'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25', 'Oct-25']
    days_per_month = [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31]
    
    # Temporary columns 
    dlt_temp_cols = []

    # Ensure all month columns exist
    missing_cols = [m for m in month_order if m not in df_prep.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing expected month columns: {missing_cols}")
    
    # Calculate average demand during lead time (DDLT) for each month
    for i, month in enumerate(month_order):
        if month in df_prep.columns:
            dlt_col = f"DLT_{month}"
            # Demand During Lead Time = (Monthly Sales / Days) * Lead Time
            df_prep[dlt_col] = (df_prep[month] / days_per_month[i]) * df_prep['Lead Time (Days)']
            dlt_temp_cols.append(dlt_col)

    # Calculate Mean and Std Dev of DDLT across the 11-month window
    df_prep['Avg_DLT'] = df_prep[dlt_temp_cols].mean(axis=1)
    df_prep['Std_DLT'] = df_prep[dlt_temp_cols].std(axis=1).fillna(0)
    
    # --- STEP 2: Safety stock and MOQ logic ---
    # Convert service level to z-score
    z_score = norm.ppf(service_level)

    # Calculate Safety Stock
    df_prep['Safety_Stock'] = (z_score * df_prep['Std_DLT']).round(2)
    
    # Helper function enforcing rules for minimum Order Quantities
    def set_moq(row):
        if row['Is_Anchor'] == 1 and row['Is_By_the_Bottle'] == 1:
            return 1
        return 12 # Standard case MOQ
    
    # Create MOQ column based on business logic
    df_prep['MOQ'] = df_prep.apply(set_moq, axis=1)

    # --- STEP 3: Menu Generation and Cost Pre-Calculation ---
    optimization_params = {
        'Q_options': {},  
        'C_ik': {}        
    }

    # Loops through each wine to generate order menus and pre-calculate costs
    for idx, row in df_prep.iterrows():
        # Generate order quantity options
        options = generate_order_menu(row, admin_cost, holding_cost_pct)

        # Store order quantity options in dictionary
        optimization_params['Q_options'][idx] = options
        
        # Loop through each option to calculate total annual cost C_ik
        for k, Q in enumerate(options):
            # C_ik = Annual Holding Cost + Annual Ordering Cost
            # Annual Holding Cost: (Cycle Stock (Q/2) + Safety Stock) * unit_holding_cost
            h_unit = row['COST'] * holding_cost_pct
            holding_annual = ((Q / 2) + row['Safety_Stock']) * h_unit
            
            # Ordering Cost: (Annual Demand / Q) * Fixed Cost per Order
            ordering_annual = (row['AdjustedAnnualSales'] / Q) * admin_cost
            
            optimization_params['C_ik'][(idx, k)] = holding_annual + ordering_annual

    # Cleanup temporary columns
    df_prep = df_prep.drop(columns=dlt_temp_cols)
    
    return df_prep, optimization_params

def run_optimization(df: pd.DataFrame, params: dict, v_max: int) -> pd.DataFrame:
    """
    Solves inventory optimization MILP using PuLP.
    Args:
        df (pd.DataFrame): DataFrame containing prepared wine data.
        params (dict): Dictionary with optimization parameters (Q_options, C_ik).
        v_max (int): Maximum storage capacity constraint.
    Returns:
        pd.DataFrame: DataFrame updated with optimization results.
    """
    # Initialize the Problem
    prob = pulp.LpProblem("Inventory_MILP", pulp.LpMinimize)

    # Extract Indices and Parameters
    Q_options = params['Q_options']
    C_ik = params['C_ik']
    wine_indices = df.index.tolist()

    # Ensure index is unique before starting optimization
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
   
    # Identify unique direct wineries for Constraint 5
    direct_wineries = df[df['Wholesale vs Direct'] == 'Direct']['Winery'].unique().tolist()

    # --- Decision Variables ----
    # y[i, k]: Binary (1 if option k selected for wine i)
    y = pulp.LpVariable.dicts("y", 
                             ((i, k) for i in wine_indices for k in range(len(Q_options[i]))), 
                             cat=pulp.LpBinary)

    # R[i]: Reorder point for wine i (Integer)
    R = pulp.LpVariable.dicts("R", wine_indices, lowBound=0, cat=pulp.LpInteger)

    # M[i]: Order-up-to level for wine i (Integer)
    M = pulp.LpVariable.dicts("M", wine_indices, lowBound=0, cat=pulp.LpInteger)

    # z[w]: Binary (1 if winery w is used)
    z = pulp.LpVariable.dicts("use_winery", range(len(direct_wineries)), cat=pulp.LpBinary)

    # --- Objective Function: Minimize total annual cost ---
    prob += pulp.lpSum([C_ik[(i, k)] * y[(i, k)] 
                        for i in wine_indices for k in range(len(Q_options[i]))])

    # --- Constraints ---
    
    # Constraint 1: Single Choice (Choose one option per wine)
    for i in wine_indices:
        prob += pulp.lpSum([y[(i, k)] for k in range(len(Q_options[i]))]) == 1, f"SingleChoice_{i}"

    # Constraint 2: Linking M - R = Q
    # The gap between max level (M) and reorder point (R) must equal chosen order quantity (Q)
    for i in wine_indices:
        prob += M[i] - R[i] == pulp.lpSum([Q_options[i][k] * y[(i, k)] 
                                         for k in range(len(Q_options[i]))]), f"Linking_{i}"

    # Constraint 3: Safety Stock
    # Reorder point must be at least the demand during lead time plus safety stock
    for i in wine_indices:
        prob += R[i] >= df.loc[i, 'Safety_Stock'] + df.loc[i, 'Avg_DLT'], f"SafetyStock_{i}"

    # Constraint 4: Storage Capacity
    # Sum of all the max inventory levels must not exceed v_max
    prob += pulp.lpSum([M[i] for i in wine_indices]) <= v_max, "StorageCapacity"

    # Constraint 5: Winery MOQs (Big M Formulation)
    # If a winery is used (z[w] = 1), total order quantity from that winery >= 60
    # If not used (z[w] = 0), total order quantity = 0F
    for w_idx, winery_code in enumerate(direct_wineries):
        # Filter for wines belonging to this winery
        winery_indices = df[(df['Winery'] == winery_code) & 
                            (df['Wholesale vs Direct'] == 'Direct')].index.tolist()
        
        if not winery_indices:
            continue
            
        # Calculate total quantity of wines from a winery
        total_qty_winery = pulp.lpSum([Q_options[i][k] * y[(i, k)] 
                                      for i in winery_indices for k in range(len(Q_options[i]))])
        
        # Calculate big M value
        big_m_winery = sum([max(Q_options[i]) for i in winery_indices])

        # Constraint A: If a winery is used, total quantity >= 60
        prob += total_qty_winery >= 60 * z[w_idx], f"moq_floor_{winery_code}"
        
        # Constraint B: If winery not used, total quantity = 0
        prob += total_qty_winery <= big_m_winery * z[w_idx], f"force_zero_{winery_code}"

    # 6. Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # 7. Extract Results
    order_results = []
    r_points = []
    m_levels = []
    
    for i in wine_indices:
        r_points.append(pulp.value(R[i]))
        m_levels.append(pulp.value(M[i]))
        
        # Find chosen Q using a threshold check
        chosen_q = 0
        for k in range(len(Q_options[i])):
            val = pulp.value(y[(i, k)])
            if val is not None and val > 0.5:
                chosen_q = Q_options[i][k]
                break
            
        order_results.append(chosen_q)

    # Update DataFrame with results
    df['Recommended_Order'] = order_results
    df['Reorder_Point_R'] = r_points
    df['Order_Up_To_M'] = m_levels
    
    return df

