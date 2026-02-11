import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Tuple, Any, Optional
import pulp

def validate_data_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that the input DataFrame has the correct schema and data types
    required for the optimization solver.
    
    Args:
        df (pd.DataFrame): Input DataFrame to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    # Required month columns
    month_columns = ['Dec-24', 'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 
                     'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25', 'Oct-25']
    
    # Required core columns
    required_columns = [
        'AdjustedAnnualSales',
        'Lead Time (Days)',
        'COST',
        'Wholesale vs Direct',
        'Winery',
        'Is_Anchor',
        'Is_By_the_Bottle'
    ]
    
    # Check for required columns
    missing_columns = []
    for col in month_columns + required_columns:
        if col not in df.columns:
            missing_columns.append(col)
    
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return False, errors
    
    # Check data types and values for month columns
    for col in month_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric")
        else:
            # Check for negative values
            if (df[col] < 0).any():
                errors.append(f"Column '{col}' contains negative values (sales cannot be negative)")
    
    # Check AdjustedAnnualSales
    if not pd.api.types.is_numeric_dtype(df['AdjustedAnnualSales']):
        errors.append("Column 'AdjustedAnnualSales' must be numeric")
    else:
        if (df['AdjustedAnnualSales'] <= 0).any():
            errors.append("Column 'AdjustedAnnualSales' must contain only positive values")
        if df['AdjustedAnnualSales'].isna().any():
            errors.append("Column 'AdjustedAnnualSales' contains null values")
    
    # Check Lead Time (Days)
    if not pd.api.types.is_numeric_dtype(df['Lead Time (Days)']):
        errors.append("Column 'Lead Time (Days)' must be numeric")
    else:
        if (df['Lead Time (Days)'] <= 0).any():
            errors.append("Column 'Lead Time (Days)' must contain only positive values")
        if df['Lead Time (Days)'].isna().any():
            errors.append("Column 'Lead Time (Days)' contains null values")
    
    # Check COST
    if not pd.api.types.is_numeric_dtype(df['COST']):
        errors.append("Column 'COST' must be numeric")
    else:
        if (df['COST'] <= 0).any():
            errors.append("Column 'COST' must contain only positive values")
        if df['COST'].isna().any():
            errors.append("Column 'COST' contains null values")
    
    # Check Wholesale vs Direct
    if df['Wholesale vs Direct'].isna().any():
        errors.append("Column 'Wholesale vs Direct' contains null values")
    else:
        valid_values = {'Wholesaler', 'Direct'}
        invalid_values = set(df['Wholesale vs Direct'].unique()) - valid_values
        if invalid_values:
            errors.append(f"Column 'Wholesale vs Direct' contains invalid values: {', '.join(map(str, invalid_values))}. Must be 'Wholesaler' or 'Direct'")
    
    # Check Winery (can have nulls/NA, but should be string type)
    if not pd.api.types.is_object_dtype(df['Winery']) and not pd.api.types.is_string_dtype(df['Winery']):
        errors.append("Column 'Winery' must be string/object type")
    
    # Check Is_Anchor
    if not pd.api.types.is_numeric_dtype(df['Is_Anchor']):
        errors.append("Column 'Is_Anchor' must be numeric")
    else:
        if df['Is_Anchor'].isna().any():
            errors.append("Column 'Is_Anchor' contains null values")
        invalid_values = set(df['Is_Anchor'].unique()) - {0, 1}
        if invalid_values:
            errors.append(f"Column 'Is_Anchor' contains invalid values: {', '.join(map(str, invalid_values))}. Must be 0 or 1")
    
    # Check Is_By_the_Bottle
    if not pd.api.types.is_numeric_dtype(df['Is_By_the_Bottle']):
        errors.append("Column 'Is_By_the_Bottle' must be numeric")
    else:
        if df['Is_By_the_Bottle'].isna().any():
            errors.append("Column 'Is_By_the_Bottle' contains null values")
        invalid_values = set(df['Is_By_the_Bottle'].unique()) - {0, 1}
        if invalid_values:
            errors.append(f"Column 'Is_By_the_Bottle' contains invalid values: {', '.join(map(str, invalid_values))}. Must be 0 or 1")
    
    # Check if DataFrame is empty
    if len(df) == 0:
        errors.append("DataFrame is empty - no rows to process")
    
    is_valid = len(errors) == 0
    return is_valid, errors

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

def run_optimization(df: pd.DataFrame, params: dict, v_max: int) -> Tuple[pd.DataFrame, float]:
    """
    Solves inventory optimization MILP using PuLP.
    Args:
        df (pd.DataFrame): DataFrame containing prepared wine data.
        params (dict): Dictionary with optimization parameters (Q_options, C_ik).
        v_max (int): Maximum storage capacity constraint.
    Returns:
        Tuple[pd.DataFrame, float]: DataFrame updated with optimization results and objective value.
    """
    # Initialize the Problem
    prob = pulp.LpProblem("Inventory_MILP", pulp.LpMinimize)

    # Extract Indices and Parameters
    Q_options = params['Q_options']
    C_ik = params['C_ik']
    wine_indices = df.index.tolist()

    if len(wine_indices) == 0:
        raise ValueError("No data to optimize.")

    # Ensure index is unique before starting optimization
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
        wine_indices = df.index.tolist()

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
    # If not used (z[w] = 0), total order quantity = 0
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

    if prob.status != pulp.LpStatusOptimal:
        raise ValueError(
            "Optimization failed: problem is infeasible or could not be solved."
        )

    # 7. Extract Objective Value
    objective_value = pulp.value(prob.objective)

    # 8. Extract Results
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
    
    return df, objective_value

def extract_metrics_from_results(results_df: pd.DataFrame, objective_value: float, 
                                capacity: int = None,
                                fixed_order_cost: float = None,
                                holding_cost_pct: float = None) -> Dict[str, Any]:
    """
    Extract aggregate metrics from optimization results DataFrame.
    
    Args:
        results_df: DataFrame with optimization results (must have Recommended_Order, 
                   Reorder_Point_R, Order_Up_To_M, AdjustedAnnualSales, COST, Safety_Stock columns)
        objective_value: Total objective value from optimization
        capacity: Storage capacity (optional, for calculating Storage_%)
        fixed_order_cost: Fixed cost per order (optional, for calculating Ordering_Cost)
        holding_cost_pct: Holding cost as decimal (optional, for calculating Holding_Cost)
        
    Returns:
        Dictionary with metrics: Total_Cost, Holding_Cost (if params provided), 
        Ordering_Cost (if params provided), Orders/Yr, Storage_%, Avg_Order_Size, 
        Avg_Reorder_Pt, Avg_Order_Up_To, Avg_Days_Supply
    """
    # Calculate ordering frequency
    orders_per_year = (results_df['AdjustedAnnualSales'] / results_df['Recommended_Order']).replace([np.inf, -np.inf], 0)
    total_orders = orders_per_year.sum()
    
    # Calculate average metrics
    avg_order_size = results_df['Recommended_Order'].mean()
    avg_reorder_pt = results_df['Reorder_Point_R'].mean()
    avg_order_up_to = results_df['Order_Up_To_M'].mean()
    
    # Average days supply
    days_supply = (results_df['Recommended_Order'] / results_df['AdjustedAnnualSales'] * 365).replace([np.inf, -np.inf], 0)
    avg_days_supply = days_supply.mean()
    
    # Storage utilization
    if capacity is not None and capacity > 0:
        total_storage_used = results_df['Order_Up_To_M'].sum()
        storage_pct = (total_storage_used / capacity) * 100
    else:
        storage_pct = np.nan
    
    # Initialize metrics dictionary
    metrics = {
        'Total_Cost': objective_value,
        'Orders/Yr': total_orders,
        'Storage_%': storage_pct,
        'Avg_Order_Size (Q)': avg_order_size,
        'Avg_Reorder_Pt (R)': avg_reorder_pt,
        'Avg_Order_Up_To (M)': avg_order_up_to,
        'Avg_Days_Supply': avg_days_supply
    }
    
    # Calculate cost components if parameters are provided
    if fixed_order_cost is not None and holding_cost_pct is not None:
        holding_cost, ordering_cost = calculate_cost_components(
            results_df, fixed_order_cost, holding_cost_pct
        )
        metrics['Holding_Cost'] = holding_cost
        metrics['Ordering_Cost'] = ordering_cost
    
    return metrics

def calculate_cost_components(results_df: pd.DataFrame, fixed_order_cost: float, 
                             holding_cost_pct: float) -> Tuple[float, float]:
    """
    Calculate holding cost and ordering cost from results.
    
    Args:
        results_df: DataFrame with Recommended_Order, AdjustedAnnualSales, COST, Safety_Stock columns
        fixed_order_cost: Fixed cost per order
        holding_cost_pct: Holding cost as decimal (e.g., 0.25 for 25%)
        
    Returns:
        Tuple of (total_holding_cost, total_ordering_cost)
    """
    total_holding_cost = 0
    total_ordering_cost = 0
    
    for idx, row in results_df.iterrows():
        Q = row['Recommended_Order']
        demand = row['AdjustedAnnualSales']
        unit_cost = row['COST']
        safety_stock = row.get('Safety_Stock', 0)
        
        if Q > 0 and demand > 0:
            # Annual holding cost: (Cycle Stock (Q/2) + Safety Stock) * unit_holding_cost
            h_unit = unit_cost * holding_cost_pct
            holding_annual = ((Q / 2) + safety_stock) * h_unit
            total_holding_cost += holding_annual
            
            # Ordering cost: (Annual Demand / Q) * Fixed Cost per Order
            ordering_annual = (demand / Q) * fixed_order_cost
            total_ordering_cost += ordering_annual
    
    return total_holding_cost, total_ordering_cost


def get_reorder_max_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve reorder point and max level column names (handles 'Reorder Point'/'Reorder_Point' and 'Max Level'/'Max_Level').
    Returns (r_col, m_col); either may be None if not found.
    """
    r_col = 'Reorder Point' if 'Reorder Point' in df.columns else ('Reorder_Point' if 'Reorder_Point' in df.columns else None)
    m_col = 'Max Level' if 'Max Level' in df.columns else ('Max_Level' if 'Max_Level' in df.columns else None)
    return r_col, m_col


def calculate_current_policy_costs(input_df: pd.DataFrame, fixed_order_cost: float, holding_cost_pct: float) -> Optional[Tuple[float, float, float]]:
    """
    Calculate costs for the current inventory policy from input data.
    
    Args:
        input_df: DataFrame with current policy data (Reorder Point, Max Level, AdjustedAnnualSales, COST)
        fixed_order_cost: Fixed cost per order
        holding_cost_pct: Holding cost as decimal (e.g., 0.25 for 25%)
        
    Returns:
        Tuple of (total_current_cost, current_holding_cost, current_ordering_cost) or None if data missing
    """
    r_col, m_col = get_reorder_max_columns(input_df)
    if r_col is None or m_col is None:
        return None
    
    # Check for other required columns
    required_cols = ['AdjustedAnnualSales', 'COST']
    if not all(col in input_df.columns for col in required_cols):
        return None
    
    total_holding_cost = 0
    total_ordering_cost = 0
    
    for idx, row in input_df.iterrows():
        r_act = row[r_col] if pd.notna(row[r_col]) else 0
        m_act = row[m_col] if pd.notna(row[m_col]) else 0
        q_act = m_act - r_act
        
        # Check for valid policy
        if q_act <= 0:
            continue
        
        # Calculate holding cost: avg_inv = (R + M) / 2, then h_cost = avg_inv * unit_holding_cost
        avg_inv_act = (r_act + m_act) / 2
        unit_cost = row['COST']
        h_unit = unit_cost * holding_cost_pct
        h_cost_act = avg_inv_act * h_unit
        
        # Calculate ordering cost: (Annual Demand / Q) * Fixed Cost per Order
        demand = row['AdjustedAnnualSales']
        if q_act > 0 and demand > 0:
            o_cost_act = (demand / q_act) * fixed_order_cost
        else:
            o_cost_act = 0
        
        total_holding_cost += h_cost_act
        total_ordering_cost += o_cost_act
    
    total_current_cost = total_holding_cost + total_ordering_cost
    return total_current_cost, total_holding_cost, total_ordering_cost

def run_sensitivity_analyses(
    df: pd.DataFrame,
    enabled_analyses: Dict[str, Dict],
    base_params: Dict,
    progress_callback=None
) -> Dict[str, pd.DataFrame]:
    """
    Run sensitivity analyses for enabled parameter types.
    
    Args:
        df: Original input DataFrame (before prepare_optimization_data)
        enabled_analyses: Dict with analysis type as key and parameter dict as value
                         e.g., {'fixed_order_cost': {'low': 25, 'high': 200, 'step': 25}, ...}
        base_params: Dict with base parameters:
                    {'service_level': float (decimal),
                     'fixed_order_cost': float,
                     'storage_capacity': int,
                     'holding_cost_pct': float (decimal)}
        progress_callback: Optional callback function(name, progress) for progress updates
        
    Returns:
        Dict of DataFrames, one per enabled analysis type
    """
    def _sensitivity_range(low: float, high: float, step: float):
        """Return 1D array from low to high (inclusive) in steps of step, robust to float rounding."""
        num = max(1, int(np.round((high - low) / step)) + 1)
        return np.linspace(low, high, num=num)

    results = {}
    
    # Extract base parameters
    base_service_level = base_params['service_level']
    base_fixed_order_cost = base_params['fixed_order_cost']
    base_storage_capacity = base_params['storage_capacity']
    base_holding_cost_pct = base_params['holding_cost_pct']
    
    # Run Fixed Order Cost sensitivity analysis
    if 'fixed_order_cost' in enabled_analyses:
        if progress_callback:
            progress_callback('fixed_order_cost', 0)
        
        params = enabled_analyses['fixed_order_cost']
        s_values = _sensitivity_range(params['low'], params['high'], params['step'])
        sensitivity_results = []
        
        # Prepare base data once (won't change for this analysis)
        df_prep, base_opt_params = prepare_optimization_data(
            df, base_service_level, base_fixed_order_cost, base_holding_cost_pct
        )
        
        for s_value in s_values:
            try:
                # Recalculate C_ik with new S value
                new_C_ik = {}
                Q_options = base_opt_params['Q_options']
                
                for idx in df_prep.index:
                    row = df_prep.loc[idx]
                    h_unit = row['COST'] * base_holding_cost_pct
                    safety_stock = row['Safety_Stock']
                    
                    for k, Q in enumerate(Q_options[idx]):
                        # Recalculate costs with new S
                        holding_annual = ((Q / 2) + safety_stock) * h_unit
                        ordering_annual = (row['AdjustedAnnualSales'] / Q) * s_value if Q > 0 else 0
                        new_C_ik[(idx, k)] = holding_annual + ordering_annual
                
                # Update params with new costs
                new_params = {'Q_options': Q_options, 'C_ik': new_C_ik}
                
                # Run optimization
                results_df, objective_value = run_optimization(
                    df_prep, new_params, base_storage_capacity
                )
                
                # Extract metrics (including cost components)
                metrics = extract_metrics_from_results(
                    results_df, objective_value, base_storage_capacity,
                    fixed_order_cost=s_value, holding_cost_pct=base_holding_cost_pct
                )
                
                sensitivity_results.append({
                    'S_Value': s_value,
                    **metrics
                })
            except Exception as e:
                # If optimization fails, skip this value
                continue
        
        if sensitivity_results:
            results['fixed_order_cost'] = pd.DataFrame(sensitivity_results)
        
        if progress_callback:
            progress_callback('fixed_order_cost', 1)
    
    # Run Storage Capacity sensitivity analysis
    if 'storage_capacity' in enabled_analyses:
        if progress_callback:
            progress_callback('storage_capacity', 0)
        
        params = enabled_analyses['storage_capacity']
        v_values = _sensitivity_range(params['low'], params['high'], params['step']).astype(int)
        sensitivity_results = []
        
        # Prepare base data once
        df_prep, base_opt_params = prepare_optimization_data(
            df, base_service_level, base_fixed_order_cost, base_holding_cost_pct
        )
        
        for v_max in v_values:
            try:
                # Run optimization with new capacity
                results_df, objective_value = run_optimization(
                    df_prep, base_opt_params, v_max
                )
                
                # Extract metrics (including cost components)
                metrics = extract_metrics_from_results(
                    results_df, objective_value, v_max,
                    fixed_order_cost=base_fixed_order_cost, holding_cost_pct=base_holding_cost_pct
                )
                
                sensitivity_results.append({
                    'V_max': v_max,
                    'Status': 'Optimal',
                    **metrics
                })
            except Exception:
                # If optimization fails, mark as infeasible
                sensitivity_results.append({
                    'V_max': v_max,
                    'Total_Cost': np.nan,
                    'Holding_Cost': np.nan,
                    'Ordering_Cost': np.nan,
                    'Status': 'Infeasible',
                    'Orders/Yr': np.nan,
                    'Storage_%': np.nan,
                    'Avg_Order_Size (Q)': np.nan,
                    'Avg_Reorder_Pt (R)': np.nan,
                    'Avg_Order_Up_To (M)': np.nan,
                    'Avg_Days_Supply': np.nan
                })
        
        if sensitivity_results:
            results['storage_capacity'] = pd.DataFrame(sensitivity_results)
        
        if progress_callback:
            progress_callback('storage_capacity', 1)
    
    # Run Service Level sensitivity analysis
    if 'service_level' in enabled_analyses:
        if progress_callback:
            progress_callback('service_level', 0)
        
        params = enabled_analyses['service_level']
        # Convert percentages to decimals
        service_low = params['low'] / 100.0
        service_high = params['high'] / 100.0
        service_step = params['step'] / 100.0
        confidence_levels = _sensitivity_range(service_low, service_high, service_step)
        sensitivity_results = []
        
        for conf_level in confidence_levels:
            try:
                # Prepare data with new service level (this recalculates safety stock)
                df_prep, opt_params = prepare_optimization_data(
                    df, conf_level, base_fixed_order_cost, base_holding_cost_pct
                )
                
                # Run optimization
                results_df, objective_value = run_optimization(
                    df_prep, opt_params, base_storage_capacity
                )
                
                # Extract metrics (including cost components)
                metrics = extract_metrics_from_results(
                    results_df, objective_value, base_storage_capacity,
                    fixed_order_cost=base_fixed_order_cost, holding_cost_pct=base_holding_cost_pct
                )
                
                # Calculate total safety stock
                total_ss_bottles = df_prep['Safety_Stock'].sum()
                
                # Calculate Z-score
                z_score = norm.ppf(conf_level)
                
                sensitivity_results.append({
                    'Confidence': conf_level * 100,  # Store as percentage for display
                    'Z_Score': z_score,
                    'Total_Cost': objective_value,
                    'Holding_Cost': metrics['Holding_Cost'],
                    'Total_SS_Bottles': total_ss_bottles
                })
            except Exception:
                # If optimization fails, skip
                continue
        
        if sensitivity_results:
            results['service_level'] = pd.DataFrame(sensitivity_results)
        
        if progress_callback:
            progress_callback('service_level', 1)
    
    # Run Holding Cost sensitivity analysis
    if 'holding_cost_pct' in enabled_analyses:
        if progress_callback:
            progress_callback('holding_cost_pct', 0)
        
        params = enabled_analyses['holding_cost_pct']
        # Convert percentages to decimals
        holding_low = params['low'] / 100.0
        holding_high = params['high'] / 100.0
        holding_step = params['step'] / 100.0
        holding_values = _sensitivity_range(holding_low, holding_high, holding_step)
        sensitivity_results = []
        
        for holding_cost_pct in holding_values:
            try:
                # Prepare data with new holding cost
                df_prep, opt_params = prepare_optimization_data(
                    df, base_service_level, base_fixed_order_cost, holding_cost_pct
                )
                
                # Run optimization
                results_df, objective_value = run_optimization(
                    df_prep, opt_params, base_storage_capacity
                )
                
                # Extract metrics (including cost components)
                metrics = extract_metrics_from_results(
                    results_df, objective_value, base_storage_capacity,
                    fixed_order_cost=base_fixed_order_cost, holding_cost_pct=holding_cost_pct
                )
                
                sensitivity_results.append({
                    'Holding_Cost_Pct': holding_cost_pct * 100,  # Store as percentage for display
                    **metrics
                })
            except Exception:
                # If optimization fails, skip
                continue
        
        if sensitivity_results:
            results['holding_cost_pct'] = pd.DataFrame(sensitivity_results)
        
        if progress_callback:
            progress_callback('holding_cost_pct', 1)
    
    return results

