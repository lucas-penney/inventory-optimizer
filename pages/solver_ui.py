import streamlit as st
import pandas as pd
import os
from src.solver_logic import validate_data_schema, prepare_optimization_data, run_optimization, run_sensitivity_analyses
from utils.ui_components import page_title


def _render_sensitivity_range(section_label, low_label, high_label, step_label, low_key, high_key, step_key,
                              default_low, default_high, default_step, low_min, step_min):
    """Render three number inputs for a sensitivity range; returns (low, high, step)."""
    st.markdown(f"**{section_label}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        low = st.number_input(low_label, min_value=low_min, value=default_low, step=step_min, key=low_key,
                              help="Low end of range (must be > 0)")
    with col2:
        min_high = low + step_min if isinstance(low, (int, float)) else low_min + step_min
        high = st.number_input(high_label, min_value=min_high, value=max(default_high, min_high), step=step_min,
                               key=high_key, help="High end of range (must be > low end)")
        if high <= low:
            st.error("High end must be greater than low end")
    with col3:
        step = st.number_input(step_label, min_value=step_min, value=default_step, step=step_min, key=step_key,
                               help="Increment between values")
    return low, high, step


page_title("Configure Optimization")

# User type from dashboard (client or guest); default to guest if not set
user_type = st.session_state.get('user_type', 'guest')

# Initialize session state for data source if not exists
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'client' if user_type == 'client' else 'dummy'

# Initialize session state for parameters
if 'holding_cost_pct' not in st.session_state:
    st.session_state.holding_cost_pct = 25.0
if 'fixed_order_cost' not in st.session_state:
    st.session_state.fixed_order_cost = 50.0
# Default storage capacity
client_storage_cap = 2000
try:
    client_storage_cap = int(st.secrets.get("client", {}).get("storage_capacity", 2000))
except (TypeError, ValueError):
    pass
if 'storage_capacity' not in st.session_state:
    st.session_state.storage_capacity = client_storage_cap if user_type == 'client' else 2000
if 'service_level' not in st.session_state:
    st.session_state.service_level = 95.0

# Data Source section: only show for client users
if user_type == 'client':
    st.subheader("Data Source")

    st.session_state.data_source = 'client'
    uploaded_file = st.file_uploader(
        "Upload Data",
        type=['csv'],
        key='data_upload'
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            is_valid, errors = validate_data_schema(df)
            if is_valid:
                st.session_state.input_data = df
                st.success("Data uploaded and validated successfully.")
            else:
                st.error("Data validation failed. Please fix the following issues:")
                for error in errors:
                    st.error(f"  • {error}")
                if 'input_data' in st.session_state:
                    del st.session_state.input_data
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            if 'input_data' in st.session_state:
                del st.session_state.input_data
    else:
        if 'input_data' in st.session_state:
            del st.session_state.input_data

    st.divider()

elif user_type == 'guest':
    # Guest: Load anonymized dataset
    st.session_state.data_source = 'dummy'
    try:
        dummy_data_path = os.path.join('data', 'dummy_data.csv')
        if os.path.exists(dummy_data_path):
            df_dummy = pd.read_csv(dummy_data_path)
            st.session_state.input_data = df_dummy
        else:
            if 'input_data' in st.session_state:
                del st.session_state.input_data
    except Exception as e:
        st.error(f"Error loading anonymized data: {str(e)}")
        if 'input_data' in st.session_state:
            del st.session_state.input_data

# Optimization Parameters Section
st.subheader("Optimization Parameters")

# Default storage capacity: from secrets in Client mode, 2000 in demo (guest) mode
default_storage = client_storage_cap if user_type == 'client' else 2000
if 'storage_capacity_initialized' not in st.session_state:
    st.session_state.storage_capacity = default_storage
    st.session_state.storage_capacity_initialized = True
else:
    # Update default only if user hasn't customized it and mode changed
    if st.session_state.storage_capacity in [client_storage_cap, 2000]:
        st.session_state.storage_capacity = default_storage

# Holding Cost Percentage
col1, col2 = st.columns([3, 2])
with col1:
    holding_cost_pct = st.slider(
        "Holding Cost %",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.holding_cost_pct,
        step=0.1,
        key='holding_cost_slider',
        help="Percentage of item cost that represents the annual holding cost"
    )
    st.session_state.holding_cost_pct = holding_cost_pct
with col2:
    st.markdown("**Impact:** Higher holding costs encourage smaller order quantities to reduce inventory carrying costs, while lower costs allow for larger bulk orders to minimize ordering frequency.")

# Fixed Order Cost
col1, col2 = st.columns([3, 2])
with col1:
    fixed_order_cost = st.slider(
        "Fixed Order Cost ($)",
        min_value=0.0,
        max_value=500.0,
        value=st.session_state.fixed_order_cost,
        step=1.0,
        key='fixed_order_slider',
        help="Fixed administrative cost per order, regardless of order size"
    )
    st.session_state.fixed_order_cost = fixed_order_cost
with col2:
    st.markdown("**Impact:** Higher fixed costs incentivize larger order quantities to spread costs over more units, reducing total ordering frequency and annual ordering costs.")

# Service Level
col1, col2 = st.columns([3, 2])
with col1:
    service_level = st.slider(
        "Service Level (%)",
        min_value=50.0,
        max_value=99.9,
        value=st.session_state.service_level,
        step=0.1,
        key='service_level_slider',
        help="Target probability of not experiencing a stockout during lead time"
    )
    st.session_state.service_level = service_level
with col2:
    st.markdown("**Impact:** Higher service levels require more safety stock to protect against demand uncertainty, increasing inventory holding costs but reducing stockout frequency and improving customer satisfaction.")

# Storage Capacity
col1, col2 = st.columns([3, 2])
with col1:
    storage_capacity_input = st.text_input(
        "Storage Capacity (bottles)",
        value=str(int(st.session_state.storage_capacity)),
        key='storage_input',
        help="Maximum total number of bottles that can be stored in inventory"
    )
    # Validate and convert to integer
    try:
        storage_capacity = int(storage_capacity_input)
        if storage_capacity <= 0:
            st.error("Storage capacity must be a positive integer.")
            storage_capacity = st.session_state.storage_capacity
        else:
            st.session_state.storage_capacity = storage_capacity
    except ValueError:
        st.error("Storage capacity must be a valid integer.")
        storage_capacity = st.session_state.storage_capacity
with col2:
    st.markdown("**Impact:** Tighter capacity constraints force the optimizer to prioritize higher-velocity items and may limit stock levels for slower-moving wines, potentially increasing stockout risk for less popular items.")

# Sensitivity Analysis Expander
st.divider()
with st.expander("Sensitivity Analysis", expanded=False):
    st.markdown("Sensitivity analysis evaluates how changes in key parameters affect optimization results. This helps identify which parameters have the greatest impact on total costs and inventory strategies.")
    
    # Initialize session state for checkboxes if not exists
    if 'sens_enable_fixed_order_cost' not in st.session_state:
        st.session_state.sens_enable_fixed_order_cost = False
    if 'sens_enable_storage_capacity' not in st.session_state:
        st.session_state.sens_enable_storage_capacity = False
    if 'sens_enable_service_level' not in st.session_state:
        st.session_state.sens_enable_service_level = False
    if 'sens_enable_holding_cost' not in st.session_state:
        st.session_state.sens_enable_holding_cost = False
    
    # Checkboxes for enabling each analysis
    st.markdown("**Enable Sensitivity Analyses:**")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.sens_enable_fixed_order_cost = st.checkbox(
            "Enable Fixed Order Cost Sensitivity Analysis",
            value=st.session_state.sens_enable_fixed_order_cost,
            key='cb_fixed_order_cost'
        )
        st.session_state.sens_enable_storage_capacity = st.checkbox(
            "Enable Storage Capacity Sensitivity Analysis",
            value=st.session_state.sens_enable_storage_capacity,
            key='cb_storage_capacity'
        )
    with col2:
        st.session_state.sens_enable_service_level = st.checkbox(
            "Enable Service Level Sensitivity Analysis",
            value=st.session_state.sens_enable_service_level,
            key='cb_service_level'
        )
        st.session_state.sens_enable_holding_cost = st.checkbox(
            "Enable Holding Cost Sensitivity Analysis",
            value=st.session_state.sens_enable_holding_cost,
            key='cb_holding_cost'
        )
    
    st.markdown("---")
    
    # Initialize default values (will be overridden if checkboxes are enabled)
    order_low = order_high = order_step = None
    storage_low = storage_high = storage_step = None
    service_low = service_high = service_step = None
    holding_low = holding_high = holding_step = None
    
    # Fixed Order Cost Sensitivity
    if st.session_state.sens_enable_fixed_order_cost:
        order_low, order_high, order_step = _render_sensitivity_range(
            "Fixed Order Cost", "Low End ($)", "High End ($)", "Step Size ($)",
            'order_low', 'order_high', 'order_step',
            25.0, 100.0, 25.0, 0.01, 1.0
        )
    else:
        order_low = order_high = order_step = None

    # Storage Capacity Sensitivity
    if st.session_state.sens_enable_storage_capacity:
        storage_low, storage_high, storage_step = _render_sensitivity_range(
            "Storage Capacity", "Low End (bottles)", "High End (bottles)", "Step Size (bottles)",
            'storage_low', 'storage_high', 'storage_step',
            1500, 2500, 100, 1, 1
        )
    else:
        storage_low = storage_high = storage_step = None

    # Service Level Sensitivity
    if st.session_state.sens_enable_service_level:
        service_low, service_high, service_step = _render_sensitivity_range(
            "Service Level", "Low End (%)", "High End (%)", "Step Size (%)",
            'service_low', 'service_high', 'service_step',
            90.0, 99.0, 2.0, 0.01, 0.1
        )
    else:
        service_low = service_high = service_step = None

    # Holding Cost Sensitivity
    if st.session_state.sens_enable_holding_cost:
        holding_low, holding_high, holding_step = _render_sensitivity_range(
            "Holding Cost %", "Low End (%)", "High End (%)", "Step Size (%)",
            'holding_low', 'holding_high', 'holding_step',
            15.0, 35.0, 5.0, 0.01, 0.1
        )
    else:
        holding_low = holding_high = holding_step = None
    
    # Store enabled sensitivity parameters (only when low, high, step valid and high > low, step > 0)
    def _valid_sens_params(low, high, step):
        return (
            low is not None and high is not None and step is not None
            and high > low and step > 0
        )

    enabled_analyses = {}
    if st.session_state.sens_enable_fixed_order_cost and _valid_sens_params(order_low, order_high, order_step):
        enabled_analyses['fixed_order_cost'] = {
            'low': order_low,
            'high': order_high,
            'step': order_step
        }
    if st.session_state.sens_enable_storage_capacity and _valid_sens_params(storage_low, storage_high, storage_step):
        enabled_analyses['storage_capacity'] = {
            'low': int(storage_low),
            'high': int(storage_high),
            'step': int(storage_step)
        }
    if st.session_state.sens_enable_service_level and _valid_sens_params(service_low, service_high, service_step):
        enabled_analyses['service_level'] = {
            'low': service_low,
            'high': service_high,
            'step': service_step
        }
    if st.session_state.sens_enable_holding_cost and _valid_sens_params(holding_low, holding_high, holding_step):
        enabled_analyses['holding_cost_pct'] = {
            'low': holding_low,
            'high': holding_high,
            'step': holding_step
        }
    
    # Count total number of sensitivity analysis scenarios
    total_scenarios = 0
    scenario_breakdown = {}
    MAX_SCENARIOS = 75
    
    for analysis_type, params in enabled_analyses.items():
        num_scenarios = int((params['high'] - params['low']) / params['step']) + 1
        scenario_breakdown[analysis_type] = num_scenarios
        total_scenarios += num_scenarios
    
    # Display scenario count and validation
    if enabled_analyses:
        st.markdown("---")
        st.markdown(f"**Total Sensitivity Analysis Scenarios:** {total_scenarios}")
        
        if total_scenarios > MAX_SCENARIOS:
            st.error(
                f"⚠️ **Too many scenarios configured ({total_scenarios} > {MAX_SCENARIOS})**\n\n"
                f"Running {total_scenarios} sensitivity analysis scenarios would take too long to solve. "
                f"Please reduce the number of scenarios by:\n"
                f"- Reducing the range (high - low)\n"
                f"- Increasing the step size\n"
                f"- Disabling one or more sensitivity analyses\n\n"
                f"**Current breakdown:**\n" + 
                "\n".join([f"- {name.replace('_', ' ').title()}: {count} scenarios" 
                           for name, count in scenario_breakdown.items()])
            )
            # Don't store parameters if over limit
            if 'sensitivity_params' in st.session_state:
                del st.session_state.sensitivity_params
        else:
            # Show breakdown if multiple analyses enabled
            if len(enabled_analyses) > 1:
                breakdown_text = " | ".join([f"{name.replace('_', ' ').title()}: {count}" 
                                            for name, count in scenario_breakdown.items()])
                st.caption(f"Breakdown: {breakdown_text}")
            
            # Store enabled analyses in session state
            st.session_state.sensitivity_params = enabled_analyses
    else:
        if 'sensitivity_params' in st.session_state:
            del st.session_state.sensitivity_params

# Bottom Navigation
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    # Back button
    if st.button("← Back",  width='stretch'):
        st.switch_page("pages/dashboard.py")

with col3:
    # Run Optimization button
    if st.button("Run Optimization →", type="primary", key='run_optimization', width='stretch'):
        # Check if input data is available
        if 'input_data' not in st.session_state or st.session_state.input_data is None:
            st.error("No data available. Please upload a valid CSV file (as a client) or ensure anonymized data has loaded.")
        else:
            try:
                # Store parameters in session state for the optimization run
                optimization_params_dict = {
                    'holding_cost_pct': st.session_state.holding_cost_pct / 100.0,  # Convert to decimal
                    'fixed_order_cost': st.session_state.fixed_order_cost,
                    'storage_capacity': int(st.session_state.storage_capacity),  # Ensure integer
                    'service_level': st.session_state.service_level / 100.0,  # Convert to decimal
                    'data_source': st.session_state.data_source
                }
                st.session_state.optimization_params = optimization_params_dict
                
                # Run optimization with spinner
                with st.spinner("Running optimization... This may take a moment."):
                    # Prepare optimization data
                    df_prep, opt_params = prepare_optimization_data(
                        df=st.session_state.input_data,
                        service_level=optimization_params_dict['service_level'],
                        admin_cost=optimization_params_dict['fixed_order_cost'],
                        holding_cost_pct=optimization_params_dict['holding_cost_pct']
                    )
                    
                    # Run optimization
                    results_df, objective_value = run_optimization(
                        df=df_prep,
                        params=opt_params,
                        v_max=optimization_params_dict['storage_capacity']
                    )
                    
                    # Store results in session state
                    st.session_state.optimization_results = results_df
                    st.session_state.optimization_objective_value = objective_value
                    st.session_state.original_input_data = st.session_state.input_data.copy()
                    
                    # Store baseline parameters for sensitivity analysis visualizations
                    st.session_state.sensitivity_baseline_fixed_order_cost = optimization_params_dict['fixed_order_cost']
                    st.session_state.sensitivity_baseline_storage_capacity = optimization_params_dict['storage_capacity']
                    st.session_state.sensitivity_baseline_service_level = optimization_params_dict['service_level'] * 100  # Store as percentage
                    st.session_state.sensitivity_baseline_holding_cost_pct = optimization_params_dict['holding_cost_pct'] * 100  # Store as percentage
                
                # Run sensitivity analyses if enabled
                if 'sensitivity_params' in st.session_state and st.session_state.sensitivity_params:
                    enabled_analyses = st.session_state.sensitivity_params
                    
                    # Clear previous sensitivity results
                    for key in ['sensitivity_results_fixed_order_cost', 'sensitivity_results_storage_capacity',
                               'sensitivity_results_service_level', 'sensitivity_results_holding_cost_pct']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Run sensitivity analyses sequentially
                    enabled_list = list(enabled_analyses.keys())
                    total_analyses = len(enabled_list)
                    
                    for i, analysis_name in enumerate(enabled_list):
                        analysis_display_name = analysis_name.replace('_', ' ').title()
                        with st.status(f"Running {analysis_display_name} sensitivity analysis... ({i+1}/{total_analyses} analyses complete)"):
                            try:
                                sensitivity_results = run_sensitivity_analyses(
                                    df=st.session_state.input_data,
                                    enabled_analyses={analysis_name: enabled_analyses[analysis_name]},
                                    base_params=optimization_params_dict
                                )
                                
                                # Store results in session state
                                if analysis_name in sensitivity_results:
                                    session_key = f'sensitivity_results_{analysis_name}'
                                    st.session_state[session_key] = sensitivity_results[analysis_name]
                                    st.success(f"Completed {analysis_display_name} analysis")
                            except Exception as e:
                                st.error(f"Error running {analysis_display_name} sensitivity analysis: {str(e)}")
                                st.exception(e)
                
                st.switch_page("pages/results.py")
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                st.exception(e)
