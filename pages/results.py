import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.solver_logic import calculate_cost_components, calculate_current_policy_costs, get_reorder_max_columns
from utils.ui_components import page_title

page_title("Optimization Results")

# KPI styling: borders around metric cards (minimal CSS)
st.markdown("""<style>div[data-testid="stMetric"]{border:1px solid #cbd5e1;border-radius:8px;padding:1rem;}</style>""", unsafe_allow_html=True)

# Check if optimization results exist
if 'optimization_results' not in st.session_state or st.session_state.optimization_results is None:
    st.info("No optimization results available. Please run the optimization on the Inventory Optimizer page first.")
else:
    results_df = st.session_state.optimization_results
    
    # KPI Bar Section
    if 'optimization_objective_value' in st.session_state:
        objective_value = st.session_state.optimization_objective_value
        
        # Get parameters for calculations
        fixed_order_cost = st.session_state.get('fixed_order_cost', 50.0)
        holding_cost_pct = st.session_state.get('holding_cost_pct', 25.0) / 100.0  # Convert to decimal
        storage_capacity = st.session_state.get('storage_capacity', 2000)
        
        # Calculate optimized cost components
        opt_holding_cost, opt_ordering_cost = calculate_cost_components(
            results_df, fixed_order_cost, holding_cost_pct
        )
        
        # Calculate space utilization
        total_storage_used = results_df['Order_Up_To_M'].sum()
        space_utilization_pct = (total_storage_used / storage_capacity * 100) if storage_capacity > 0 else 0
        
        # Try to calculate current policy costs
        current_policy_costs = None
        if 'original_input_data' in st.session_state:
            current_policy_costs = calculate_current_policy_costs(
                st.session_state.original_input_data,
                fixed_order_cost,
                holding_cost_pct
            )
        elif 'input_data' in st.session_state:
            current_policy_costs = calculate_current_policy_costs(
                st.session_state.input_data,
                fixed_order_cost,
                holding_cost_pct
            )
        
        # Determine which KPIs to show
        show_comparison_kpis = current_policy_costs is not None
        
        # Create KPI columns
        if show_comparison_kpis:
            kpi_cols = st.columns(4)
        else:
            kpi_cols = st.columns(2)
        
        # KPI 1: Total Annual Cost (no delta)
        with kpi_cols[0]:
            st.metric(
                "Total Annual Cost",
                f"${round(objective_value):,}",
                help="The minimized total annual cost from the optimization solver"
            )
        
        # KPI 2 & 3: Comparison KPIs (only show if current policy data available)
        if show_comparison_kpis:
            current_total_cost, current_holding_cost, current_ordering_cost = current_policy_costs
            
            # Calculate dollar savings
            annual_savings_value = current_total_cost - objective_value
            
            # Calculate percentage reduction
            total_cost_reduction_pct = ((current_total_cost - objective_value) / current_total_cost * 100) if current_total_cost > 0 else 0
            
            with kpi_cols[1]:
                st.metric(
                    "Annual Savings",
                    f"${round(annual_savings_value):,}",
                    help="Dollar savings in total annual cost compared to current policy"
                )
            
            with kpi_cols[2]:
                st.metric(
                    "Cost Reduction",
                    f"{total_cost_reduction_pct:.1f}%",
                    help="Percentage reduction in total annual cost compared to current policy"
                )
            
            # KPI 4: Space Utilization (Comparison case)
            with kpi_cols[3]:
                st.metric(
                    "Space Utilization",
                    f"{space_utilization_pct:.1f}%",
                    help="Percentage of storage capacity utilized by optimized inventory levels"
                )
        else:
            # KPI 2: Space Utilization (Non-comparison case)
            with kpi_cols[1]:
                st.metric(
                    "Space Utilization",
                    f"{space_utilization_pct:.1f}%",
                    help="Percentage of storage capacity utilized by optimized inventory levels"
                )
        
        # View Optimized Inventory Policy expander
        r_col, m_col = get_reorder_max_columns(results_df)
        orig_df = results_df
        if r_col is None or m_col is None:
            for src in [st.session_state.get('original_input_data'), st.session_state.get('input_data')]:
                if src is not None:
                    ro, mo = get_reorder_max_columns(src)
                    if ro is not None or mo is not None:
                        orig_df = src
                        r_col, m_col = ro, mo
                        break
        product_col = results_df['PRODUCT'] if 'PRODUCT' in results_df.columns else results_df.index.astype(str)
        opt_r = results_df['Reorder_Point_R'].astype('Int64')
        opt_m = results_df['Order_Up_To_M'].astype('Int64')
        orig_r = orig_df[r_col] if r_col and r_col in orig_df.columns else pd.Series([pd.NA] * len(results_df), index=results_df.index)
        orig_m = orig_df[m_col] if m_col and m_col in orig_df.columns else pd.Series([pd.NA] * len(results_df), index=results_df.index)
        if not orig_r.index.equals(results_df.index):
            orig_r = orig_r.reindex(results_df.index)
            orig_m = orig_m.reindex(results_df.index)
        change_r = opt_r.astype(float) - orig_r.astype(float)
        change_m = opt_m.astype(float) - orig_m.astype(float)
        annual_savings_list = []
        for idx in results_df.index:
            row = results_df.loc[idx]
            orig_r_val = orig_r.loc[idx]
            orig_m_val = orig_m.loc[idx]
            if pd.isna(orig_r_val) or pd.isna(orig_m_val):
                annual_savings_list.append(pd.NA)
                continue
            q_act = float(orig_m_val) - float(orig_r_val)
            if q_act <= 0:
                annual_savings_list.append(pd.NA)
                continue
            demand = row['AdjustedAnnualSales']
            unit_cost = row['COST']
            if demand <= 0:
                annual_savings_list.append(pd.NA)
                continue
            r_act, m_act = float(orig_r_val), float(orig_m_val)
            avg_inv_act = (r_act + m_act) / 2
            current_h_cost = avg_inv_act * (unit_cost * holding_cost_pct)
            current_o_cost = (demand / q_act) * fixed_order_cost
            Q_opt = row['Recommended_Order']
            if Q_opt <= 0:
                annual_savings_list.append(pd.NA)
                continue
            opt_h_cost = ((Q_opt / 2) + row.get('Safety_Stock', 0)) * (unit_cost * holding_cost_pct)
            opt_o_cost = (demand / Q_opt) * fixed_order_cost
            savings = (current_h_cost + current_o_cost) - (opt_h_cost + opt_o_cost)
            annual_savings_list.append(savings)
        annual_savings = pd.Series(annual_savings_list, index=results_df.index)
        policy_df = pd.DataFrame({
            'Product': product_col.values,
            'Optimized Reorder Point': opt_r.values,
            'Original Reorder Point': orig_r.values,
            'Change in Reorder Point': change_r.values,
            'Optimized Max Level': opt_m.values,
            'Original Max Level': orig_m.values,
            'Change in Max Level': change_m.values,
            'Annual Savings': annual_savings.values,
        })
        def _fmt_na_int(x):
            return '—' if pd.isna(x) else int(x)
        def _fmt_na_int_round(x):
            return '—' if pd.isna(x) else int(round(x))
        policy_formats = [
            ('Original Reorder Point', _fmt_na_int),
            ('Change in Reorder Point', _fmt_na_int_round),
            ('Original Max Level', _fmt_na_int),
            ('Change in Max Level', _fmt_na_int_round),
            ('Annual Savings', lambda x: '—' if pd.isna(x) else f"${x:,.2f}"),
        ]
        for col_name, formatter in policy_formats:
            policy_df[col_name] = policy_df[col_name].apply(formatter)
        with st.expander("View Optimized Inventory Policy", expanded=False):
            buf = BytesIO()
            policy_df.to_excel(buf, index=False, engine='openpyxl')
            buf.seek(0)
            st.download_button(
                "Download inventory policy as Excel",
                data=buf,
                file_name="inventory_policy.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_policy_excel",
            )
            st.dataframe(policy_df, width='stretch', hide_index=True)
        
        # Comparative Visualizations Section
        if show_comparison_kpis:
            st.divider()
            st.subheader("Comparative Analysis: Current vs. Optimized")
            
            tab_costs, tab_freq = st.tabs(["Inventory Costs", "Order Frequency"])
            
            # --- TAB 1: Current vs Optimized Annual Inventory Costs ---
            with tab_costs:
                current_total_cost, current_holding_cost, current_ordering_cost = current_policy_costs
                
                fig_costs = go.Figure()
                
                categories = ['Holding Cost', 'Ordering Cost', 'Total Cost']
                current_vals = [current_holding_cost, current_ordering_cost, current_total_cost]
                opt_vals = [opt_holding_cost, opt_ordering_cost, objective_value]
                
                fig_costs.add_trace(go.Bar(
                    x=categories,
                    y=current_vals,
                    name='Current',
                    marker_color='#d9534f', # Reddish
                    text=[f"${v:,.0f}" for v in current_vals],
                    textposition='auto',
                ))
                
                fig_costs.add_trace(go.Bar(
                    x=categories,
                    y=opt_vals,
                    name='Optimized (Model)',
                    marker_color='#5bc0de', # Blueish
                    text=[f"${v:,.0f}" for v in opt_vals],
                    textposition='auto',
                ))
                
                fig_costs.update_layout(
                    title="Current vs Optimized Annual Inventory Costs",
                    xaxis_title="Cost Category",
                    yaxis_title="Annual Cost ($)",
                    barmode='group',
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='white'
                )
                fig_costs.update_yaxes(showgrid=True, gridcolor='lightgray')
                
                st.plotly_chart(fig_costs, width='stretch')
            
            # --- TAB 2: Reduction in Order Frequency of Top 5 Cost-Saving Wines ---
            with tab_freq:
                # Recalculate per-wine savings to find top 5
                comparison_data = []
                r_col_comp, m_col_comp = get_reorder_max_columns(results_df)
                if r_col_comp and m_col_comp:
                    for idx, row in results_df.iterrows():
                        r_act = row[r_col_comp] if pd.notna(row[r_col_comp]) else 0
                        m_act = row[m_col_comp] if pd.notna(row[m_col_comp]) else 0
                        q_act = m_act - r_act
                        
                        # Demand and cost are required
                        demand = row['AdjustedAnnualSales']
                        unit_cost = row['COST']
                        
                        if q_act > 0 and demand > 0:
                            # Current costs
                            avg_inv_act = (r_act + m_act) / 2
                            current_h_cost = avg_inv_act * (unit_cost * holding_cost_pct)
                            current_o_cost = (demand / q_act) * fixed_order_cost
                            current_orders = demand / q_act
                            
                            # Optimized costs
                            Q_opt = row['Recommended_Order']
                            if Q_opt > 0:
                                opt_h_cost = ((Q_opt / 2) + row.get('Safety_Stock', 0)) * (unit_cost * holding_cost_pct)
                                opt_o_cost = (demand / Q_opt) * fixed_order_cost
                                opt_orders = demand / Q_opt
                                
                                savings = (current_h_cost + current_o_cost) - (opt_h_cost + opt_o_cost)
                                
                                # Try to find a descriptive name
                                wine_name = None
                                name_priority = ['PRODUCT', 'Wine', 'Description', 'Item', 'Wine Name', 'Product']
                                for col_name in name_priority:
                                    if col_name in row and pd.notna(row[col_name]):
                                        wine_name = str(row[col_name])
                                        break
                                
                                if wine_name is None:
                                    # Fallback to case-insensitive
                                    for col_name in row.index:
                                        if col_name.lower() in [p.lower() for p in name_priority]:
                                            if pd.notna(row[col_name]):
                                                wine_name = str(row[col_name])
                                                break
                                
                                if wine_name is None:
                                    wine_name = f"Item {idx}"
                                
                                # Use Item ID/Index as a prefix to ensure uniqueness in Plotly, but keep it subtle
                                unique_label = f"{wine_name} ({idx})"
                                
                                comparison_data.append({
                                    'Wine': unique_label,
                                    'Display Name': wine_name,
                                    'Current Orders': current_orders,
                                    'Optimized Orders': opt_orders,
                                    'Savings': savings
                                })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    # Keep top 5 by savings
                    top_5 = comp_df.nlargest(5, 'Savings')
                    
                    fig_freq = go.Figure()
                    
                    fig_freq.add_trace(go.Bar(
                        x=top_5['Wine'],
                        y=top_5['Current Orders'],
                        name='Actual Orders/Yr',
                        marker_color='#d62728', # Reddish
                        opacity=0.8,
                        customdata=top_5['Display Name'],
                        hovertemplate="<b>%{customdata}</b><br>Actual: %{y:.1f} orders/yr<extra></extra>"
                    ))
                    
                    fig_freq.add_trace(go.Bar(
                        x=top_5['Wine'],
                        y=top_5['Optimized Orders'],
                        name='Optimized Orders/Yr',
                        marker_color='#1f77b4', # Blueish
                        opacity=0.8,
                        customdata=top_5['Display Name'],
                        hovertemplate="<b>%{customdata}</b><br>Optimized: %{y:.1f} orders/yr<extra></extra>"
                    ))
                    
                    # Add annotations for savings, centred over the Actual Orders (red) bars
                    for i, row in top_5.iterrows():
                        fig_freq.add_annotation(
                            x=row['Wine'],
                            y=row['Current Orders'],
                            text=f"Save:<br>${row['Savings']:,.0f}",
                            showarrow=False,
                            font=dict(color='black', size=10, weight="bold"),
                            yshift=20,
                            xshift=-20
                        )
                    
                    fig_freq.update_layout(
                        title="Reduction in Order Frequency of Top 5 Cost-Saving Wines",
                        xaxis_title="Wine",
                        yaxis_title="Orders Per Year",
                        barmode='group',
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        plot_bgcolor='white',
                        margin=dict(t=80) # Space for annotations
                    )
                    
                    # Clean up X-axis labels to show only the wine name, even if IDs are unique underneath
                    fig_freq.update_xaxes(
                        tickvals=top_5['Wine'],
                        ticktext=top_5['Display Name'],
                        showgrid=False
                    )
                    
                    fig_freq.update_yaxes(showgrid=True, gridcolor='lightgray')
                    
                    st.plotly_chart(fig_freq, width='stretch')
    else:
        st.warning("Objective value not available. Please re-run the optimization.")
    
    # Sensitivity Analysis Results Section
    # Build ordered list of available analyses
    sens_order = [
        ('service_level', 'Service Level'),
        ('fixed_order_cost', 'Fixed Order Cost'),
        ('holding_cost_pct', 'Holding Cost'),
        ('storage_capacity', 'Storage Capacity'),
    ]
    available_sens = [(key, label) for key, label in sens_order
                       if f'sensitivity_results_{key}' in st.session_state]
    
    if available_sens:
        st.divider()
        st.subheader("Sensitivity Analysis Results")
        
        # Create tabs if 2+ analyses, otherwise use a plain container
        if len(available_sens) >= 2:
            sens_tabs = st.tabs([label for _, label in available_sens])
            sens_containers = {key: sens_tabs[i] for i, (key, _) in enumerate(available_sens)}
        else:
            sens_containers = {available_sens[0][0]: st.container()}
        
        # Fixed Order Cost Sensitivity Analysis
        if 'fixed_order_cost' in sens_containers:
            with sens_containers['fixed_order_cost']:
                df_sens = st.session_state.sensitivity_results_fixed_order_cost
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Total Cost vs. Ordering Cost", "Holding vs. Ordering Cost Trade-off"),
                    horizontal_spacing=0.15,
                    vertical_spacing=0.15
                )
                
                # Panel 1: Total Cost
                fig.add_trace(
                    go.Scatter(
                        x=df_sens['S_Value'],
                        y=df_sens['Total_Cost'],
                        mode='lines+markers',
                        name='Total Cost',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Add baseline vertical line
                baseline_label_p1 = None
                if 'sensitivity_baseline_fixed_order_cost' in st.session_state:
                    baseline = st.session_state.sensitivity_baseline_fixed_order_cost
                    baseline_label_p1 = f'Baseline S=${baseline:.0f}'
                    fig.add_vline(
                        x=baseline,
                        line_dash="dash",
                        line_color="grey",
                        line_width=2,
                        opacity=0.7,
                        row=1, col=1
                    )
                
                fig.update_xaxes(title_text="Ordering Cost S ($)", row=1, col=1, showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(title_text="Total Annual Cost ($)", row=1, col=1, showgrid=True, gridcolor='lightgray')
                
                # Panel 2: Cost Components
                fig.add_trace(
                    go.Scatter(
                        x=df_sens['S_Value'],
                        y=df_sens['Holding_Cost'],
                        mode='lines+markers',
                        name='Holding Cost',
                        line=dict(color='green', width=2),
                        marker=dict(symbol='square', size=8),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_sens['S_Value'],
                        y=df_sens['Ordering_Cost'],
                        mode='lines+markers',
                        name='Ordering Cost',
                        line=dict(color='red', width=2),
                        marker=dict(symbol='triangle-up', size=8),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # Add baseline vertical line (panel 2)
                if 'sensitivity_baseline_fixed_order_cost' in st.session_state:
                    baseline = st.session_state.sensitivity_baseline_fixed_order_cost
                    fig.add_vline(
                        x=baseline,
                        line_dash="dash",
                        line_color="gray",
                        opacity=0.5,
                        row=1, col=2
                    )
                
                fig.update_xaxes(title_text="Ordering Cost S ($)", row=1, col=2, showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(title_text="Cost Component ($)", row=1, col=2, showgrid=True, gridcolor='lightgray')
                
                # Annotations: baseline in left panel, cost component legend in top-left of right panel
                annotations = list(fig.layout.annotations) if fig.layout.annotations else []
                if baseline_label_p1:
                    annotations.append(
                        dict(
                            x=0.01, y=0.98,
                            xref='paper', yref='paper',
                            text=f'<span style="color:grey;">━━</span> {baseline_label_p1}',
                            showarrow=False,
                            xanchor='left',
                            yanchor='top',
                            font=dict(size=12),
                            bgcolor='white',
                            bordercolor='lightgray',
                            borderwidth=1,
                            borderpad=4
                        )
                    )
                # Ordering Cost on top, Holding Cost below — top-left of right panel
                annotations.append(
                    dict(
                        x=0.58, y=0.95,
                        xref='paper', yref='paper',
                        text='<span style="color:red;">━━▲</span> Ordering Cost<br><span style="color:green;">━━■</span> Holding Cost',
                        showarrow=False,
                        xanchor='left',
                        yanchor='top',
                        font=dict(size=12),
                        bgcolor='white',
                        bordercolor='lightgray',
                        borderwidth=1,
                        borderpad=4
                    )
                )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=False,
                    annotations=annotations,
                    margin=dict(t=60, b=80, l=60, r=60)
                )
                
                st.plotly_chart(fig, width='stretch')
        
        # Storage Capacity Sensitivity Analysis
        if 'storage_capacity' in sens_containers:
            with sens_containers['storage_capacity']:
                df_sens = st.session_state.sensitivity_results_storage_capacity
                
                # Filter for Optimal solutions only
                df_plot = df_sens[df_sens['Status'] == 'Optimal'].copy() if 'Status' in df_sens.columns else df_sens
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Value of Extra Capacity (RHS Sensitivity)", "Warehouse Utilization"),
                    horizontal_spacing=0.15,
                    vertical_spacing=0.15
                )
                
                # Panel 1: Total Cost
                fig.add_trace(
                    go.Scatter(
                        x=df_plot['V_max'],
                        y=df_plot['Total_Cost'],
                        mode='lines+markers',
                        name='Total Cost',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Add baseline vertical line with centred annotation
                if 'sensitivity_baseline_storage_capacity' in st.session_state:
                    baseline = st.session_state.sensitivity_baseline_storage_capacity
                    fig.add_vline(
                        x=baseline,
                        line_dash="dash",
                        line_color="grey",
                        line_width=2,
                        opacity=0.7,
                        row=1, col=1
                    )
                
                fig.update_xaxes(title_text="Storage Capacity (bottles)", row=1, col=1, showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(title_text="Total Annual Cost ($)", row=1, col=1, showgrid=True, gridcolor='lightgray')
                
                # Panel 2: Storage Utilization (Bar chart)
                fig.add_trace(
                    go.Bar(
                        x=df_plot['V_max'].astype(str),
                        y=df_plot['Storage_%'],
                        name='Utilization',
                        marker_color='steelblue',
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # Add 100% horizontal line
                fig.add_hline(
                    y=100,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.7,
                    annotation_text='Full Capacity',
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Storage Capacity (bottles)", row=1, col=2, showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(title_text="Utilization (%)", row=1, col=2, showgrid=True, gridcolor='lightgray')
                
                # Centre baseline annotation over the vertical line
                annotations = list(fig.layout.annotations) if fig.layout.annotations else []
                if 'sensitivity_baseline_storage_capacity' in st.session_state:
                    baseline = st.session_state.sensitivity_baseline_storage_capacity
                    annotations.append(
                        dict(
                            x=baseline,
                            y=0.95,
                            xref='x',
                            yref='paper',
                            text=f'Baseline V = {baseline}',
                            showarrow=False,
                            xanchor='center',
                            yanchor='top',
                            font=dict(size=12, color='grey'),
                            bgcolor='white',
                            borderpad=2
                        )
                    )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=False,
                    annotations=annotations,
                    margin=dict(t=60, b=80, l=60, r=60)
                )
                
                st.plotly_chart(fig, width='stretch')
        
        # Service Level Sensitivity Analysis
        if 'service_level' in sens_containers:
            with sens_containers['service_level']:
                df_sens = st.session_state.sensitivity_results_service_level
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Total Cost vs. Service Level", "Required Safety Stock Inventory"),
                    horizontal_spacing=0.15,
                    vertical_spacing=0.15
                )
                
                # Panel 1: Total Cost
                fig.add_trace(
                    go.Scatter(
                        x=df_sens['Confidence'],
                        y=df_sens['Total_Cost'],
                        mode='lines+markers',
                        name='Total Cost',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Add baseline vertical line
                baseline_label_sl = None
                if 'sensitivity_baseline_service_level' in st.session_state:
                    baseline = st.session_state.sensitivity_baseline_service_level
                    baseline_label_sl = f'Baseline {baseline:.1f}%'
                    fig.add_vline(
                        x=baseline,
                        line_dash="dash",
                        line_color="grey",
                        line_width=2,
                        opacity=0.7,
                        row=1, col=1
                    )
                
                fig.update_xaxes(title_text="Target Service Level (%)", row=1, col=1, showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(title_text="Total Annual Cost ($)", row=1, col=1, showgrid=True, gridcolor='lightgray')
                
                # Panel 2: Safety Stock (Bar chart)
                fig.add_trace(
                    go.Bar(
                        x=df_sens['Confidence'].astype(str),
                        y=df_sens['Total_SS_Bottles'],
                        name='Safety Stock',
                        marker_color='steelblue',
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Target Service Level (%)", row=1, col=2, showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(title_text="Total Safety Stock (Bottles)", row=1, col=2, showgrid=True, gridcolor='lightgray')
                
                # Combined baseline + Total Cost legend annotation inside left panel
                annotations = list(fig.layout.annotations) if fig.layout.annotations else []
                legend_text = '<span style="color:blue;">━━●</span> Total Cost'
                if baseline_label_sl:
                    legend_text = f'<span style="color:grey;">━━</span> {baseline_label_sl}<br>{legend_text}'
                annotations.append(
                    dict(
                        x=0.01, y=0.98,
                        xref='paper', yref='paper',
                        text=legend_text,
                        showarrow=False,
                        xanchor='left',
                        yanchor='top',
                        font=dict(size=12),
                        bgcolor='white',
                        bordercolor='lightgray',
                        borderwidth=1,
                        borderpad=4
                    )
                )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=False,
                    annotations=annotations,
                    margin=dict(t=60, b=80, l=60, r=60)
                )
                
                st.plotly_chart(fig, width='stretch')
        
        # Holding Cost Sensitivity Analysis
        if 'holding_cost_pct' in sens_containers:
            with sens_containers['holding_cost_pct']:
                df_sens = st.session_state.sensitivity_results_holding_cost_pct
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Total Cost vs. Holding Cost %", "Holding vs. Ordering Cost Trade-off"),
                    horizontal_spacing=0.15,
                    vertical_spacing=0.15
                )
                
                # Panel 1: Total Cost
                fig.add_trace(
                    go.Scatter(
                        x=df_sens['Holding_Cost_Pct'],
                        y=df_sens['Total_Cost'],
                        mode='lines+markers',
                        name='Total Cost',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Add baseline vertical line
                baseline_label_hc = None
                if 'sensitivity_baseline_holding_cost_pct' in st.session_state:
                    baseline = st.session_state.sensitivity_baseline_holding_cost_pct
                    baseline_label_hc = f'Baseline {baseline:.1f}%'
                    fig.add_vline(
                        x=baseline,
                        line_dash="dash",
                        line_color="grey",
                        line_width=2,
                        opacity=0.7,
                        row=1, col=1
                    )
                
                fig.update_xaxes(title_text="Holding Cost %", row=1, col=1, showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(title_text="Total Annual Cost ($)", row=1, col=1, showgrid=True, gridcolor='lightgray')
                
                # Panel 2: Cost Components
                fig.add_trace(
                    go.Scatter(
                        x=df_sens['Holding_Cost_Pct'],
                        y=df_sens['Holding_Cost'],
                        mode='lines+markers',
                        name='Holding Cost',
                        line=dict(color='green', width=2),
                        marker=dict(symbol='square', size=8),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_sens['Holding_Cost_Pct'],
                        y=df_sens['Ordering_Cost'],
                        mode='lines+markers',
                        name='Ordering Cost',
                        line=dict(color='red', width=2),
                        marker=dict(symbol='triangle-up', size=8),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # Add baseline vertical line (panel 2)
                if 'sensitivity_baseline_holding_cost_pct' in st.session_state:
                    baseline = st.session_state.sensitivity_baseline_holding_cost_pct
                    fig.add_vline(
                        x=baseline,
                        line_dash="dash",
                        line_color="gray",
                        opacity=0.5,
                        row=1, col=2
                    )
                
                fig.update_xaxes(title_text="Holding Cost %", row=1, col=2, showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(title_text="Cost Component ($)", row=1, col=2, showgrid=True, gridcolor='lightgray')
                
                # Annotations: baseline in left panel, cost component legend in bottom-right of right panel
                annotations = list(fig.layout.annotations) if fig.layout.annotations else []
                if baseline_label_hc:
                    annotations.append(
                        dict(
                            x=0.01, y=0.98,
                            xref='paper', yref='paper',
                            text=f'<span style="color:grey;">━━</span> {baseline_label_hc}',
                            showarrow=False,
                            xanchor='left',
                            yanchor='top',
                            font=dict(size=12),
                            bgcolor='white',
                            bordercolor='lightgray',
                            borderwidth=1,
                            borderpad=4
                        )
                    )
                # Holding Cost on top, Ordering Cost below — bottom-right of right panel
                annotations.append(
                    dict(
                        x=0.99, y=0.05,
                        xref='paper', yref='paper',
                        text='<span style="color:green;">━━■</span> Holding Cost<br><span style="color:red;">━━▲</span> Ordering Cost',
                        showarrow=False,
                        xanchor='right',
                        yanchor='bottom',
                        font=dict(size=12),
                        bgcolor='white',
                        bordercolor='lightgray',
                        borderwidth=1,
                        borderpad=4
                    )
                )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=False,
                    annotations=annotations,
                    margin=dict(t=60, b=80, l=60, r=60)
                )
                
                st.plotly_chart(fig, width='stretch')
    
    # Navigation buttons at the bottom
    st.divider()
    col1, col_space, col2 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("Back", width='stretch'):
            st.switch_page("pages/solver_ui.py")

    with col_space:
        pass
    
    with col2:
        if st.button("View Methodology", width='stretch'):
            st.switch_page("pages/methodology.py")
