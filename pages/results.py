import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.solver_logic import calculate_cost_components, calculate_current_policy_costs

st.title("Optimization Results")

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
        
        # KPI 1: Total Annual Cost
        with kpi_cols[0]:
            # Calculate delta if current policy data is available
            delta_value = None
            if show_comparison_kpis:
                current_total_cost, _, _ = current_policy_costs
                # Delta = optimized - current (negative means cost decreased = savings)
                delta_value = round(objective_value - current_total_cost,2)
            
            st.metric(
                "Total Annual Cost",
                f"${round(objective_value):,}",
                delta=delta_value,
                delta_color="inverse" if delta_value is not None else "normal",
                help="The minimized total annual cost from the optimization solver"
            )
        
        # KPI 2 & 3: Comparison KPIs (only show if current policy data available)
        if show_comparison_kpis:
            current_total_cost, current_holding_cost, current_ordering_cost = current_policy_costs
            
            # Calculate percentage savings
            total_cost_savings_pct = ((current_total_cost - objective_value) / current_total_cost * 100) if current_total_cost > 0 else 0
            
            # Calculate admin cost reduction percentage
            admin_reduction_pct = ((current_ordering_cost - opt_ordering_cost) / current_ordering_cost * 100) if current_ordering_cost > 0 else 0
            
            with kpi_cols[1]:
                st.metric(
                    "Total Cost Saved",
                    f"{total_cost_savings_pct:.1f}%",
                    help="Percentage reduction in total annual cost compared to current policy"
                )
            
            with kpi_cols[2]:
                st.metric(
                    "Admin Cost Reduction",
                    f"{admin_reduction_pct:.1f}%",
                    help="Percentage reduction in ordering/admin costs compared to current policy"
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
        
        # Comparative Visualizations Section
        if show_comparison_kpis:
            st.divider()
            st.subheader("Comparative Analysis: Current vs. Optimized")
            
            # --- GRAPH 1: Current vs Optimized Annual Inventory Costs ---
            current_total_cost, current_holding_cost, current_ordering_cost = current_policy_costs
            
            fig_costs = go.Figure()
            
            categories = ['Holding Cost', 'Ordering Cost', 'Total Cost']
            current_vals = [current_holding_cost, current_ordering_cost, current_total_cost]
            opt_vals = [opt_holding_cost, opt_ordering_cost, objective_value]
            
            fig_costs.add_trace(go.Bar(
                x=categories,
                y=current_vals,
                name='Current (Skyline)',
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
            
            st.plotly_chart(fig_costs, use_container_width=True)
            
        # --- GRAPH 2: Reduction in Order Frequency of Top 5 Cost-Saving Wines ---
        # Recalculate per-wine savings to find top 5
        comparison_data = []
        
        # Detect current policy columns (case-insensitive and common variations)
        r_col = None
        m_col = None
        
        for col in results_df.columns:
            col_lower = col.lower()
            if col_lower in ['reorder point', 'reorder_point', 'rop', 'rp']:
                r_col = col
            if col_lower in ['max level', 'max_level', 'max_inv', 'max']:
                m_col = col
            
        if r_col and m_col:
            for idx, row in results_df.iterrows():
                r_act = row[r_col] if pd.notna(row[r_col]) else 0
                m_act = row[m_col] if pd.notna(row[m_col]) else 0
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
                
                # Add annotations for savings
                for i, row in top_5.iterrows():
                    fig_freq.add_annotation(
                        x=row['Wine'],
                        y=max(row['Current Orders'], row['Optimized Orders']),
                        text=f"Save:<br>${row['Savings']:,.0f}",
                        showarrow=False,
                        font=dict(color="green", size=10, weight="bold"),
                        yshift=20
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
                
                st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.warning("Objective value not available. Please re-run the optimization.")
    
    # Sensitivity Analysis Results Section
    st.divider()
    
    # Check for sensitivity analysis results
    sensitivity_results_available = False
    if 'sensitivity_results_fixed_order_cost' in st.session_state:
        sensitivity_results_available = True
    if 'sensitivity_results_storage_capacity' in st.session_state:
        sensitivity_results_available = True
    if 'sensitivity_results_service_level' in st.session_state:
        sensitivity_results_available = True
    if 'sensitivity_results_holding_cost_pct' in st.session_state:
        sensitivity_results_available = True
    
    if sensitivity_results_available:
        st.subheader("Sensitivity Analysis Results")
        
        # Fixed Order Cost Sensitivity Analysis
        if 'sensitivity_results_fixed_order_cost' in st.session_state:
            st.markdown("### Fixed Order Cost Sensitivity Analysis")
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
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add baseline vertical line (hidden from legend, will show in left annotation)
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
                    showlegend=True
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
                    showlegend=True
                ),
                row=1, col=2
            )
            
            # Add baseline vertical line (no legend entry for panel 2)
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
            
            # Create annotations for left-side baseline legend
            # Preserve existing annotations (subplot titles) and add baseline legend
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
            
            fig.update_layout(
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="right", x=1),
                annotations=annotations,
                margin=dict(t=60, b=80, l=60, r=60)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Storage Capacity Sensitivity Analysis
        if 'sensitivity_results_storage_capacity' in st.session_state:
            st.markdown("### Storage Capacity Sensitivity Analysis")
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
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add baseline vertical line (hidden from legend, will show in left annotation)
            baseline_label_sc = None
            if 'sensitivity_baseline_storage_capacity' in st.session_state:
                baseline = st.session_state.sensitivity_baseline_storage_capacity
                baseline_label_sc = f'Baseline V={baseline}'
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
            
            # Create annotations for left-side baseline legend
            # Preserve existing annotations (subplot titles) and add baseline legend
            annotations = list(fig.layout.annotations) if fig.layout.annotations else []
            if baseline_label_sc:
                annotations.append(
                    dict(
                        x=0.01, y=0.98,
                        xref='paper', yref='paper',
                        text=f'<span style="color:grey;">━━</span> {baseline_label_sc}',
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
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="right",
                    x=1
                ),
                annotations=annotations,
                margin=dict(t=60, b=80, l=60, r=60)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Service Level Sensitivity Analysis
        if 'sensitivity_results_service_level' in st.session_state:
            st.markdown("### Service Level Sensitivity Analysis")
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
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add baseline vertical line (hidden from legend, will show in left annotation)
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
            
            # Create annotations for left-side baseline legend
            # Preserve existing annotations (subplot titles) and add baseline legend
            annotations = list(fig.layout.annotations) if fig.layout.annotations else []
            if baseline_label_sl:
                annotations.append(
                    dict(
                        x=0.01, y=0.98,
                        xref='paper', yref='paper',
                        text=f'<span style="color:grey;">━━</span> {baseline_label_sl}',
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
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="right",
                    x=1
                ),
                annotations=annotations,
                margin=dict(t=60, b=80, l=60, r=60)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Holding Cost Sensitivity Analysis
        if 'sensitivity_results_holding_cost_pct' in st.session_state:
            st.markdown("### Holding Cost Sensitivity Analysis")
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
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add baseline vertical line (hidden from legend, will show in left annotation)
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
                    showlegend=True
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
                    showlegend=True
                ),
                row=1, col=2
            )
            
            # Add baseline vertical line (no legend entry for panel 2)
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
            
            # Create annotations for left-side baseline legend
            # Preserve existing annotations (subplot titles) and add baseline legend
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
            
            fig.update_layout(
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="right", x=1),
                annotations=annotations,
                margin=dict(t=60, b=80, l=60, r=60)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
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
