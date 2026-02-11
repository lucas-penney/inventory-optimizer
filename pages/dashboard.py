import streamlit as st
from utils.ui_components import get_shared_page_styles, page_title

# Initialize session state variables
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Page configuration with custom styling
st.markdown(get_shared_page_styles(), unsafe_allow_html=True)

# Authentication Section
if st.session_state.user_type is None:
    page_title("Wine Inventory Optimizer")
    st.markdown("### Welcome")
    img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
    with img_col2:
        st.image("images/inventory_icon.png", width='stretch')
    st.markdown("The application can be viewed in a generic demo mode or using client-specific data.")
    st.markdown("Please select your access level to continue:")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_selection = st.radio(
            "Access Type",
            options=["Guest", "Client"],
            label_visibility="collapsed",
            horizontal=True
        )
        
        if user_selection == "Client":
            password = st.text_input(
                "Enter Client Password",
                type="password",
                key="client_password_input"
            )
            
            password_expected = (st.secrets.get("executive_summary") or {}).get("password")
            if not password_expected:
                st.info("Client access is not available. Please sign in as a guest to use the app with demo data.")
                if st.button("Back", width='stretch'):
                    st.session_state.user_type = None
                    st.rerun()
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Submit", type="primary", width='stretch'):
                        if password == password_expected:
                            st.session_state.user_type = "client"
                            st.session_state.authenticated = True
                            st.rerun()
                        else:
                            st.error("Incorrect password. Please try again.")
                with col_b:
                    if st.button("Back", width='stretch'):
                        st.session_state.user_type = None
                        st.rerun()
        else:
            if st.button("Continue as Guest", type="primary", width='stretch'):
                st.session_state.user_type = "guest"
                st.session_state.authenticated = True
                st.rerun()

else:
    # Main Executive Summary Content
    page_title("Executive Summary")

    # Project Context Section
    st.markdown('<p class="section-header">Project Context</p>', unsafe_allow_html=True)
    st.markdown("""
    Wine inventory management presents a complex optimization challenge balancing competing objectives. 
    Restaurants and wine bars must maintain sufficient stock to meet customer demand while minimizing the holding 
    costs of excess inventory. Overstocking ties up working capital and wastes valuable storage space, 
    while understocking leads to lost sales and damaged customer experience. Traditional inventory 
    management relies on experience and intuition, but this approach struggles with diverse product 
    portfolios, variable demand patterns, and complex supplier constraints. This tool applies quantitative 
    optimization techniques to provide data-driven reorder policies that minimize total costs while 
    respecting operational realities.
    """)
    
    # Solution Section
    st.markdown('<p class="section-header">Solution</p>', unsafe_allow_html=True)
    st.markdown("""
    The Wine Inventory Optimizer transforms raw operational data into actionable inventory policies 
    through a sophisticated two-stage process. First, the data ingestion pipeline processes historical 
    sales records, product costs, supplier lead times, and business constraints to calculate demand rates, 
    variability, and safety stock requirements. Second, the optimization engine formulates the inventory 
    problem as a Mixed-Integer Linear Program (MILP) that determines optimal reorder points and order 
    quantities for each wine. The mathematical model incorporates Economic Order Quantity (EOQ) principles, 
    balancing holding costs against ordering costs while respecting capacity constraints and supplier 
    minimum order quantities. The tool provides sensitivity analysis capabilities, allowing users to 
    understand how changes in cost parameters, service levels, or capacity constraints affect optimal decisions.
    """)
    
    # Benefits Section
    st.markdown('<p class="section-header">Benefits</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="benefit-card">
            <div class="benefit-title">Cost Savings</div>
            <p>Reduce total inventory costs through optimized ordering strategies. The model minimizes 
            the combined impact of holding costs and ordering costs, typically achieving 30-40% reductions 
            in total annual inventory expenses.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="benefit-card">
            <div class="benefit-title">Space Management</div>
            <p>Maximize efficient use of limited storage capacity by intelligently allocating space to 
            high-velocity items. The optimizer considers physical constraints and prioritizes inventory 
            decisions that deliver the greatest return per bottle stored.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="benefit-card">
            <div class="benefit-title">Risk Reduction</div>
            <p>Maintain high service levels while minimizing stockout risks through calculated safety 
            stock policies. The model balances the cost of carrying buffer inventory against the 
            probability and impact of running out of critical products.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional benefits row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="benefit-card">
            <div class="benefit-title">Reduced Administrative Burden</div>
            <p>Decrease order frequency and streamline purchasing operations by placing larger, 
            less frequent orders. This frees up valuable time for strategic activities like 
            supplier development and quality management.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="benefit-card">
            <div class="benefit-title">Improved Cash Flow</div>
            <p>Optimize working capital allocation by reducing excess inventory levels while 
            maintaining operational effectiveness. Better inventory policies translate directly 
            to improved cash flow and financial flexibility.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Future Extensions Section (Client-only)
    if st.session_state.user_type == "client":
        st.markdown('<p class="section-header">Future Extensions</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="future-extension-card">
            <div class="extension-title">Database Migration & Scalability</div>
            <p>Transition from CSV-based data management to a centralized PostgreSQL relational database 
            system. This migration will standardize product naming conventions across categories, eliminate 
            manual data cleaning bottlenecks, and enable seamless scaling of optimization logic to other 
            product categories including spirits, beer and food. A unified database architecture creates a 
            single source of truth for all inventory operations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="future-extension-card">
            <div class="extension-title">Enhanced Cost Parameter Estimation</div>
            <p>Collaborate with the accounting department to develop precise estimates of holding costs 
            and administrative order costs based on actual financial data. Current estimates rely on 
            industry benchmarks and management consultation, but access to detailed cost accounting 
            data will improve model accuracy. More precise cost parameters lead to more accurate 
            optimization results and greater realized savings.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="future-extension-card">
            <div class="extension-title">Real-Time Data Integration</div>
            <p>Integrate daily sales data feeds to enable dynamic reorder point adjustments based on 
            current demand patterns. Rather than relying solely on historical averages, the system 
            would continuously update demand forecasts and safety stock requirements. This real-time 
            approach improves responsiveness to demand shifts, seasonal variations, and emerging 
            trends in customer preferences.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation Section
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col3:
        if st.button("Configure Optimization →", type="primary", width='stretch'):
            st.switch_page("pages/solver_ui.py")
