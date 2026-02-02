import streamlit as st
import matplotlib.pyplot as plt

from utils.ui_components import build_etl_flowchart, page_title


# Page configuration with custom styling
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1e7b34;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #2c5aa0;
        text-align: center;
        margin-bottom: 3rem;
        font-style: italic;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e7b34;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2c5aa0;
        padding-bottom: 0.5rem;
    }
    .methodology-card {
        background-color: #f0f8ff;
        border-left: 4px solid #2c5aa0;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .methodology-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e7b34;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e8f5e9;
        border-left: 4px solid #1e7b34;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Page Title
page_title("Methodology & Pipeline")

# Section 1: ETL Pipeline Diagram
st.markdown('<p class="section-header">1. ETL Pipeline</p>', unsafe_allow_html=True)

st.markdown("""
The flowchart below documents the ETL process implemented in `scripts/local_etl.py`. Three raw .xslx files are ingested and the result is a  clean .csv file for input into the PuLP optimization solver.
""")

# ETL flowchart
etl_fig = build_etl_flowchart()
st.pyplot(etl_fig)
plt.close(etl_fig)

# Section 2: Data Anonymization
st.markdown('<p class="section-header">2. Data Anonymization</p>', unsafe_allow_html=True)

st.markdown("""
Documentation of how client data is anonymized for demo use.
""")

# Placeholder for data optimization content
st.info("**Coming soon** — Data anonymization flow chart.")

# Section 3: Mathematical Model
st.markdown('<p class="section-header">3. Mathematical Model</p>', unsafe_allow_html=True)

st.markdown("""
The inventory optimization problem is formulated as a Mixed-Integer Linear Program (MILP) that determines optimal reorder points and order quantities for each wine SKU.
""")

with st.container(border=True):
    st.markdown('<p class="methodology-title">Problem Structure</p>', unsafe_allow_html=True)
    st.markdown("""
    The inventory system operates under a **continuous review (R, M) policy**:
    - **R** (Reorder Point): Inventory level that triggers a new order
    - **M** (Order-Up-To Level): Target inventory level after replenishment
    - **Q = M - R**: Order quantity (gap between M and R)
    
    When inventory drops to or below R, an order is placed to bring inventory up to level M.
    """)

with st.container(border=True):
    st.markdown('<p class="methodology-title">Sets and Indices</p>', unsafe_allow_html=True)
    st.markdown("""
    - $I$: Set of all wines in the portfolio, indexed by $i \\in \\{1, 2, \\ldots, n\\}$
    - $K_i$: Set of valid order quantity options for wine $i$, indexed by $k$
    - $W$: Set of direct wineries from which the company can order
    - $I_w \\subseteq I$: Subset of wines sourced from winery $w \\in W$
    """)

with st.container(border=True):
    st.markdown('<p class="methodology-title">Parameters</p>', unsafe_allow_html=True)
    st.markdown(r"""
    **Demand and Lead Time Parameters:**
    - $D_i$: Annual demand for wine $i$ (bottles per year)
    - $\mu_{LT,i}$: Mean demand during lead time for wine $i$
    - $\sigma_{LT,i}$: Standard deviation of demand during lead time for wine $i$

    **Cost Parameters:**
    - $H_i$: Annual holding cost per bottle for wine $i$ (calculated as 25% of unit cost)
    - $S$: Fixed administrative cost per order (\$50)

    **Capacity and Policy Parameters:**
    - $V_{\max}$: Maximum storage capacity (bottles), configured in the optimizer
    - $V_i$: Storage space required per bottle of wine $i$ (assumed to be 1 for all wines)
    - $Z_{LT,i}$: Standard normal quantile for the target service level for wine $i$ (lead-time context; $Z_{LT,i} \approx 1.645$ for 95%)
    - $SS_i$: Safety stock for wine $i$, calculated as $SS_i = Z_{LT,i} \times \sigma_{LT,i}$

    **Pre-computed Values:**
    - $Q_{ik}$: The order quantity associated with option $k$ for wine $i$
    - $C_{ik}$: The pre-computed total annual cost for wine $i$ if option $k$ is selected
    """)

with st.container(border=True):
    st.markdown('<p class="methodology-title">Decision Variables</p>', unsafe_allow_html=True)
    st.markdown("""
    - $y_{ik} \\in \\{0, 1\\}$: Binary variable equal to 1 if order quantity option $k$ is selected for wine $i$, and 0 otherwise
    - $R_i \\geq 0$: Integer variable representing the reorder point for wine $i$
    - $M_i \\geq 0$: Integer variable representing the order-up-to level for wine $i$
    - $z_w \\in \\{0, 1\\}$: Binary variable equal to 1 if winery $w$ is active (i.e., at least one wine is ordered from winery $w$), and 0 otherwise
    """)

with st.container(border=True):
    st.markdown('<p class="methodology-title">Objective Function</p>', unsafe_allow_html=True)
    st.latex(r"""
    \text{Minimize } Z = \sum_{i \in I} \sum_{k \in K_i} C_{ik} \times y_{ik}
    """)
    st.markdown("""
    The objective is to minimize the total annual inventory cost across all wines. Since exactly one option k must be selected for each wine i, this objective sums the costs of the selected order quantity options.
    """)

with st.container(border=True):
    st.markdown('<p class="methodology-title">Constraints</p>', unsafe_allow_html=True)
    st.markdown("""
    **Constraint 1 – Single Choice:**
    """)
    st.latex(r"""
    \sum_{k \in K_i} y_{ik} = 1 \quad \text{for all } i \in I
    """)
    st.markdown("For each wine, exactly one order quantity option must be selected.")
    
    st.markdown("""
    **Constraint 2 – Order Quantity Linking:**
    """)
    st.latex(r"""
    M_i - R_i = \sum_{k \in K_i} Q_{ik} \times y_{ik} \quad \text{for all } i \in I
    """)
    st.markdown("The order quantity (gap between order-up-to level and reorder point) must equal the selected option.")
    
    st.markdown("""
    **Constraint 3 – Safety Stock Requirement:**
    """)
    st.latex(r"""
    R_i \geq 0.5 \times SS_i + \mu_{LT,i} \quad \text{for all } i \in I
    """)
    st.markdown("The reorder point must be sufficient to cover expected demand during lead time plus a portion of the safety stock to maintain the target service level.")
    
    st.markdown("""
    **Constraint 4 – Storage Capacity:**
    """)
    st.latex(r"""
    \sum_{i \in I} M_i \times V_i \leq V_{\max}
    """)
    st.markdown("The total inventory at its maximum level cannot exceed the warehouse capacity.")
    
    st.markdown("""
    **Constraint 5 – Non-negativity and Integrality:**
    """)
    st.latex(r"""
    R_i, M_i \geq 0 \text{ and integer for all } i \in I \\
    y_{ik} \in \{0, 1\} \text{ for all } i \in I, k \in K_i \\
    z_w \in \{0, 1\} \text{ for all } w \in W
    """)
    
    st.markdown("""
    **Constraints 6-7 – Big-M Formulation for Winery Minimum Order Quantities:**
    """)
    st.latex(r"""
    \sum_{i \in I_w} \sum_{k \in K_i} Q_{ik} \times y_{ik} \geq 60 \times z_w \quad \text{for all } w \in W
    """)
    st.latex(r"""
    \sum_{i \in I_w} \sum_{k \in K_i} Q_{ik} \times y_{ik} \leq M_{big} \times z_w \quad \text{for all } w \in W
    """)
    st.markdown(r"""
    These constraints ensure that for each direct winery, the company either orders at least 60 bottles total or orders nothing at all. $M_{\mathrm{big}}$ is set equal to $V_{\max}$ to strengthen the linear relaxation.
    """)

with st.container(border=True):
    st.markdown('<p class="methodology-title">Cost Calculation Formula</p>', unsafe_allow_html=True)
    st.latex(r"""
    C_{ik} = \left(\frac{Q_{ik}}{2} + SS_i\right) \times H_i + \frac{D_i}{Q_{ik}} \times S
    """)
    st.markdown("""
    Where:
    - **First term**: Annual holding cost = (Average cycle stock + Safety stock) × Unit holding cost
    - **Second term**: Annual ordering cost = (Annual demand / Order quantity) × Fixed cost per order
    
    The average cycle stock is $Q_{ik}/2$ because inventory oscillates between 0 and $Q_{ik}$ under the (R, M) policy.
    """)

st.markdown("---")

# Section 4: Optimization Solver
st.markdown('<p class="section-header">4. Optimization Solver: Gurobi vs PuLP</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="methodology-card">
        <div class="methodology-title">Gurobi Optimizer</div>
        <p><strong>Original Implementation</strong></p>
        <ul>
            <li>Industry standard commercial solver</li>
            <li>Requires academic or commercial license</li>
            <li>Used in the original research and analysis</li>
            <li>Superior performance for large-scale MILP problems</li>
            <li>Advanced presolving and cutting plane techniques</li>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="methodology-card">
        <div class="methodology-title">PuLP with CBC</div>
        <p><strong>Current Implementation</strong></p>
        <ul>
            <li>Open-source alternative using COIN-OR CBC solver</li>
            <li>No licensing restrictions for demo/portfolio purposes</li>
            <li>Sufficient performance for this problem size (~60-100 wines)</li>
            <li>Makes the tool accessible without solver licensing</li>
            <li>Free and open-source (BSD license)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>Why PuLP for This Portfolio Project?</strong><br/>
    While Gurobi is the industry standard and was used in the original research, PuLP was selected for this portfolio demonstration to ensure the tool is fully accessible without requiring solver licenses. For production deployments with larger problem sizes or performance-critical applications, Gurobi would be the recommended choice. The mathematical formulation remains identical regardless of solver choice.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Section 5: Business Logic & Rules
st.markdown('<p class="section-header">5. Business Logic & Rules</p>', unsafe_allow_html=True)

st.markdown("""
The optimization model incorporates several operational constraints and business rules specific to the client's wine procurement process.
""")

with st.container(border=True):
    st.markdown('<p class="methodology-title">Minimum Order Quantity (MOQ) Rules</p>', unsafe_allow_html=True)
    st.markdown("""
    **Standard Wines:**
    - Minimum order: 1 case (12 bottles)
    - Order quantities must be multiples of 12 bottles
    - Options range from MOQ up to approximately 3× the Economic Order Quantity (EOQ)
    
    **Anchor Wines (By-the-Bottle):**
    - Minimum order: 1 bottle
    - Order quantity options: [1, 2, 3, ..., 12] bottles
    - Only applies to anchor wines that can be purchased by the bottle
    
    **Direct Wineries:**
    - Minimum total order: 60 bottles (5 cases) across all wines from that winery
    - If any wine is ordered from a direct winery, the total must be ≥ 60 bottles
    - If not ordering from a winery, total order must be 0 bottles
    - Enforced using Big-M formulation in the optimization model
    """)

with st.container(border=True):
    st.markdown('<p class="methodology-title">Anchor Wine Logic</p>', unsafe_allow_html=True)
    st.markdown("""
    **What are Anchor Wines?**
    - Premium wines sold by the glass
    - Strategically priced to redirect customers from the cheapest menu options
    - Typically have low annual demand
    - Used as a psychological pricing tool
    
    **Ordering Rules:**
    - If an anchor wine can be ordered by the bottle (Is_By_the_Bottle = 1), order quantities are [1-12] bottles
    - If an anchor wine must be ordered by the case, standard MOQ rules apply
    - This flexibility allows for minimal inventory of these low-demand items
    """)

with st.container(border=True):
    st.markdown('<p class="methodology-title">Order Quantity Menu Generation</p>', unsafe_allow_html=True)
    st.markdown("""
    The optimization model pre-generates a "menu" of valid order quantity options for each wine:
    
    **Standard Wines:**
    1. Calculate Economic Order Quantity (EOQ): √(2 × D × S / H)
    2. Set upper bound: max(MOQ, 3 × EOQ)
    3. Generate options: [MOQ, MOQ+12, MOQ+24, ..., upper_bound] in multiples of 12
    
    **Anchor Wines (By-the-Bottle):**
    - Options: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] bottles
    
    **Direct Wineries:**
    - Standard menu generation applies to individual wines
    - Big-M constraints ensure winery-level minimums are met
    
    **Cost Pre-calculation:**
    - For each wine $i$ and each option $k$, calculate $C_{ik} = (Q_{ik}/2 + SS_i) \times H_i + (D_i/Q_{ik}) \times S$
    - This allows the objective function to be linear (sum of selected costs)
    """)

with st.container(border=True):
    st.markdown('<p class="methodology-title">Service Level Target</p>', unsafe_allow_html=True)
    st.markdown("""
    - **Target Service Level**: 95% (probability of not experiencing a stockout during lead time)
    - **Z-score**: $Z_{LT,i} \approx 1.645$ (from standard normal distribution)
    - **Safety Stock Calculation**: $SS_i = Z_{LT,i} \times \sigma_{LT,i}$
    - Higher service levels require more safety stock, increasing holding costs
    - The model balances service level requirements with cost minimization
    """)

st.markdown("---")

# Section 6: Key Assumptions & Limitations
st.markdown('<p class="section-header">6. Key Assumptions & Limitations</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="methodology-card">
        <div class="methodology-title">Model Assumptions</div>
        <ul>
            <li><strong>Deterministic Lead Times:</strong> Lead times are assumed fixed (stochastic extension discussed in report)</li>
            <li><strong>Stationary Demand:</strong> Single-period model assumes average annual conditions</li>
            <li><strong>Fixed Holding Cost:</strong> 25% of unit cost (industry benchmark)</li>
            <li><strong>Fixed Order Cost:</strong> $50 per order (estimated from consultation)</li>
            <li><strong>Normal Distribution:</strong> Demand during lead time follows normal distribution</li>
            <li><strong>Uniform Storage:</strong> All bottles require same storage space (V_i = 1)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="methodology-card">
        <div class="methodology-title">Model Limitations</div>
        <ul>
            <li><strong>Single-Echelon:</strong> Single warehouse, no multi-location optimization</li>
            <li><strong>No Quantity Discounts:</strong> Base model assumes fixed unit costs (extension available)</li>
            <li><strong>No Seasonal Planning:</strong> No multi-period/seasonal demand modeling</li>
            <li><strong>Binding Capacity:</strong> Storage constraint is typically binding, limiting flexibility</li>
            <li><strong>Static Parameters:</strong> Cost parameters don't vary over time</li>
            <li><strong>No Supplier Constraints:</strong> Assumes unlimited supplier capacity</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Section 7: Parameter Explanations
st.markdown('<p class="section-header">7. Parameter Explanations</p>', unsafe_allow_html=True)

st.markdown("""
Detailed explanations of user-configurable optimization parameters and their impacts.
""")

param_tabs = st.tabs(["Service Level", "Holding Cost %", "Fixed Order Cost", "Storage Capacity"])

with param_tabs[0]:
    st.markdown("""
    ### Service Level (%)
    
    **Definition:** The target probability of not experiencing a stockout during the supplier lead time period.
    
    **Calculation:**
    - Service level is converted to a Z-score using the inverse normal distribution: $Z_{LT,i} = \Phi^{-1}(\text{service level})$
    - For 95% service level: $Z_{LT,i} \approx 1.645$ for all wines $i$
    - Safety stock: $SS_i = Z_{LT,i} \times \sigma_{LT,i}$ for each wine $i$
    
    **Impact:**
    - Higher service levels require more safety stock
    - Increases holding costs (more buffer inventory)
    - Reduces stockout frequency and improves customer satisfaction
    - Typical range: 80% to 99%
    - Default: 95% (industry standard for critical inventory)
    
    **Trade-off:** Higher service levels cost more but reduce stockout risk.
    """)

with param_tabs[1]:
    st.markdown("""
    ### Holding Cost Percentage
    
    **Definition:** Annual holding cost expressed as a percentage of the unit cost per bottle.
    
    **Components:**
    - Storage costs (warehouse space, climate control)
    - Insurance and risk
    - Opportunity cost of capital
    - Obsolescence and spoilage risk
    
    **Calculation:**
    - Unit holding cost: $H_i = \\text{COST} \\times (\\text{holding cost \\%} / 100)$
    - Annual holding cost for wine $i$: $(Q/2 + SS_i) \\times H_i$
    
    **Impact:**
    - Higher holding costs encourage smaller, more frequent orders
    - Lower holding costs allow larger bulk orders
    - Default: 25% (industry benchmark for wine inventory)
    - Typical range: 15% to 35%
    
    **Trade-off:** Higher holding costs favor lower inventory levels but may increase ordering frequency.
    """)

with param_tabs[2]:
    st.markdown("""
    ### Fixed Order Cost ($)
    
    **Definition:** Administrative cost incurred per purchase order, regardless of order size.
    
    **Components:**
    - Order processing and paperwork
    - Communication with suppliers
    - Receiving and inspection
    - Administrative overhead
    
    **Calculation:**
    - Annual ordering cost for wine $i$: $(D_i / Q) \\times S$
    - Where $D_i$ is annual demand and $Q$ is order quantity
    
    **Impact:**
    - Higher fixed costs incentivize larger, less frequent orders
    - Lower fixed costs allow smaller, more frequent orders
    - Default: $50 (estimated from Purchasing Manager consultation)
    - Typical range: $25 to $200
    
    **Trade-off:** Higher fixed costs favor larger orders to spread costs over more units.
    """)

with param_tabs[3]:
    st.markdown("""
    ### Storage Capacity (bottles)
    
    **Definition:** Maximum total number of bottles that can be stored in the warehouse at any time.
    
    **Constraint Type:** Hard capacity limit (binding constraint in most solutions)
    
    **Calculation:**
    - Constraint: $\sum_i M_i \leq V_{\max}$
    - Where $M_i$ is the order-up-to level for each wine
    - Represents maximum inventory when all wines are at their M levels
    
    **Impact:**
    - Tighter capacity forces prioritization of high-velocity items
    - May limit stock levels for slower-moving wines
    - Binding constraint prevents full cost optimization
    - Default: 2000 bottles for demo
    
    **Sensitivity Analysis Finding:** Adding the first 68 bottles of capacity saves approximately $7.54 per bottle per year.
    
    **Trade-off:** More capacity enables better cost optimization but requires physical space investment.
    """)

st.markdown("---")

# Navigation Section
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("← Back", use_container_width=True):
        st.switch_page("pages/solver_ui.py")
