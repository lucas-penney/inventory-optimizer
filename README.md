# Wine Inventory Optimizer

## Project Description

**Objective/Goal**:
- To optimize wine inventory reorder policies using Mixed-Integer Linear Programming (MILP), minimizing total inventory costs while respecting storage capacity and supplier constraints.

**Sector**:
- Hospitality, Food & Beverage, Supply Chain Management

**Technologies Used**:
- **Python**: Primary programming language for data processing and optimization logic.
- **Streamlit**: Interactive web application framework for the user interface and data visualization.
- **PuLP**: Open-source linear programming library used to formulate and solve the MILP optimization problem.
- **Pandas & NumPy**: Data manipulation and numerical calculations for demand analysis and cost modeling.
- **SciPy**: Statistical functions for safety stock calculations using normal distribution quantiles.
- **Plotly & Matplotlib**: Data visualization for results dashboards and sensitivity analysis charts.

## Architecture & Data Flow

**Architecture Overview**:

<!-- PLACEHOLDER: Process Flowchart -->
![Process Flowchart](./docs/images/process_flowchart.png)
*Add a flowchart showing the end-to-end process from data ingestion through optimization to results output.*

**Data Sources**:
- Historical sales data (11 months of monthly sales by wine SKU)
- Product cost and supplier lead time information
- Supplier type and winery designations (Wholesaler vs Direct)
- Business rules (anchor wines, by-the-bottle ordering capability)

**Transformation Steps**:
1. **Data Ingestion**: Raw Excel/CSV files are processed through the ETL pipeline to standardize naming conventions and merge datasets.
2. **Demand Analysis**: Monthly sales data is transformed into demand-during-lead-time (DDLT) statistics, calculating mean and standard deviation for each wine.
3. **Safety Stock Calculation**: Using the target service level (e.g., 95%), safety stock is computed as Z × σ_LT for each SKU.
4. **Order Menu Generation**: Valid order quantity options are generated for each wine based on MOQ rules, case sizes, and EOQ calculations.
5. **Cost Pre-Calculation**: Total annual costs (holding + ordering) are pre-computed for each wine-option combination to linearize the objective function.
6. **MILP Optimization**: The PuLP solver determines optimal reorder points (R), order-up-to levels (M), and order quantities (Q) that minimize total cost subject to constraints.

**Output**:
- Recommended order quantities for each wine SKU
- Optimal reorder points and order-up-to levels
- Total cost savings compared to current policy
- Sensitivity analysis across key parameters

## Application Screenshots

### Executive Summary Dashboard
<!-- PLACEHOLDER: Executive Summary Screenshot -->
![Executive Summary](./docs/images/executive_summary.png)
*The landing page provides project context, solution overview, and key benefits of the optimization approach.*

### Inventory Optimizer Configuration
<!-- PLACEHOLDER: Solver UI Screenshot -->
![Optimizer Configuration](./docs/images/solver_ui.png)
*Users configure optimization parameters including service level, holding cost percentage, fixed order cost, and storage capacity.*

### Optimization Results
<!-- PLACEHOLDER: Results Screenshot -->
![Optimization Results](./docs/images/results.png)
*Results display recommended policies for each wine, cost breakdowns, and comparison with current inventory policies.*

### Sensitivity Analysis
<!-- PLACEHOLDER: Sensitivity Analysis Screenshot -->
![Sensitivity Analysis](./docs/images/sensitivity_analysis.png)
*Interactive charts show how optimal costs change across different parameter values.*

### Methodology & Pipeline
<!-- PLACEHOLDER: Methodology Screenshot -->
![Methodology](./docs/images/methodology.png)
*Technical documentation of the mathematical model, constraints, and business logic.*

## Mathematical Model

The optimization problem is formulated as a Mixed-Integer Linear Program (MILP) using a continuous review (R, M) inventory policy:

**Objective Function**:
```
Minimize Z = Σ Σ C_ik × y_ik
```
Where C_ik represents the total annual cost (holding + ordering) for wine i under option k.

**Key Constraints**:
- **Single Choice**: Exactly one order quantity option selected per wine
- **Linking Constraint**: M_i - R_i = Q (order quantity equals gap between levels)
- **Safety Stock**: R_i ≥ SS_i + μ_LT,i (reorder point covers expected demand plus buffer)
- **Storage Capacity**: Σ M_i ≤ V_max (total max inventory within capacity)
- **Winery MOQs**: Big-M formulation ensures minimum 60-bottle orders from direct wineries

## Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/inventory_optimizer.git
cd inventory_optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

### Project Structure
```
inventory_optimizer/
├── app.py                 # Main Streamlit application entry point
├── requirements.txt       # Python dependencies
├── pages/
│   ├── dashboard.py       # Executive summary and landing page
│   ├── solver_ui.py       # Optimization configuration interface
│   ├── results.py         # Results visualization and analysis
│   └── methodology.py     # Technical documentation
├── src/
│   └── solver_logic.py    # Core optimization logic and MILP solver
├── scripts/
│   └── anonymizer.py      # Data anonymization utilities
├── utils/
│   └── ui_components.py   # Reusable UI components
└── data/
    └── dummy_data.csv     # Sample anonymized dataset for demo
```

## Results

This project demonstrates a quantitative approach to inventory optimization that:
- **Reduces total inventory costs** by 30-40% through optimized ordering strategies
- **Balances holding and ordering costs** using Economic Order Quantity (EOQ) principles
- **Respects operational constraints** including storage capacity and supplier MOQs
- **Provides sensitivity insights** to understand the impact of parameter changes

The MILP formulation handles the discrete nature of wine ordering (case quantities, winery minimums) while efficiently finding optimal solutions for portfolios of 60-100+ wines.

## Learnings

Through this project, key learnings were achieved in:
- Formulating real-world inventory problems as Mixed-Integer Linear Programs
- Implementing (R, M) continuous review policies with safety stock considerations
- Applying Big-M formulations to handle conditional constraints (winery MOQs)
- Building interactive data applications with Streamlit for operations research tools
- Balancing model complexity with practical solver performance using PuLP/CBC

## Future Extensions

- **Database Integration**: Migrate from CSV to PostgreSQL for scalable data management
- **Real-Time Data Feeds**: Integrate daily sales data for dynamic policy adjustments
- **Multi-Period Planning**: Extend to seasonal demand modeling and rolling horizon optimization
- **Quantity Discounts**: Incorporate tiered pricing structures from suppliers
