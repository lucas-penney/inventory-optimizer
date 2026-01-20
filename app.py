import streamlit as st

# Define the pages
executive_summary = st.Page("pages/dashboard.py", title="Executive Summary", default=True)
inventory_optimizer = st.Page("pages/solver_ui.py", title="Inventory Optimizer")
results = st.Page("pages/results.py", title="Results")
methodology = st.Page("pages/methodology.py", title="Methodology and Pipeline")

# Initialize Navigation
pg = st.navigation([executive_summary, inventory_optimizer, results, methodology])

# Global Sidebar
st.sidebar.title("🍷 Inventory Optimizer")
st.sidebar.info("Portfolio Project by Lucas Penney\n\n")

# Run the selected page
pg.run()