import streamlit as st

# Primary button colour: blue (theme in .streamlit/config.toml; fallback below if theme not applied)
# Tighter gaps above and below st.divider() throughout the UI
st.markdown(
    """<style>
    button[kind="primary"]{background-color:#1f77b4!important;border-color:#1f77b4!important;color:#fff!important}
    hr { margin: 0.35rem 0 !important; }
    </style>""",
    unsafe_allow_html=True,
)

# Define the pages
executive_summary = st.Page("pages/dashboard.py", title="Executive Summary", default=True)
inventory_optimizer = st.Page("pages/solver_ui.py", title="Configure Optimization")
results = st.Page("pages/results.py", title="Results")
methodology = st.Page("pages/methodology.py", title="Methodology and Pipeline")

# Initialize Navigation
pg = st.navigation([executive_summary, inventory_optimizer, results, methodology])

# Global Sidebar
st.sidebar.title("🍷 Inventory Optimizer")
st.sidebar.info("Portfolio Project by Lucas Penney\n\n")

# Run the selected page
pg.run()