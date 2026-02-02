import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Navy blue for consistent page titles (matches Streamlit title styling)
PAGE_TITLE_COLOR = "#000080"


def page_title(title: str) -> None:
    """Render a consistent page title with optional Logout button and a divider underneath."""
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.title(title)
    with col_btn:
        if st.session_state.get("user_type") is not None:
            if st.button("Logout", use_container_width=True):
                st.session_state.user_type = None
                st.session_state.authenticated = False
                st.switch_page("pages/dashboard.py")
    st.divider()


def build_etl_flowchart():
    """Build flowchart for scripts/local_etl.py ETL process using networkx + matplotlib."""
    G = nx.DiGraph()

    # Nodes from local_etl.py: sources -> load -> step1..step5 -> solver
    nodes = [
        ("sales", "Beverage Sales\n.xlsx", 0),
        ("costing", "Wine Costing\n.xlsx", 0),
        ("misc", "Suppliers and\nBusiness Logic\n.xlsx", 0),
        ("load", "Load Data\nPandas\nread_excel()", 1),
        ("step1", "Step 1: Clean wine\n names, aggregate\nmonthly\nand annual sales", 2),
        ("step2", "Step 2: Combining\nglasses and bottles\n sold", 3),
        ("step3", "Step 3: Fuzzy\n Matching to\nmerge sales\nand costing data", 4),
        ("step4", "Step 4: Merge\nsupplier\nand business\nlogic data", 5),
        ("step5", "Step 5: Save\nOutput\n.csv", 6),
        ("solver", "Data ready\n for PuLP\nsolver", 7),
    ]
    for node_id, label, layer in nodes:
        G.add_node(node_id, label=label, subset=layer)

    # Edges: sources -> load -> steps -> solver
    G.add_edges_from([
        ("sales", "load"),
        ("costing", "load"),
        ("misc", "load"),
        ("load", "step1"),
        ("step1", "step2"),
        ("step2", "step3"),
        ("step3", "step4"),
        ("step4", "step5"),
        ("step5", "solver"),
    ])

    # Manual layered layout: three source nodes side-by-side (tight), then vertical flow
    layers = {0: ["sales", "costing", "misc"], 1: ["load"], 2: ["step1"], 3: ["step2"], 4: ["step3"], 5: ["step4"], 6: ["step5"], 7: ["solver"]}
    pos = {}
    for layer, node_ids in layers.items():
        n = len(node_ids)
        for i, nid in enumerate(node_ids):
            # Source nodes (layer 0) right next to each other; other layers centered
            x_scale = 0.55 if layer == 0 else 2.2
            x = (i - (n - 1) / 2) * x_scale if n > 1 else 0
            pos[nid] = (x, -layer * 1.4)

    # Colors by type: sources=light blue, load=orange, steps=green, output=yellow, solver=light red
    source_nodes = {"sales", "costing", "misc"}
    node_list = list(G.nodes())
    colors = []
    for n in node_list:
        if n in source_nodes:
            colors.append("#e1f5fe")
        elif n == "load":
            colors.append("#fff3e0")
        elif n == "step5":
            colors.append("#fff9c4")
        elif n == "solver":
            colors.append("#ffebee")
        else:
            colors.append("#e8f5e9")

    fig, ax = plt.subplots(figsize=(8, 10))
    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_nodes(
        G, pos, node_color=colors, node_shape="s", node_size=5000,
        edgecolors="#2c5aa0", linewidths=1.5, ax=ax
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=7, font_weight="normal",
        verticalalignment="center", horizontalalignment="center", ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edge_color="#1e7b34", arrows=True, arrowsize=24,
        arrowstyle=None, connectionstyle="arc3,rad=0.05", ax=ax
    )
    ax.axis("off")
    plt.tight_layout()
    return fig
