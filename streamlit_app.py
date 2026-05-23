import base64

import numpy as np
import streamlit as st

from src.streamlit_backend import tsp_algos, run_tsp, run_tsp_comparison
from src.streamlit_backend_nlp import run_nlp_experiment, run_lab10_task
from src.tsp_instances import INSTANCES
SVG_ICON = (
    "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC"
    "9zdmciIHZpZXdCb3g9IjAgMCA2NCA2NCIgd2lkdGg9IjY0IiBoZWlnaHQ9IjY0Ij48ZyBzdH"
    "Jva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bm"
    "QiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIG9wYWNpdHk9IjAuNiI+PGxpbmUgeDE9IjIyIi"
    "B5MT0iMjAiIHgyPSIxNCIgeTI9IjMyIi8+PGxpbmUgeDE9IjE0IiB5MT0iMzIiIHgyPSIyNC"
    "IgeTI9IjQ0Ii8+PGxpbmUgeDE9IjI0IiB5MT0iNDQiIHgyPSIzMiIgeTI9IjMyIi8+PGxpbm"
    "UgeDE9IjMyIiB5MT0iMzIiIHgyPSIyMiIgeTI9IjIwIi8+PGxpbmUgeDE9IjIyIiB5MT0iMj"
    "AiIHgyPSIyNCIgeTI9IjQ0Ii8+PGxpbmUgeDE9IjQyIiB5MT0iMjAiIHgyPSI1MCIgeTI9Ij"
    "MyIi8+PGxpbmUgeDE9IjUwIiB5MT0iMzIiIHgyPSI0MCIgeTI9IjQ0Ii8+PGxpbmUgeDE9Ij"
    "QwIiB5MT0iNDQiIHgyPSIzMiIgeTI9IjMyIi8+PGxpbmUgeDE9IjMyIiB5MT0iMzIiIHgyPS"
    "I0MiIgeTI9IjIwIi8+PGxpbmUgeDE9IjQyIiB5MT0iMjAiIHgyPSI0MCIgeTI9IjQ0Ii8+PC"
    "9nPjxnIGZpbGw9IiNmZmZmZmYiIG9wYWNpdHk9IjAuOSI+PGNpcmNsZSBjeD0iMjIiIGN5PS"
    "IyMCIgcj0iMi41Ii8+PGNpcmNsZSBjeD0iMTQiIGN5PSIzMiIgcj0iMi41Ii8+PGNpcmNsZS"
    "BjeD0iMjQiIGN5PSI0NCIgcj0iMi41Ii8+PGNpcmNsZSBjeD0iNDIiIGN5PSIyMCIgcj0iMi"
    "41Ii8+PGNpcmNsZSBjeD0iNTAiIGN5PSIzMiIgcj0iMi41Ii8+PGNpcmNsZSBjeD0iNDAiIG"
    "N5PSI0NCIgcj0iMi41Ii8+PC9nPjxnIGZpbGw9IiNmZmZmZmYiPjxjaXJjbGUgY3g9IjMyIi"
    "BjeT0iMzIiIHI9IjQuNSIvPjxwYXRoIGQ9Ik0zMSAzMWgydjJoLTJ6IiBmaWxsPSJub25lIi"
    "BzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC44Ii8+PC9nPjwvc3ZnPg=="
)

st.set_page_config(
    page_title="NeuroRoute — TSP & NLP Studio",
    page_icon=SVG_ICON,
    layout="wide",
    initial_sidebar_state="expanded",  # open the sidebar on load
)

# --------------------
# Global CSS / Theme
# --------------------
# Full page background (gradient by default). Replace with background image via base64 if desired.
st.markdown(
    """
    <style>
      /* Hide Streamlit default chrome for a cleaner look */
      header {visibility: hidden;}
      footer {visibility: hidden;}

      /* ----------------------------------------------------------------
         Keep the sidebar ALWAYS visible / always open.
         - force it open even on narrow screens (override collapse transform)
         - hide every collapse / expand control so it cannot be closed
         ---------------------------------------------------------------- */
      section[data-testid="stSidebar"] {
        transform: none !important;
        visibility: visible !important;
        min-width: 260px !important;
        width: 260px !important;
      }
      section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        visibility: visible !important;
        margin-left: 0 !important;
      }
      /* Hide the collapse arrow inside the sidebar and the expand control.
         Multiple selectors are used because the test-id for this button
         differs between Streamlit versions (and the raw "keyboard_double_
         arrow_left" icon-name text shows up when its font fails to load). */
      [data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="collapsedControl"],
      [data-testid="stSidebarHeader"],
      [data-testid="stSidebarNavCollapseButton"],
      section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] button,
      section[data-testid="stSidebar"] button[kind="header"],
      section[data-testid="stSidebar"] button[kind="headerNoPadding"] {
        display: none !important;
      }

      /* Full-page background is set dynamically by the theme system below. */

      /* Font + spacing.
         NOTE: do NOT use a broad [class*="st-"] selector here — it also
         matches Streamlit's Material-icon <span>s and overrides their icon
         font, which makes icons render as raw text (e.g. "keyboard_arrow_
         right"). Apply the UI font only to real text elements. */
      html, body, .stApp, .stMarkdown, .stText, .stTextInput,
      .stButton, button, input, textarea, select, label, p, h1, h2, h3, h4 {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      }

      /* Re-assert the Material icon font on icon elements so their ligatures
         render as glyphs instead of leaking their name as text. */
      [data-testid="stIconMaterial"],
      [data-testid="stExpanderToggleIcon"],
      span.material-icons,
      span.material-icons-outlined,
      span.material-symbols-rounded,
      span.material-symbols-outlined,
      .material-symbols-rounded {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
      }

      /* ----------------------------------------------------------------
         Cards: style the REAL Streamlit bordered container so the
         widgets are genuinely nested inside it (no orphan </div> tags,
         no overlapping elements).
         ---------------------------------------------------------------- */
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      }

      /* Inputs/buttons rounding */
      button[kind="primary"] { border-radius: 12px; }
      .stButton > button { border-radius: 12px; }
      .stSelectbox > div, .stSlider > div, .stNumberInput > div, .stCheckbox > div {
        border-radius: 12px;
      }

      /* Sidebar style */
      section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.06) !important;
        border-right: 1px solid rgba(255,255,255,0.10);
      }

      /* Typography */
      .bb-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
      }
      .bb-subtitle {
        color: rgba(255,255,255,0.70);
        margin-bottom: 1rem;
      }
      .bb-muted { color: rgba(255,255,255,0.70); }
      .bb-controls-title { font-size: 1.1rem; font-weight: 800; margin-bottom: 0.4rem; }

      /* Footer */
      .bb-footer {
        margin-top: 3rem;
        padding: 1.1rem 1rem;
        border-top: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
      }
      .bb-footer h4 { margin: 0 0 0.35rem 0; font-size: 1rem; color: rgba(255,255,255,0.92); }
      .bb-footer .muted { color: rgba(255,255,255,0.68); }

      /* Improve plot containers a bit */
      .stPlotlyChart, .stPyplot {
        border-radius: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------
# Theme definitions
# ---------------
# Each theme defines the full-page background. The selected one is injected
# into .stApp after the sidebar renders. Add a new entry here to add a theme.
THEMES = {
    "Nebula": """
        radial-gradient(1200px circle at 10% 20%, rgba(99,102,241,0.22), transparent 35%),
        radial-gradient(900px circle at 90% 25%, rgba(16,185,129,0.18), transparent 40%),
        linear-gradient(135deg, #0b1220 0%, #0f172a 45%, #0b1220 100%)
    """,
    "Neon": """
        radial-gradient(900px circle at 20% 10%, rgba(34,197,94,0.22), transparent 38%),
        radial-gradient(900px circle at 80% 25%, rgba(59,130,246,0.22), transparent 40%),
        linear-gradient(135deg, #050814 0%, #081027 45%, #050814 100%)
    """,
    "Aurora": """
        radial-gradient(1000px circle at 15% 15%, rgba(45,212,191,0.24), transparent 40%),
        radial-gradient(900px circle at 85% 20%, rgba(168,85,247,0.20), transparent 42%),
        radial-gradient(800px circle at 50% 95%, rgba(236,72,153,0.16), transparent 45%),
        linear-gradient(135deg, #07101b 0%, #0a1626 50%, #07101b 100%)
    """,
    "Sunset": """
        radial-gradient(1000px circle at 12% 18%, rgba(251,146,60,0.24), transparent 40%),
        radial-gradient(900px circle at 88% 22%, rgba(244,63,94,0.20), transparent 42%),
        linear-gradient(135deg, #190f0a 0%, #20140d 48%, #160b09 100%)
    """,
    "Oceanic": """
        radial-gradient(1000px circle at 15% 18%, rgba(14,165,233,0.24), transparent 40%),
        radial-gradient(900px circle at 85% 25%, rgba(6,182,212,0.18), transparent 42%),
        linear-gradient(135deg, #06121c 0%, #081a2a 48%, #06121c 100%)
    """,
    "Mono Slate": """
        radial-gradient(1000px circle at 20% 15%, rgba(148,163,184,0.16), transparent 42%),
        radial-gradient(900px circle at 85% 25%, rgba(100,116,139,0.12), transparent 44%),
        linear-gradient(135deg, #0c0f14 0%, #11151c 50%, #0c0f14 100%)
    """,
}

# ---------------
# App header
# ---------------

st.markdown(
    f"""
    <div class="bb-title" style="display: flex; align-items: center; gap: 10px;">
        <img src="{SVG_ICON}" width="80" height="80" style="vertical-align: middle; filter: brightness(1);"/>
        NeuroRoute
    </div>
    <div class="bb-subtitle">🚀 TSP Algorithms + 🧠 NLP Text Classification</div>
    """,
    unsafe_allow_html=True,
)

# ---------------
# Sidebar
# ---------------
# The sidebar is forced open via CSS above. Each block below is a normal
# Streamlit element (no unclosed raw-HTML divs), so nothing overlaps.
with st.sidebar:
    st.markdown('<div class="bb-controls-title">Controls</div>', unsafe_allow_html=True)

    theme_mode = st.selectbox(
        "🎨 Look & feel",
        options=list(THEMES.keys()),
        index=0,
        help="Pick a color theme for the whole app.",
    )
    compact_ui = st.checkbox("Compact view (less padding)", value=False)

    st.markdown("---")
    st.markdown("### Team")
    st.markdown(
        """
        <div class="bb-muted"> <b>Eroare 404</b></div>
        <div class="bb-muted">Ilisescu Adrian Corneliu, Nistor Robert Cristian, Ilisoi Fineas</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Tip: Choose an algorithm/model, then press Run.")

# Apply the selected theme's background to the whole app.
st.markdown(
    f"""
    <style>
      .stApp {{
        background: {THEMES.get(theme_mode, THEMES["Nebula"])};
        background-attachment: fixed;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Compact view: tighten the vertical spacing between elements
if compact_ui:
    st.markdown(
        """
        <style>
          div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
          div[data-testid="stVerticalBlockBorderWrapper"] > div > div { gap: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------
# City generation
# ---------------
# Available spatial distributions for generating TSP city coordinates.
CITY_GEN_MODES = [
    "Random uniform",
    "Clustered",
    "Circle",
    "Grid",
    "Gaussian blob",
]

NLP_DATASET_META = {
    "20newsgroups_full": {
        "desc": "Complete 20 Newsgroups dataset — all 20 categories.",
        "n_classes": 20,
        "train_docs": 11_314,
        "test_docs": 7_532,
        "classes": [
            "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
            "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
            "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
            "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
            "sci.space", "soc.religion.christian", "talk.politics.guns",
            "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc",
        ],
    },
    "20newsgroups_medium": {
        "desc": "20 Newsgroups — all 20 categories (medium variant).",
        "n_classes": 20,
        "train_docs": 11_314,
        "test_docs": 7_532,
        "classes": [
            "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
            "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
            "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
            "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
            "sci.space", "soc.religion.christian", "talk.politics.guns",
            "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc",
        ],
    },
}


def generate_cities(mode: str, *, n: int, seed: int, span: float = 1000.0) -> np.ndarray:
    """Generate `n` 2D city coordinates using the chosen spatial distribution.

    Returns an (n, 2) float array. The result is fully deterministic for a
    given (mode, n, seed), so every run is reproducible.

    Modes:
      - "Random uniform": points spread uniformly over the whole area.
      - "Clustered":      points grouped into a few random clusters.
      - "Circle":         points placed around a ring with light jitter.
      - "Grid":           points on a near-regular grid with light jitter.
      - "Gaussian blob":  points concentrated around the center.
    """
    rng = np.random.default_rng(seed)

    if mode == "Clustered":
        n_clusters = max(2, min(8, n // 6))
        centers = rng.uniform(0.15 * span, 0.85 * span, size=(n_clusters, 2))
        labels = rng.integers(0, n_clusters, size=n)
        spread = 0.06 * span
        points = centers[labels] + rng.normal(0.0, spread, size=(n, 2))

    elif mode == "Circle":
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n))
        radius = 0.40 * span
        center = span / 2.0
        jitter = rng.normal(0.0, 0.02 * span, size=(n, 2))
        points = (
            np.column_stack(
                [center + radius * np.cos(angles), center + radius * np.sin(angles)]
            )
            + jitter
        )

    elif mode == "Grid":
        side = int(np.ceil(np.sqrt(n)))
        axis = np.linspace(0.10 * span, 0.90 * span, side)
        grid_x, grid_y = np.meshgrid(axis, axis)
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:n]
        points = grid + rng.normal(0.0, 0.015 * span, size=grid.shape)

    elif mode == "Gaussian blob":
        center = np.array([span / 2.0, span / 2.0])
        points = center + rng.normal(0.0, 0.16 * span, size=(n, 2))

    else:  # "Random uniform" (default)
        points = rng.uniform(0.0, span, size=(n, 2))

    return np.clip(points, 0.0, span).astype(float)


def _format_dist_matrix(coords: np.ndarray) -> str:
    """Return TSP input as text: first line N, then NxN rounded Euclidean distance matrix."""
    n = int(coords.shape[0])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    lines = [str(n)]
    for row in dist:
        lines.append(" ".join(str(int(round(v))) for v in row))
    return "\n".join(lines)


def _tsp_params_ui(algo: str, *, n: int) -> dict:
    params = {}
    if algo == "BKT (Backtracking)":
        st.warning(
            "⚠️ Backtracking is an exact method — cost grows factorially. "
            "Keep N small (≤ 12). Runs for at least 120 seconds, returning the best cost found."
        )

    elif algo == "HC (Hill Climbing)":
        restarts = st.slider("Restarts", 1, 80, value=30, step=1)
        iterations = st.slider("Iterations per restart", 200, 10000, value=2000, step=200)
        params["restarts"] = int(restarts)
        params["iterations"] = int(iterations)

    elif algo == "NN (Nearest Neighbor)":
        start_city = st.slider("Start city", 0, n - 1, value=0, step=1)
        params["start_city"] = int(start_city)

    elif algo == "SA (Simulated Annealing)":
        init = st.selectbox("Initialization", options=["nn", "random"], index=0)
        t_max = st.number_input("t_max", min_value=1.0, value=10000.0, step=1000.0)
        t_min = st.number_input("t_min", min_value=0.1, value=1.0, step=0.1)
        alpha = st.slider("alpha", min_value=0.90, max_value=0.999, value=0.995, step=0.001)
        iters_per_temp = st.slider("iters/temperature", min_value=10, max_value=500, value=100, step=10)
        params.update(
            {
                "init": init,
                "t_max": float(t_max),
                "t_min": float(t_min),
                "alpha": float(alpha),
                "iters_per_temp": int(iters_per_temp),
            }
        )

    elif algo == "GA (Genetic Algorithm - PyGAD)":
        pop_size = st.slider("Population Size", min_value=20, max_value=300, value=120, step=10)
        generations = st.slider("Generations", min_value=50, max_value=2000, value=400, step=50)
        mutation_rate = st.slider("Mutation Rate %", min_value=1, max_value=100, value=45, step=1)
        params.update(
            {
                "pop_size": int(pop_size),
                "generations": int(generations),
                "mutation_rate_percent": int(mutation_rate),
            }
        )

    return params


# --------------------
# TSP / NLP tabs
# --------------------
tab_tsp, tab_nlp = st.tabs(["🗺️ TSP", "📝 NLP"])


with tab_tsp:
    # Card 1: TSP settings + output  (real bordered container -> widgets are inside it)
    with st.container(border=True):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("⚙️ TSP Settings")
            algo = st.selectbox("Algorithm", options=tsp_algos, index=0, format_func=lambda x: x)

            data_source = st.radio(
                "Input data",
                options=["Random generation", "Predefined instance"],
                index=0,
                horizontal=True,
            )

            if data_source == "Predefined instance":
                instance_key = st.selectbox(
                    "TSPLIB instance",
                    options=list(INSTANCES.keys()),
                    help="Classic benchmark instances with real coordinates.",
                )
                inst = INSTANCES[instance_key]
                tsp_coords = inst.coords
                n = int(tsp_coords.shape[0])
                seed = st.number_input("Seed (for algorithm)", min_value=0, value=42, step=1, key="tsp_seed")
            else:
                n = st.slider("Number of cities (N)", min_value=4, max_value=100, value=10, step=1)
                gen_mode = st.selectbox(
                    "City generation mode",
                    options=CITY_GEN_MODES,
                    index=0,
                    help="How the city coordinates are laid out on the map.",
                )
                seed = st.number_input("Seed", min_value=0, value=42, step=1, key="tsp_seed")
                tsp_coords = generate_cities(gen_mode, n=n, seed=int(seed))

            # --- Input summary box ---
            st.markdown("---")
            if data_source == "Predefined instance":
                optimal_str = f"{inst.optimal:,.0f}" if inst.optimal is not None else "unknown"
                st.markdown(
                    f"""
                    <div style="background:rgba(99,102,241,0.13);border:1px solid rgba(99,102,241,0.35);
                                border-radius:10px;padding:0.7rem 1rem;font-size:0.88rem;line-height:1.7;">
                      <b>Input data:</b> Predefined instance (TSPLIB)<br>
                      <b>Instance:</b> {inst.name}<br>
                      <b>Number of cities (N):</b> {n}<br>
                      <b>Known optimum:</b> {optimal_str}<br>
                      <b>Algorithm seed:</b> {int(seed)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background:rgba(16,185,129,0.10);border:1px solid rgba(16,185,129,0.30);
                                border-radius:10px;padding:0.7rem 1rem;font-size:0.88rem;line-height:1.7;">
                      <b>Input data:</b> Random generation<br>
                      <b>Distribution mode:</b> {gen_mode}<br>
                      <b>Number of cities (N):</b> {n}<br>
                      <b>Seed:</b> {int(seed)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            params = _tsp_params_ui(algo, n=n)
            params["coords"] = tsp_coords.tolist()

        with col2:
            st.subheader("📦 Output")
            run_btn = st.button("Run", type="primary", width="stretch")

            if run_btn:
                if data_source != "Predefined instance":
                    st.session_state["tsp_matrix"] = _format_dist_matrix(tsp_coords)

                with st.status("⚙️ Running algorithm...", expanded=True) as status:

                    def _bkt_progress(elapsed: float, remaining: float, nr_sol: int, best: int) -> None:
                        status.update(
                            label=(
                                f"⏱️ {elapsed:.0f}s — "
                                f"{nr_sol} solutions found — "
                                f"current cost: {best}"
                            ),
                            state="running",
                        )

                    res = run_tsp(
                        algo, n=n, seed=int(seed),
                        log_fn=st.write,
                        progress_fn=_bkt_progress,
                        **params,
                    )
                    status.update(
                        label=f"✅ Done — Cost: {res['cost']:.3f} | Time: {res['duration_s']:.4f}s",
                        state="complete",
                        expanded=False,
                    )

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("💸 Cost", value=f"{res['cost']:.3f}")
                with m2:
                    st.metric("⏱️ Time (sec)", value=f"{res['duration_s']:.4f}")

                with st.expander("🧩 Route (details)", expanded=False):
                    st.code(res["route_str"], language="text")

                if res.get("plot") is not None:
                    st.pyplot(res["plot"], clear_figure=True, width="stretch")

                if res.get("metrics"):
                    st.json(res["metrics"])

        # --- Distance matrix display (full-width, inside the card) ---
        st.markdown("---")
        st.markdown("##### Input data — Distance matrix")
        if data_source == "Predefined instance":
            matrix_text = _format_dist_matrix(tsp_coords)
            st.text_area(
                "Distance matrix", value=matrix_text, height=200, disabled=True, key="tsp_matrix_area",
                label_visibility="collapsed",
                help="First line = N (number of cities). Followed by the NxN Euclidean distance matrix.",
            )
        elif "tsp_matrix" in st.session_state:
            st.text_area(
                "Distance matrix", value=st.session_state["tsp_matrix"], height=200, disabled=True, key="tsp_matrix_area",
                label_visibility="collapsed",
                help="First line = N. Followed by the NxN Euclidean distance matrix.",
            )
        else:
            st.caption("Press **Run** to visualize the distance matrix of the generated instance.")

    st.divider()

    # Card 2: Quick comparison
    with st.container(border=True):
        st.subheader("⚡ Quick Comparison — All Algorithms")

        comp_data_source = st.radio(
            "Input data (comparison)",
            options=["Random generation", "Predefined instance"],
            index=0,
            horizontal=True,
            key="comp_data_src",
        )

        if comp_data_source == "Predefined instance":
            comp_instance_key = st.selectbox(
                "TSPLIB instance",
                options=list(INSTANCES.keys()),
                key="comp_instance",
            )
            comp_inst = INSTANCES[comp_instance_key]
            comp_coords = comp_inst.coords
            comp_n = int(comp_coords.shape[0])
            comp_seed = st.number_input("Comparison seed", min_value=0, value=42, step=1)
        else:
            col_l, col_r = st.columns([1, 1])
            with col_l:
                comp_n = st.slider("N for comparison", min_value=4, max_value=100, value=12, step=1)
            with col_r:
                comp_seed = st.number_input("Comparison seed", min_value=0, value=42, step=1)
            comp_gen_mode = st.selectbox(
                "City generation mode",
                options=CITY_GEN_MODES,
                index=0,
                key="comp_gen_mode",
                help="How the city coordinates are laid out for the comparison.",
            )
            comp_coords = generate_cities(comp_gen_mode, n=int(comp_n), seed=int(comp_seed))

        # --- Comparison input summary box ---
        if comp_data_source == "Predefined instance":
            comp_optimal_str = f"{comp_inst.optimal:,.0f}" if comp_inst.optimal is not None else "unknown"
            st.markdown(
                f"""
                <div style="background:rgba(99,102,241,0.13);border:1px solid rgba(99,102,241,0.35);
                            border-radius:10px;padding:0.7rem 1rem;font-size:0.88rem;line-height:1.7;">
                  <b>Input data:</b> Predefined instance (TSPLIB)<br>
                  <b>Instance:</b> {comp_inst.name}<br>
                  <b>Number of cities (N):</b> {comp_n}<br>
                  <b>Known optimum:</b> {comp_optimal_str}<br>
                  <b>Algorithm seed:</b> {int(comp_seed)}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background:rgba(16,185,129,0.10);border:1px solid rgba(16,185,129,0.30);
                            border-radius:10px;padding:0.7rem 1rem;font-size:0.88rem;line-height:1.7;">
                  <b>Input data:</b> Random generation<br>
                  <b>Distribution mode:</b> {comp_gen_mode}<br>
                  <b>Number of cities (N):</b> {comp_n}<br>
                  <b>Seed:</b> {int(comp_seed)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        run_comp = st.button("Compare 🤹‍♂️", key="comp", width="stretch")
        if run_comp:
            if comp_data_source != "Predefined instance":
                st.session_state["comp_matrix"] = _format_dist_matrix(comp_coords)

            with st.status("⚙️ Comparing algorithms...", expanded=True) as comp_status:
                comp = run_tsp_comparison(
                    n=int(comp_n), seed=int(comp_seed), coords=comp_coords.tolist(),
                    log_fn=st.write,
                )
                comp_status.update(
                    label=f"✅ Comparison complete — Winner: {comp['best']}",
                    state="complete",
                    expanded=False,
                )

            col_1, col_2 = st.columns([1, 1])
            with col_1:
                st.dataframe(comp["table"], width="stretch")
            with col_2:
                st.pyplot(comp["plot"], clear_figure=True, width="stretch")

        # --- Comparison distance matrix display ---
        st.markdown("---")
        st.markdown("##### Input data — Distance matrix")
        if comp_data_source == "Predefined instance":
            comp_matrix_text = _format_dist_matrix(comp_coords)
            st.text_area(
                "Comparison distance matrix", value=comp_matrix_text, height=200, disabled=True, key="comp_matrix_area",
                label_visibility="collapsed",
                help="First line = N. Followed by the NxN Euclidean distance matrix.",
            )
        elif "comp_matrix" in st.session_state:
            st.text_area(
                "Comparison distance matrix", value=st.session_state["comp_matrix"], height=200, disabled=True, key="comp_matrix_area",
                label_visibility="collapsed",
                help="First line = N. Followed by the NxN Euclidean distance matrix.",
            )
        else:
            st.caption("Press **Compare** to visualize the distance matrix of the generated instance.")


with tab_nlp:
    # Card: NLP settings + output
    with st.container(border=True):
        colA, colB = st.columns([1, 1])

        with colA:
            st.subheader("📝 NLP Settings")
            dataset = st.selectbox(
                "Dataset",
                options=["20newsgroups_full", "20newsgroups_medium"],
                index=0,
                help="Mapped to categories exactly as in the lab scripts.",
            )
            seed = st.number_input("Seed", min_value=0, value=42, step=1, key="nlp_seed")
            no_plots = st.checkbox("No plots (faster)", value=False)

            algo_choice = st.selectbox(
                "Method",
                options=[
                    "Naive Bayes (default TF-IDF)",
                    "Comparison of classifiers",
                    "N-gram range (SVM)",
                    "Max features (SVM)",
                    "Grid Search ngram × max_features (SVM)",
                ],
                index=0,
            )

            st.markdown("### 🧪 TF-IDF parameters (for Naive Bayes)")
            c_tf1, c_tf2 = st.columns(2)
            with c_tf1:
                ng_min = st.slider("ngram min", 1, 3, value=1, step=1)
            with c_tf2:
                ng_max = st.slider("ngram max", 1, 3, value=1, step=1)

            mf_options = [None, 100, 500, 1000, 5000, 10000]
            max_features = st.selectbox("max_features", options=mf_options, index=0)

            # --- NLP input summary box ---
            st.markdown("---")
            meta = NLP_DATASET_META.get(dataset, {})
            classes_list = "".join(
                f"<span style='display:inline-block;background:rgba(255,255,255,0.08);"
                f"border-radius:5px;padding:1px 7px;margin:2px 3px 2px 0;"
                f"font-size:0.78rem;'>{c}</span>"
                for c in meta.get("classes", [])
            )
            ngram_display = f"({ng_min}, {ng_max})" if algo_choice == "Naive Bayes (default TF-IDF)" else "—"
            mf_display = str(max_features) if max_features is not None else "None (all)"
            st.markdown(
                f"""
                <div style="background:rgba(99,102,241,0.13);border:1px solid rgba(99,102,241,0.35);
                            border-radius:10px;padding:0.75rem 1rem;font-size:0.88rem;line-height:1.8;">
                  <b>Dataset:</b> {dataset}<br>
                  <b>Description:</b> {meta.get('desc', '—')}<br>
                  <b>Classes:</b> {meta.get('n_classes', '?')}&nbsp;&nbsp;
                  <b>Train:</b> {meta.get('train_docs', '?'):,} docs&nbsp;&nbsp;
                  <b>Test:</b> {meta.get('test_docs', '?'):,} docs<br>
                  <b>Method:</b> {algo_choice}<br>
                  <b>ngram_range:</b> {ngram_display}&nbsp;&nbsp;
                  <b>max_features:</b> {mf_display}&nbsp;&nbsp;
                  <b>Seed:</b> {int(seed)}<br>
                  <div style="margin-top:0.4rem;"><b>Classes:</b><br>{classes_list}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with colB:
            st.subheader("📊 Output")
            out_tabs = st.tabs(["Output", "Report", "Plots"])
            with out_tabs[0]:
                st.info("Choose a method and press Run.")

            def _render_result(res: dict):
                with out_tabs[0]:
                    if res.get("task") == 2:
                        st.subheader("Results table")
                        st.dataframe(res.get("table", []), width="stretch")
                        st.success(f"🏆 Best: {res.get('best')} | Accuracy={res.get('best_accuracy')}")
                    elif res.get("task") in (3, 4):
                        key = "ngram_range" if res.get("task") == 3 else "max_features"
                        st.subheader(f"Results table ({key})")
                        st.dataframe(res.get("table", []), width="stretch")
                    elif res.get("task") == 5:
                        if res.get("error"):
                            st.error(res["error"])
                        else:
                            st.subheader("Heatmap (SVM grid)")
                    else:
                        st.subheader("Results")
                        st.write(f"Accuracy: {res.get('accuracy')}")

                with out_tabs[1]:
                    if res.get("report"):
                        st.text_area("Classification report", value=res["report"], height=240)
                    else:
                        st.caption("No textual report available for this output.")

                with out_tabs[2]:
                    if res.get("plot_cm") is not None:
                        st.pyplot(res["plot_cm"], clear_figure=True, width="stretch")
                    if res.get("plot") is not None:
                        st.pyplot(res["plot"], clear_figure=True, width="stretch")
                    if res.get("heatmap") is not None:
                        st.pyplot(res["heatmap"], clear_figure=True, width="stretch")

            mapping = {
                "Naive Bayes (default TF-IDF)": 1,
                "Comparison of classifiers": 2,
                "N-gram range (SVM)": 3,
                "Max features (SVM)": 4,
                "Grid Search ngram × max_features (SVM)": 5,
            }

            btn_run = st.button(" Run selected", type="primary", width="stretch", key="run_selected")
            btn_all = st.button(" Run ALL (Task 1..5)", width="stretch", key="all")

            if btn_all:
                with st.status("⚙️ Running all tasks (1–5)…", expanded=True) as nlp_status_all:
                    for task_id in [1, 2, 3, 4, 5]:
                        st.write(f"▶️ Task {task_id}/5…")
                        res = run_lab10_task(
                            task_id=task_id, dataset=dataset,
                            no_plots=no_plots, seed=int(seed), log_fn=st.write,
                        )
                        _render_result(res)
                    nlp_status_all.update(
                        label="✅ All tasks complete!", state="complete", expanded=False
                    )

            elif btn_run:
                task_id = mapping[algo_choice]
                with st.status(f"⚙️ Running: {algo_choice}…", expanded=True) as nlp_status:
                    if task_id == 1:
                        res = run_nlp_experiment(
                            dataset=dataset,
                            model_name="Naive Bayes",
                            ngram_range=(int(ng_min), int(ng_max)),
                            max_features=max_features,
                            no_plots=no_plots,
                            seed=int(seed),
                            log_fn=st.write,
                        )
                        res = {**res, "task": 1}
                    else:
                        res = run_lab10_task(
                            task_id=task_id, dataset=dataset,
                            no_plots=no_plots, seed=int(seed), log_fn=st.write,
                        )
                    acc = res.get("accuracy") or res.get("best_accuracy")
                    label = f"✅ Done — Accuracy: {acc:.4f}" if acc else "✅ Done!"
                    nlp_status.update(label=label, state="complete", expanded=False)
                _render_result(res)

# ---------------
# Footer
# ---------------
st.markdown(
    """
    <div class="bb-footer">
      <h4>Eroare 404</h4>
      <div class="muted">Ilisescu Adrian Corneliu, Nistor Robert Cristian, Ilisoi Fineas</div>
    </div>
    """,
    unsafe_allow_html=True,
)