import base64
import streamlit as st

from src.streamlit_backend import tsp_algos, run_tsp, run_tsp_comparison
from src.streamlit_backend_nlp import run_nlp_experiment, run_lab10_task


st.set_page_config(
    page_title="NeuroRoute — TSP & NLP Studio",
    page_icon="🧭",
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
        font-size: 2rem;
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
    """
    <div class="bb-title">🧭 NeuroRoute</div>
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


def _tsp_params_ui(algo: str, *, n: int) -> dict:
    params = {}
    if algo == "BKT (Backtracking)":
        st.info("⚠️ Backtracking can be expensive — use smaller N.")
        bt_mode = st.selectbox("Backtracking mode", options=["first", "all"], index=0)
        params["bt_mode"] = bt_mode
        bt_time_limit = st.slider("Time limit (sec) — soft", min_value=1, max_value=60, value=15, step=1)
        params["bt_time_limit"] = bt_time_limit

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
            n = st.slider("Number of cities (N)", min_value=4, max_value=18, value=10, step=1)
            seed = st.number_input("Seed", min_value=0, value=42, step=1, key="tsp_seed")
            params = _tsp_params_ui(algo, n=n)

        with col2:
            st.subheader("📦 Output")
            run_btn = st.button("Run", type="primary", use_container_width=True)

            if run_btn:
                with st.spinner("Running… (may take time for BKT/GA)"):
                    res = run_tsp(algo, n=n, seed=int(seed), **params)

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("💸 Cost", value=f"{res['cost']:.3f}")
                with m2:
                    st.metric("⏱️ Time (sec)", value=f"{res['duration_s']:.4f}")

                with st.expander("🧩 Route (details)", expanded=False):
                    st.code(res["route_str"], language="text")

                if res.get("plot") is not None:
                    st.pyplot(res["plot"], clear_figure=True, use_container_width=True)

                if res.get("metrics"):
                    st.json(res["metrics"])

    st.divider()

    # Card 2: Quick comparison
    with st.container(border=True):
        st.subheader("⚡ Quick Comparison — All Algorithms")
        col_l, col_r = st.columns([1, 1])
        with col_l:
            comp_n = st.slider("N for comparison", min_value=4, max_value=16, value=12, step=1)
        with col_r:
            comp_seed = st.number_input("Comparison seed", min_value=0, value=42, step=1)

        run_comp = st.button("Compare 🤹‍♂️", key="comp", use_container_width=True)
        if run_comp:
            with st.spinner("Running comparison…"):
                comp = run_tsp_comparison(n=int(comp_n), seed=int(comp_seed))

            col_1, col_2 = st.columns([1, 1])
            with col_1:
                st.dataframe(comp["table"], use_container_width=True)
            with col_2:
                st.pyplot(comp["plot"], clear_figure=True, use_container_width=True)


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
                    "Naive Bayes (TF-IDF implicite)",
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

        with colB:
            st.subheader("📊 Output")
            out_tabs = st.tabs(["Output", "Report", "Plots"])
            with out_tabs[0]:
                st.info("Choose a method and press Run.")

            def _render_result(res: dict):
                with out_tabs[0]:
                    if res.get("task") == 2:
                        st.subheader("Results table")
                        st.dataframe(res.get("table", []), use_container_width=True)
                        st.success(f"🏆 Best: {res.get('best')} | Accuracy={res.get('best_accuracy')}")
                    elif res.get("task") in (3, 4):
                        key = "ngram_range" if res.get("task") == 3 else "max_features"
                        st.subheader(f"Results table ({key})")
                        st.dataframe(res.get("table", []), use_container_width=True)
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
                        st.pyplot(res["plot_cm"], clear_figure=True, use_container_width=True)
                    if res.get("plot") is not None:
                        st.pyplot(res["plot"], clear_figure=True, use_container_width=True)
                    if res.get("heatmap") is not None:
                        st.pyplot(res["heatmap"], clear_figure=True, use_container_width=True)

            mapping = {
                "Naive Bayes (TF-IDF implicite)": 1,
                "Comparison of classifiers": 2,
                "N-gram range (SVM)": 3,
                "Max features (SVM)": 4,
                "Grid Search ngram × max_features (SVM)": 5,
            }

            btn_run = st.button(" Run selected", type="primary", use_container_width=True, key="run_selected")
            btn_all = st.button(" Run ALL (Task 1..5)", use_container_width=True, key="all")

            if btn_all:
                with st.spinner("Running all tasks… (may take time)"):
                    for task_id in [1, 2, 3, 4, 5]:
                        res = run_lab10_task(task_id=task_id, dataset=dataset, no_plots=no_plots, seed=int(seed))
                        _render_result(res)

            elif btn_run:
                task_id = mapping[algo_choice]
                st.info(f"Processing NLP… (task_id={task_id})")

                with st.spinner("Training / evaluating classifier (may take time)…"):
                    if task_id == 1:
                        res = run_nlp_experiment(
                            dataset=dataset,
                            model_name="Naive Bayes",
                            ngram_range=(int(ng_min), int(ng_max)),
                            max_features=max_features,
                            no_plots=no_plots,
                            seed=int(seed),
                        )
                        res = {**res, "task": 1}
                    else:
                        res = run_lab10_task(task_id=task_id, dataset=dataset, no_plots=no_plots, seed=int(seed))

                st.success("✅ Done")
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