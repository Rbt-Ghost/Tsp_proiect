import streamlit as st

from src.streamlit_backend import tsp_algos, run_tsp, run_tsp_comparison
from src.streamlit_backend_nlp import run_nlp_experiment, run_lab10_task



st.set_page_config(page_title="AI Dashboard - TSP & NLP", layout="wide")

st.markdown(
    """
    <style>
      .app-title {font-size: 2rem; font-weight: 800; margin-bottom: 0.2rem;}
      .app-subtitle {color: #6b7280; margin-bottom: 1.2rem;}
      .card {border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; padding: 1rem;}
      .muted {color: #6b7280;}

      /* Slightly nicer UI but keep it simple */
      div[data-testid="stVerticalBlock"] {
        border-radius: 14px;
      }

      .stButton > button {
        border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="app-title">AI Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">TSP Algorithms + NLP Text Classification</div>',
    unsafe_allow_html=True,
)

tab_tsp, tab_nlp = st.tabs(["TSP", "NLP"])


def _tsp_params_ui(algo: str, *, n: int) -> dict:
    params = {}
    if algo == "BKT (Backtracking)":
        st.info("Backtracking can be expensive. It is recommended for smaller values of N.")
        bt_mode = st.selectbox("Backtracking mode", options=["first", "all"], index=0)
        params["bt_mode"] = bt_mode
        bt_time_limit = st.slider(
            "Time limit (sec) - soft", min_value=1, max_value=60, value=15, step=1
        )
        params["bt_time_limit"] = bt_time_limit

    elif algo == "HC (Hill Climbing)":
        restarts = st.slider("Restarts", 1, 80, value=30, step=1)
        iterations = st.slider(
            "Iterations per restart", 200, 10000, value=2000, step=200
        )
        params["restarts"] = int(restarts)
        params["iterations"] = int(iterations)

    elif algo == "NN (Nearest Neighbor)":
        start_city = st.slider("Start city", 0, n - 1, value=0, step=1)
        params["start_city"] = int(start_city)

    elif algo == "SA (Simulated Annealing)":
        init = st.selectbox("Initialization", options=["nn", "random"], index=0)
        t_max = st.number_input("t_max", min_value=1.0, value=10000.0, step=1000.0)
        t_min = st.number_input("t_min", min_value=0.1, value=1.0, step=0.1)
        alpha = st.slider("alpha ", min_value=0.90, max_value=0.999, value=0.995, step=0.001)
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


with tab_tsp:
    st.header("TSP - Algorithms & comparisons")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Settings")
        algo = st.selectbox("Algorithm", options=tsp_algos, index=0, format_func=lambda x: x)
        n = st.slider("Number of cities (N)", min_value=4, max_value=18, value=10, step=1)
        seed = st.number_input("Seed", min_value=0, value=42, step=1, key="tsp_seed")

        params = _tsp_params_ui(algo, n=n)

    with col2:
        st.subheader("Output")
        run_btn = st.button("Run", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Running... (may take time for BKT/GA)"):
                res = run_tsp(algo, n=n, seed=int(seed), **params)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Cost", value=f"{res['cost']:.3f}")
            with m2:
                st.metric("Time (sec)", value=f"{res['duration_s']:.4f}")

            with st.expander("Route", expanded=False):
                st.code(res["route_str"], language="text")

            if res.get("plot") is not None:
                st.pyplot(res["plot"], clear_figure=True, use_container_width=True)

            if res.get("metrics"):
                st.json(res["metrics"])

    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.subheader("Quick Comparison - All Algorithms")
    comp_n = st.slider("N for comparison", min_value=4, max_value=16, value=12, step=1)
    comp_seed = st.number_input("Comparison seed", min_value=0, value=42, step=1)

    if st.button("Run comparison", key="comp", use_container_width=True):
        with st.spinner("Running comparison..."):
            comp = run_tsp_comparison(n=int(comp_n), seed=int(comp_seed))

        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.dataframe(comp["table"], use_container_width=True)
        with col_r:
            st.pyplot(comp["plot"], clear_figure=True, use_container_width=True)


with tab_nlp:
    st.header("NLP - Lab 10: Text Classification")
    st.caption("Choose method from dropdown (configurable) + Run")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    dataset = st.selectbox(
        "Dataset",
        options=["20newsgroups_full", "20newsgroups_medium"],
        index=0,
        help="Mapped to categories exactly as in the lab scripts.",
    )
    seed = st.number_input("Seed", min_value=0, value=42, step=1, key="nlp_seed")
    no_plots = st.checkbox("No plots (faster)", value=False)

    st.divider()

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

    # Inputs suplimentare doar când are sens (pentru uniformitate păstrăm ngram_range/max_features în UI,
    # iar backend-ul Lab10 le folosește exact când rulează Sarcina 1; pentru sarcinile 3..5, backend-ul
    # are propriile valori conform laboratorului).
    st.subheader("TF-IDF parameters (for Naive Bayes)")
    c_tf1, c_tf2 = st.columns(2)
    with c_tf1:
        ng_min = st.slider("ngram min", 1, 3, value=1, step=1)
    with c_tf2:
        ng_max = st.slider("ngram max", 1, 3, value=1, step=1)

    mf_options = [None, 100, 500, 1000, 5000, 10000]
    max_features = st.selectbox("max_features", options=mf_options, index=0)

    st.subheader("Run")

    btn_run = st.button(
        "Run selected",
        type="primary",
        width="stretch",
        key="run_selected",
    )

    out_tabs = st.tabs(["Output", "Report", "Plots"])

    with out_tabs[0]:
        st.info("Choose method and click Run.")

    def _render_result(res: dict):
        with out_tabs[0]:
            if res.get("task") == 2:
                st.subheader("Results table")
                st.dataframe(res.get("table", []), use_container_width=True)
                st.success(f"Best: {res.get('best')} | Accuracy={res.get('best_accuracy')}")
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

    executed = False

    mapping = {
        "Naive Bayes (TF-IDF implicite)": 1,
        "Comparare clasificatori": 2,
        "N-gram range (SVM)": 3,
        "Max features (SVM)": 4,
        "Grid Search ngram × max_features (SVM)": 5,
    }

    btn_all = st.button("Run ALL (Task 1..5)", key="all", width="stretch")

    if btn_all:

        with st.spinner("Run ALL (may take time)..."):
            for task_id in [1, 2, 3, 4, 5]:
                res = run_lab10_task(task_id=task_id, dataset=dataset, no_plots=no_plots, seed=int(seed))
                _render_result(res)
                executed = True

    if not executed and btn_run:
        task_id = mapping[algo_choice]
        # pentru task 1 respectăm UI: rulăm prin run_lab10_task dacă vrei implicit; însă tu ai cerut input
        # configurabil, deci pentru task 1 folosim run_nlp_experiment general.
        st.info(f"Processing NLP… (task_id={task_id})")
        with st.spinner("Training / evaluating classifier (may take time)..."):
            if task_id == 1:
                # Backend general: model Naive Bayes + TF-IDF parametri
                res = run_nlp_experiment(
                    dataset=dataset,
                    model_name="Naive Bayes",
                    ngram_range=(int(ng_min), int(ng_max)),
                    max_features=max_features,
                    no_plots=no_plots,
                    seed=int(seed),
                )
                # convertim într-un format compatibil UI
                res = {**res, "task": 1}
            else:
                res = run_lab10_task(
                    task_id=task_id,
                    dataset=dataset,
                    no_plots=no_plots,
                    seed=int(seed),
                )

            st.info(f"Done: task_id={task_id}")
            _render_result(res)

    st.markdown('</div>', unsafe_allow_html=True)




