from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt

from .utils.backtracking import rezolva_tsp_backtracking, rezolva_tsp_backtracking_extins
from .utils.hill_climbing_tsp import rezolva_tsp_hc
from .utils.nearest_neighbor import (
    rezolva_tsp_nn,
    rezolva_tsp_nn_multistart,
    rezolva_tsp_nn_timp,
)
from .utils.simulated_annealing_tsp import SimulatedAnnealingTSP
from .utils.genetic_algorithm_tsp import CitySet, random_cityset, run_ga, build_distance_matrix
from .utils.performance import genereaza_instanta_tsp
from .streamlit_backend_nlp import run_nlp_experiment


from pathlib import Path
import os
from typing import Sequence


# -----------------------
# Streamlit artifacts
# -----------------------

STREAMLIT_OUTDIR = Path(__file__).resolve().parents[1] / "streamlit_out"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _list_images(folder: Path, *, exts: Sequence[str] = (".png", ".jpg", ".jpeg")) -> List[str]:
    if not folder.exists():
        return []
    imgs: List[str] = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in set(e.lower() for e in exts):
            imgs.append(str(f))
    return imgs


# -----------------------
# TSP helpers
# -----------------------


tsp_algos = ["BKT (Backtracking)", "HC (Hill Climbing)", "NN (Nearest Neighbor)", "SA (Simulated Annealing)", "GA (Genetic Algorithm - PyGAD)"]


def _route_str(route: List[int]) -> str:
    if not route:
        return "(empty)"
    return " -> ".join(str(x) for x in route + [route[0]]) if route else ""


def _plot_tour(coords: np.ndarray, route: List[int], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if coords.size == 0 or not route:
        ax.set_title(title)
        return fig

    ordered = route + [route[0]]
    xs = coords[ordered, 0]
    ys = coords[ordered, 1]

    ax.plot(xs, ys, "-o", linewidth=2, markersize=5)
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), fontsize=10)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def run_tsp(algo: str, *, n: int, seed: int = 42, log_fn=None, progress_fn=None, **params: Any) -> Dict[str, Any]:
    """Runs a single TSP solver and returns route/cost/timing/plot."""

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    rng_seed = seed + int(n)

    coords_input = params.pop("coords", None)
    if coords_input is not None:
        coords = np.array(coords_input, dtype=float)
        n = int(coords.shape[0])
        _log(f"📥 Input data loaded — {n} cities")
        _log(f"📐 Building Euclidean distance matrix ({n}×{n})...")
        dist = build_distance_matrix(coords)
        names = [str(i) for i in range(n)]
        cityset = CitySet(names=names, coords=coords)
    else:
        _log(f"🎲 Generating random instance — {n} cities, seed={rng_seed}")
        _log(f"📐 Generating distance matrix ({n}×{n})...")
        dist = genereaza_instanta_tsp(n, rng=np.random.RandomState(rng_seed))  # type: ignore[arg-type]
        cityset = random_cityset(n, seed=rng_seed)
        coords = cityset.coords

    start = time.perf_counter()
    plot_fig: Optional[plt.Figure] = None

    if algo == "BKT (Backtracking)":
        BKT_TIME = 120.0
        _log(f"🔁 Starting Backtracking — run time: {BKT_TIME:.0f}s")
        _log("⏳ Exploring solution tree (branch-and-bound)...")

        route, cost, nr_sol, dur = rezolva_tsp_backtracking_extins(
            n, dist, mod="timp", timp_max=BKT_TIME, progress_fn=progress_fn
        )
        _log(f"✔️ Backtracking done — cost={cost:.2f}, {nr_sol} solutions in {dur:.2f}s")

    elif algo == "HC (Hill Climbing)":
        restarts = int(params.get("restarts", 30))
        iterations = int(params.get("iterations", 2000))
        _log(f"🏔️ Starting Hill Climbing | restarts={restarts}, iterations/restart={iterations}")
        _log(f"⏳ Searching local optimum ({restarts * iterations:,} total iterations)...")
        route, cost = rezolva_tsp_hc(n, dist, reporniri=restarts, iteratii=iterations, seed=seed)
        _log(f"✔️ Hill Climbing done — cost={cost:.2f}")

    elif algo == "NN (Nearest Neighbor)":
        start_city = int(params.get("start_city", 0))
        _log(f"📍 Starting Nearest Neighbor | start city={start_city}")
        _log("⏳ Building greedy tour (nearest neighbor)...")
        route, cost = rezolva_tsp_nn(n, dist, start=start_city)
        _log(f"✔️ Nearest Neighbor done — cost={cost:.2f}")

    elif algo == "SA (Simulated Annealing)":
        t_max = float(params.get("t_max", 10000.0))
        t_min = float(params.get("t_min", 1.0))
        alpha = float(params.get("alpha", 0.995))
        iters_per_temp = int(params.get("iters_per_temp", 100))
        init = params.get("init", "nn")
        _log(f"🌡️ Starting Simulated Annealing | T_max={t_max}, T_min={t_min}, α={alpha}")
        _log(f"   Iterations/temperature={iters_per_temp}, initialization='{init}'")
        n_steps = 0
        T = t_max
        while T > t_min:
            n_steps += iters_per_temp
            T *= alpha
        _log(f"⏳ Simulating cooling ({n_steps:,} estimated steps)...")
        sa = SimulatedAnnealingTSP(
            dist,
            t_max=t_max,
            t_min=t_min,
            alpha=alpha,
            iterations_per_temp=iters_per_temp,
            seed=seed,
            fix_start=True,
        )
        start_city = int(params.get("start_city", 0))
        res = sa.solve(init=init, start_city=start_city)
        route, cost = res.best_tour, res.best_cost
        _log(f"✔️ Simulated Annealing done — cost={cost:.2f}")

    elif algo == "GA (Genetic Algorithm - PyGAD)":
        pop_size = int(params.get("pop_size", 120))
        generations = int(params.get("generations", 400))
        mutation_rate_percent = int(params.get("mutation_rate_percent", 45))
        _log(f"🧬 Starting Genetic Algorithm | population={pop_size}, generations={generations}, mutation={mutation_rate_percent}%")
        _log(f"⏳ Evolving population ({pop_size * generations:,} fitness evaluations)...")
        ga_res = run_ga(
            cityset,
            pop_size=pop_size,
            n_generations=generations,
            mutation_rate_percent=mutation_rate_percent,
            seed=seed,
            verbose=False,
        )
        route, cost = ga_res.best_tour, ga_res.best_distance
        _log(f"✔️ Genetic Algorithm done — cost={cost:.2f}")

    else:
        raise ValueError(f"Unknown algo: {algo}")

    duration_s = time.perf_counter() - start

    _log("🗺️ Generating route chart...")
    plot_fig = _plot_tour(coords, route, f"{algo} - route (N={n})")
    _log(f"✅ Complete! Cost={cost:.2f} | Time={duration_s:.4f}s")

    return {
        "route": route,
        "route_str": _route_str(route),
        "cost": float(cost),
        "duration_s": float(duration_s),
        "plot": plot_fig,
        "metrics": {},
    }


def run_tsp_comparison(*, n: int, seed: int = 42, coords: Optional[List] = None, log_fn=None) -> Dict[str, Any]:
    """Quick comparison table + plot for all algorithms."""

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    if coords is not None:
        coords_arr = np.array(coords, dtype=float)
        n = int(coords_arr.shape[0])
        _log(f"📥 Input data loaded — {n} cities")
        _log(f"📐 Building Euclidean distance matrix ({n}×{n})...")
        dist = build_distance_matrix(coords_arr)
        cityset = CitySet(names=[str(i) for i in range(n)], coords=coords_arr)
        coords_plot = coords_arr
    else:
        _log(f"🎲 Generating random instance — {n} cities")
        _log(f"📐 Generating distance matrix ({n}×{n})...")
        dist = genereaza_instanta_tsp(n, rng=np.random.RandomState(seed + n))  # type: ignore[arg-type]
        cityset = random_cityset(n, seed=seed + n)
        coords_plot = cityset.coords

    results = []

    # BT
    _log("🔁 [1/5] Starting Backtracking...")
    t0 = time.perf_counter()
    bt_route, bt_cost = rezolva_tsp_backtracking(n, dist)
    bt_dur = time.perf_counter() - t0
    results.append(("Backtracking", bt_cost, bt_dur, bt_route))
    _log(f"   ✔️ Backtracking — cost={bt_cost:.2f}, time={bt_dur:.4f}s")

    # HC
    _log("🏔️ [2/5] Starting Hill Climbing (restarts=10, iterations=1500)...")
    t0 = time.perf_counter()
    hc_route, hc_cost = rezolva_tsp_hc(n, dist, reporniri=10, iteratii=1500, seed=seed)
    hc_dur = time.perf_counter() - t0
    results.append(("Hill Climbing", hc_cost, hc_dur, hc_route))
    _log(f"   ✔️ Hill Climbing — cost={hc_cost:.2f}, time={hc_dur:.4f}s")

    # NN
    _log("📍 [3/5] Starting Nearest Neighbor (start=0)...")
    t0 = time.perf_counter()
    nn_route, nn_cost = rezolva_tsp_nn(n, dist, start=0)
    nn_dur = time.perf_counter() - t0
    results.append(("Nearest Neighbor", nn_cost, nn_dur, nn_route))
    _log(f"   ✔️ Nearest Neighbor — cost={nn_cost:.2f}, time={nn_dur:.4f}s")

    # SA
    _log("🌡️ [4/5] Starting Simulated Annealing (T_max=2000, α=0.995)...")
    t0 = time.perf_counter()
    sa = SimulatedAnnealingTSP(dist, t_max=2000.0, t_min=1.0, alpha=0.995, iterations_per_temp=50, seed=seed, fix_start=True)
    sa_res = sa.solve(init="nn", start_city=0)
    sa_dur = time.perf_counter() - t0
    results.append(("Simulated Annealing", sa_res.best_cost, sa_dur, sa_res.best_tour))
    _log(f"   ✔️ Simulated Annealing — cost={sa_res.best_cost:.2f}, time={sa_dur:.4f}s")

    # GA
    _log("🧬 [5/5] Starting Genetic Algorithm (pop=80, generations=200)...")
    t0 = time.perf_counter()
    ga_res = run_ga(cityset, pop_size=80, n_generations=200, mutation_rate_percent=45, seed=seed, verbose=False)
    ga_dur = time.perf_counter() - t0
    results.append(("Genetic Algorithm", ga_res.best_distance, ga_dur, ga_res.best_tour))
    _log(f"   ✔️ Genetic Algorithm — cost={ga_res.best_distance:.2f}, time={ga_dur:.4f}s")

    table = [
        {"Algorithm": name, "Cost": float(cost), "Time_s": float(dur)}
        for (name, cost, dur, _route) in results
    ]

    best = min(results, key=lambda x: float(x[1]))
    best_name, best_cost, _best_dur, best_route = best

    _log(f"🏆 Winner: {best_name} with cost={best_cost:.2f}")
    _log("🗺️ Generating optimal route chart...")
    fig = _plot_tour(coords_plot, best_route, f"Best route: {best_name} (cost={best_cost:.2f})")
    _log("✅ Comparison complete!")

    return {"table": table, "plot": fig, "best": best_name}


# -----------------------
# NLP
# -----------------------


def run_nlp_experiment(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    # kept for stable import from streamlit_app
    return run_nlp_experiment(*args, **kwargs)

