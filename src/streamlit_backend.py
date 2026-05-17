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
from .utils.genetic_algorithm_tsp import CitySet, random_cityset, run_ga
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


def run_tsp(algo: str, *, n: int, seed: int = 42, **params: Any) -> Dict[str, Any]:
    """Runs a single TSP solver and returns route/cost/timing/plot."""
    rng_seed = seed + int(n)

    # Use existing distance-matrix generator for consistency with other labs.
    dist = genereaza_instanta_tsp(n, rng=np.random.RandomState(rng_seed))  # type: ignore[arg-type]

    # Also keep coords for plotting (GA uses coords; others just use dist).
    # We'll generate a matching coordinate set from seed.
    cityset = random_cityset(n, seed=rng_seed)
    coords = cityset.coords

    start = time.perf_counter()
    plot_fig: Optional[plt.Figure] = None

    if algo == "BKT (Backtracking)":
        bt_mode = params.get("bt_mode", "toate")
        bt_time_limit = float(params.get("bt_time_limit", 15))
        if bt_mode == "toate":
            route, cost = rezolva_tsp_backtracking_extins(n, dist, mod="toate")[0:2]  # type: ignore[misc]
        else:
            route, cost, nr_sol, dur = rezolva_tsp_backtracking_extins(n, dist, mod="timp", timp_max=bt_time_limit)

    elif algo == "HC (Hill Climbing)":
        restarts = int(params.get("restarts", 30))
        iterations = int(params.get("iterations", 2000))
        route, cost = rezolva_tsp_hc(n, dist, reporniri=restarts, iteratii=iterations, seed=seed)

    elif algo == "NN (Nearest Neighbor)":
        start_city = int(params.get("start_city", 0))
        route, cost = rezolva_tsp_nn(n, dist, start=start_city)

    elif algo == "SA (Simulated Annealing)":
        sa = SimulatedAnnealingTSP(
            dist,
            t_max=float(params.get("t_max", 10000.0)),
            t_min=float(params.get("t_min", 1.0)),
            alpha=float(params.get("alpha", 0.995)),
            iterations_per_temp=int(params.get("iters_per_temp", 100)),
            seed=seed,
            fix_start=True,
        )
        init = params.get("init", "nn")
        start_city = int(params.get("start_city", 0))
        res = sa.solve(init=init, start_city=start_city)
        route, cost = res.best_tour, res.best_cost

    elif algo == "GA (Genetic Algorithm - PyGAD)":
        pop_size = int(params.get("pop_size", 120))
        generations = int(params.get("generations", 400))
        mutation_rate_percent = int(params.get("mutation_rate_percent", 45))

        ga_res = run_ga(
            cityset,
            pop_size=pop_size,
            n_generations=generations,
            mutation_rate_percent=mutation_rate_percent,
            seed=seed,
            verbose=False,
        )
        route, cost = ga_res.best_tour, ga_res.best_distance

    else:
        raise ValueError(f"Unknown algo: {algo}")

    duration_s = time.perf_counter() - start

    plot_fig = _plot_tour(coords, route, f"{algo} - route (N={n})")

    return {
        "route": route,
        "route_str": _route_str(route),
        "cost": float(cost),
        "duration_s": float(duration_s),
        "plot": plot_fig,
        "metrics": {},
    }


def run_tsp_comparison(*, n: int, seed: int = 42) -> Dict[str, Any]:

    """Quick comparison table + plot for all algorithms."""
    dist = genereaza_instanta_tsp(n, rng=np.random.RandomState(seed + n))  # type: ignore[arg-type]
    cityset = random_cityset(n, seed=seed + n)
    coords = cityset.coords

    results = []

    # BT
    t0 = time.perf_counter()
    bt_route, bt_cost = rezolva_tsp_backtracking(n, dist)
    bt_dur = time.perf_counter() - t0
    results.append(("Backtracking", bt_cost, bt_dur, bt_route))

    # HC
    t0 = time.perf_counter()
    hc_route, hc_cost = rezolva_tsp_hc(n, dist, reporniri=10, iteratii=1500, seed=seed)
    hc_dur = time.perf_counter() - t0
    results.append(("Hill Climbing", hc_cost, hc_dur, hc_route))

    # NN
    t0 = time.perf_counter()
    nn_route, nn_cost = rezolva_tsp_nn(n, dist, start=0)
    nn_dur = time.perf_counter() - t0
    results.append(("Nearest Neighbor", nn_cost, nn_dur, nn_route))

    # SA
    t0 = time.perf_counter()
    sa = SimulatedAnnealingTSP(dist, t_max=2000.0, t_min=1.0, alpha=0.995, iterations_per_temp=50, seed=seed, fix_start=True)
    sa_res = sa.solve(init="nn", start_city=0)
    sa_dur = time.perf_counter() - t0
    results.append(("Simulated Annealing", sa_res.best_cost, sa_dur, sa_res.best_tour))

    # GA
    t0 = time.perf_counter()
    ga_res = run_ga(cityset, pop_size=80, n_generations=200, mutation_rate_percent=45, seed=seed, verbose=False)
    ga_dur = time.perf_counter() - t0
    results.append(("Genetic Algorithm", ga_res.best_distance, ga_dur, ga_res.best_tour))

    table = [
        {"Algorithm": name, "Cost": float(cost), "Time_s": float(dur)}
        for (name, cost, dur, _route) in results
    ]

    best = min(results, key=lambda x: float(x[1]))
    best_name, best_cost, _best_dur, best_route = best

    fig = _plot_tour(coords, best_route, f"Best route: {best_name} (cost={best_cost:.2f})")

    return {"table": table, "plot": fig, "best": best_name}


# -----------------------
# NLP
# -----------------------


def run_nlp_experiment(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    # kept for stable import from streamlit_app
    return run_nlp_experiment(*args, **kwargs)

