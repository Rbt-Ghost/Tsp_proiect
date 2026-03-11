"""Punct de intrare pentru proiectul TSP (laborator #03).

Ofera:
  - rezolvare TSP prin backtracking (optim) sau hill climbing (euristic)
  - rularea experimentului comparativ si generarea graficului PNG
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from utils.backtracking import rezolva_tsp_backtracking
from utils.hill_climbing_tsp import rezolva_tsp_hc
from utils.io_utils import citeste_matrice, formateaza_traseu, salveaza_rezultat
from utils.performance import ruleaza_experiment


def _cmd_solve(args: argparse.Namespace) -> int:
	n, matrix = citeste_matrice(args.input)

	start = time.perf_counter()
	if args.algo == "bt":
		route, cost = rezolva_tsp_backtracking(n, matrix)
		algo_name = "backtracking"
	else:
		route, cost = rezolva_tsp_hc(
			n,
			matrix,
			reporniri=args.restarts,
			iteratii=args.iterations,
			seed=args.seed,
		)
		algo_name = "hill_climbing_random_restarts"
	duration = time.perf_counter() - start

	print(f"Numar de orase: {n}")
	print(f"Algoritm: {algo_name}")
	print(f"Traseu: {formateaza_traseu(route)}")
	print(f"Cost: {cost}")
	print(f"Timp: {duration:.6f} secunde")

	if args.output:
		salveaza_rezultat(args.output, route, cost, duration, algo_name)
		print(f"Rezultat salvat in: {args.output}")

	return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
	out = ruleaza_experiment(
		output_png=args.output,
		seed=args.seed,
		reporniri_hc=args.restarts,
		iteratii_hc=args.iterations,
		bt_time_limit_s=args.bt_time_limit,
	)
	out = Path(out)
	print(f"Grafic salvat: {out.resolve()}")
	return 0


def build_parser() -> argparse.ArgumentParser:
	"""Construieste parserul CLI."""
	parser = argparse.ArgumentParser(prog="tsp_proiect", description="TSP: backtracking vs hill climbing")
	sub = parser.add_subparsers(dest="command", required=True)

	p_solve = sub.add_parser("solve", help="Rezolva o instanta TSP din fisier")
	p_solve.add_argument("input", help="Fisier input cu matricea NxN")
	p_solve.add_argument("--algo", choices=["bt", "hc"], default="bt", help="Algoritm: bt/backtracking sau hc/hill climbing")
	p_solve.add_argument("--restarts", type=int, default=30, help="Reporniri (doar pentru hc)")
	p_solve.add_argument("--iterations", type=int, default=2000, help="Iteratii per repornire (doar pentru hc)")
	p_solve.add_argument("--seed", type=int, default=42, help="Seed pentru hc/experiment")
	p_solve.add_argument("--output", help="Fisier de iesire (optional)")
	p_solve.set_defaults(func=_cmd_solve)

	p_exp = sub.add_parser("experiment", help="Ruleaza experimentul si genereaza comparare_performanta.png")
	p_exp.add_argument("--output", default="comparare_performanta.png", help="Cale PNG output")
	p_exp.add_argument("--seed", type=int, default=42, help="Seed pentru generarea instantelor")
	p_exp.add_argument("--restarts", type=int, default=30, help="Reporniri hill climbing")
	p_exp.add_argument("--iterations", type=int, default=2000, help="Iteratii per repornire")
	p_exp.add_argument("--bt-time-limit", type=float, default=30.0, help="Prag (sec) pentru backtracking")
	p_exp.set_defaults(func=_cmd_experiment)

	return parser


def main(argv: list[str] | None = None) -> int:
	"""Entry point pentru rulare ca script."""
	parser = build_parser()
	args = parser.parse_args(argv)
	return int(args.func(args))


if __name__ == "__main__":
	raise SystemExit(main())
