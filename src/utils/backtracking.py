from __future__ import annotations

import random
import time
import heapq
from typing import List, Optional, Tuple
from typing import List, Literal, Sequence, Tuple


Matrix = List[List[int]]

ModOprire = Literal["prima", "toate", "timp", "y_solutii"]

def rezolva_tsp_backtracking(n: int, matrice: Matrix, mod: str = 'exhaustiv', Y: Optional[int] = None, time_limit_s: Optional[float] = None) -> Tuple[List[int], int]:
    "Rezolva TSP cu backtracking: 'prima', 'y_solutii', 'exhaustiv'."
    if n <= 0:
        raise ValueError("n > 0 required")
    if len(matrice) != n or any(len(row) != n for row in matrice):
        raise ValueError("matrice NxN required")

    if n == 1:
        return [0], 0

    if mod not in ['prima', 'y_solutii', 'exhaustiv']:
        raise ValueError("Invalid mod")

    if Y is None:
        Y = n if mod == 'y_solutii' else 1

    top_solutions = []
    found_first = None
    visited = [False] * n
    route = [0]
    visited[0] = True
    start_time = time.perf_counter() if time_limit_s else 0

    def backtrack(current, cost_cur):
        nonlocal found_first
        if time_limit_s and time.perf_counter() - start_time > time_limit_s:
            return

        if len(route) == n:
            cost_total = cost_cur + matrice[current][0]
            if mod == 'prima':
                if found_first is None:
                    found_first = (route[:], cost_total)
                return
            tup = (cost_total, route[:])
            heapq.heappush(top_solutions, tup)
            if len(top_solutions) > Y:
                heapq.heappop(top_solutions)
            return

        for next_city in range(n):
            if visited[next_city]:
                continue
            cost_new = cost_cur + matrice[current][next_city]
            if top_solutions and cost_new >= top_solutions[0][0]:
                continue
            visited[next_city] = True
            route.append(next_city)
            backtrack(next_city, cost_new)
            route.pop()
            visited[next_city] = False

    backtrack(0, 0)

    if mod == 'prima' and found_first:
        return found_first
    if top_solutions:
        _, route_best = top_solutions[0]
        return route_best, top_solutions[0][0]
    return [0], 0

def _cost_tur(traseu: Sequence[int], matrice: Matrix) -> int:
	"""Calculeaza costul turului, incluzand revenirea la orasul de start."""
	if not traseu:
		return 0
	cost = 0
	for i in range(len(traseu) - 1):
		cost += matrice[traseu[i]][traseu[i + 1]]
	cost += matrice[traseu[-1]][traseu[0]]
	return cost


def rezolva_tsp_backtracking_extins(
	n: int,
	matrice: Matrix,
	*,
	mod: ModOprire = "toate",
	timp_max: float | None = None,
	y_max: int | None = None,
	progress_fn=None,
) -> Tuple[List[int], int, int, float]:
	"""Rezolva TSP prin backtracking cu moduri configurabile de oprire.

	Moduri:
		- prima: se opreste la primul tur complet gasit
		- toate: exploreaza exhaustiv cu prunere branch-and-bound (optim garantat)
		- timp: se opreste dupa cel mult `timp_max` secunde (returneaza cel mai bun tur gasit)
		- y_solutii: se opreste dupa gasirea a `y_max` tururi complete (returneaza cel mai bun)

	Args:
		n: Numarul de orase.
		matrice: Matricea de distante NxN (simetrica, diagonala 0).
		mod: Modul de oprire.
		timp_max: Limita de timp in secunde (doar pentru mod="timp").
		y_max: Numar maxim de solutii complete (doar pentru mod="y_solutii").

	Returns:
		(traseu, cost, nr_solutii_gasite, durata_secunde)

	Raises:
		ValueError: Daca parametrii sunt invalizi.
	"""
	if n <= 0:
		raise ValueError("n trebuie sa fie > 0")
	if len(matrice) != n or any(len(row) != n for row in matrice):
		raise ValueError("matrice trebuie sa fie NxN")
	if mod not in ("prima", "toate", "timp", "y_solutii"):
		raise ValueError("mod invalid")
	if mod == "timp":
		if timp_max is None or timp_max <= 0:
			raise ValueError("timp_max trebuie sa fie > 0 pentru mod='timp'")
	if mod == "y_solutii":
		if y_max is None or y_max <= 0:
			raise ValueError("y_max trebuie sa fie > 0 pentru mod='y_solutii'")

	start_time = time.perf_counter()
	deadline = (start_time + float(timp_max)) if mod == "timp" and timp_max is not None else None

	if n == 1:
		duration = time.perf_counter() - start_time
		return [0], 0, 1, duration

	# Bound initial (un tur valid rapid) pentru a avea mereu o solutie completa.
	traseu_initial = list(range(n))
	best_cost = [_cost_tur(traseu_initial, matrice)]
	best_route: List[int] = traseu_initial.copy()

	visited = [False] * n
	visited[0] = True
	route = [0]

	# Ordinea in care backtrack exploreaza orasele urmatoare.
	# La restart se amesteca (shuffle) pentru a explora ramuri diferite.
	city_order = list(range(1, n))

	nr_solutii = [0]
	oprire = [False]
	call_count = [0]
	last_progress_time = [start_time]

	def timp_expirat() -> bool:
		return deadline is not None and time.perf_counter() >= deadline

	def backtrack(current_city: int, current_cost: int) -> None:
		if oprire[0]:
			return
		if deadline is not None and timp_expirat():
			oprire[0] = True
			return

		# Raportare progres aproximativ o data pe secunda
		call_count[0] += 1
		if progress_fn is not None and call_count[0] % 20_000 == 0:
			now = time.perf_counter()
			if now - last_progress_time[0] >= 1.0:
				last_progress_time[0] = now
				elapsed = now - start_time
				remaining = max(0.0, float(timp_max or 0) - elapsed)
				progress_fn(elapsed, remaining, nr_solutii[0], best_cost[0])

		if len(route) == n:
			nr_solutii[0] += 1
			total_cost = current_cost + matrice[current_city][0]
			if total_cost < best_cost[0]:
				best_cost[0] = total_cost
				best_route.clear()
				best_route.extend(route)
			if mod == "prima":
				oprire[0] = True
				return
			if mod == "y_solutii" and y_max is not None and nr_solutii[0] >= y_max:
				oprire[0] = True
				return
			return

		for next_city in city_order:
			if visited[next_city]:
				continue

			new_cost = current_cost + matrice[current_city][next_city]
			if new_cost >= best_cost[0]:
				continue

			visited[next_city] = True
			route.append(next_city)
			backtrack(next_city, new_cost)
			route.pop()
			visited[next_city] = False

			if oprire[0]:
				return

	# --- Prima rulare (ordine naturala) ---
	backtrack(0, 0)

	# --- Restart-uri cu ordine randomizata pana la deadline ---
	# Fiecare restart amesteca ordinea de explorare a oraselor,
	# astfel branch-and-bound taie ramuri diferite si poate gasi
	# solutii mai bune. best_cost / best_route se pastreaza intre rulari.
	if mod == "timp" and deadline is not None:
		rng = random.Random(42)
		while not timp_expirat():
			oprire[0] = False
			rng.shuffle(city_order)
			for i in range(n):
				visited[i] = False
			visited[0] = True
			route.clear()
			route.append(0)
			backtrack(0, 0)

	duration = time.perf_counter() - start_time
	return best_route, int(best_cost[0]), int(nr_solutii[0]), duration


def rezolva_tsp_backtracking(n: int, matrice: Matrix) -> Tuple[List[int], int]:
	"""Rezolva problema comis-voiajorului (TSP) optim, folosind backtracking.

	Algoritmul fixeaza orasul de start la 0 (optimizare pentru TSP simetric) si
	foloseste prunere branch-and-bound: daca un cost partial depaseste cea mai buna
	solutie cunoscuta, ramura este abandonata.

	Args:
		n: Numarul de orase.
		matrice: Matricea de distante NxN (simetrica, diagonala 0).

	Returns:
		Un tuplu (traseu_optim, cost_minim), unde traseu_optim este lista de orase
		in ordinea vizitarii (incepe cu 0). Costul include revenirea la orasul 0.

	Raises:
		ValueError: Daca n este invalid sau matricea nu are dimensiunea corecta.
	"""
	route, cost, _, _ = rezolva_tsp_backtracking_extins(n, matrice, mod="toate")
	return route, cost
