"""Rezolvarea TSP prin backtracking recursiv (branch-and-bound).

Implementare structurata (fara variabile globale) conform cerintelor din laborator.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


Matrix = List[List[int]]


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
	if n <= 0:
		raise ValueError("n trebuie sa fie > 0")
	if len(matrice) != n or any(len(row) != n for row in matrice):
		raise ValueError("matrice trebuie sa fie NxN")

	if n == 1:
		return [0], 0

	best_cost = [float("inf")]
	best_route: List[int] = []

	visited = [False] * n
	visited[0] = True
	route = [0]

	def backtrack(current_city: int, current_cost: int) -> None:
		if len(route) == n:
			total_cost = current_cost + matrice[current_city][0]
			if total_cost < best_cost[0]:
				best_cost[0] = total_cost
				best_route.clear()
				best_route.extend(route)
			return

		for next_city in range(n):
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

	backtrack(0, 0)
	return best_route, int(best_cost[0])
