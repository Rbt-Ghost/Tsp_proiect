"""Rezolvarea TSP cu algoritmul alpinistului (Hill Climbing) folosind `simpleai`.

Modelare:
- Starea este un tuplu cu ordinea oraselor (ex. (0, 3, 1, 2)).
- Vecinatatea foloseste miscari 2-opt: inversarea unui segment din tur.
- `simpleai` maximizeaza `value`, deci folosim `value(state) = -cost(state)`.

Conform cerintelor laboratorului se recomanda varianta cu reporniri aleatorii.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Sequence, Tuple

from simpleai.search import SearchProblem
from simpleai.search.local import hill_climbing_random_restarts


Matrix = List[List[int]]
State = Tuple[int, ...]
Action = Tuple[int, int]  # (i, j) -> inverseaza segmentul [i, j]


def _tsp_cost(state: Sequence[int], matrice: Matrix) -> int:
    """Calculeaza costul total al turului, incluzand revenirea la start."""
    n = len(state)
    if n == 0:
        return 0
    cost = 0
    for k in range(n - 1):
        cost += matrice[state[k]][state[k + 1]]
    cost += matrice[state[-1]][state[0]]
    return cost


@dataclass
class TSPHillClimbing(SearchProblem):
    """Problema TSP pentru cautare locala in `simpleai`.

    Starea este un tur complet (permutare) cu orasul 0 fixat pe prima pozitie.

    Args:
        matrice: Matricea de distante NxN.
        seed: Seed pentru random, pentru reproductibilitate.

    Notes:
        - Orasul 0 este fixat pe prima pozitie pentru a elimina rotatiile echivalente.
        - Vecinii sunt generati prin 2-opt cu indecsi in [1, n-1] (nu mutam orasul 0).
    """

    matrice: Matrix
    seed: int = 42

    def __post_init__(self) -> None:
        n = len(self.matrice)
        initial_state = tuple(range(n))
        super().__init__(initial_state=initial_state)
        self._rng = random.Random(self.seed)

    @property
    def n(self) -> int:
        """Numarul de orase."""
        return len(self.matrice)

    def actions(self, state: State) -> Iterable[Action]:
        """Genereaza toate miscarile 2-opt posibile."""
        n = len(state)
        # i si j pornind de la 1 ca sa pastram orasul 0 pe pozitia 0.
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                yield (i, j)

    def result(self, state: State, action: Action) -> State:
        """Aplica o miscare 2-opt: inverseaza segmentul [i, j]."""
        i, j = action
        if i <= 0:
            # safety: nu ar trebui sa se intample, dar pastram state[0] fix.
            i = 1
        if j <= i:
            return state
        new_state = list(state)
        new_state[i : j + 1] = reversed(new_state[i : j + 1])
        return tuple(new_state)

    def value(self, state: State) -> float:
        """Functia de evaluare maximizata de `simpleai`.

        Returns:
            -cost(tur)
        """
        return -float(_tsp_cost(state, self.matrice))

    def generate_random_state(self) -> State:
        """Genereaza o stare initiala aleatoare (tur aleator), cu 0 fixat."""
        if self.n <= 1:
            return tuple(range(self.n))
        rest = list(range(1, self.n))
        self._rng.shuffle(rest)
        return tuple([0] + rest)

    # Unele versiuni/simpleai pot apela si `random_state`; il oferim ca alias.
    def random_state(self) -> State:
        """Alias pentru generate_random_state() pentru compatibilitate."""
        return self.generate_random_state()


def rezolva_tsp_hc(
    n: int,
    matrice: Matrix,
    reporniri: int = 30,
    iteratii: int = 2000,
    seed: int = 42,
) -> Tuple[List[int], int]:
    """Rezolva TSP prin hill climbing cu reporniri aleatorii.

    Args:
        n: Numarul de orase.
        matrice: Matricea de distante NxN.
        reporniri: Numarul de reporniri aleatorii (random restarts).
        iteratii: Limita de iteratii per repornire (iterations_limit in simpleai).
        seed: Seed pentru reproductibilitate (afecteaza starile initiale).

    Returns:
        Un tuplu (traseu, cost) compatibil cu backtracking.

    Raises:
        ValueError: Daca n/matrice sunt invalide.
    """
    if n != len(matrice):
        raise ValueError("n nu corespunde cu dimensiunea matricei")

    problem = TSPHillClimbing(matrice=matrice, seed=seed)

    result_state = hill_climbing_random_restarts(
        problem,
        restarts_limit=reporniri,
        iterations_limit=iteratii,
    )

    # `simpleai` poate intoarce fie starea, fie un Node-like; acceptam ambele.
    state = getattr(result_state, "state", result_state)
    route = list(state)
    cost = _tsp_cost(route, matrice)
    return route, cost
