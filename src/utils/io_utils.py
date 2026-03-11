"""Utilitare pentru citirea/scrierea datelor si rezultatelor pentru TSP.

Modulul contine functii pentru:
  - citirea matricei de distante din fisier (formatul laboratorului)
  - formatarea unui traseu TSP pentru afisare
  - salvarea rezultatului intr-un fisier text (optional)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple


Matrix = List[List[int]]


def citeste_matrice(cale_fisier: str | Path) -> Tuple[int, Matrix]:
	"""Citeste matricea de distante dintr-un fisier text.

	Formatul fisierului:
		N
		D[0][0] D[0][1] ... D[0][N-1]
		...
		D[N-1][0] ...      D[N-1][N-1]

	Args:
		cale_fisier: Calea catre fisierul de intrare.

	Returns:
		Un tuplu (n, matrice) unde:
		  - n este numarul de orase
		  - matrice este lista NxN de intregi

	Raises:
		FileNotFoundError: Daca fisierul nu exista.
		ValueError: Daca formatul sau dimensiunile sunt invalide.
	"""
	path = Path(cale_fisier)
	text = path.read_text(encoding="utf-8").splitlines()
	lines = [line.strip() for line in text if line.strip()]
	if not lines:
		raise ValueError("Fisierul este gol.")

	try:
		n = int(lines[0])
	except ValueError as exc:
		raise ValueError("Prima linie trebuie sa fie un intreg N.") from exc

	if n <= 0:
		raise ValueError("N trebuie sa fie > 0.")

	if len(lines) - 1 != n:
		raise ValueError(f"Sunt necesare {n} linii pentru matrice, dar am gasit {len(lines) - 1}.")

	matrix: Matrix = []
	for i in range(n):
		parts = lines[i + 1].split()
		if len(parts) != n:
			raise ValueError(f"Linia {i + 2} trebuie sa contina {n} valori.")
		try:
			row = [int(x) for x in parts]
		except ValueError as exc:
			raise ValueError(f"Linia {i + 2} contine valori non-intregi.") from exc
		matrix.append(row)

	_valideaza_matrice(n, matrix)
	return n, matrix


def _valideaza_matrice(n: int, matrice: Matrix) -> None:
	"""Valideaza proprietatile de baza ale matricei de distante."""
	if len(matrice) != n or any(len(row) != n for row in matrice):
		raise ValueError("Matricea trebuie sa fie NxN.")

	for i in range(n):
		if matrice[i][i] != 0:
			raise ValueError("Diagonala matricei trebuie sa fie 0.")
		for j in range(n):
			if i == j:
				continue
			if matrice[i][j] <= 0:
				raise ValueError("Toate distantele off-diagonal trebuie sa fie strict pozitive.")
			if matrice[i][j] != matrice[j][i]:
				raise ValueError("Matricea trebuie sa fie simetrica (TSP simetric).")


def formateaza_traseu(traseu: Sequence[int]) -> str:
	"""Formateaza traseul pentru afisare, inchizand turul (revenire la start).

	Args:
		traseu: Secventa de orase in ordinea vizitarii (de obicei incepe cu 0).

	Returns:
		Un string de forma "0 -> 1 -> 3 -> 2 -> 0".
	"""
	if not traseu:
		return ""
	parts = [str(x) for x in traseu]
	parts.append(str(traseu[0]))
	return " -> ".join(parts)


def salveaza_rezultat(
	cale_fisier: str | Path,
	traseu: Sequence[int],
	cost: int,
	durata_secunde: float,
	algoritm: str,
) -> None:
	"""Salveaza rezultatul intr-un fisier text.

	Args:
		cale_fisier: Calea catre fisierul de iesire.
		traseu: Traseul (lista/tuplu) de orase.
		cost: Costul total al turului.
		durata_secunde: Timpul de executie.
		algoritm: Denumirea algoritmului (ex. "backtracking", "hill_climbing").
	"""
	path = Path(cale_fisier)
	path.write_text(
		"\n".join(
			[
				f"Algoritm: {algoritm}",
				f"Traseu: {formateaza_traseu(traseu)}",
				f"Cost: {cost}",
				f"Timp: {durata_secunde:.6f} secunde",
			]
		)
		+ "\n",
		encoding="utf-8",
	)
