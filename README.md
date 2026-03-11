# TSP Proiect — Laborator #03 (Backtracking vs Hill Climbing)

Proiect Python structurat pentru rezolvarea problemei Comis-Voiajorului (TSP):

- **Backtracking (optim, exhaustiv + branch-and-bound)**
- **Hill Climbing (euristic) cu `simpleai` + random restarts + vecinătate 2-opt**
- **Experiment comparativ** cu grafic PNG (scară liniară + logaritmică)

## Cerințe

- Python 3.x
- Dependențe: `simpleai`, `matplotlib`, `seaborn`

Instalare dependențe:

```bash
pip install -r requirements.txt
```

## Rulare (Windows)

### 1) Activare venv (opțional, recomandat)

Dacă folosești PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Dacă folosești Command Prompt:

```bat
.venv\Scripts\activate
```

### 2) Comenzi CLI

Scriptul de intrare este: `src/main.py`.

Ajutor:

```bash
python src/main.py --help
python src/main.py solve --help
python src/main.py experiment --help
```

Rezolvare TSP din fișier (backtracking):

```bash
python src/main.py solve <fisier_intrare> --algo bt
```

Rezolvare TSP din fișier (hill climbing):

```bash
python src/main.py solve <fisier_intrare> --algo hc --restarts 30 --iterations 2000 --seed 42
```

Salvare rezultat într-un fișier text:

```bash
python src/main.py solve <fisier_intrare> --algo bt --output rezultat.txt
```

Rulare experiment + generare grafic:

```bash
python src/main.py experiment --output comparare_performanta.png
```

## Format fișier de intrare (ex. `orase.txt`)

Fișierul de intrare este un text simplu:

1. Prima linie: `N` (numărul de orașe)
2. Următoarele `N` linii: matricea `N×N` de distanțe (simetrică, diagonala 0)

Exemplu:

```text
4
0 10 15 20
10 0 35 25
15 35 0 30
20 25 30 0
```

## Output

Comanda `solve` afișează:

- numărul de orașe
- algoritmul folosit
- traseul (închis: revine la start)
- costul total
- timpul de execuție

Comanda `experiment` salvează fișierul PNG cu două subploturi (liniar + `semilogy`).

## Structura proiectului

```text
requirements.txt
src/
	main.py
	utils/
		backtracking.py
		hill_climbing_tsp.py
		io_utils.py
		performance.py
```
