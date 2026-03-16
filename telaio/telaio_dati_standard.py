"""
telaio_dati_standard.py
Materiali e sezioni standard pre-caricati per il modulo telaio FEM.

Unità di misura utilizzate nell'interfaccia:
  - Modulo elastico E       → MPa  (N/mm²)
  - Modulo taglio G         → MPa  (N/mm²)
  - Densità rho             → kg/m³
  - Area A                  → mm²
  - Inerzie Iy, Iz          → mm⁴
  - Modulo torsione J       → mm⁴
  - Posizioni nodi          → m
  - Carichi concentrati     → kN, kNm
  - Carichi distribuiti     → kN/m

ATTENZIONE: i valori E, G vengono convertiti in Pa (×1e6) e
A, Iy, Iz, J in m² / m⁴ (÷1e6 / ÷1e12) prima di passarli al
motore FEM (che lavora in unità SI: N, m, Pa).
"""

# ──────────────────────────────────────────────────────────────────────
# MATERIALI STANDARD
# Chiave  → id univoco stringa
# E, G    → MPa
# rho     → kg/m³
# ──────────────────────────────────────────────────────────────────────
MATERIALI_STANDARD: dict[str, dict] = {
    "mat_0": {
        "nome": "Acciaio S275",
        "E":    210_000.0,   # MPa
        "G":     81_000.0,   # MPa
        "rho":    7_850.0,   # kg/m³
    },
    "mat_1": {
        "nome": "Acciaio S355",
        "E":    210_000.0,
        "G":     81_000.0,
        "rho":    7_850.0,
    },
    "mat_2": {
        "nome": "Cls C25/30",
        "E":     31_000.0,
        "G":     12_900.0,
        "rho":    2_500.0,
    },
}

# ──────────────────────────────────────────────────────────────────────
# SEZIONI STANDARD  (profili laminati / equivalenti tipici)
# A   → mm²
# Iy  → mm⁴   (inerzia flessione piano xz  → Vz, My)
# Iz  → mm⁴   (inerzia flessione piano xy  → Vy, Mz)
# J   → mm⁴   (costante torsionale)
# ──────────────────────────────────────────────────────────────────────
SEZIONI_STANDARD: dict[str, dict] = {
    "sez_0": {
        "nome":      "HEB 200",
        "A":         7_810.0,
        "Iy":    57_000_000.0,
        "Iz":    20_000_000.0,
        "J":       123_000.0,
        "materiale": "mat_0",
    },
    "sez_1": {
        "nome":      "IPE 200",
        "A":         2_848.0,
        "Iy":    19_430_000.0,
        "Iz":       512_700.0,
        "J":         6_920.0,
        "materiale": "mat_0",
    },
    "sez_2": {
        "nome":      "IPE 300",
        "A":         5_381.0,
        "Iy":    83_560_000.0,
        "Iz":     2_190_000.0,
        "J":        20_100.0,
        "materiale": "mat_0",
    },
}
