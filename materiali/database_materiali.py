"""
database_materiali.py
---------------------
Genera il database standard dei materiali (EC2 / NTC2018) e lo persiste
come  materiali/database_materiali.json  (creato automaticamente al primo avvio).

Il JSON è il punto di riferimento per i valori di default:
  - quando si crea un nuovo progetto, il database viene copiato nella sezione
    "materiali" del file .scprj
  - qualunque modifica successiva vive solo nel .scprj, non qui

Convenzione deformazioni (Tipica Cemento Armato):
  ε > 0, σ > 0  →  compressione
  ε < 0, σ < 0  →  trazione

Le formule usano 'x' come variabile e vengono valutate con numpy in grafico_materiali.py
"""

import json
import os

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "database_materiali.json")

# Incrementa questo numero ogni volta che modifichi le formule per forzare
# l'aggiornamento automatico del file .json sul disco degli utenti.
_DB_VERSION = 2  

# ================================================================
#  FUNZIONI DI CALCOLO (EC2 / NTC2018)
# ================================================================

def _ecm(fck: float) -> int:
    return int(round(22000 * ((fck + 8) / 10) ** 0.3))

def _gcm(ecm: int, nu: float = 0.2) -> int:
    return int(round(ecm / (2 * (1 + nu))))

def _fcd(fck: float, alpha: float = 0.85, gamma_c: float = 1.5) -> float:
    return round(alpha * fck / gamma_c, 4)

def _eps_c2(fck: float) -> float:
    if fck <= 50:
        return 0.002
    # Formula EC2 in per mille. Dividiamo per 1000 per la deformazione reale
    return round((2.0 + 0.085 * ((fck - 50) ** 0.53)) / 1000, 6)

def _eps_cu2(fck: float) -> float:
    if fck <= 50:
        return 0.0035
    # Formula EC2 in per mille. Dividiamo per 1000 per la deformazione reale
    return round((2.6 + 35.0 * ((90 - fck) / 100) ** 4) / 1000, 6)

def _n_cls(fck: float) -> float:
    if fck <= 50:
        return 2.0
    return round(1.4 + 23.4 * ((90 - fck) / 100) ** 4, 3)

# ================================================================
#  COSTRUTTORI MATERIALI
# ================================================================

def _crea_calcestruzzo(fck: float, alpha: float = 0.85, gamma_c: float = 1.5) -> dict:
    fcd  = _fcd(fck, alpha, gamma_c)
    ecm  = _ecm(fck)
    ec2  = _eps_c2(fck)
    ecu2 = _eps_cu2(fck)
    n    = _n_cls(fck)
    return {
        "tipo": "calcestruzzo", "standard": True, "fck": fck,
        "gamma": gamma_c, "alpha": alpha, "densita": 2500,
        "poisson": 0.2, "m_elastico": ecm, "m_taglio": _gcm(ecm),
        "slu": [
            # Ramo parabolico (compressione nel quadrante positivo)
            {"formula": f"{fcd} * (1 - (1 - x / {ec2}) ** {n})",
             "eps_min": 0.0,  "eps_max": ec2},
            # Ramo plastico (compressione nel quadrante positivo)
            {"formula": f"{fcd}",
             "eps_min": ec2,  "eps_max": ecu2},
        ],
        "sle": [
            # Modello elastico lineare centrato sullo zero
            {"formula": f"{ecm} * x", "eps_min": -0.001, "eps_max": 0.001},
        ],
    }


def _crea_barra(fyk: float, ftk: float, eps_uk: float, nome_classe: str,
                gamma_s: float = 1.15, Es: int = 200_000) -> dict:
    fyd    = round(fyk / gamma_s, 4)
    ftd    = round(ftk / gamma_s, 4)
    eps_yd = round(fyd / Es, 8)
    k      = round(ftd / fyd, 6)
    return {
        "tipo": "barre", "standard": True, "classe": nome_classe,
        "fyk": fyk, "ftk": ftk, "gamma": gamma_s, "alpha": 1.0,
        "densita": 7850, "poisson": 0.3, "m_elastico": Es,
        "m_taglio": int(round(Es / (2 * 1.3))),
        "slu": [
            # Lato negativo (ora considerato "trazione" nella nostra convenzione)
            {"formula": f"-{fyd} + ({k} - 1) * (-{fyd}) * ((-x) - {eps_yd}) / ({eps_uk} - {eps_yd})",
             "eps_min": -eps_uk, "eps_max": -eps_yd},
            # Ramo elastico simmetrico
            {"formula": f"{Es} * x",
             "eps_min": -eps_yd, "eps_max": eps_yd},
            # Lato positivo (ora considerato "compressione")
            {"formula": f"{fyd} + ({k} - 1) * {fyd} * (x - {eps_yd}) / ({eps_uk} - {eps_yd})",
             "eps_min": eps_yd,  "eps_max": eps_uk},
        ],
        "sle": [
            {"formula": f"{Es} * x", "eps_min": -0.002, "eps_max": 0.002},
        ],
    }


def _crea_acciaio_profilo(nome: str, fy: float, fu: float,
                          E: int = 210_000, gamma_m0: float = 1.05) -> dict:
    fyd    = round(fy / gamma_m0, 4)
    eps_yd = round(fyd / E, 8)
    nu     = 0.3
    return {
        "tipo": "acciaio", "standard": True, "nome": nome,
        "fy": fy, "fu": fu, "gamma": gamma_m0, "alpha": 1.0,
        "densita": 7850, "poisson": nu, "m_elastico": E,
        "m_taglio": int(round(E / (2 * (1 + nu)))),
        "slu": [
            {"formula": f"-{fyd}",        "eps_min": -0.15,   "eps_max": -eps_yd},
            {"formula": f"{E} * x",       "eps_min": -eps_yd, "eps_max": eps_yd},
            {"formula": f"{fyd}",         "eps_min": eps_yd,  "eps_max": 0.15},
        ],
        "sle": [
            {"formula": f"{E} * x", "eps_min": -0.002, "eps_max": 0.002},
        ],
    }


def _crea_personalizzato() -> dict:
    return {
        "tipo": "personalizzato", "standard": False,
        "gamma": 1.0, "alpha": 1.0, "densita": 0.0,
        "poisson": 0.0, "m_elastico": 0.0, "m_taglio": 0.0,
        "slu": [], "sle": [],
    }

# ================================================================
#  GENERAZIONE COMPLETA
# ================================================================

def _genera_database() -> dict:
    return {
        "_version": _DB_VERSION,
        "calcestruzzo": {
            "C16/20":  _crea_calcestruzzo(16),
            "C20/25":  _crea_calcestruzzo(20),
            "C25/30":  _crea_calcestruzzo(25),
            "C28/35":  _crea_calcestruzzo(28),
            "C30/37":  _crea_calcestruzzo(30),
            "C32/40":  _crea_calcestruzzo(32),
            "C35/45":  _crea_calcestruzzo(35),
            "C40/50":  _crea_calcestruzzo(40),
            "C45/55":  _crea_calcestruzzo(45),
            "C50/60":  _crea_calcestruzzo(50),
            "C55/67":  _crea_calcestruzzo(55),
            "C60/75":  _crea_calcestruzzo(60),
            "C70/85":  _crea_calcestruzzo(70),
            "C80/95":  _crea_calcestruzzo(80),
            "C90/105": _crea_calcestruzzo(90),
        },
        "barre": {
            "B450C": _crea_barra(450, 540, 0.075, "B450C"),
            "B450B": _crea_barra(450, 540, 0.050, "B450B"),
            "B500B": _crea_barra(500, 550, 0.050, "B500B"),
            "B500C": _crea_barra(500, 600, 0.075, "B500C"),
            "B600B": _crea_barra(600, 660, 0.050, "B600B"),
            "B700B": _crea_barra(700, 770, 0.050, "B700B"),
        },
        "acciaio": {
            "S235": _crea_acciaio_profilo("S235", 235, 360),
            "S275": _crea_acciaio_profilo("S275", 275, 430),
            "S355": _crea_acciaio_profilo("S355", 355, 510),
            "S420": _crea_acciaio_profilo("S420", 420, 520),
            "S460": _crea_acciaio_profilo("S460", 460, 550),
            "S690": _crea_acciaio_profilo("S690", 690, 770),
        },
        "personalizzati": {},
    }

# ================================================================
#  API PUBBLICA
# ================================================================

def carica_database() -> dict:
    """
    Carica il database standard dal JSON.
    Se il file non esiste, è corrotto o obsoleto, lo rigenera e lo salva.
    """
    if not os.path.exists(_DB_PATH):
        return _rigenera_e_salva()
    try:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
            
            if db.get("_version") != _DB_VERSION:
                print(f"INFO  Aggiornamento database_materiali.json alla versione {_DB_VERSION}")
                return _rigenera_e_salva()
                
            return db
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARN  database_materiali.json corrotto – rigenerazione: {e}")
        return _rigenera_e_salva()


def _rigenera_e_salva() -> dict:
    db = _genera_database()
    try:
        with open(_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"WARN  Impossibile scrivere database_materiali.json: {e}")
    return db


def nuovo_materiale_personalizzato() -> dict:
    """Restituisce un dict vuoto per un materiale personalizzato."""
    return _crea_personalizzato()