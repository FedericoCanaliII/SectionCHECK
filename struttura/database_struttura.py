"""
database_struttura.py – Database strutture standard.

Genera il file  struttura/database_struttura.json  al primo avvio.
Contiene template di strutture-tipo (telaio RC, telaio acciaio, …)
con testo pre-compilato compatibile OpenSees.
"""

import json
import os

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "database_struttura.json")

_DB_VERSION = 3


# ================================================================
#  TEMPLATE TESTO STRUTTURE STANDARD
# ================================================================

_TELAIO_RC_2D = """\
# ═══════════════════════════════════════
#  TELAIO IN C.A. – 2 campate, 2 piani
# ═══════════════════════════════════════

# ─── MATERIALI ───
# Riferimento al database del programma (il nome diventa verde se riconosciuto).
material  1  'C25/30'

# ─── SEZIONI ───
# Sezioni inline: Area, Iy, Iz e materiale (id o nome, diventa verde se valido).
section  1  'Pilastro 30x30'  0.09  6.75e-4   6.75e-4   material: 'C25/30'
section  2  'Trave 30x50'     0.15  3.125e-3  1.125e-3  material: 1

# ─── NODI ───
node  1    0.0    0.0    0.0
node  2    5.0    0.0    0.0
node  3   10.0    0.0    0.0
node  4    0.0    0.0    3.5
node  5    5.0    0.0    3.5
node  6   10.0    0.0    3.5
node  7    0.0    0.0    7.0
node  8    5.0    0.0    7.0
node  9   10.0    0.0    7.0

# ─── ASTE ───
# Le aste fanno riferimento alla sezione per id o per nome.
# Pilastri piano terra
beam  1   1  4   section: 1
beam  2   2  5   section: 1
beam  3   3  6   section: 1
# Travi piano 1
beam  4   4  5   section: 'Trave 30x50'
beam  5   5  6   section: 'Trave 30x50'
# Pilastri piano 1
beam  6   4  7   section: 1
beam  7   5  8   section: 1
beam  8   6  9   section: 1
# Travi piano 2
beam  9   7  8   section: 'Trave 30x50'
beam 10   8  9   section: 'Trave 30x50'

# ─── VINCOLI ───
fix  1   1 1 1  1 1 1
fix  2   1 1 1  1 1 1
fix  3   1 1 1  1 1 1

# ─── CARICHI ───
nodeLoad  7    10.0   0.0   0.0
nodeLoad  8    10.0   0.0   0.0
nodeLoad  9    10.0   0.0   0.0
beamLoad  4    0.0    0.0  -12.0
beamLoad  5    0.0    0.0  -12.0
beamLoad  9    0.0    0.0  -10.0
beamLoad 10    0.0    0.0  -10.0
"""

_TELAIO_ACCIAIO_2D = """\
# ═══════════════════════════════════════
#  TELAIO IN ACCIAIO – 1 campata, 1 piano
# ═══════════════════════════════════════

# ─── MATERIALI ───
material  1  'S355'

# ─── SEZIONI ───
# Riferimenti a profilari standard del database (HEA240, IPE300).
section  1  'HEA240'
section  2  'IPE300'

# ─── NODI ───
node  1    0.0    0.0    0.0
node  2    8.0    0.0    0.0
node  3    0.0    0.0    4.0
node  4    8.0    0.0    4.0

# ─── ASTE ───
beam  1   1  3   section: 'HEA240'
beam  2   2  4   section: 'HEA240'
beam  3   3  4   section: 'IPE300'

# ─── VINCOLI ───
fix  1   1 1 1  1 1 1
fix  2   1 1 1  1 1 1

# ─── CARICHI ───
nodeLoad  3    5.0   0.0   0.0
beamLoad  3    0.0   0.0  -15.0
"""

_MENSOLA_RC = """\
# ═══════════════════════════════════════
#  MENSOLA IN C.A.
# ═══════════════════════════════════════

# ─── MATERIALI ───
material  1  'C25/30'

# ─── SEZIONI ───
section  1  'Trave 30x50'  0.15  3.125e-3  1.125e-3  material: 'C25/30'

# ─── NODI ───
node  1    0.0    0.0    0.0
node  2    3.0    0.0    0.0

# ─── ASTE ───
beam  1   1  2   section: 'Trave 30x50'

# ─── VINCOLI ───
fix  1   1 1 1  1 1 1

# ─── CARICHI ───
nodeLoad  2    0.0   0.0  -20.0
"""

_TRAVE_CONTINUA = """\
# ═══════════════════════════════════════
#  TRAVE CONTINUA SU 3 APPOGGI
# ═══════════════════════════════════════

# ─── MATERIALI ───
material  1  'C25/30'

# ─── SEZIONI ───
section  1  'Trave 30x50'  0.15  3.125e-3  1.125e-3  material: 'C25/30'

# ─── NODI ───
node  1    0.0    0.0    0.0
node  2    5.0    0.0    0.0
node  3   10.0    0.0    0.0

# ─── ASTE ───
beam  1   1  2   section: 1
beam  2   2  3   section: 'Trave 30x50'

# ─── VINCOLI ───
fix  1   1 1 1  0 0 0
fix  2   0 1 1  0 0 0
fix  3   0 1 1  0 0 0

# ─── CARICHI ───
beamLoad  1    0.0   0.0  -15.0
beamLoad  2    0.0   0.0  -15.0
"""

_TELAIO_3D = """\
# ═══════════════════════════════════════
#  TELAIO 3D – 1 campata per direzione
# ═══════════════════════════════════════

# ─── MATERIALI ───
material  1  'C25/30'
# Esempio di materiale inline (densità, E, G, ν):
material  2  'ClsCustom'  2500  31476  13115  0.2

# ─── SEZIONI ───
section  1  'Pilastro 30x30'  0.09  6.75e-4   6.75e-4   material: 'C25/30'
section  2  'Trave 30x50'     0.15  3.125e-3  1.125e-3  material: 1
section  3  'Trave 25x40'     0.10  1.333e-3  5.208e-4  material: 'C25/30'

# ─── NODI ───
node  1    0.0    0.0    0.0
node  2    6.0    0.0    0.0
node  3    0.0    5.0    0.0
node  4    6.0    5.0    0.0
node  5    0.0    0.0    3.5
node  6    6.0    0.0    3.5
node  7    0.0    5.0    3.5
node  8    6.0    5.0    3.5

# ─── ASTE ───
# Pilastri
beam  1   1  5   section: 1
beam  2   2  6   section: 1
beam  3   3  7   section: 'Pilastro 30x30'
beam  4   4  8   section: 'Pilastro 30x30'
# Travi X
beam  5   5  6   section: 'Trave 30x50'
beam  6   7  8   section: 2
# Travi Y
beam  7   5  7   section: 'Trave 25x40'
beam  8   6  8   section: 3

# ─── SHELL ───
# Shell quadrilatera (4 nodi). Il materiale può essere id o nome.
shell  1   5 6 8 7   thickness: 0.20   material: 'C25/30'

# ─── VINCOLI ───
fix  1   1 1 1  1 1 1
fix  2   1 1 1  1 1 1
fix  3   1 1 1  1 1 1
fix  4   1 1 1  1 1 1

# ─── CARICHI ───
beamLoad  5    0.0   0.0  -12.0
beamLoad  6    0.0   0.0  -12.0
beamLoad  7    0.0   0.0  -10.0
beamLoad  8    0.0   0.0  -10.0
"""


# ================================================================
#  GENERAZIONE DATABASE
# ================================================================

def _genera_database() -> dict:
    return {
        "_version": _DB_VERSION,
        "calcestruzzo": {
            "Telaio RC 2 campate": {
                "standard": True,
                "testo": _TELAIO_RC_2D,
            },
            "Mensola RC": {
                "standard": True,
                "testo": _MENSOLA_RC,
            },
            "Trave continua 3 appoggi": {
                "standard": True,
                "testo": _TRAVE_CONTINUA,
            },
            "Telaio 3D": {
                "standard": True,
                "testo": _TELAIO_3D,
            },
        },
        "acciaio": {
            "Telaio acciaio": {
                "standard": True,
                "testo": _TELAIO_ACCIAIO_2D,
            },
        },
        "personalizzate": {},
    }


# ================================================================
#  STRUTTURA VUOTA (per nuove strutture personalizzate)
# ================================================================

_TEMPLATE_VUOTO = """\
# ─── MATERIALI ───
# material  <id>  '<nome>'                             (riferimento al database)
# material  <id>  '<nome>'  <densita>  <E>  <G>  <J>   (definizione inline)

# ─── SEZIONI ───
# section  <id>  '<nome>'                                                (riferimento al database)
# section  <id>  '<nome>'  <Area>  <Iy>  <Iz>  material: <id_o_nome>     (definizione inline)

# ─── NODI ───
# node  <id>  <x>  <y>  <z>

# ─── ASTE ───
# beam  <id>  <nodo_i>  <nodo_j>   section: <id_o_nome>

# ─── SHELL ───
# shell  <id>  <n1> <n2> <n3> [<n4>]   thickness: <t>   material: <id_o_nome>

# ─── VINCOLI ───
# fix  <nodo_id>   <dx> <dy> <dz>  [<rx> <ry> <rz>]

# ─── CARICHI ───
# nodeLoad  <nodo_id>   <Fx> <Fy> <Fz>
# beamLoad  <asta_id>   <wx> <wy> <wz>
"""


def nuovo_struttura_vuota(cat: str) -> dict:
    """Crea il dizionario per una struttura personalizzata vuota."""
    return {
        "standard": False,
        "testo": _TEMPLATE_VUOTO,
    }


# ================================================================
#  API PUBBLICA
# ================================================================

def carica_database() -> dict:
    if not os.path.exists(_DB_PATH):
        return _rigenera_e_salva()
    try:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARN  database_struttura.json corrotto – rigenerazione: {e}")
        return _rigenera_e_salva()
    if db.get("_version") != _DB_VERSION:
        print(f"INFO  Aggiornamento database_struttura.json alla versione {_DB_VERSION}")
        return _rigenera_e_salva()
    return db


def _rigenera_e_salva() -> dict:
    db = _genera_database()
    try:
        with open(_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"WARN  Impossibile scrivere database_struttura.json: {e}")
    return db
