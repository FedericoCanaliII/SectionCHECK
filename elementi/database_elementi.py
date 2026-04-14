"""
database_elementi.py – Database elementi strutturali standard.

Genera il file  elementi/database_elementi.json  al primo avvio.
Il JSON è il riferimento: se manca o è corrotto viene rigenerato automaticamente.

Unità: metri (m).

Sistema di riferimento locale per ogni elemento:
  X → asse longitudinale (lunghezza)
  Y → larghezza della sezione trasversale
  Z → altezza della sezione (o spessore per elementi piani)

Convenzioni armatura (valori in m):
  copriferro netto = 0.03 m (travi/pilastri), 0.04 m (fondazioni), 0.025 m (solai)
  centro barra Φ16: cop + Φstaffa + r_barra = 0.030 + 0.008 + 0.008 = 0.046 m
  centro barra Φ14: 0.030 + 0.008 + 0.007 = 0.045 m
  centro barra Φ12: 0.025 + 0.000 + 0.006 = 0.031 m  (solaio, strato inf.)
  centro barra Φ10: 0.031 + 0.012 = 0.043 m           (solaio, strato sup.)
"""

import json
import os

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "database_elementi.json")


# ================================================================
#  HELPER COSTRUTTORI OGGETTI
# ================================================================

def _base(obj_id, nome, tipo, geometria, mat=""):
    return {
        "id":              obj_id,
        "nome":            nome,
        "tipo":            tipo,
        "geometria":       geometria,
        "materiale":       mat,
        "posizione":       [0.0, 0.0, 0.0],
        "rotazione":       [0.0, 0.0, 0.0],
        "custom_geometry": False,
        "visibile":        True,
        "selezionabile":   True,
        "vertice_ref":     0,
    }


def _parall(oid, nome, L, B, H, mat=""):
    return _base(oid, nome, "parallelepipedo",
                 {"lunghezza": float(L), "base": float(B), "altezza": float(H)}, mat)


def _barra(oid, nome, phi, p1, p2, mat=""):
    return _base(oid, nome, "barra",
                 {"diametro": float(phi),
                  "punti": [[float(c) for c in p1], [float(c) for c in p2]]}, mat)


def _staffa(oid, nome, phi, punti, mat=""):
    return _base(oid, nome, "staffa",
                 {"diametro": float(phi),
                  "punti": [[float(c) for c in p] for p in punti]}, mat)


def _elemento(eid, nome, tipo, oggetti):
    return {
        "id":       eid,
        "nome":     nome,
        "tipo":     tipo,
        "standard": True,
        "oggetti":  oggetti,
    }


# ================================================================
#  GENERAZIONE DATABASE
# ================================================================

def _genera_database() -> dict:
    oid = [0]
    eid = [0]

    def noid():
        oid[0] += 1; return oid[0]

    def neid():
        eid[0] += 1; return eid[0]

    # ================================================================
    #  1 – TRAVE  0.30×0.50 m   L=5.0 m
    # ================================================================
    #  c16 = 0.030 + 0.008 + 0.008 = 0.046 m
    c16 = 0.046
    BY, BZ = 0.30, 0.50

    objs_t1 = [
        _parall(noid(), "Parallelepipedo.001", 5.0, BY, BZ, "C25/30"),
        _barra (noid(), "Barra.001", 0.016, [0, c16,       c16      ], [5.0, c16,       c16      ], "B450C"),
        _barra (noid(), "Barra.002", 0.016, [0, BY - c16,  c16      ], [5.0, BY - c16,  c16      ], "B450C"),
        _barra (noid(), "Barra.003", 0.016, [0, c16,       BZ - c16 ], [5.0, c16,       BZ - c16 ], "B450C"),
        _barra (noid(), "Barra.004", 0.016, [0, BY - c16,  BZ - c16 ], [5.0, BY - c16,  BZ - c16 ], "B450C"),
    ]
    # 25 staffe Φ8, passo 0.20 m, da x=0.10 a x=4.90
    for i, xi in enumerate(
            [round(0.10 + 0.20 * k, 4) for k in range(25)], 1):
        objs_t1.append(_staffa(
            noid(), f"Staffa.{i:03d}", 0.008,
            [[xi, 0.03, 0.03], [xi, BY-0.03, 0.03],
             [xi, BY-0.03, BZ-0.03], [xi, 0.03, BZ-0.03]],
            "B450C"
        ))

    trave_1 = _elemento(neid(), "Trave 0.30×0.50", "trave", objs_t1)

    # ================================================================
    #  2 – TRAVE  0.25×0.40 m   L=3.50 m
    # ================================================================
    #  c14 = 0.030 + 0.008 + 0.007 = 0.045 m
    c14 = 0.045
    BY2, BZ2 = 0.25, 0.40

    objs_t2 = [
        _parall(noid(), "Parallelepipedo.001", 3.5, BY2, BZ2, "C25/30"),
        _barra (noid(), "Barra.001", 0.014, [0, c14,        c14       ], [3.5, c14,        c14       ], "B450C"),
        _barra (noid(), "Barra.002", 0.014, [0, BY2 - c14,  c14       ], [3.5, BY2 - c14,  c14       ], "B450C"),
        _barra (noid(), "Barra.003", 0.014, [0, c14,        BZ2 - c14 ], [3.5, c14,        BZ2 - c14 ], "B450C"),
        _barra (noid(), "Barra.004", 0.014, [0, BY2 - c14,  BZ2 - c14 ], [3.5, BY2 - c14,  BZ2 - c14 ], "B450C"),
    ]
    # 17 staffe Φ8, passo 0.20 m, da x=0.10 a x=3.30
    for i, xi in enumerate(
            [round(0.10 + 0.20 * k, 4) for k in range(17)], 1):
        objs_t2.append(_staffa(
            noid(), f"Staffa.{i:03d}", 0.008,
            [[xi, 0.03, 0.03], [xi, BY2-0.03, 0.03],
             [xi, BY2-0.03, BZ2-0.03], [xi, 0.03, BZ2-0.03]],
            "B450C"
        ))

    trave_2 = _elemento(neid(), "Trave 0.25×0.40", "trave", objs_t2)

    # ================================================================
    #  3 – PILASTRO  0.30×0.30 m   H=3.0 m
    # ================================================================
    LP, BP, HP = 0.30, 0.30, 3.0

    objs_p = [
        _parall(noid(), "Parallelepipedo.001", LP, BP, HP, "C25/30"),
        _barra (noid(), "Barra.001", 0.016, [c16,      c16,      0  ], [c16,      c16,      HP], "B450C"),
        _barra (noid(), "Barra.002", 0.016, [LP - c16, c16,      0  ], [LP - c16, c16,      HP], "B450C"),
        _barra (noid(), "Barra.003", 0.016, [c16,      BP - c16, 0  ], [c16,      BP - c16, HP], "B450C"),
        _barra (noid(), "Barra.004", 0.016, [LP - c16, BP - c16, 0  ], [LP - c16, BP - c16, HP], "B450C"),
    ]
    # 15 staffe Φ8, passo 0.20 m, da z=0.10 a z=2.90
    for i, zi in enumerate(
            [round(0.10 + 0.20 * k, 4) for k in range(15)], 1):
        objs_p.append(_staffa(
            noid(), f"Staffa.{i:03d}", 0.008,
            [[0.03, 0.03, zi], [LP-0.03, 0.03, zi],
             [LP-0.03, BP-0.03, zi], [0.03, BP-0.03, zi]],
            "B450C"
        ))

    pilastro = _elemento(neid(), "Pilastro 0.30×0.30", "pilastro", objs_p)

    # ================================================================
    #  4 – PLINTO ISOLATO  1.20×1.20×0.60 m
    # ================================================================
    LP2, BP2, HP2 = 1.20, 1.20, 0.60
    c_f = 0.04
    r14 = 0.007
    z_x = c_f + r14               # 0.047 m  (strato X, inferiore)
    z_y = z_x + 2 * r14           # 0.061 m  (strato Y, superiore)
    cl  = c_f
    yl_s = [0.10, 0.30, 0.50, 0.70, 0.90, 1.10]
    xl_s = [0.10, 0.30, 0.50, 0.70, 0.90, 1.10]

    objs_f = [_parall(noid(), "Parallelepipedo.001", LP2, BP2, HP2, "C25/30")]
    for i, yi in enumerate(yl_s, 1):
        objs_f.append(_barra(noid(), f"Barra.{i:03d}", 0.014,
                             [c_f, yi, z_x], [LP2 - c_f, yi, z_x], "B450C"))
    for i, xi in enumerate(xl_s, 7):
        objs_f.append(_barra(noid(), f"Barra.{i:03d}", 0.014,
                             [xi, c_f, z_y], [xi, BP2 - c_f, z_y], "B450C"))
    objs_f += [
        _staffa(noid(), "Staffa.001", 0.008,
                [[cl, cl, cl], [LP2-cl, cl, cl],
                 [LP2-cl, cl, HP2-cl], [cl, cl, HP2-cl]], "B450C"),
        _staffa(noid(), "Staffa.002", 0.008,
                [[cl, BP2-cl, cl], [LP2-cl, BP2-cl, cl],
                 [LP2-cl, BP2-cl, HP2-cl], [cl, BP2-cl, HP2-cl]], "B450C"),
        _staffa(noid(), "Staffa.003", 0.008,
                [[cl, cl, cl], [cl, BP2-cl, cl],
                 [cl, BP2-cl, HP2-cl], [cl, cl, HP2-cl]], "B450C"),
        _staffa(noid(), "Staffa.004", 0.008,
                [[LP2-cl, cl, cl], [LP2-cl, BP2-cl, cl],
                 [LP2-cl, BP2-cl, HP2-cl], [LP2-cl, cl, HP2-cl]], "B450C"),
    ]

    plinto = _elemento(neid(), "Plinto 1.20×1.20", "fondazione", objs_f)

    # ================================================================
    #  5 – SOLAIO  4.00×1.20×0.20 m
    # ================================================================
    LS, BS, HS = 4.00, 1.20, 0.20
    c_s  = 0.025
    z_s1 = c_s + 0.006            # 0.031 m  (Φ12, strato inf.)
    z_s2 = z_s1 + 0.012           # 0.043 m  (Φ10, strato sup.)
    yl_s1 = [0.10, 0.30, 0.50, 0.70, 0.90, 1.10]
    xl_s2 = [0.20, 0.60, 1.00, 1.40, 1.80, 2.20, 2.60, 3.00, 3.40, 3.80]

    objs_s = [_parall(noid(), "Parallelepipedo.001", LS, BS, HS, "C25/30")]
    for i, yi in enumerate(yl_s1, 1):
        objs_s.append(_barra(noid(), f"Barra.{i:03d}", 0.012,
                             [0, yi, z_s1], [LS, yi, z_s1], "B450C"))
    for i, xi in enumerate(xl_s2, 7):
        objs_s.append(_barra(noid(), f"Barra.{i:03d}", 0.010,
                             [xi, 0, z_s2], [xi, BS, z_s2], "B450C"))
    for i, xi in enumerate([0.50, 1.50, 2.50, 3.50], 1):
        objs_s.append(_staffa(
            noid(), f"Staffa.{i:03d}", 0.008,
            [[xi, c_s, c_s], [xi, BS-c_s, c_s],
             [xi, BS-c_s, HS-c_s], [xi, c_s, HS-c_s]],
            "B450C"
        ))

    solaio = _elemento(neid(), "Solaio 4.00×1.20", "solaio", objs_s)

    # ================================================================
    #  DIZIONARIO FINALE
    # ================================================================
    return {
        "trave":      [trave_1, trave_2],
        "pilastro":   [pilastro],
        "fondazione": [plinto],
        "solaio":     [solaio],
    }


# ================================================================
#  API PUBBLICA
# ================================================================

def carica_database() -> dict:
    """
    Carica il database standard dal JSON.
    Se il file non esiste o è corrotto lo rigenera e lo salva.
    """
    if not os.path.exists(_DB_PATH):
        return _rigenera_e_salva()
    try:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARN  database_elementi.json corrotto – rigenerazione: {e}")
        return _rigenera_e_salva()


def _rigenera_e_salva() -> dict:
    db = _genera_database()
    try:
        with open(_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"WARN  Impossibile scrivere database_elementi.json: {e}")
    return db
