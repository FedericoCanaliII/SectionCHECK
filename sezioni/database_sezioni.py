"""
database_sezioni.py
-------------------
Database standard delle sezioni trasversali.
Genera il file  sezioni/database_sezioni.json  al primo avvio.

Sezioni C.A. incluse:
  Rettangolari  R 200×400 (Ø14/Ø8)  –  R 400×600 (Ø18/Ø10)
  A T           T Standard Nuova
  Circolari     CIR Ø400 (8Ø16/Ø8)

Profili in acciaio:
  IPE 80÷600 · HEA 100÷600 · HEB 100÷500

Unità: mm.
"""

import json
import math as _math
import os

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "database_sezioni.json")

_DB_VERSION = 7   # incrementare ad ogni modifica strutturale del DB

_C = 30  # copriferro nominale [mm]


# ================================================================
#  GEOMETRIA PROFILI IN ACCIAIO  (a I, 12 vertici)
# ================================================================

def _ipe_punti(h, b, tw, tf):
    hh = h / 2.0
    return [
        [-b/2, -hh],      [ b/2, -hh],
        [ b/2, -hh + tf], [ tw/2, -hh + tf],
        [ tw/2,  hh - tf],[ b/2,  hh - tf],
        [ b/2,  hh],      [-b/2,  hh],
        [-b/2,  hh - tf], [-tw/2,  hh - tf],
        [-tw/2, -hh + tf],[-b/2, -hh + tf],
    ]


# ================================================================
#  DATI PROFILI IN ACCIAIO  (h, b, tw, tf)  [mm]
# ================================================================

_IPE = {
    "IPE80":  (80,  46,  3.8, 5.2), "IPE100": (100, 55,  4.1, 5.7),
    "IPE120": (120, 64,  4.4, 6.3), "IPE140": (140, 73,  4.7, 6.9),
    "IPE160": (160, 82,  5.0, 7.4), "IPE180": (180, 91,  5.3, 8.0),
    "IPE200": (200, 100, 5.6, 8.5), "IPE220": (220, 110, 5.9, 9.2),
    "IPE240": (240, 120, 6.2, 9.8), "IPE270": (270, 135, 6.6, 10.2),
    "IPE300": (300, 150, 7.1, 10.7),"IPE330": (330, 160, 7.5, 11.5),
    "IPE360": (360, 170, 8.0, 12.7),"IPE400": (400, 180, 8.6, 13.5),
    "IPE450": (450, 190, 9.4, 14.6),"IPE500": (500, 200, 10.2,16.0),
    "IPE550": (550, 210, 11.1,17.2),"IPE600": (600, 220, 12.0,19.0),
}
_HEA = {
    "HEA100": (96,  100, 5.0,  8.0), "HEA120": (114, 120, 5.0,  8.0),
    "HEA140": (133, 140, 5.5,  8.5), "HEA160": (152, 160, 6.0,  9.0),
    "HEA180": (171, 180, 6.0,  9.5), "HEA200": (190, 200, 6.5, 10.0),
    "HEA220": (210, 220, 7.0, 11.0), "HEA240": (230, 240, 7.5, 12.0),
    "HEA260": (250, 260, 7.5, 12.5), "HEA280": (270, 280, 8.0, 13.0),
    "HEA300": (290, 300, 8.5, 14.0), "HEA320": (310, 300, 9.0, 15.5),
    "HEA340": (330, 300, 9.5, 16.5), "HEA360": (350, 300,10.0, 17.5),
    "HEA400": (390, 300,11.0, 19.0), "HEA450": (440, 300,11.5, 21.0),
    "HEA500": (490, 300,12.0, 23.0), "HEA550": (540, 300,12.5, 24.0),
    "HEA600": (590, 300,13.0, 25.0),
}
_HEB = {
    "HEB100": (100, 100, 6.0, 10.0), "HEB120": (120, 120, 6.5, 11.0),
    "HEB140": (140, 140, 7.0, 12.0), "HEB160": (160, 160, 8.0, 13.0),
    "HEB180": (180, 180, 8.5, 14.0), "HEB200": (200, 200, 9.0, 15.0),
    "HEB220": (220, 220, 9.5, 16.0), "HEB240": (240, 240,10.0, 17.0),
    "HEB260": (260, 260,10.0, 17.5), "HEB280": (280, 280,10.5, 18.0),
    "HEB300": (300, 300,11.0, 19.0), "HEB320": (320, 300,11.5, 20.5),
    "HEB340": (340, 300,12.0, 21.5), "HEB360": (360, 300,12.5, 22.5),
    "HEB400": (400, 300,13.5, 24.0), "HEB450": (450, 300,14.0, 26.0),
    "HEB500": (500, 300,14.5, 28.0),
}


# ================================================================
#  COSTRUTTORE SEZIONE PROFILO IN ACCIAIO
# ================================================================

def _crea_profilo(nome, h, b, tw, tf, materiale="S355"):
    return {
        "tipo_categoria": "profili", "standard": True, "nome": nome,
        "dimensioni": {"h": h, "b": b, "tw": tw, "tf": tf},
        "materiale_default": materiale,
        "elementi": {
            "carpenteria": [{"id": "carp_1", "tipo": "poligono",
                             "geometria": {"punti": _ipe_punti(h, b, tw, tf)},
                             "materiale": materiale}],
            "barre": [], "staffe": [],
        }
    }


# ================================================================
#  PRIMITIVE ELEMENTI C.A.
# ================================================================

def _off_b(r_b, r_s):
    """Offset centro barra da faccia: copriferro + Østaffa + r_barra."""
    return _C + 2.0 * r_s + r_b

def _off_s(r_s):
    """Offset centro staffa da faccia: copriferro + r_staffa."""
    return _C + r_s

def _barra(bid, cx, cy, r_b, mat="B450C"):
    return {"id": f"barra_{bid}", "tipo": "barra",
            "geometria": {"cx": round(float(cx), 1),
                          "cy": round(float(cy), 1),
                          "r":  float(r_b)},
            "materiale": mat}

def _staffa(sid, punti, r_s, mat="B450C"):
    pts = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in punti]
    if pts[0] != pts[-1]:
        pts.append(list(pts[0]))
    return {"id": f"staffa_{sid}", "tipo": "staffa",
            "geometria": {"punti": pts, "r": float(r_s)},
            "materiale": mat}

def _carp_rett(cid, x0, y0, x1, y1, mat="C30/37"):
    return {"id": f"carp_{cid}", "tipo": "rettangolo",
            "geometria": {"x0": float(x0), "y0": float(y0),
                          "x1": float(x1), "y1": float(y1)},
            "materiale": mat}

def _carp_poly(cid, punti, mat="C30/37"):
    return {"id": f"carp_{cid}", "tipo": "poligono",
            "geometria": {"punti": [[round(float(p[0]), 1),
                                      round(float(p[1]), 1)] for p in punti]},
            "materiale": mat}

def _carp_circ(cid, r, mat="C30/37"):
    return {"id": f"carp_{cid}", "tipo": "cerchio",
            "geometria": {"cx": 0.0, "cy": 0.0, "rx": float(r), "ry": float(r)},
            "materiale": mat}

def _linspace(x0, x1, n):
    if n == 1:
        return [(x0 + x1) / 2.0]
    return [x0 + i * (x1 - x0) / (n - 1) for i in range(n)]


# ================================================================
#  SEZIONI RETTANGOLARI
# ================================================================

def _crea_sez_rettangolare(nome, b, h, n_bot, n_top, n_side, r_b, r_s):
    ob  = _off_b(r_b, r_s)
    os_ = _off_s(r_s)

    y_b = -h/2 + ob;  y_t = h/2 - ob
    x_l = -b/2 + ob;  x_r = b/2 - ob

    barre = []; bid = 1
    for x in _linspace(x_l, x_r, n_bot):
        barre.append(_barra(bid, x, y_b, r_b)); bid += 1
    for x in _linspace(x_l, x_r, n_top):
        barre.append(_barra(bid, x, y_t, r_b)); bid += 1
    if n_side > 0:
        dy = (y_t - y_b) / (n_side + 1)
        for i in range(1, n_side + 1):
            y = y_b + i * dy
            barre.append(_barra(bid, x_l, y, r_b)); bid += 1
            barre.append(_barra(bid, x_r, y, r_b)); bid += 1

    xs = b/2 - os_;  ys = h/2 - os_
    staffe = [_staffa(1, [(-xs, -ys), (xs, -ys), (xs, ys), (-xs, ys)], r_s)]

    return {
        "tipo_categoria": "calcestruzzo_armato", "standard": True, "nome": nome,
        "dimensioni": {"b": b, "h": h}, "materiale_default": "C30/37",
        "elementi": {"carpenteria": [_carp_rett(1, -b/2, -h/2, b/2, h/2)],
                     "barre": barre, "staffe": staffe},
    }


# ================================================================
#  SEZIONI A T
# ================================================================

def _crea_sez_T(nome, bf, bw, H, hf,
                n_bot, n_top, n_side,
                r_b_bot, r_b_top, r_s):
    punti = [
        (-bw/2, -H/2), ( bw/2, -H/2),
        ( bw/2,  H/2-hf), ( bf/2,  H/2-hf),
        ( bf/2,  H/2),    (-bf/2,  H/2),
        (-bf/2,  H/2-hf), (-bw/2,  H/2-hf),
    ]

    ob_bot = _off_b(r_b_bot, r_s)
    ob_top = _off_b(r_b_top, r_s)
    os_    = _off_s(r_s)

    xs     = bw/2 - os_
    ys_bot = -H/2      + os_
    ys_top =  H/2 - hf - os_
    staffe = [_staffa(1, [(-xs, ys_bot), (xs, ys_bot),
                           (xs, ys_top), (-xs, ys_top)], r_s)]

    barre = []; bid = 1

    y_b  = -H/2  + ob_bot
    x_lw = -bw/2 + ob_bot;  x_rw = bw/2 - ob_bot
    for x in _linspace(x_lw, x_rw, n_bot):
        barre.append(_barra(bid, x, y_b, r_b_bot)); bid += 1

    r_b_mont = max(6, min(r_b_bot, r_b_top) - 1)
    ob_mont  = _off_b(r_b_mont, r_s)
    y_mont   = H/2 - hf - ob_mont
    x_lm     = -bw/2 + ob_mont;  x_rm = bw/2 - ob_mont
    barre.append(_barra(bid, x_lm, y_mont, r_b_mont)); bid += 1
    barre.append(_barra(bid, x_rm, y_mont, r_b_mont)); bid += 1

    y_t  =  H/2  - ob_top
    x_lf = -bf/2 + ob_top;  x_rf = bf/2 - ob_top
    for x in _linspace(x_lf, x_rf, n_top):
        barre.append(_barra(bid, x, y_t, r_b_top)); bid += 1

    r_b_side = (r_b_bot + r_b_top) / 2.0
    if n_side > 0:
        ob_s    = _off_b(r_b_side, r_s)
        x_lws   = -bw/2 + ob_s;  x_rws = bw/2 - ob_s
        y_top_w = y_mont - ob_s
        if y_top_w > y_b:
            dy = (y_top_w - y_b) / (n_side + 1)
            for i in range(1, n_side + 1):
                y = y_b + i * dy
                barre.append(_barra(bid, x_lws, y, r_b_side)); bid += 1
                barre.append(_barra(bid, x_rws, y, r_b_side)); bid += 1

    return {
        "tipo_categoria": "calcestruzzo_armato", "standard": True, "nome": nome,
        "dimensioni": {"bf": bf, "bw": bw, "H": H, "hf": hf},
        "materiale_default": "C30/37",
        "elementi": {"carpenteria": [_carp_poly(1, punti)],
                     "barre": barre, "staffe": staffe},
    }


# ================================================================
#  SEZIONI A DOPPIA T  (profilo I in c.a.)
# ================================================================

def _crea_sez_2T(nome, bf, bw, H, hf_top, hf_bot,
                 n_bot, n_top, n_side,
                 r_b_bot, r_b_top, r_s):
    y0 = -H / 2
    punti = [
        (-bf/2, y0),              ( bf/2, y0),
        ( bf/2, y0 + hf_bot),     ( bw/2, y0 + hf_bot),
        ( bw/2, H/2 - hf_top),   ( bf/2, H/2 - hf_top),
        ( bf/2, H/2),             (-bf/2, H/2),
        (-bf/2, H/2 - hf_top),   (-bw/2, H/2 - hf_top),
        (-bw/2, y0 + hf_bot),    (-bf/2, y0 + hf_bot),
    ]

    ob_bot = _off_b(r_b_bot, r_s)
    ob_top = _off_b(r_b_top, r_s)
    os_    = _off_s(r_s)

    xs     = bw/2 - os_
    ys_bot = y0  + hf_bot + os_
    ys_top = H/2 - hf_top - os_
    staffe = [_staffa(1, [(-xs, ys_bot), (xs, ys_bot),
                           (xs, ys_top), (-xs, ys_top)], r_s)]

    barre = []; bid = 1

    y_b  = y0 + hf_bot + ob_bot
    x_lw = -bw/2 + ob_bot;  x_rw = bw/2 - ob_bot
    for x in _linspace(x_lw, x_rw, n_bot):
        barre.append(_barra(bid, x, y_b, r_b_bot)); bid += 1

    y_b_fl = y0 + ob_bot
    for x in [-bf/2 + ob_bot, bf/2 - ob_bot]:
        barre.append(_barra(bid, x, y_b_fl, r_b_bot)); bid += 1

    y_t  = H/2 - hf_top - ob_top
    x_lt = -bw/2 + ob_top;  x_rt = bw/2 - ob_top
    for x in _linspace(x_lt, x_rt, n_top):
        barre.append(_barra(bid, x, y_t, r_b_top)); bid += 1

    y_t_fl = H/2 - ob_top
    for x in [-bf/2 + ob_top, bf/2 - ob_top]:
        barre.append(_barra(bid, x, y_t_fl, r_b_top)); bid += 1

    r_b_side = (r_b_bot + r_b_top) / 2.0
    if n_side > 0:
        ob_s = _off_b(r_b_side, r_s)
        x_ls = -bw/2 + ob_s;  x_rs = bw/2 - ob_s
        y_bw = y_b  + ob_s
        y_tw = y_t  - ob_s
        if y_tw > y_bw:
            dy = (y_tw - y_bw) / (n_side + 1)
            for i in range(1, n_side + 1):
                y = y_bw + i * dy
                barre.append(_barra(bid, x_ls, y, r_b_side)); bid += 1
                barre.append(_barra(bid, x_rs, y, r_b_side)); bid += 1

    return {
        "tipo_categoria": "calcestruzzo_armato", "standard": True, "nome": nome,
        "dimensioni": {"bf": bf, "bw": bw, "H": H,
                       "hf_top": hf_top, "hf_bot": hf_bot},
        "materiale_default": "C30/37",
        "elementi": {"carpenteria": [_carp_poly(1, punti)],
                     "barre": barre, "staffe": staffe},
    }


# ================================================================
#  SEZIONI CIRCOLARI
# ================================================================

def _crea_sez_circolare(nome, D, n_barre, r_b, r_s):
    R      = D / 2.0
    r_bars = R - _off_b(r_b, r_s)
    r_stir = R - _off_s(r_s)

    barre = []
    for i in range(n_barre):
        ang = _math.pi / 2 - 2 * _math.pi * i / n_barre
        barre.append(_barra(i + 1,
                            r_bars * _math.cos(ang),
                            r_bars * _math.sin(ang), r_b))

    N   = 36
    pts = [(r_stir * _math.cos(2 * _math.pi * k / N),
            r_stir * _math.sin(2 * _math.pi * k / N)) for k in range(N)]
    staffe = [_staffa(1, pts, r_s)]

    return {
        "tipo_categoria": "calcestruzzo_armato", "standard": True, "nome": nome,
        "dimensioni": {"D": D, "n_barre": n_barre},
        "materiale_default": "C30/37",
        "elementi": {"carpenteria": [_carp_circ(1, R)],
                     "barre": barre, "staffe": staffe},
    }


# ================================================================
#  SEZIONE COMPOSITA  (cls rettangolare + profilo IPE annegato)
# ================================================================

def _crea_sez_composita(nome, b_c, h_c,
                         h_ipe, b_ipe, tw_ipe, tf_ipe,
                         n_bot, n_top, n_side,
                         r_b, r_s):
    punti_ipe = _ipe_punti(h_ipe, b_ipe, tw_ipe, tf_ipe)
    carp = [
        _carp_rett(1, -b_c/2, -h_c/2, b_c/2, h_c/2, mat="C30/37"),
        _carp_poly(2, punti_ipe, mat="S355"),
    ]

    ob  = _off_b(r_b, r_s)
    os_ = _off_s(r_s)

    y_b = -h_c/2 + ob;  y_t = h_c/2 - ob
    x_l = -b_c/2 + ob;  x_r = b_c/2 - ob

    barre = []; bid = 1
    for x in _linspace(x_l, x_r, n_bot):
        barre.append(_barra(bid, x, y_b, r_b)); bid += 1
    for x in _linspace(x_l, x_r, n_top):
        barre.append(_barra(bid, x, y_t, r_b)); bid += 1
    if n_side > 0:
        dy = (y_t - y_b) / (n_side + 1)
        for i in range(1, n_side + 1):
            y = y_b + i * dy
            barre.append(_barra(bid, x_l, y, r_b)); bid += 1
            barre.append(_barra(bid, x_r, y, r_b)); bid += 1

    xs = b_c/2 - os_;  ys = h_c/2 - os_
    staffe = [_staffa(1, [(-xs, -ys), (xs, -ys), (xs, ys), (-xs, ys)], r_s)]

    return {
        "tipo_categoria": "calcestruzzo_armato", "standard": True, "nome": nome,
        "dimensioni": {"b_c": b_c, "h_c": h_c,
                       "h_ipe": h_ipe, "b_ipe": b_ipe},
        "materiale_default": "C30/37",
        "elementi": {"carpenteria": carp, "barre": barre, "staffe": staffe},
    }


# ================================================================
#  GENERAZIONE DATABASE
# ================================================================

def _genera_database():
    db = {
        "__version__":         _DB_VERSION,
        "calcestruzzo_armato": {},
        "profili":             {},
        "precompresso":        {},
        "personalizzate":      {},
    }

    # --------------------------------------------------------
    #  SEZIONI RETTANGOLARI MANTENUTE
    # --------------------------------------------------------
    _RETT = {
        "R 200×400": (200, 400, 2, 2, 0,  7, 4),   # Ø14 / Ø8
        "R 400×600": (400, 600, 3, 3, 1,  9, 5),   # Ø18 / Ø10
    }
    for nome, (b, h, nb, nt, ns, rb, rs) in _RETT.items():
        db["calcestruzzo_armato"][nome] = _crea_sez_rettangolare(
            nome, b, h, nb, nt, ns, rb, rs)

    # --------------------------------------------------------
    #  SEZIONE CIRCOLARE MANTENUTA
    # --------------------------------------------------------
    db["calcestruzzo_armato"]["CIR Ø400"] = _crea_sez_circolare(
        "CIR Ø400", D=400, n_barre=8,  r_b=8,  r_s=4)   # 8Ø16  / Ø8

    # --------------------------------------------------------
    #  NUOVA SEZIONE A T STANDARD (Poligono + Ferri/Staffe custom)
    # --------------------------------------------------------
    punti_t = [
        (-250.0, 0.0), (-250.0, 200.0), (250.0, 200.0), (250.0, 0.0),
        (100.0, 0.0), (100.0, -250.0), (-100.0, -250.0), (-100.0, 0.0)
    ]
    
    # Raggi barre: diametro / 2
    barre_t = [
        _barra(1, -210.0, 160.0, 7.0),
        _barra(2, 210.0, 160.0, 7.0),
        _barra(3, 210.0, 40.0, 7.0),
        _barra(4, -210.0, 40.0, 7.0),
        _barra(5, -58.0, 42.0, 7.0),
        _barra(6, 58.0, 42.0, 7.0),
        _barra(7, -58.0, 158.0, 7.0),
        _barra(8, 58.0, 158.0, 7.0),
        _barra(9, 55.0, -210.0, 10.0),
        _barra(10, 0.0, -210.0, 10.0),
        _barra(11, -55.0, -210.0, 10.0),
    ]

    # Raggi staffe: diametro / 2
    staffe_t = [
        _staffa(1, [(-220.0, 170.0), (220.0, 170.0), (220.0, 30.0), (-220.0, 30.0)], 3.0),
        _staffa(2, [(-70.0, 170.0), (-70.0, -225.0), (70.0, -225.0), (70.0, 170.0)], 4.0),
    ]

    db["calcestruzzo_armato"]["T 400x550"] = {
        "tipo_categoria": "calcestruzzo_armato",
        "standard": True,
        "nome": "T 400x550",
        "dimensioni": {},
        "materiale_default": "C30/37",
        "elementi": {
            "carpenteria": [_carp_poly(1, punti_t)],
            "barre": barre_t,
            "staffe": staffe_t
        }
    }

    # --------------------------------------------------------
    #  Profili in acciaio
    # --------------------------------------------------------
    for nome, (h, b, tw, tf) in _IPE.items():
        db["profili"][nome] = _crea_profilo(nome, h, b, tw, tf)
    for nome, (h, b, tw, tf) in _HEA.items():
        db["profili"][nome] = _crea_profilo(nome, h, b, tw, tf)
    for nome, (h, b, tw, tf) in _HEB.items():
        db["profili"][nome] = _crea_profilo(nome, h, b, tw, tf)

    return db


# ================================================================
#  API PUBBLICA
# ================================================================

def _db_obsoleto(db):
    if db.get("__version__", 0) < _DB_VERSION:
        return True
    for key, cat_data in db.items():
        if not isinstance(cat_data, dict) or key.startswith("__"):
            continue
        for sez in cat_data.values():
            if not isinstance(sez, dict):
                continue
            for el in sez.get("elementi", {}).get("carpenteria", []):
                if el.get("id") == "poly_001":
                    return True
    return False


def carica_database():
    if not os.path.exists(_DB_PATH):
        return _rigenera_e_salva()
    try:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
        if _db_obsoleto(db):
            print(f"INFO  database_sezioni.json v{db.get('__version__',0)} "
                  f"→ rigenero v{_DB_VERSION}")
            return _rigenera_e_salva()
        return db
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARN  database_sezioni.json corrotto – rigenerazione: {e}")
        return _rigenera_e_salva()


def _rigenera_e_salva():
    db = _genera_database()
    try:
        with open(_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"WARN  Impossibile scrivere database_sezioni.json: {e}")
    return db


def nuova_sezione_vuota(tipo_categoria="personalizzate"):
    return {
        "tipo_categoria":    tipo_categoria,
        "standard":          False,
        "materiale_default": "",
        "elementi": {"carpenteria": [], "barre": [], "staffe": []},
    }