"""
modello_carichi_vincoli.py – Data model for loads (carichi) and constraints (vincoli).

A CaricoVincolo is a parallelepipedo with:
  • sottotipo: "vincolo" (green) | "carico" (purple)
  • caratteristiche: physical properties (cedimenti or forze in kN)
  • Same geometry / position / rotation interface as Oggetto3D for tool compatibility
"""

import copy


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def _geometria_default() -> dict:
    return {"lunghezza": 0.3, "base": 0.3, "altezza": 0.3}


def _caratteristiche_default(sottotipo: str) -> dict:
    if sottotipo == "vincolo":
        return {"sx": 0.0, "sy": 0.0, "sz": 0.0}   # cedimenti [m]
    return {"fx": 0.0, "fy": 0.0, "fz": 0.0}        # forze [kN]


# ---------------------------------------------------------------------------
# CaricoVincolo
# ---------------------------------------------------------------------------

class CaricoVincolo:
    """
    A load or constraint object, always represented as a parallelepipedo.

    Attributes shared with Oggetto3D (duck-typing for tool/renderer compatibility):
      tipo, geometria, posizione, rotazione, custom_geometry,
      visibile, selezionabile, vertice_ref, materiale, nome, id
    """

    _id_counter: int  = 0
    _nome_count: dict = {}

    def __init__(self, sottotipo: str):
        CaricoVincolo._id_counter += 1
        self.id = CaricoVincolo._id_counter

        self.sottotipo = sottotipo          # "vincolo" | "carico"
        self.tipo      = "parallelepipedo"  # always – renderer compatibility
        self.materiale = ""                 # unused but Outliner expects it

        nome_base = "Vincolo" if sottotipo == "vincolo" else "Carico"
        CaricoVincolo._nome_count[nome_base] = (
            CaricoVincolo._nome_count.get(nome_base, 0) + 1
        )
        self.nome = f"{nome_base}.{CaricoVincolo._nome_count[nome_base]:03d}"

        self.geometria       = _geometria_default()
        self.posizione       = [0.0, 0.0, 0.0]
        self.rotazione       = [0.0, 0.0, 0.0]   # rx, ry, rz [deg]
        self.custom_geometry = False
        self.visibile        = True
        self.selezionabile   = True
        self.vertice_ref     = 0

        self.caratteristiche = _caratteristiche_default(sottotipo)

    # ---------------------------------------------------------------- vertices

    def get_vertices_local(self) -> list:
        if self.custom_geometry:
            return [list(v) for v in self.geometria.get("vertici_custom", [[0, 0, 0]])]
        L = float(self.geometria.get("lunghezza", 0.3))
        B = float(self.geometria.get("base",      0.3))
        A = float(self.geometria.get("altezza",   0.3))
        return [
            [0.0, 0.0, 0.0], [L,   0.0, 0.0], [L,   B, 0.0], [0.0, B, 0.0],
            [0.0, 0.0, A],   [L,   0.0, A],   [L,   B, A],   [0.0, B, A],
        ]

    def get_vertices_world(self) -> list:
        from .modello_3d import trasforma_punto
        return [trasforma_punto(v, self.posizione, self.rotazione)
                for v in self.get_vertices_local()]

    def get_vertex_ref_world(self) -> list:
        verts = self.get_vertices_local()
        if not verts:
            return list(self.posizione)
        from .modello_3d import trasforma_punto
        idx = self.vertice_ref if self.vertice_ref < len(verts) else 0
        return trasforma_punto(verts[idx], self.posizione, self.rotazione)

    def set_vertices_custom(self, vertici: list):
        self.custom_geometry = True
        self.geometria["vertici_custom"] = [list(v) for v in vertici]

    # ---------------------------------------------------------------- serialise

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "nome":            self.nome,
            "sottotipo":       self.sottotipo,
            "geometria":       copy.deepcopy(self.geometria),
            "posizione":       list(self.posizione),
            "rotazione":       list(self.rotazione),
            "custom_geometry": self.custom_geometry,
            "visibile":        self.visibile,
            "selezionabile":   self.selezionabile,
            "vertice_ref":     self.vertice_ref,
            "caratteristiche": copy.deepcopy(self.caratteristiche),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CaricoVincolo":
        cv               = cls.__new__(cls)
        cv.id            = d["id"]
        cv.nome          = d["nome"]
        cv.sottotipo     = d["sottotipo"]
        cv.tipo          = "parallelepipedo"
        cv.materiale     = ""
        cv.geometria     = copy.deepcopy(d["geometria"])
        cv.posizione     = list(d.get("posizione",  [0, 0, 0]))
        cv.rotazione     = list(d.get("rotazione",  [0, 0, 0]))
        cv.custom_geometry = d.get("custom_geometry", False)
        cv.visibile      = d.get("visibile",        True)
        cv.selezionabile = d.get("selezionabile",   True)
        cv.vertice_ref   = d.get("vertice_ref",     0)
        cv.caratteristiche = copy.deepcopy(
            d.get("caratteristiche", _caratteristiche_default(d["sottotipo"]))
        )
        return cv

    def duplica(self) -> "CaricoVincolo":
        nuovo = copy.deepcopy(self)
        CaricoVincolo._id_counter += 1
        nuovo.id = CaricoVincolo._id_counter
        nome_base = "Vincolo" if self.sottotipo == "vincolo" else "Carico"
        CaricoVincolo._nome_count[nome_base] = (
            CaricoVincolo._nome_count.get(nome_base, 0) + 1
        )
        nuovo.nome = f"{nome_base}.{CaricoVincolo._nome_count[nome_base]:03d}"
        return nuovo
