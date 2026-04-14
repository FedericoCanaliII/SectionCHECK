"""
modello_3d.py – Data model for 3D structural elements.

Defines Oggetto3D and all vertex-computation / transformation helpers.
"""

import math
import copy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIPO_STRUTTURALE = {"parallelepipedo", "cilindro", "sfera"}
TIPO_ARMATURA    = {"barra", "staffa"}

_TIPO_A_NOME = {
    "parallelepipedo": "Parallelepipedo",
    "cilindro":        "Cilindro",
    "sfera":           "Sfera",
    "barra":           "Barra",
    "staffa":          "Staffa",
}

# Number of segments used when discretising circles
_CYL_N = 24


# ---------------------------------------------------------------------------
# Geometry defaults
# ---------------------------------------------------------------------------

def _geometria_default(tipo: str) -> dict:
    # Unità: rigorosamente metri (m) dappertutto
    if tipo == "parallelepipedo":
        # Es: trave 30x50 cm, lunga 5 metri
        return {"lunghezza": 5.0, "base": 0.3, "altezza": 0.5}
    if tipo == "cilindro":
        # Es: pilastro circolare diametro 40 cm, alto 3 metri
        return {"altezza": 3.0, "raggio": 0.2}
    if tipo == "sfera":
        # Es: nodo/sfera di diametro 40 cm
        return {"raggio": 0.2}
    if tipo == "barra":
        # Es: barra dritto fi 16 mm (0.016 m), lunga 5 m
        return {"diametro": 0.016,
                "punti": [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]}
    if tipo == "staffa":
        # Es: staffa fi 8 mm (0.008 m) per trave 30x50 con copriferro 2.5 cm -> (25x45 cm)
        return {"diametro": 0.008,
                "punti": [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0],
                          [0.25, 0.45, 0.0], [0.0, 0.45, 0.0]]}
    return {}


# ---------------------------------------------------------------------------
# Vertex computation
# ---------------------------------------------------------------------------

def calcola_vertici(tipo: str, geometria: dict) -> list:
    """Return list of [x, y, z] vertices in local (object) space."""

    if tipo == "parallelepipedo":
        L = float(geometria.get("lunghezza", 5.0))
        B = float(geometria.get("base",      0.3))
        A = float(geometria.get("altezza",   0.5))
        return [
            [0.0, 0.0, 0.0], [L,   0.0, 0.0], [L,   B, 0.0], [0.0, B, 0.0],
            [0.0, 0.0, A],   [L,   0.0, A],   [L,   B, A],   [0.0, B, A],
        ]

    if tipo == "cilindro":
        A = float(geometria.get("altezza", 3.0))
        R = float(geometria.get("raggio",  0.2))
        N = _CYL_N
        verts = [[0.0, 0.0, 0.0]]          # v0 = bottom centre (reference)
        for i in range(N):
            t = 2 * math.pi * i / N
            verts.append([R * math.cos(t), R * math.sin(t), 0.0])
        verts.append([0.0, 0.0, A])        # top centre
        for i in range(N):
            t = 2 * math.pi * i / N
            verts.append([R * math.cos(t), R * math.sin(t), A])
        return verts

    if tipo == "sfera":
        R  = float(geometria.get("raggio", 0.2))
        NL = 8; NM = 12
        verts = [[0.0, 0.0, 0.0]]          # v0 = centre (reference)
        for il in range(NL + 1):
            phi = math.pi * (il / NL - 0.5)
            for im in range(NM):
                theta = 2 * math.pi * im / NM
                verts.append([
                    R * math.cos(phi) * math.cos(theta),
                    R * math.cos(phi) * math.sin(theta),
                    R * math.sin(phi),
                ])
        return verts

    if tipo in ("barra", "staffa"):
        return [list(map(float, p)) for p in geometria.get("punti", [[0, 0, 0]])]

    return [[0.0, 0.0, 0.0]]


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def ruota_punto(pt: list, rx_deg: float, ry_deg: float, rz_deg: float) -> list:
    """Rotate a point by Euler angles (X → Y → Z order)."""
    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    # X
    y1 = y * math.cos(rx) - z * math.sin(rx)
    z1 = y * math.sin(rx) + z * math.cos(rx)
    # Y
    x2 =  x * math.cos(ry) + z1 * math.sin(ry)
    z2 = -x * math.sin(ry) + z1 * math.cos(ry)
    # Z
    x3 = x2 * math.cos(rz) - y1 * math.sin(rz)
    y3 = x2 * math.sin(rz) + y1 * math.cos(rz)
    return [x3, y3, z2]


def trasforma_punto(pt: list, posizione: list, rotazione: list) -> list:
    """Apply rotation then translation to a point."""
    r = ruota_punto(pt, *rotazione)
    return [r[0] + posizione[0], r[1] + posizione[1], r[2] + posizione[2]]


def deruota_punto(pt: list, rx_deg: float, ry_deg: float, rz_deg: float) -> list:
    """Inverse rotation: Z⁻¹ → Y⁻¹ → X⁻¹ (reverses ruota_punto)."""
    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    # Inverse Z
    x1 =  x * math.cos(rz) + y * math.sin(rz)
    y1 = -x * math.sin(rz) + y * math.cos(rz)
    z1 = z
    # Inverse Y
    x2 = x1 * math.cos(ry) - z1 * math.sin(ry)
    y2 = y1
    z2 = x1 * math.sin(ry) + z1 * math.cos(ry)
    # Inverse X
    x3 = x2
    y3 =  y2 * math.cos(rx) + z2 * math.sin(rx)
    z3 = -y2 * math.sin(rx) + z2 * math.cos(rx)
    return [x3, y3, z3]


def detrasforma_punto(pt_world: list, posizione: list, rotazione: list) -> list:
    """Convert a world-space point back to local object space."""
    shifted = [pt_world[i] - posizione[i] for i in range(3)]
    return deruota_punto(shifted, *rotazione)


# ---------------------------------------------------------------------------
# Oggetto3D
# ---------------------------------------------------------------------------

class Oggetto3D:
    """A single 3D structural element (body or reinforcement)."""

    _id_counter: int  = 0
    _nome_count: dict = {}

    def __init__(self, tipo: str, geometria: dict = None, materiale: str = ""):
        Oggetto3D._id_counter += 1
        self.id = Oggetto3D._id_counter

        nome_base = _TIPO_A_NOME.get(tipo, "Oggetto")
        Oggetto3D._nome_count[nome_base] = Oggetto3D._nome_count.get(nome_base, 0) + 1
        self.nome = f"{nome_base}.{Oggetto3D._nome_count[nome_base]:03d}"

        self.tipo            = tipo
        self.geometria       = geometria if geometria is not None else _geometria_default(tipo)
        self.materiale       = materiale
        self.posizione       = [0.0, 0.0, 0.0]
        self.rotazione       = [0.0, 0.0, 0.0]   # rx, ry, rz  [degrees]
        self.custom_geometry = False
        self.visibile        = True
        self.selezionabile   = True
        self.vertice_ref     = 0   # index of the reference/pivot vertex

    # ---------------------------------------------------------------- vertices

    def get_vertices_local(self) -> list:
        if self.custom_geometry:
            return [list(v) for v in self.geometria.get("vertici_custom", [[0, 0, 0]])]
        return calcola_vertici(self.tipo, self.geometria)

    def get_vertices_world(self) -> list:
        return [trasforma_punto(v, self.posizione, self.rotazione)
                for v in self.get_vertices_local()]

    def get_vertex_ref_world(self) -> list:
        verts = self.get_vertices_local()
        if not verts:
            return list(self.posizione)
        idx = self.vertice_ref if self.vertice_ref < len(verts) else 0
        return trasforma_punto(verts[idx], self.posizione, self.rotazione)

    def set_vertices_custom(self, vertici: list):
        """Switch to custom geometry with the given vertex list."""
        self.custom_geometry = True
        self.geometria["vertici_custom"] = [list(v) for v in vertici]

    # ---------------------------------------------------------------- text repr

    def get_properties_text(self) -> str:
        verts = self.get_vertices_local()
        return ", ".join(
            f"v{i+1}({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})"
            for i, v in enumerate(verts)
        )

    # ---------------------------------------------------------------- serialise

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "nome":            self.nome,
            "tipo":            self.tipo,
            "geometria":       copy.deepcopy(self.geometria),
            "materiale":       self.materiale,
            "posizione":       list(self.posizione),
            "rotazione":       list(self.rotazione),
            "custom_geometry": self.custom_geometry,
            "visibile":        self.visibile,
            "selezionabile":   self.selezionabile,
            "vertice_ref":     self.vertice_ref,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Oggetto3D":
        obj               = cls.__new__(cls)
        obj.id            = d["id"]
        obj.nome          = d["nome"]
        obj.tipo          = d["tipo"]
        obj.geometria     = copy.deepcopy(d["geometria"])
        obj.materiale     = d.get("materiale",       "")
        obj.posizione     = list(d.get("posizione",  [0, 0, 0]))
        obj.rotazione     = list(d.get("rotazione",  [0, 0, 0]))
        obj.custom_geometry = d.get("custom_geometry", False)
        obj.visibile      = d.get("visibile",        True)
        obj.selezionabile = d.get("selezionabile",   True)
        obj.vertice_ref   = d.get("vertice_ref",     0)
        return obj

    def duplica(self) -> "Oggetto3D":
        """Deep-copy with a new unique ID and auto-generated name."""
        nuovo = copy.deepcopy(self)
        Oggetto3D._id_counter += 1
        nuovo.id = Oggetto3D._id_counter
        nome_base = _TIPO_A_NOME.get(self.tipo, "Oggetto")
        Oggetto3D._nome_count[nome_base] = Oggetto3D._nome_count.get(nome_base, 0) + 1
        nuovo.nome = f"{nome_base}.{Oggetto3D._nome_count[nome_base]:03d}"
        return nuovo


# ---------------------------------------------------------------------------
# Elemento (structural member container)
# ---------------------------------------------------------------------------

_TIPO_A_NOME_ELEM = {
    "trave":      "Trave",
    "pilastro":   "Pilastro",
    "fondazione": "Fondazione",
    "solaio":     "Solaio",
}


class Elemento:
    """A structural member (beam, column, foundation, slab)."""

    _id_counter: int  = 0
    _nome_count: dict = {}

    def __init__(self, tipo: str):
        Elemento._id_counter += 1
        self.id       = Elemento._id_counter
        self.tipo     = tipo
        self.standard = False   # True solo per elementi del database standard

        nome_base = _TIPO_A_NOME_ELEM.get(tipo, "Elemento")
        Elemento._nome_count[nome_base] = Elemento._nome_count.get(nome_base, 0) + 1
        self.nome    = f"{nome_base}.{Elemento._nome_count[nome_base]:03d}"
        self.oggetti: list = []   # list[Oggetto3D]

    def to_dict(self) -> dict:
        return {
            "id":       self.id,
            "nome":     self.nome,
            "tipo":     self.tipo,
            "standard": self.standard,
            "oggetti":  [o.to_dict() for o in self.oggetti],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Elemento":
        el          = cls.__new__(cls)
        el.id       = d["id"]
        el.nome     = d["nome"]
        el.tipo     = d["tipo"]
        el.standard = d.get("standard", False)
        el.oggetti  = [Oggetto3D.from_dict(od) for od in d.get("oggetti", [])]
        return el