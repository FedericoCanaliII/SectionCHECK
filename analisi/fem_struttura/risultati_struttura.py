"""
risultati_struttura.py
----------------------
Classi dati per i risultati dell'analisi FEM della struttura.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SforziAsta:
    """Sforzi interni lungo un'asta (beam) originale.

    Ogni lista ha (N_divisioni + 1) valori, uno per ciascun nodo
    della discretizzazione (da nodo_i a nodo_j dell'asta originale).
    """
    asta_id: int
    posizioni: list[float] = field(default_factory=list)   # ascissa curvilinea [m]
    N: list[float] = field(default_factory=list)            # sforzo normale [kN]
    Vy: list[float] = field(default_factory=list)           # taglio y [kN]
    Vz: list[float] = field(default_factory=list)           # taglio z [kN]
    My: list[float] = field(default_factory=list)           # momento y [kNm]
    Mz: list[float] = field(default_factory=list)           # momento z [kNm]
    T: list[float] = field(default_factory=list)            # torsione [kNm]
    # Tensione equivalente stimata in ogni posizione [MPa]
    # (|sigma_N| + sqrt(sigma_My^2 + sigma_Mz^2) al raggio d'inerzia)
    sigma_eq: list[float] = field(default_factory=list)


@dataclass
class SpostamentoNodo:
    """Spostamento e rotazione di un nodo."""
    nodo_tag: int
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    @property
    def modulo(self) -> float:
        return (self.dx ** 2 + self.dy ** 2 + self.dz ** 2) ** 0.5


@dataclass
class TensioniShell:
    """Tensioni in un elemento shell."""
    shell_tag: int
    shell_originale: int
    # Tensioni medie ai nodi: {nodo_tag: (sx, sy, sxy, sigma_vm)}
    tensioni_nodi: dict[int, tuple[float, float, float, float]] = \
        field(default_factory=dict)


@dataclass
class RisultatiFEMStruttura:
    """Contenitore di tutti i risultati dell'analisi."""
    # Sforzi per ogni asta originale
    sforzi_aste: dict[int, SforziAsta] = field(default_factory=dict)

    # Spostamenti per ogni nodo mesh
    spostamenti: dict[int, SpostamentoNodo] = field(default_factory=dict)

    # Tensioni shell
    tensioni_shell: list[TensioniShell] = field(default_factory=list)

    # Reazioni vincolari: {nodo_tag: (Fx, Fy, Fz, Mx, My, Mz)}
    reazioni: dict[int, tuple] = field(default_factory=dict)

    # Flag successo
    successo: bool = False
    messaggio: str = ""

    # ---- Proprieta' utili ----

    @property
    def max_spostamento(self) -> float:
        if not self.spostamenti:
            return 0.0
        return max(s.modulo for s in self.spostamenti.values())

    def max_sforzo(self, componente: str) -> float:
        """Valore massimo assoluto di una componente sforzo (N, Vy, Vz, My, Mz, T)."""
        val_max = 0.0
        for sf in self.sforzi_aste.values():
            vals = getattr(sf, componente, [])
            if vals:
                val_max = max(val_max, max(abs(v) for v in vals))
        return val_max

    def max_sigma_vm_shell(self) -> float:
        """Massima tensione di Von Mises nelle shell."""
        vm_max = 0.0
        for ts in self.tensioni_shell:
            for _, tens in ts.tensioni_nodi.items():
                vm_max = max(vm_max, abs(tens[3]))
        return vm_max

    def max_sigma_eq_beam(self) -> float:
        """Massima tensione equivalente nelle aste."""
        s_max = 0.0
        for sf in self.sforzi_aste.values():
            if sf.sigma_eq:
                s_max = max(s_max, max(abs(s) for s in sf.sigma_eq))
        return s_max

    def max_tensione_globale(self) -> float:
        """Massimo tra tensione sulle aste e sigma_vm shell."""
        return max(self.max_sigma_eq_beam(), self.max_sigma_vm_shell())
