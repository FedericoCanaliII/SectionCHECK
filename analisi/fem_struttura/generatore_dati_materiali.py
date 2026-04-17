"""
generatore_dati_materiali.py
----------------------------
Risolve i materiali definiti nella struttura (sia inline che per riferimento)
e li converte nel formato necessario per l'analisi FEM:

    {mid: {"nome": str, "densita": float, "E": float, "G": float, "J": float}}

Per materiali definiti come 'riferimento' (solo il nome), cerca nel database
materiali del programma e ne estrae le proprieta' meccaniche.
"""
from __future__ import annotations

from typing import Optional

from analisi.raccolta_dati import RaccoltaDati


class GeneratoreDatiMateriali:
    """Risolve i materiali della struttura per l'analisi."""

    def __init__(self, main_window) -> None:
        self._raccolta = RaccoltaDati(main_window)

    def risolvi_materiali(self, materiali_struttura: dict) -> dict[int, dict]:
        """
        Converte i materiali parsati dalla struttura nel formato analisi.

        Parametri
        ---------
        materiali_struttura : dict
            Dizionario {mid: {...}} dal parser della struttura.

        Ritorna
        -------
        dict[int, dict]
            {mid: {"nome", "densita", "E", "G", "J"}}
        """
        risultato: dict[int, dict] = {}

        for mid, mat_data in materiali_struttura.items():
            tipo = mat_data.get("tipo", "inline")
            nome = mat_data.get("nome", f"Mat_{mid}")

            if tipo == "inline":
                risultato[mid] = {
                    "nome":    nome,
                    "densita": float(mat_data.get("densita", 0.0)),
                    "E":       float(mat_data.get("E", 0.0)),
                    "G":       float(mat_data.get("G", 0.0)),
                    "J":       float(mat_data.get("J", 0.0)),
                }
            else:
                # Riferimento: cerca nel database del programma
                risolto = self._risolvi_riferimento(nome)
                if risolto is not None:
                    risultato[mid] = risolto
                else:
                    print(f"WARN  Materiale '{nome}' (id={mid}) non trovato nel "
                          f"database. Verra' usato un materiale di default.")
                    risultato[mid] = self._materiale_default(nome)

        return risultato

    def _risolvi_riferimento(self, nome: str) -> Optional[dict]:
        """
        Cerca il materiale per nome nel database del programma e ne estrae
        le proprieta' necessarie per l'analisi della struttura.
        """
        dati_mat = self._raccolta.dati_materiale(nome)
        if dati_mat is None:
            return None

        # Estrai proprieta' dal formato del database materiali
        densita = float(dati_mat.get("densita", 0.0))
        E = float(dati_mat.get("m_elastico", 0.0))
        G = float(dati_mat.get("m_taglio", 0.0))
        poisson = float(dati_mat.get("poisson", 0.3))

        # Se G non e' specificato, calcolalo da E e poisson
        if G <= 0 and E > 0:
            G = E / (2.0 * (1.0 + poisson))

        # Modulo torsionale: per materiali isotropi coincide con G
        J_mod = G

        return {
            "nome":    nome,
            "densita": densita,
            "E":       E,
            "G":       G,
            "J":       J_mod,
        }

    @staticmethod
    def _materiale_default(nome: str) -> dict:
        """Materiale di fallback (calcestruzzo C25/30 tipico)."""
        E = 31000.0     # MPa
        poisson = 0.2
        G = E / (2.0 * (1.0 + poisson))
        return {
            "nome":    nome,
            "densita": 2500.0,
            "E":       E,
            "G":       G,
            "J":       G,
        }
