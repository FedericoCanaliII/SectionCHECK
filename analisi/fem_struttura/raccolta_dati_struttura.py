"""
raccolta_dati_struttura.py
--------------------------
Raccoglie i dati della struttura selezionata nella combobox e li prepara
per l'analisi FEM.  Legge il testo della struttura dal progetto attivo,
lo parsa con ``parse_struttura`` e restituisce i dati grezzi arricchiti
delle informazioni necessarie al modulo di analisi.
"""
from __future__ import annotations

from typing import Optional

from struttura.testo_struttura import parse_struttura


# Categorie di strutture nel progetto
_CAT_STRUTTURE = ("calcestruzzo", "acciaio", "personalizzate")


class RaccoltaDatiStruttura:
    """Interfaccia per leggere le strutture disponibili dal progetto attivo."""

    def __init__(self, main_window) -> None:
        self._main = main_window

    # ------------------------------------------------------------------
    #  LISTA STRUTTURE (per la combobox)
    # ------------------------------------------------------------------

    def lista_strutture(self) -> list[tuple[str, str]]:
        """
        Ritorna una lista di (categoria, nome) per tutte le strutture
        disponibili nel progetto corrente.
        """
        risultato: list[tuple[str, str]] = []
        if not self._main.ha_progetto():
            return risultato

        strutture = self._main.get_sezione("strutture")
        for cat in _CAT_STRUTTURE:
            cat_dict = strutture.get(cat, {})
            for nome in sorted(cat_dict.keys()):
                risultato.append((cat, nome))
        return risultato

    # ------------------------------------------------------------------
    #  TESTO STRUTTURA
    # ------------------------------------------------------------------

    def _leggi_testo(self, cat: str, nome: str) -> Optional[str]:
        """Legge il testo della struttura dal progetto."""
        if not self._main.ha_progetto():
            return None
        strutture = self._main.get_sezione("strutture")
        cat_dict = strutture.get(cat, {})
        entry = cat_dict.get(nome)
        if entry is None:
            return None
        return entry.get("testo", "")

    # ------------------------------------------------------------------
    #  DATI PARSATI
    # ------------------------------------------------------------------

    def dati_struttura(self, cat: str, nome: str) -> Optional[dict]:
        """
        Parsa il testo della struttura e restituisce i dati strutturali.

        Ritorna None se la struttura non esiste o il testo e' vuoto.
        Il dizionario restituito contiene le chiavi standard di
        ``parse_struttura`` piu' metadati utili.
        """
        testo = self._leggi_testo(cat, nome)
        if not testo or not testo.strip():
            return None

        dati, errori = parse_struttura(testo)

        if errori:
            for riga, msg in errori:
                print(f"WARN  Struttura '{nome}' riga {riga}: {msg}")

        # Aggiungi metadati
        dati["_nome"] = nome
        dati["_categoria"] = cat
        dati["_errori"] = errori

        return dati
