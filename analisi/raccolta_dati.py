"""
raccolta_dati.py
----------------
Raccoglie e struttura i dati di sezioni e materiali dal progetto attivo,
esponendoli ai sotto-moduli di analisi in un formato pronto all'uso.

Le sezioni standard del database sono sempre visibili anche se non ancora
esplicitamente salvate nel progetto; i dati del progetto hanno priorità
(override) su quelli del database, coerentemente con gestione_sezioni._leggi.
"""
from __future__ import annotations

import copy
from typing import Optional


# Categorie riconosciute dal progetto
_CAT_SEZIONI   = ("calcestruzzo_armato", "profili", "precompresso", "personalizzate")
_CAT_MATERIALI = ("calcestruzzo", "barre", "acciaio", "personalizzati")


def _carica_db_sezioni() -> dict:
    """Carica il database sezioni standard (lazy, con fallback silenzioso)."""
    try:
        from sezioni.database_sezioni import carica_database
        return carica_database()
    except Exception as e:
        print(f"WARN  raccolta_dati: impossibile caricare database sezioni ({e})")
        return {}


def _carica_db_materiali() -> dict:
    """Carica il database materiali standard (lazy, con fallback silenzioso)."""
    try:
        from materiali.database_materiali import carica_database
        return carica_database()
    except Exception as e:
        print(f"WARN  raccolta_dati: impossibile caricare database materiali ({e})")
        return {}


class RaccoltaDati:
    """
    Interfaccia unica per leggere sezioni e materiali dal progetto attivo.

    Parametri
    ---------
    main_window : MainWindow
        Finestra principale del programma (espone get_sezione / ha_progetto).
    """

    def __init__(self, main_window) -> None:
        self._main = main_window
        # Database standard (caricati una volta sola)
        self._db_sez: dict | None = None
        self._db_mat: dict | None = None

    # ------------------------------------------------------------------
    # ACCESSO AI DATABASE STANDARD
    # ------------------------------------------------------------------

    def _db_sezioni(self) -> dict:
        if self._db_sez is None:
            self._db_sez = _carica_db_sezioni()
        return self._db_sez

    def _db_materiali(self) -> dict:
        if self._db_mat is None:
            self._db_mat = _carica_db_materiali()
        return self._db_mat

    # ------------------------------------------------------------------
    # SEZIONI
    # ------------------------------------------------------------------

    def lista_sezioni(self) -> list[str]:
        """
        Ritorna la lista ordinata dei nomi di tutte le sezioni disponibili:
        – sezioni del database standard (IPE, HEA, HEB, R …, T …, ecc.)
        – sezioni salvate nel progetto (override o personalizzate)

        Se nessun progetto è aperto, restituisce comunque le sezioni standard
        del database in modo che si possano sfogliare.
        """
        nomi: set[str] = set()

        # 1. Sezioni dal database standard
        db = self._db_sezioni()
        for cat in _CAT_SEZIONI:
            for nome in db.get(cat, {}):
                nomi.add(nome)

        # 2. Sezioni dal progetto (aggiunge personalizzate e override)
        if self._main.ha_progetto():
            sezioni_prj = self._main.get_sezione("sezioni")
            for cat in _CAT_SEZIONI:
                for nome in sezioni_prj.get(cat, {}):
                    nomi.add(nome)

        return sorted(nomi)

    def dati_sezione(self, nome: str) -> Optional[dict]:
        """
        Ritorna il dizionario della sezione richiesta.
        Priorità: progetto  →  database standard.
        Ritorna None se non trovata.
        """
        # Prima cerca nel progetto (override utente)
        if self._main.ha_progetto():
            sezioni_prj = self._main.get_sezione("sezioni")
            for cat in _CAT_SEZIONI:
                cat_dict = sezioni_prj.get(cat, {})
                if nome in cat_dict:
                    return cat_dict[nome]

        # Poi cerca nel database standard
        db = self._db_sezioni()
        for cat in _CAT_SEZIONI:
            cat_dict = db.get(cat, {})
            if nome in cat_dict:
                return copy.deepcopy(cat_dict[nome])

        return None

    # ------------------------------------------------------------------
    # MATERIALI
    # ------------------------------------------------------------------

    def dati_materiale(self, nome: str) -> Optional[dict]:
        """
        Ritorna il dizionario del materiale richiesto.
        Priorità: progetto  →  database standard.
        Ritorna None se non trovato.
        """
        # Prima cerca nel progetto
        if self._main.ha_progetto():
            materiali_prj = self._main.get_sezione("materiali")
            for cat in _CAT_MATERIALI:
                cat_dict = materiali_prj.get(cat, {})
                if nome in cat_dict:
                    return cat_dict[nome]

        # Poi cerca nel database standard
        db = self._db_materiali()
        for cat in _CAT_MATERIALI:
            cat_dict = db.get(cat, {})
            if nome in cat_dict:
                return copy.deepcopy(cat_dict[nome])

        return None

    # ------------------------------------------------------------------
    # DATI ANALISI  (sezione + materiali risolti)
    # ------------------------------------------------------------------

    def dati_per_analisi(self, nome_sezione: str) -> Optional[dict]:
        """
        Restituisce un dizionario con:
          - 'sezione'   : dict grezzo della sezione
          - 'materiali' : {nome_mat: dict_materiale}  – solo i materiali usati

        Ritorna None se la sezione non esiste.
        """
        sez = self.dati_sezione(nome_sezione)
        if sez is None:
            return None

        # Raccoglie tutti i nomi di materiale usati dalla sezione
        nomi_mat: set[str] = set()
        elementi = sez.get("elementi", {})
        for gruppo in ("carpenteria", "barre"):
            for elem in elementi.get(gruppo, []):
                mat_nome = elem.get("materiale")
                if mat_nome:
                    nomi_mat.add(mat_nome)

        # Risolve ogni materiale (progetto → database standard)
        materiali: dict[str, dict] = {}
        for nome in nomi_mat:
            dati = self.dati_materiale(nome)
            if dati is not None:
                materiali[nome] = dati
            else:
                print(f"WARN  Materiale «{nome}» non trovato nel progetto né nel database.")

        return {
            "sezione":   sez,
            "materiali": materiali,
        }

    # ------------------------------------------------------------------
    # ANALISI SALVATA / CARICATA
    # ------------------------------------------------------------------

    def carica_impostazioni_pressoflessione(self) -> dict:
        """Legge le impostazioni salvate per il modulo pressoflessione."""
        analisi = self._main.get_sezione("analisi")
        return analisi.get("pressoflessione", {})

    def salva_impostazioni_pressoflessione(self, impostazioni: dict) -> None:
        """Salva le impostazioni del modulo pressoflessione nel progetto."""
        analisi = self._main.get_sezione("analisi")
        analisi["pressoflessione"] = impostazioni
        self._main.set_sezione("analisi", analisi)
