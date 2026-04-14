"""
raccolta_dati_elemento_fem.py
------------------------------
Raccoglie e struttura i dati degli elementi 3D (Oggetto3D), dei materiali
e dei carichi/vincoli dal progetto attivo, esponendoli al modulo FEM
in un formato pronto all'uso per la generazione della mesh e l'analisi.
"""
from __future__ import annotations

import copy
from typing import Optional

from elementi.modello_3d import Oggetto3D, Elemento, TIPO_STRUTTURALE, TIPO_ARMATURA
from elementi.modello_carichi_vincoli import CaricoVincolo


_CAT_MATERIALI = ("calcestruzzo", "barre", "acciaio", "personalizzati")


def _carica_db_materiali() -> dict:
    try:
        from materiali.database_materiali import carica_database
        return carica_database()
    except Exception as e:
        print(f"WARN  raccolta_dati_elemento_fem: impossibile caricare database materiali ({e})")
        return {}


class RaccoltaDatiElementoFEM:
    """
    Interfaccia per leggere elementi 3D, materiali e carichi/vincoli
    dal progetto attivo, destinata al modulo FEM.
    """

    def __init__(self, main_window) -> None:
        self._main = main_window
        self._db_mat: dict | None = None

    def _db_materiali(self) -> dict:
        if self._db_mat is None:
            self._db_mat = _carica_db_materiali()
        return self._db_mat

    # ------------------------------------------------------------------
    # LISTA ELEMENTI
    # ------------------------------------------------------------------

    def lista_elementi(self) -> list[tuple[str, str, int]]:
        """
        Ritorna la lista di tutti gli elementi strutturali definiti nel progetto.
        Ogni voce e' (tipo, nome, id).
        """
        risultato = []
        if not self._main.ha_progetto():
            return risultato

        # Accede al modulo elementi
        gestione_elem = getattr(self._main, '_elementi', None)
        if gestione_elem is None:
            return risultato

        lista_ctrl = getattr(gestione_elem, '_lista_ctrl', None)
        if lista_ctrl is None:
            return risultato

        elementi_dict = lista_ctrl.get_elementi()
        for tipo, lista in elementi_dict.items():
            for el in lista:
                risultato.append((tipo, el.nome, el.id))
        return risultato

    def get_elemento_by_id(self, el_id: int) -> Optional[Elemento]:
        """Ritorna l'oggetto Elemento dato il suo id."""
        gestione_elem = getattr(self._main, '_elementi', None)
        if gestione_elem is None:
            return None
        lista_ctrl = getattr(gestione_elem, '_lista_ctrl', None)
        if lista_ctrl is None:
            return None
        for lista in lista_ctrl.get_elementi().values():
            for el in lista:
                if el.id == el_id:
                    return el
        return None

    def get_elemento_by_nome(self, nome: str) -> Optional[Elemento]:
        """Ritorna l'oggetto Elemento dato il suo nome."""
        gestione_elem = getattr(self._main, '_elementi', None)
        if gestione_elem is None:
            return None
        lista_ctrl = getattr(gestione_elem, '_lista_ctrl', None)
        if lista_ctrl is None:
            return None
        for lista in lista_ctrl.get_elementi().values():
            for el in lista:
                if el.nome == nome:
                    return el
        return None

    # ------------------------------------------------------------------
    # MATERIALI
    # ------------------------------------------------------------------

    def dati_materiale(self, nome: str) -> Optional[dict]:
        """
        Ritorna il dizionario del materiale richiesto.
        Priorita': progetto -> database standard.
        """
        if self._main.ha_progetto():
            materiali_prj = self._main.get_sezione("materiali")
            for cat in _CAT_MATERIALI:
                cat_dict = materiali_prj.get(cat, {})
                if nome in cat_dict:
                    return cat_dict[nome]

        db = self._db_materiali()
        for cat in _CAT_MATERIALI:
            cat_dict = db.get(cat, {})
            if nome in cat_dict:
                return copy.deepcopy(cat_dict[nome])
        return None

    # ------------------------------------------------------------------
    # CARICHI E VINCOLI
    # ------------------------------------------------------------------

    def get_carichi_vincoli(self, el_id: int) -> list[CaricoVincolo]:
        """
        Ritorna la lista dei CaricoVincolo associati all'elemento dato.
        Legge direttamente dalla sezione 'carichi' del progetto.
        """
        if not self._main.ha_progetto():
            return []
        sezione = self._main.get_sezione("carichi")
        raw = sezione.get(str(el_id), [])
        return [CaricoVincolo.from_dict(d) for d in raw]

    # ------------------------------------------------------------------
    # DATI COMPLETI PER FEM
    # ------------------------------------------------------------------

    def dati_per_fem(self, el_id: int) -> Optional[dict]:
        """
        Restituisce un dizionario completo per la generazione FEM:
          - 'elemento'      : Elemento
          - 'oggetti'       : list[Oggetto3D]  (tutti gli oggetti dell'elemento)
          - 'carpenteria'   : list[Oggetto3D]  (solo strutturali)
          - 'barre'         : list[Oggetto3D]  (solo barre longitudinali)
          - 'staffe'        : list[Oggetto3D]  (solo staffe)
          - 'materiali'     : {nome_mat: dict_materiale}
          - 'carichi'       : list[CaricoVincolo]
          - 'vincoli'       : list[CaricoVincolo]
        """
        el = self.get_elemento_by_id(el_id)
        if el is None:
            return None

        carpenteria = []
        barre = []
        staffe = []

        for obj in el.oggetti:
            if obj.tipo in TIPO_STRUTTURALE:
                carpenteria.append(obj)
            elif obj.tipo == "barra":
                barre.append(obj)
            elif obj.tipo == "staffa":
                staffe.append(obj)

        # Raccoglie materiali usati
        nomi_mat: set[str] = set()
        for obj in el.oggetti:
            if obj.materiale:
                nomi_mat.add(obj.materiale)

        materiali = {}
        for nome in nomi_mat:
            dati = self.dati_materiale(nome)
            if dati is not None:
                materiali[nome] = dati
            else:
                print(f"WARN  Materiale '{nome}' non trovato.")

        # Carichi e vincoli
        cv_lista = self.get_carichi_vincoli(el_id)
        carichi = [cv for cv in cv_lista if cv.sottotipo == "carico"]
        vincoli = [cv for cv in cv_lista if cv.sottotipo == "vincolo"]

        return {
            "elemento":    el,
            "oggetti":     el.oggetti,
            "carpenteria": carpenteria,
            "barre":       barre,
            "staffe":      staffe,
            "materiali":   materiali,
            "carichi":     carichi,
            "vincoli":     vincoli,
        }
