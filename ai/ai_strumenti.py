# -*- coding: utf-8 -*-
"""
ai_strumenti.py
---------------
Strumenti che l'agente AI può usare per interagire con SectionCHECK.
Ogni metodo pubblico restituisce (successo: bool, messaggio: str, dati: Any).
Il dispatcher `esegui()` è il punto di ingresso usato dall'agente.
"""

import uuid
from typing import Any


# ================================================================
#  SCHEMA STRUMENTI  (usato per costruire il system prompt)
# ================================================================

TOOLS_SCHEMA: list[dict] = [
    {
        "name": "elenca_materiali",
        "description": "Elenca tutti i materiali disponibili nel progetto corrente "
                       "(calcestruzzo, barre, acciaio, personalizzati).",
        "params": {},
    },
    {
        "name": "get_info_materiale",
        "description": "Restituisce le proprietà di un materiale specifico.",
        "params": {
            "nome": "str – nome esatto del materiale",
        },
    },
    {
        "name": "crea_materiale_personalizzato",
        "description": "Crea un nuovo materiale personalizzato nel progetto.",
        "params": {
            "nome":       "str – nome univoco",
            "gamma":      "float – fattore di sicurezza (default 1.0)",
            "alpha":      "float – fattore di riduzione (default 1.0)",
            "densita":    "float – densità kg/m³ (default 0)",
            "poisson":    "float – coefficiente di Poisson (default 0)",
            "m_elastico": "float – modulo elastico E [MPa] (default 0)",
            "m_taglio":   "float – modulo di taglio G [MPa] (default 0)",
            "slu":        "list  – segmenti σ-ε SLU: [{\"formula\":\"...\",\"eps_min\":…,\"eps_max\":…}] (opz.)",
            "sle":        "list  – segmenti σ-ε SLE: stessa struttura (opz.)",
        },
    },
    {
        "name": "modifica_materiale",
        "description": "Modifica le proprietà di un materiale non-standard.",
        "params": {
            "nome":       "str – nome del materiale",
            "categoria":  "str – calcestruzzo | barre | acciaio | personalizzati",
            "proprieta":  "dict – coppie chiave→valore da aggiornare "
                          "(gamma, alpha, densita, poisson, m_elastico, m_taglio, slu, sle)",
        },
    },
    {
        "name": "elimina_materiale_personalizzato",
        "description": "Elimina un materiale dalla categoria 'personalizzati'.",
        "params": {
            "nome": "str – nome del materiale personalizzato",
        },
    },
    {
        "name": "elenca_sezioni",
        "description": "Elenca tutte le sezioni disponibili nel progetto corrente "
                       "(calcestruzzo_armato, profili, precompresso, personalizzate).",
        "params": {},
    },
    {
        "name": "get_info_sezione",
        "description": "Restituisce la struttura di una sezione (tipo, elementi, materiali).",
        "params": {
            "nome": "str – nome esatto della sezione",
        },
    },
    {
        "name": "crea_sezione_personalizzata",
        "description": "Crea una nuova sezione personalizzata vuota.",
        "params": {
            "nome":      "str – nome univoco",
            "categoria": "str – calcestruzzo_armato | profili | precompresso | personalizzate "
                         "(default: personalizzate)",
        },
    },
    {
        "name": "aggiungi_rettangolo",
        "description": "Aggiunge un elemento rettangolare (carpenteria) a una sezione personalizzata.",
        "params": {
            "nome_sezione": "str – nome della sezione",
            "x0":           "float – coord X angolo inferiore-sinistro [mm]",
            "y0":           "float – coord Y angolo inferiore-sinistro [mm]",
            "x1":           "float – coord X angolo superiore-destro [mm]",
            "y1":           "float – coord Y angolo superiore-destro [mm]",
            "materiale":    "str  – nome materiale da assegnare (opz., default \"\")",
        },
    },
    {
        "name": "aggiungi_cerchio_carpenteria",
        "description": "Aggiunge un elemento circolare o ellittico (carpenteria) a una sezione personalizzata. "
                       "Per cerchio usa solo 'r'; per ellisse usa 'rx' e 'ry' (raggi lungo X e Y).",
        "params": {
            "nome_sezione": "str",
            "cx":           "float – coordinata X del centro [mm]",
            "cy":           "float – coordinata Y del centro [mm]",
            "r":            "float – raggio (cerchio, imposta rx=ry=r) [mm] (opz. se rx/ry forniti)",
            "rx":           "float – raggio semiasse X [mm] (opz., sovrascrive r)",
            "ry":           "float – raggio semiasse Y [mm] (opz., sovrascrive r)",
            "materiale":    "str  (opz.)",
        },
    },
    {
        "name": "aggiungi_poligono",
        "description": "Aggiunge un elemento poligonale (carpenteria) a una sezione personalizzata.",
        "params": {
            "nome_sezione": "str",
            "punti":        "list – [[x1,y1],[x2,y2],…] almeno 3 vertici [mm]",
            "materiale":    "str  (opz.)",
        },
    },
    {
        "name": "aggiungi_barra",
        "description": "Aggiunge una barra di armatura longitudinale. "
                       "REGOLE ANTI-SOVRAPPOSIZIONE: Le barre non devono mai sovrapporsi fisicamente alle staffe o ad altre barre. "
                       "Se la barra si trova nell'angolo di una staffa, calcola il suo centro (cx, cy) spostandolo "
                       "dal vertice della staffa verso l'interno della sezione. L'offset esatto da applicare al vertice "
                       "è (raggio_staffa + raggio_barra) sia lungo X che lungo Y, a seconda del quadrante dell'angolo.",
        "params": {
            "nome_sezione": "str",
            "cx":           "float – coordinata X del centro [mm] (applica offset se vicino a una staffa)",
            "cy":           "float – coordinata Y del centro [mm] (applica offset se vicino a una staffa)",
            "r":            "float – raggio della barra [mm]  (es. ⌀16 → r=8)",
            "materiale":    "str  – es. \"B500B\" (opz.)",
        },
    },
    {
        "name": "aggiungi_staffa",
        "description": "Aggiunge una staffa (stirrup) a una sezione personalizzata. "
                       "REGOLE CRITICHE: (1) La staffa deve essere chiusa (primo e ultimo punto coincidono). "
                       "(2) Le aree di staffe e barre NON DEVONO MAI SOVRAPPORSI. "
                       "(3) La staffa avvolge le barre longitudinali dall'ESTERNO, sfiorandole. "
                       "(4) In ogni vertice della staffa va posizionata una barra, ma rigorosamente all'INTERNO della staffa. "
                       "Il centro della barra NON DEVE COINCIDERE col vertice della staffa, ma deve essere traslato "
                       "verso l'interno della sezione di una quantità pari a (raggio_staffa + raggio_barra).",
        "params": {
            "nome_sezione": "str",
            "punti":        "list – [[x1,y1],…,[x1,y1]] vertici (asse del filo) incl. chiusura [mm]",
            "r":            "float – raggio del filo staffa [mm] (default 5)",
            "materiale":    "str  (opz.)",
        },
    },
    {
        "name": "aggiungi_foro_rettangolo",
        "description": "Aggiunge un foro rettangolare a una sezione personalizzata.",
        "params": {
            "nome_sezione": "str",
            "x0":           "float – coord X angolo inferiore-sinistro [mm]",
            "y0":           "float – coord Y angolo inferiore-sinistro [mm]",
            "x1":           "float – coord X angolo superiore-destro [mm]",
            "y1":           "float – coord Y angolo superiore-destro [mm]",
        },
    },
    {
        "name": "aggiungi_foro_cerchio",
        "description": "Aggiunge un foro circolare o ellittico a una sezione personalizzata. "
                       "Per foro circolare usa 'r'; per foro ellittico usa 'rx' e 'ry'.",
        "params": {
            "nome_sezione": "str",
            "cx":           "float – coordinata X del centro [mm]",
            "cy":           "float – coordinata Y del centro [mm]",
            "r":            "float – raggio (foro circolare) [mm] (opz. se rx/ry forniti)",
            "rx":           "float – raggio semiasse X [mm] (opz.)",
            "ry":           "float – raggio semiasse Y [mm] (opz.)",
        },
    },
    {
        "name": "aggiungi_foro_poligono",
        "description": "Aggiunge un foro poligonale a una sezione personalizzata.",
        "params": {
            "nome_sezione": "str",
            "punti":        "list – [[x1,y1],[x2,y2],…] almeno 3 vertici [mm]",
        },
    },
    {
        "name": "elimina_sezione_personalizzata",
        "description": "Elimina una sezione non-standard dal progetto.",
        "params": {
            "nome": "str – nome della sezione",
        },
    },

    # ── ELEMENTI 3D ────────────────────────────────────────────────────────
    {
        "name": "elenca_elementi",
        "description": "Elenca tutti gli elementi 3D nel progetto corrente "
                       "(travi, pilastri, fondazioni, solai), indicando nome e se sono standard.",
        "params": {},
    },
    {
        "name": "get_info_elemento",
        "description": "Restituisce la struttura dettagliata di un elemento 3D: "
                       "lista degli oggetti con tipo, geometria, materiale, posizione e rotazione.",
        "params": {
            "nome": "str – nome esatto dell'elemento",
        },
    },
    {
        "name": "crea_elemento",
        "description": "Crea un nuovo elemento 3D personalizzato vuoto nel progetto.",
        "params": {
            "nome": "str – nome univoco",
            "tipo": "str – trave | pilastro | fondazione | solaio",
        },
    },
    {
        "name": "aggiungi_oggetto_elemento",
        "description": (
            "Aggiunge un oggetto 3D a un elemento personalizzato. "
            "UNITÀ: metri [m]. "
            "Sistema di riferimento locale: X = asse longitudinale (lunghezza), "
            "Y = larghezza (base), Z = altezza/spessore. "
            "Per il pilastro l'asse longitudinale è Z. "
            "Tipi strutturali: "
            "  parallelepipedo → geometria: {lunghezza, base, altezza}; "
            "  cilindro        → geometria: {altezza, raggio}; "
            "  sfera           → geometria: {raggio}. "
            "Tipi armatura: "
            "  barra  → geometria: {diametro, punti:[[x,y,z],…]} (linea dell'asse); "
            "  staffa → geometria: {diametro, punti:[[x,y,z],…]} (percorso chiuso: ultimo punto = primo). "
            "Copriferro tipico: 0.030 m travi/pilastri, 0.040 m fondazioni, 0.025 m solai. "
            "Centro barra Φ16: copriferro + r_staffa + r_barra = 0.030+0.008+0.008 = 0.046 m. "
            "Ordine consigliato: (1) carpenteria, (2) staffe, (3) barre."
        ),
        "params": {
            "nome_elemento": "str – nome dell'elemento",
            "tipo_oggetto":  "str – parallelepipedo | cilindro | sfera | barra | staffa",
            "nome_oggetto":  "str – nome dell'oggetto (opz., default: auto-generato es. 'Barra.001')",
            "geometria":     "dict – parametri geometrici (es. {\"lunghezza\":5.0,\"base\":0.30,\"altezza\":0.50})",
            "materiale":     "str  – nome materiale (opz., es. \"C25/30\", \"B450C\")",
            "posizione":     "list – [x,y,z] in m (default [0,0,0])",
            "rotazione":     "list – [rx,ry,rz] gradi Euler X→Y→Z (default [0,0,0])",
        },
    },
    {
        "name": "modifica_oggetto_elemento",
        "description": "Modifica geometria, materiale, posizione o rotazione di un oggetto "
                       "in un elemento personalizzato (aggiornamento parziale: specifica solo i campi da cambiare).",
        "params": {
            "nome_elemento": "str – nome dell'elemento",
            "nome_oggetto":  "str – nome esatto dell'oggetto da modificare",
            "geometria":     "dict – nuovi parametri geometrici (opz., aggiornamento parziale)",
            "materiale":     "str  – nuovo materiale (opz.)",
            "posizione":     "list – nuova posizione [x,y,z] m (opz.)",
            "rotazione":     "list – nuova rotazione [rx,ry,rz] gradi (opz.)",
        },
    },
    {
        "name": "elimina_oggetto_elemento",
        "description": "Rimuove un oggetto da un elemento personalizzato.",
        "params": {
            "nome_elemento": "str – nome dell'elemento",
            "nome_oggetto":  "str – nome esatto dell'oggetto da rimuovere",
        },
    },
    {
        "name": "elimina_elemento",
        "description": "Elimina un elemento personalizzato dal progetto.",
        "params": {
            "nome": "str – nome dell'elemento",
        },
    },

    # ── CARICHI / VINCOLI ──────────────────────────────────────────────
    {
        "name": "elenca_carichi_vincoli",
        "description": "Elenca tutti i carichi e vincoli associati a un elemento 3D.",
        "params": {
            "nome_elemento": "str – nome dell'elemento",
        },
    },
    {
        "name": "aggiungi_carico_vincolo",
        "description": (
            "Aggiunge un carico o vincolo a un elemento 3D. "
            "Il carico/vincolo è un parallelepipedo: tutti i nodi della mesh FEM "
            "che ricadono al suo interno ricevono le forze/cedimenti specificati. "
            "UNITÀ: geometria in m, forze in kN, cedimenti in m. "
            "Sottotipi: 'vincolo' → caratteristiche {sx, sy, sz} [m] (cedimenti); "
            "'carico' → caratteristiche {fx, fy, fz} [kN] (forze). "
            "Il parallelepipedo può essere posizionato e ruotato liberamente. "
            "Dopo la creazione si può alterare la forma con modifica_vertici_oggetto."
        ),
        "params": {
            "nome_elemento":  "str – nome dell'elemento",
            "sottotipo":      "str – 'vincolo' | 'carico'",
            "geometria":      "dict – {lunghezza, base, altezza} in m (default 0.3 ciascuno)",
            "caratteristiche": "dict – {sx,sy,sz} per vincolo [m] o {fx,fy,fz} per carico [kN] "
                              "(default tutti 0.0)",
            "posizione":      "list – [x,y,z] in m (default [0,0,0])",
            "rotazione":      "list – [rx,ry,rz] gradi (default [0,0,0])",
        },
    },
    {
        "name": "modifica_carico_vincolo",
        "description": "Modifica geometria, caratteristiche, posizione o rotazione di un carico/vincolo "
                       "esistente (aggiornamento parziale: specifica solo i campi da cambiare).",
        "params": {
            "nome_elemento":   "str – nome dell'elemento",
            "nome_cv":         "str – nome esatto del carico/vincolo (es. 'Vincolo.001')",
            "geometria":       "dict – {lunghezza, base, altezza} (opz.)",
            "caratteristiche": "dict – {sx,sy,sz} o {fx,fy,fz} (opz.)",
            "posizione":       "list – [x,y,z] m (opz.)",
            "rotazione":       "list – [rx,ry,rz] gradi (opz.)",
        },
    },
    {
        "name": "elimina_carico_vincolo",
        "description": "Rimuove un carico/vincolo da un elemento.",
        "params": {
            "nome_elemento": "str – nome dell'elemento",
            "nome_cv":       "str – nome esatto del carico/vincolo",
        },
    },

    # ── MANIPOLAZIONE VERTICI ──────────────────────────────────────────
    {
        "name": "get_vertici_oggetto",
        "description": (
            "Restituisce la lista dei vertici locali di un oggetto 3D o di un carico/vincolo. "
            "Utile per ispezionare la forma prima di modificarla. "
            "Parallelepipedo: 8 vertici (ordine: 4 faccia inferiore + 4 faccia superiore). "
            "Cilindro: centro_basso + 24 punti cerchio basso + centro_alto + 24 punti cerchio alto. "
            "Sfera: centro + griglia lat/lon. "
            "Barra/Staffa: lista punti della polyline."
        ),
        "params": {
            "nome_elemento": "str – nome dell'elemento",
            "nome_oggetto":  "str – nome dell'oggetto o del carico/vincolo",
            "target":        "str – 'oggetto' (default) | 'carico_vincolo' – indica se cercare "
                             "tra gli oggetti strutturali o tra i carichi/vincoli",
        },
    },
    {
        "name": "modifica_vertici_oggetto",
        "description": (
            "Modifica uno o più vertici di un oggetto 3D o carico/vincolo, "
            "attivando la geometria custom. Permette di trasformare la forma "
            "(es. parallelepipedo → tronco di piramide, deformare cilindro/sfera, ecc.). "
            "Specificare TUTTI i vertici (lista completa) oppure solo quelli da modificare "
            "tramite il parametro 'modifiche' (dizionario indice→[x,y,z]). "
            "ATTENZIONE: usare prima get_vertici_oggetto per conoscere la geometria attuale."
        ),
        "params": {
            "nome_elemento": "str – nome dell'elemento",
            "nome_oggetto":  "str – nome dell'oggetto o del carico/vincolo",
            "target":        "str – 'oggetto' (default) | 'carico_vincolo'",
            "vertici":       "list – lista completa [[x,y,z],…] di TUTTI i vertici (opz., alternativo a 'modifiche')",
            "modifiche":     "dict – {indice: [x,y,z], …} vertici da modificare per indice (opz., alternativo a 'vertici')",
        },
    },
]


# ================================================================
#  CLASSE PRINCIPALE
# ================================================================

class AIStrumenti:
    """
    Wrapper che espone operazioni sicure sul progetto corrente.
    Viene istanziato con un riferimento alla MainWindow.
    """

    def __init__(self, main_window):
        self._main = main_window

    # ------------------------------------------------------------------
    #  DISPATCHER
    # ------------------------------------------------------------------

    def esegui(self, nome: str, params: dict) -> tuple[bool, str]:
        """
        Punto di ingresso usato dall'agente: chiama il metodo
        corrispondente a `nome` con i `params` forniti.
        Ritorna (successo, messaggio_testuale).
        """
        _mappa = {
            "elenca_materiali":               lambda p: self.elenca_materiali(),
            "get_info_materiale":             lambda p: self.get_info_materiale(**p),
            "crea_materiale_personalizzato":  lambda p: self.crea_materiale_personalizzato(**p),
            "modifica_materiale":             lambda p: self.modifica_materiale(**p),
            "elimina_materiale_personalizzato": lambda p: self.elimina_materiale_personalizzato(**p),
            "elenca_sezioni":                 lambda p: self.elenca_sezioni(),
            "get_info_sezione":               lambda p: self.get_info_sezione(**p),
            "crea_sezione_personalizzata":    lambda p: self.crea_sezione_personalizzata(**p),
            "aggiungi_rettangolo":            lambda p: self.aggiungi_rettangolo(**p),
            "aggiungi_cerchio_carpenteria":   lambda p: self.aggiungi_cerchio_carpenteria(**p),
            "aggiungi_poligono":              lambda p: self.aggiungi_poligono(**p),
            "aggiungi_barra":                 lambda p: self.aggiungi_barra(**p),
            "aggiungi_staffa":                lambda p: self.aggiungi_staffa(**p),
            "aggiungi_foro_rettangolo":       lambda p: self.aggiungi_foro_rettangolo(**p),
            "aggiungi_foro_cerchio":          lambda p: self.aggiungi_foro_cerchio(**p),
            "aggiungi_foro_poligono":         lambda p: self.aggiungi_foro_poligono(**p),
            "elimina_sezione_personalizzata": lambda p: self.elimina_sezione_personalizzata(**p),
            # Elementi 3D
            "elenca_elementi":           lambda p: self.elenca_elementi(),
            "get_info_elemento":         lambda p: self.get_info_elemento(**p),
            "crea_elemento":             lambda p: self.crea_elemento(**p),
            "aggiungi_oggetto_elemento": lambda p: self.aggiungi_oggetto_elemento(**p),
            "modifica_oggetto_elemento": lambda p: self.modifica_oggetto_elemento(**p),
            "elimina_oggetto_elemento":  lambda p: self.elimina_oggetto_elemento(**p),
            "elimina_elemento":          lambda p: self.elimina_elemento(**p),
            # Carichi / Vincoli
            "elenca_carichi_vincoli":    lambda p: self.elenca_carichi_vincoli(**p),
            "aggiungi_carico_vincolo":   lambda p: self.aggiungi_carico_vincolo(**p),
            "modifica_carico_vincolo":   lambda p: self.modifica_carico_vincolo(**p),
            "elimina_carico_vincolo":    lambda p: self.elimina_carico_vincolo(**p),
            # Manipolazione vertici
            "get_vertici_oggetto":       lambda p: self.get_vertici_oggetto(**p),
            "modifica_vertici_oggetto":  lambda p: self.modifica_vertici_oggetto(**p),
        }
        fn = _mappa.get(nome)
        if fn is None:
            return False, f"Strumento '{nome}' non riconosciuto."
        try:
            ok, msg, _ = fn(params)
            return ok, msg
        except TypeError as e:
            return False, f"Parametri non validi per '{nome}': {e}"
        except Exception as e:
            return False, f"Errore durante '{nome}': {e}"

    # ------------------------------------------------------------------
    #  HELPERS INTERNI
    # ------------------------------------------------------------------

    def _mat(self) -> dict:
        return self._main.get_sezione("materiali")

    def _sez(self) -> dict:
        return self._main.get_sezione("sezioni")

    def _salva_mat(self, mat: dict):
        self._main.set_sezione("materiali", mat)

    def _salva_sez(self, sez: dict):
        self._main.set_sezione("sezioni", sez)

    def _ricarica_materiali(self):
        if hasattr(self._main, "_materiali"):
            try:
                self._main._materiali.ricarica_da_progetto()
            except Exception as e:
                print(f"WARN  AI – ricarica materiali: {e}")

    def _ricarica_sezioni(self):
        if hasattr(self._main, "_sezioni"):
            try:
                self._main._sezioni.ricarica_da_progetto()
            except Exception as e:
                print(f"WARN  AI – ricarica sezioni: {e}")

    def _trova_sezione_modificabile(self, nome: str) -> tuple[bool, str, Any, Any]:
        """Ritorna (ok, msg, sez_data, sez_root) per una sezione non-standard."""
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None, None
        sez = self._sez()
        for cat in ("calcestruzzo_armato", "profili", "precompresso", "personalizzate"):
            if nome in sez.get(cat, {}):
                s = sez[cat][nome]
                if s.get("standard", False):
                    return False, (
                        f"'{nome}' è una sezione standard e non può essere modificata. "
                        "Crea prima una sezione personalizzata."
                    ), None, None
                return True, "", s, sez
        return False, f"Sezione '{nome}' non trovata nel progetto.", None, None

    # ------------------------------------------------------------------
    #  MATERIALI – LETTURA
    # ------------------------------------------------------------------

    def elenca_materiali(self) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        mat = self._mat()
        righe = ["Materiali nel progetto:"]
        trovati = False
        for cat in ("calcestruzzo", "barre", "acciaio", "personalizzati"):
            nomi = list(mat.get(cat, {}).keys())
            if nomi:
                trovati = True
                righe.append(f"  • {cat}: " + ", ".join(nomi))
        if not trovati:
            return True, "Nessun materiale trovato nel progetto.", {}
        return True, "\n".join(righe), mat

    def get_info_materiale(self, nome: str) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        mat = self._mat()
        for cat in ("calcestruzzo", "barre", "acciaio", "personalizzati"):
            if nome in mat.get(cat, {}):
                p = mat[cat][nome]
                lines = [f"Materiale: {nome}  [{cat}]"]
                for k in ("tipo", "gamma", "alpha", "densita", "poisson",
                          "m_elastico", "m_taglio"):
                    if k in p:
                        lines.append(f"  {k}: {p[k]}")
                lines.append(f"  SLU: {len(p.get('slu', []))} segmenti")
                lines.append(f"  SLE: {len(p.get('sle', []))} segmenti")
                lines.append(f"  standard: {p.get('standard', True)}")
                return True, "\n".join(lines), p
        return False, f"Materiale '{nome}' non trovato.", None

    # ------------------------------------------------------------------
    #  MATERIALI – SCRITTURA
    # ------------------------------------------------------------------

    def crea_materiale_personalizzato(
        self, nome: str,
        gamma: float = 1.0, alpha: float = 1.0,
        densita: float = 0.0, poisson: float = 0.0,
        m_elastico: float = 0.0, m_taglio: float = 0.0,
        slu: list = None, sle: list = None,
    ) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        mat = self._mat()
        # Controlla unicità del nome
        for cat in ("calcestruzzo", "barre", "acciaio", "personalizzati"):
            if nome in mat.get(cat, {}):
                return False, f"Nome '{nome}' già presente nella categoria '{cat}'.", None
        if "personalizzati" not in mat:
            mat["personalizzati"] = {}
        mat["personalizzati"][nome] = {
            "tipo":       "personalizzato",
            "standard":   False,
            "gamma":      float(gamma),
            "alpha":      float(alpha),
            "densita":    float(densita),
            "poisson":    float(poisson),
            "m_elastico": float(m_elastico),
            "m_taglio":   float(m_taglio),
            "slu":        slu if slu is not None else [],
            "sle":        sle if sle is not None else [],
        }
        self._salva_mat(mat)
        self._ricarica_materiali()
        return True, f"Materiale personalizzato '{nome}' creato.", mat["personalizzati"][nome]

    def modifica_materiale(
        self, nome: str, categoria: str, proprieta: dict
    ) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        _alias = {
            "cls": "calcestruzzo", "calcestruzzo": "calcestruzzo",
            "barre": "barre", "barra": "barre",
            "acciaio": "acciaio",
            "personalizzato": "personalizzati", "personalizzati": "personalizzati",
            "custom": "personalizzati",
        }
        cat = _alias.get(categoria.lower(), categoria.lower())
        mat = self._mat()
        if cat not in mat or nome not in mat[cat]:
            return False, f"Materiale '{nome}' non trovato in categoria '{categoria}'.", None
        m = mat[cat][nome]
        if m.get("standard", True) and cat != "personalizzati":
            return False, (
                f"'{nome}' è un materiale standard. "
                "Puoi modificare solo materiali nella categoria 'personalizzati'."
            ), None
        _campi = {"gamma", "alpha", "densita", "poisson", "m_elastico", "m_taglio", "slu", "sle", "tipo"}
        aggiornati = []
        for k, v in proprieta.items():
            if k in _campi:
                m[k] = v
                aggiornati.append(k)
        if not aggiornati:
            return False, "Nessun campo valido da aggiornare.", None
        self._salva_mat(mat)
        self._ricarica_materiali()
        return True, f"Materiale '{nome}' aggiornato: {', '.join(aggiornati)}.", m

    def elimina_materiale_personalizzato(self, nome: str) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        mat = self._mat()
        pers = mat.get("personalizzati", {})
        if nome not in pers:
            return False, f"Materiale personalizzato '{nome}' non trovato.", None
        del pers[nome]
        self._salva_mat(mat)
        self._ricarica_materiali()
        return True, f"Materiale personalizzato '{nome}' eliminato.", None

    # ------------------------------------------------------------------
    #  SEZIONI – LETTURA
    # ------------------------------------------------------------------

    def elenca_sezioni(self) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        sez = self._sez()
        righe = ["Sezioni nel progetto:"]
        trovate = False
        for cat in ("calcestruzzo_armato", "profili", "precompresso", "personalizzate"):
            nomi = list(sez.get(cat, {}).keys())
            if nomi:
                trovate = True
                righe.append(f"  • {cat}: " + ", ".join(nomi))
        if not trovate:
            return True, "Nessuna sezione trovata nel progetto.", {}
        return True, "\n".join(righe), sez

    def get_info_sezione(self, nome: str) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        sez = self._sez()
        for cat in ("calcestruzzo_armato", "profili", "precompresso", "personalizzate"):
            if nome in sez.get(cat, {}):
                s = sez[cat][nome]
                el = s.get("elementi", {})
                carp = el.get("carpenteria", [])
                barre = el.get("barre", [])
                staffe = el.get("staffe", [])
                lines = [
                    f"Sezione: {nome}  [{cat}]",
                    f"  standard: {s.get('standard', False)}",
                    f"  materiale_default: {s.get('materiale_default', '')}",
                    f"  carpenteria: {len(carp)} element{'o' if len(carp)==1 else 'i'}",
                    f"  barre: {len(barre)} element{'a' if len(barre)==1 else 'e'}",
                    f"  staffe: {len(staffe)} staff{'a' if len(staffe)==1 else 'e'}",
                ]
                if carp:
                    lines.append("  Carpenteria:")
                    for e in carp:
                        geo = e.get("geometria", {})
                        lines.append(f"    [{e.get('id','')}] {e.get('tipo','')} – mat: {e.get('materiale','')} – geo: {geo}")
                if barre:
                    lines.append("  Barre:")
                    for e in barre:
                        geo = e.get("geometria", {})
                        lines.append(f"    [{e.get('id','')}] r={geo.get('r','')}mm  cx={geo.get('cx','')}  cy={geo.get('cy','')} – mat: {e.get('materiale','')}")
                return True, "\n".join(lines), s
        return False, f"Sezione '{nome}' non trovata.", None

    # ------------------------------------------------------------------
    #  SEZIONI – SCRITTURA
    # ------------------------------------------------------------------

    def crea_sezione_personalizzata(
        self, nome: str, categoria: str = "personalizzate"
    ) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        _alias = {
            "ca": "calcestruzzo_armato", "rc": "calcestruzzo_armato",
            "cls": "calcestruzzo_armato", "calcestruzzo_armato": "calcestruzzo_armato",
            "calcestruzzo armato": "calcestruzzo_armato",
            "profili": "profili", "profilo": "profili",
            "acciaio": "profili", "steel": "profili",
            "precompresso": "precompresso", "prestressed": "precompresso",
            "personalizzate": "personalizzate", "personalizzata": "personalizzate",
            "custom": "personalizzate",
        }
        cat = _alias.get(categoria.lower(), "personalizzate")
        sez = self._sez()
        # Unicità nome tra tutte le categorie
        for c in ("calcestruzzo_armato", "profili", "precompresso", "personalizzate"):
            if nome in sez.get(c, {}):
                return False, f"Nome '{nome}' già usato nella categoria '{c}'.", None
        if cat not in sez:
            sez[cat] = {}
        sez[cat][nome] = {
            "tipo_categoria":    cat,
            "standard":          False,
            "materiale_default": "",
            "elementi":          {"carpenteria": [], "barre": [], "staffe": []},
        }
        self._salva_sez(sez)
        self._ricarica_sezioni()
        return True, f"Sezione personalizzata '{nome}' creata in categoria '{cat}'.", sez[cat][nome]

    def aggiungi_rettangolo(
        self, nome_sezione: str,
        x0: float, y0: float, x1: float, y1: float,
        materiale: str = "",
    ) -> tuple[bool, str, Any]:
        ok, msg, s, sez = self._trova_sezione_modificabile(nome_sezione)
        if not ok:
            return False, msg, None
        elem = {
            "id":       str(uuid.uuid4())[:8],
            "tipo":     "rettangolo",
            "geometria": {"x0": float(x0), "y0": float(y0),
                          "x1": float(x1), "y1": float(y1)},
            "materiale": materiale,
        }
        s["elementi"]["carpenteria"].append(elem)
        self._salva_sez(sez)
        self._ricarica_sezioni()
        mat_str = f" – mat: {materiale}" if materiale else ""
        return True, (
            f"Rettangolo ({x0},{y0})→({x1},{y1}){mat_str} aggiunto a '{nome_sezione}'."
        ), elem

    def aggiungi_cerchio_carpenteria(
        self, nome_sezione: str,
        cx: float, cy: float,
        r: float = None, rx: float = None, ry: float = None,
        materiale: str = "",
    ) -> tuple[bool, str, Any]:
        ok, msg, s, sez = self._trova_sezione_modificabile(nome_sezione)
        if not ok:
            return False, msg, None
        # Risolvi rx/ry: r è shorthand per cerchio, rx/ry per ellisse
        if rx is None and ry is None:
            if r is None:
                return False, "Specificare almeno 'r' (cerchio) oppure 'rx' e 'ry' (ellisse).", None
            rx = ry = float(r)
        else:
            rx = float(rx) if rx is not None else float(r or 0)
            ry = float(ry) if ry is not None else float(r or 0)
        if rx <= 0 or ry <= 0:
            return False, "I raggi devono essere > 0.", None
        elem = {
            "id":        str(uuid.uuid4())[:8],
            "tipo":      "cerchio",
            "geometria": {"cx": float(cx), "cy": float(cy), "rx": rx, "ry": ry},
            "materiale": materiale,
        }
        s["elementi"]["carpenteria"].append(elem)
        self._salva_sez(sez)
        self._ricarica_sezioni()
        mat_str = f" – mat: {materiale}" if materiale else ""
        forma = f"Cerchio ⌀{2*rx}mm" if rx == ry else f"Ellisse rx={rx}mm ry={ry}mm"
        return True, f"{forma} in ({cx},{cy}){mat_str} aggiunto a '{nome_sezione}'.", elem

    def aggiungi_poligono(
        self, nome_sezione: str,
        punti: list, materiale: str = "",
    ) -> tuple[bool, str, Any]:
        ok, msg, s, sez = self._trova_sezione_modificabile(nome_sezione)
        if not ok:
            return False, msg, None
        if len(punti) < 3:
            return False, "Il poligono richiede almeno 3 vertici.", None
        elem = {
            "id":        str(uuid.uuid4())[:8],
            "tipo":      "poligono",
            "geometria": {"punti": [list(p) for p in punti]},
            "materiale": materiale,
        }
        s["elementi"]["carpenteria"].append(elem)
        self._salva_sez(sez)
        self._ricarica_sezioni()
        mat_str = f" – mat: {materiale}" if materiale else ""
        return True, f"Poligono ({len(punti)} vertici){mat_str} aggiunto a '{nome_sezione}'.", elem

    def aggiungi_barra(
        self, nome_sezione: str,
        cx: float, cy: float, r: float,
        materiale: str = "",
    ) -> tuple[bool, str, Any]:
        ok, msg, s, sez = self._trova_sezione_modificabile(nome_sezione)
        if not ok:
            return False, msg, None
        elem = {
            "id":        str(uuid.uuid4())[:8],
            "tipo":      "barra",
            "geometria": {"cx": float(cx), "cy": float(cy), "r": float(r)},
            "materiale": materiale,
        }
        s["elementi"]["barre"].append(elem)
        self._salva_sez(sez)
        self._ricarica_sezioni()
        mat_str = f" – mat: {materiale}" if materiale else ""
        return True, f"Barra ⌀{2*r}mm in ({cx},{cy}){mat_str} aggiunta a '{nome_sezione}'.", elem

    def aggiungi_staffa(
        self, nome_sezione: str,
        punti: list, r: float = 5.0,
        materiale: str = "",
    ) -> tuple[bool, str, Any]:
        ok, msg, s, sez = self._trova_sezione_modificabile(nome_sezione)
        if not ok:
            return False, msg, None
        elem = {
            "id":        str(uuid.uuid4())[:8],
            "tipo":      "staffa",
            "geometria": {"punti": punti, "r": float(r)},
            "materiale": materiale,
        }
        s["elementi"]["staffe"].append(elem)
        self._salva_sez(sez)
        self._ricarica_sezioni()
        mat_str = f" – mat: {materiale}" if materiale else ""
        return True, (
            f"Staffa ({len(punti)} vertici, r={r}mm){mat_str} aggiunta a '{nome_sezione}'."
        ), elem

    def aggiungi_foro_rettangolo(
        self, nome_sezione: str,
        x0: float, y0: float, x1: float, y1: float,
    ) -> tuple[bool, str, Any]:
        ok, msg, s, sez = self._trova_sezione_modificabile(nome_sezione)
        if not ok:
            return False, msg, None
        elem = {
            "id":        str(uuid.uuid4())[:8],
            "tipo":      "foro_rettangolo",
            "geometria": {"x0": float(min(x0, x1)), "y0": float(min(y0, y1)),
                          "x1": float(max(x0, x1)), "y1": float(max(y0, y1))},
            "materiale": "",
        }
        s["elementi"]["carpenteria"].append(elem)
        self._salva_sez(sez)
        self._ricarica_sezioni()
        return True, f"Foro rettangolare ({x0},{y0})→({x1},{y1}) aggiunto a '{nome_sezione}'.", elem

    def aggiungi_foro_cerchio(
        self, nome_sezione: str,
        cx: float, cy: float,
        r: float = None, rx: float = None, ry: float = None,
    ) -> tuple[bool, str, Any]:
        ok, msg, s, sez = self._trova_sezione_modificabile(nome_sezione)
        if not ok:
            return False, msg, None
        if rx is None and ry is None:
            if r is None:
                return False, "Specificare almeno 'r' (foro circolare) oppure 'rx' e 'ry' (foro ellittico).", None
            rx = ry = float(r)
        else:
            rx = float(rx) if rx is not None else float(r or 0)
            ry = float(ry) if ry is not None else float(r or 0)
        if rx <= 0 or ry <= 0:
            return False, "I raggi devono essere > 0.", None
        elem = {
            "id":        str(uuid.uuid4())[:8],
            "tipo":      "foro_cerchio",
            "geometria": {"cx": float(cx), "cy": float(cy), "rx": rx, "ry": ry},
            "materiale": "",
        }
        s["elementi"]["carpenteria"].append(elem)
        self._salva_sez(sez)
        self._ricarica_sezioni()
        forma = f"Foro circolare ⌀{2*rx}mm" if rx == ry else f"Foro ellittico rx={rx}mm ry={ry}mm"
        return True, f"{forma} in ({cx},{cy}) aggiunto a '{nome_sezione}'.", elem

    def aggiungi_foro_poligono(
        self, nome_sezione: str,
        punti: list,
    ) -> tuple[bool, str, Any]:
        ok, msg, s, sez = self._trova_sezione_modificabile(nome_sezione)
        if not ok:
            return False, msg, None
        if len(punti) < 3:
            return False, "Il foro poligonale richiede almeno 3 vertici.", None
        elem = {
            "id":        str(uuid.uuid4())[:8],
            "tipo":      "foro_poligono",
            "geometria": {"punti": [list(p) for p in punti]},
            "materiale": "",
        }
        s["elementi"]["carpenteria"].append(elem)
        self._salva_sez(sez)
        self._ricarica_sezioni()
        return True, f"Foro poligonale ({len(punti)} vertici) aggiunto a '{nome_sezione}'.", elem

    def elimina_sezione_personalizzata(self, nome: str) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        sez = self._sez()
        for cat in ("calcestruzzo_armato", "profili", "precompresso", "personalizzate"):
            if nome in sez.get(cat, {}):
                if sez[cat][nome].get("standard", False):
                    return False, f"'{nome}' è una sezione standard e non può essere eliminata.", None
                del sez[cat][nome]
                self._salva_sez(sez)
                self._ricarica_sezioni()
                return True, f"Sezione '{nome}' eliminata.", None
        return False, f"Sezione '{nome}' non trovata nel progetto.", None

    # ------------------------------------------------------------------
    #  ELEMENTI 3D – HELPERS
    # ------------------------------------------------------------------

    _TIPI_ELEM  = ("trave", "pilastro", "fondazione", "solaio")
    _TIPI_OBJ   = ("parallelepipedo", "cilindro", "sfera", "barra", "staffa")
    _BASE_NOME_OBJ = {
        "parallelepipedo": "Parallelepipedo",
        "cilindro":        "Cilindro",
        "sfera":           "Sfera",
        "barra":           "Barra",
        "staffa":          "Staffa",
    }
    _GEO_DEFAULT = {
        "parallelepipedo": {"lunghezza": 5.0,   "base": 0.30,  "altezza": 0.50},
        "cilindro":        {"altezza":   3.0,   "raggio": 0.15},
        "sfera":           {"raggio":    0.15},
        "barra":           {"diametro":  0.016, "punti": [[0.0,0.0,0.0],[5.0,0.0,0.0]]},
        "staffa":          {"diametro":  0.008,
                            "punti": [[0.0,0.0,0.0],[0.28,0.0,0.0],
                                      [0.28,0.48,0.0],[0.0,0.48,0.0],[0.0,0.0,0.0]]},
    }

    def _el_data(self) -> dict:
        return self._main.get_sezione("elementi") or {}

    def _salva_el(self, el_data: dict):
        self._main.set_sezione("elementi", el_data)

    def _ricarica_elementi(self):
        if hasattr(self._main, "_elementi"):
            try:
                self._main._elementi.ricarica_da_progetto()
            except Exception as e:
                print(f"WARN  AI – ricarica elementi: {e}")

    def _get_tutti_elementi(self) -> list[dict]:
        """Flat list of all element dicts (standard + custom) from the running controller."""
        if hasattr(self._main, "_elementi"):
            try:
                ctrl = self._main._elementi._lista_ctrl
                elementi = ctrl.get_elementi()
                return [el.to_dict()
                        for lista in elementi.values()
                        for el in lista]
            except Exception:
                pass
        # fallback: project JSON only
        el_data = self._el_data()
        return [el for lista in el_data.values() for el in lista]

    def _next_el_id(self) -> int:
        tutti = self._get_tutti_elementi()
        ids = [el.get("id", 0) for el in tutti if isinstance(el, dict)]
        return max(ids, default=0) + 1

    def _next_obj_id(self) -> int:
        tutti = self._get_tutti_elementi()
        ids = [obj.get("id", 0)
               for el in tutti if isinstance(el, dict)
               for obj in el.get("oggetti", [])]
        return max(ids, default=0) + 1

    def _auto_nome_obj(self, tipo_oggetto: str, oggetti_esistenti: list) -> str:
        base = self._BASE_NOME_OBJ.get(tipo_oggetto, tipo_oggetto.capitalize())
        existing = {o.get("nome", "") for o in oggetti_esistenti}
        n = 1
        while f"{base}.{n:03d}" in existing:
            n += 1
        return f"{base}.{n:03d}"

    def _trova_elem_modificabile(self, nome: str) -> tuple[bool, str, Any, Any, str]:
        """Returns (ok, msg, el_dict, all_el_data, tipo)."""
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None, None, ""
        el_data = self._el_data()
        for tipo in self._TIPI_ELEM:
            for el in el_data.get(tipo, []):
                if el.get("nome") == nome:
                    if el.get("standard", False):
                        return False, (
                            f"'{nome}' è un elemento standard e non può essere modificato. "
                            "Usa 'crea_elemento' per creare una copia personalizzata."
                        ), None, None, ""
                    return True, "", el, el_data, tipo
        return False, f"Elemento '{nome}' non trovato tra gli elementi personalizzati.", None, None, ""

    # ------------------------------------------------------------------
    #  ELEMENTI 3D – LETTURA
    # ------------------------------------------------------------------

    def elenca_elementi(self) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        tutti = self._get_tutti_elementi()
        if not tutti:
            return True, "Nessun elemento trovato nel progetto.", {}
        per_tipo: dict[str, list] = {t: [] for t in self._TIPI_ELEM}
        for el in tutti:
            t = el.get("tipo", "")
            if t in per_tipo:
                flag = " [standard]" if el.get("standard", False) else " [custom]"
                per_tipo[t].append(el.get("nome", "?") + flag)
        righe = ["Elementi 3D nel progetto:"]
        for tipo, nomi in per_tipo.items():
            if nomi:
                righe.append(f"  • {tipo}: " + ", ".join(nomi))
        return True, "\n".join(righe), per_tipo

    def get_info_elemento(self, nome: str) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        for el in self._get_tutti_elementi():
            if el.get("nome") == nome:
                oggetti = el.get("oggetti", [])
                lines = [
                    f"Elemento: {nome}  [{el.get('tipo','')}]",
                    f"  standard: {el.get('standard', False)}",
                    f"  oggetti: {len(oggetti)}",
                ]
                for o in oggetti:
                    geo = o.get("geometria", {})
                    mat = o.get("materiale", "")
                    pos = o.get("posizione", [0,0,0])
                    lines.append(
                        f"    [{o.get('nome','')}] tipo={o.get('tipo','')} "
                        f"geo={geo}  mat={mat!r}  pos={pos}"
                    )
                return True, "\n".join(lines), el
        return False, f"Elemento '{nome}' non trovato.", None

    # ------------------------------------------------------------------
    #  ELEMENTI 3D – SCRITTURA
    # ------------------------------------------------------------------

    def crea_elemento(self, nome: str, tipo: str) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        tipo = tipo.lower().strip()
        if tipo not in self._TIPI_ELEM:
            return False, f"Tipo '{tipo}' non valido. Valori: {', '.join(self._TIPI_ELEM)}.", None
        # Unicità nome
        for el in self._get_tutti_elementi():
            if el.get("nome") == nome:
                return False, f"Nome '{nome}' già presente negli elementi.", None
        el_data = self._el_data()
        nuovo_id = self._next_el_id()
        nuovo_el = {
            "id":       nuovo_id,
            "nome":     nome,
            "tipo":     tipo,
            "standard": False,
            "oggetti":  [],
        }
        if tipo not in el_data:
            el_data[tipo] = []
        el_data[tipo].append(nuovo_el)
        self._salva_el(el_data)
        self._ricarica_elementi()
        return True, f"Elemento '{nome}' ({tipo}) creato.", nuovo_el

    def aggiungi_oggetto_elemento(
        self, nome_elemento: str, tipo_oggetto: str,
        nome_oggetto: str = None,
        geometria: dict = None,
        materiale: str = "",
        posizione: list = None,
        rotazione: list = None,
    ) -> tuple[bool, str, Any]:
        ok, msg, el, el_data, _ = self._trova_elem_modificabile(nome_elemento)
        if not ok:
            return False, msg, None
        tipo_oggetto = tipo_oggetto.lower().strip()
        if tipo_oggetto not in self._TIPI_OBJ:
            return False, f"Tipo oggetto '{tipo_oggetto}' non valido. Valori: {', '.join(self._TIPI_OBJ)}.", None
        oggetti = el.get("oggetti", [])
        nome_obj = nome_oggetto or self._auto_nome_obj(tipo_oggetto, oggetti)
        geo = dict(self._GEO_DEFAULT.get(tipo_oggetto, {}))
        if geometria:
            geo.update({k: v for k, v in geometria.items()})
        nuovo_obj = {
            "id":              self._next_obj_id(),
            "nome":            nome_obj,
            "tipo":            tipo_oggetto,
            "geometria":       geo,
            "materiale":       materiale or "",
            "posizione":       list(posizione)  if posizione  else [0.0, 0.0, 0.0],
            "rotazione":       list(rotazione)  if rotazione  else [0.0, 0.0, 0.0],
            "custom_geometry": False,
            "visibile":        True,
            "selezionabile":   True,
            "vertice_ref":     0,
        }
        el["oggetti"].append(nuovo_obj)
        self._salva_el(el_data)
        self._ricarica_elementi()
        mat_str = f" – mat: {materiale}" if materiale else ""
        return True, (
            f"Oggetto '{nome_obj}' ({tipo_oggetto}){mat_str} aggiunto a '{nome_elemento}'."
        ), nuovo_obj

    def modifica_oggetto_elemento(
        self, nome_elemento: str, nome_oggetto: str,
        geometria: dict = None,
        materiale: str = None,
        posizione: list = None,
        rotazione: list = None,
    ) -> tuple[bool, str, Any]:
        ok, msg, el, el_data, _ = self._trova_elem_modificabile(nome_elemento)
        if not ok:
            return False, msg, None
        for obj in el.get("oggetti", []):
            if obj.get("nome") == nome_oggetto:
                aggiornati = []
                if geometria is not None:
                    obj["geometria"].update(geometria)
                    aggiornati.append("geometria")
                if materiale is not None:
                    obj["materiale"] = materiale
                    aggiornati.append("materiale")
                if posizione is not None:
                    obj["posizione"] = list(posizione)
                    aggiornati.append("posizione")
                if rotazione is not None:
                    obj["rotazione"] = list(rotazione)
                    aggiornati.append("rotazione")
                if not aggiornati:
                    return False, "Nessun campo da aggiornare specificato.", None
                self._salva_el(el_data)
                self._ricarica_elementi()
                return True, (
                    f"Oggetto '{nome_oggetto}' aggiornato: {', '.join(aggiornati)}."
                ), obj
        return False, f"Oggetto '{nome_oggetto}' non trovato in '{nome_elemento}'.", None

    def elimina_oggetto_elemento(
        self, nome_elemento: str, nome_oggetto: str
    ) -> tuple[bool, str, Any]:
        ok, msg, el, el_data, _ = self._trova_elem_modificabile(nome_elemento)
        if not ok:
            return False, msg, None
        prima = len(el.get("oggetti", []))
        el["oggetti"] = [o for o in el.get("oggetti", []) if o.get("nome") != nome_oggetto]
        if len(el["oggetti"]) == prima:
            return False, f"Oggetto '{nome_oggetto}' non trovato in '{nome_elemento}'.", None
        self._salva_el(el_data)
        self._ricarica_elementi()
        return True, f"Oggetto '{nome_oggetto}' rimosso da '{nome_elemento}'.", None

    def elimina_elemento(self, nome: str) -> tuple[bool, str, Any]:
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        el_data = self._el_data()
        for tipo in self._TIPI_ELEM:
            lista = el_data.get(tipo, [])
            for el in lista:
                if el.get("nome") == nome:
                    if el.get("standard", False):
                        return False, f"'{nome}' è un elemento standard e non può essere eliminato.", None
                    el_data[tipo] = [e for e in lista if e.get("nome") != nome]
                    self._salva_el(el_data)
                    self._ricarica_elementi()
                    return True, f"Elemento '{nome}' eliminato.", None
        return False, f"Elemento '{nome}' non trovato.", None

    # ------------------------------------------------------------------
    #  CARICHI / VINCOLI – HELPERS
    # ------------------------------------------------------------------

    def _cv_data(self) -> dict:
        return self._main.get_sezione("carichi") or {}

    def _salva_cv(self, cv_data: dict):
        import copy
        self._main.push_undo("Modifica carichi/vincoli", modulo="extra_elemento")
        self._main.set_sezione("carichi", copy.deepcopy(cv_data))

    def _ricarica_cv(self):
        """Reload the extra controller if it's showing the current element."""
        if hasattr(self._main, "_elementi"):
            try:
                extra = self._main._elementi._extra_ctrl
                if extra._elem_rif is not None:
                    extra._cv_list = extra._get_cv_per_elemento(extra._elem_rif.id)
                    extra._spazio.aggiorna_oggetti(extra._cv_list)
                    extra._outliner.ricarica(extra._cv_list)
            except Exception as e:
                print(f"WARN  AI – ricarica CV: {e}")

    def _trova_elemento_per_nome(self, nome: str):
        """Find an Elemento object by name from the running controller. Returns (ok, msg, Elemento)."""
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None
        if hasattr(self._main, "_elementi"):
            try:
                ctrl = self._main._elementi._lista_ctrl
                for lista in ctrl.get_elementi().values():
                    for el in lista:
                        if el.nome == nome:
                            return True, "", el
            except Exception:
                pass
        return False, f"Elemento '{nome}' non trovato.", None

    def _next_cv_id(self) -> int:
        cv_data = self._cv_data()
        max_id = 0
        for lista in cv_data.values():
            if isinstance(lista, list):
                for d in lista:
                    if isinstance(d, dict):
                        max_id = max(max_id, d.get("id", 0))
        return max_id + 1

    # ------------------------------------------------------------------
    #  CARICHI / VINCOLI – LETTURA
    # ------------------------------------------------------------------

    def elenca_carichi_vincoli(self, nome_elemento: str) -> tuple[bool, str, Any]:
        ok, msg, el = self._trova_elemento_per_nome(nome_elemento)
        if not ok:
            return False, msg, None
        cv_data = self._cv_data()
        chiave = str(el.id)
        lista = cv_data.get(chiave, [])
        if not lista:
            return True, f"Nessun carico/vincolo per '{nome_elemento}'.", []
        righe = [f"Carichi/vincoli per '{nome_elemento}':"]
        for cv in lista:
            st = cv.get("sottotipo", "?")
            nome = cv.get("nome", "?")
            car = cv.get("caratteristiche", {})
            pos = cv.get("posizione", [0, 0, 0])
            geo = cv.get("geometria", {})
            righe.append(
                f"  • [{nome}] {st} – car={car} geo={geo} pos={pos}"
            )
        return True, "\n".join(righe), lista

    # ------------------------------------------------------------------
    #  CARICHI / VINCOLI – SCRITTURA
    # ------------------------------------------------------------------

    def aggiungi_carico_vincolo(
        self, nome_elemento: str, sottotipo: str,
        geometria: dict = None,
        caratteristiche: dict = None,
        posizione: list = None,
        rotazione: list = None,
    ) -> tuple[bool, str, Any]:
        ok, msg, el = self._trova_elemento_per_nome(nome_elemento)
        if not ok:
            return False, msg, None
        sottotipo = sottotipo.lower().strip()
        if sottotipo not in ("vincolo", "carico"):
            return False, "sottotipo deve essere 'vincolo' o 'carico'.", None

        # Build CV dict
        nuovo_id = self._next_cv_id()
        nome_base = "Vincolo" if sottotipo == "vincolo" else "Carico"
        # Count existing
        cv_data = self._cv_data()
        chiave = str(el.id)
        lista = cv_data.get(chiave, [])
        count = sum(1 for cv in lista if cv.get("sottotipo") == sottotipo) + 1
        nome_cv = f"{nome_base}.{count:03d}"
        # Ensure unique name
        nomi_esistenti = {cv.get("nome", "") for cv in lista}
        while nome_cv in nomi_esistenti:
            count += 1
            nome_cv = f"{nome_base}.{count:03d}"

        geo = {"lunghezza": 0.3, "base": 0.3, "altezza": 0.3}
        if geometria:
            geo.update({k: float(v) for k, v in geometria.items()
                        if k in ("lunghezza", "base", "altezza")})

        if sottotipo == "vincolo":
            car = {"sx": 0.0, "sy": 0.0, "sz": 0.0}
        else:
            car = {"fx": 0.0, "fy": 0.0, "fz": 0.0}
        if caratteristiche:
            car.update({k: float(v) for k, v in caratteristiche.items() if k in car})

        nuovo_cv = {
            "id":              nuovo_id,
            "nome":            nome_cv,
            "sottotipo":       sottotipo,
            "geometria":       geo,
            "posizione":       list(posizione) if posizione else [0.0, 0.0, 0.0],
            "rotazione":       list(rotazione) if rotazione else [0.0, 0.0, 0.0],
            "custom_geometry": False,
            "visibile":        True,
            "selezionabile":   True,
            "vertice_ref":     0,
            "caratteristiche": car,
        }
        lista.append(nuovo_cv)
        cv_data[chiave] = lista
        self._salva_cv(cv_data)
        self._ricarica_cv()
        car_str = ", ".join(f"{k}={v}" for k, v in car.items())
        return True, (
            f"{nome_base} '{nome_cv}' aggiunto a '{nome_elemento}' ({car_str})."
        ), nuovo_cv

    def modifica_carico_vincolo(
        self, nome_elemento: str, nome_cv: str,
        geometria: dict = None,
        caratteristiche: dict = None,
        posizione: list = None,
        rotazione: list = None,
    ) -> tuple[bool, str, Any]:
        ok, msg, el = self._trova_elemento_per_nome(nome_elemento)
        if not ok:
            return False, msg, None
        cv_data = self._cv_data()
        chiave = str(el.id)
        lista = cv_data.get(chiave, [])
        for cv in lista:
            if cv.get("nome") == nome_cv:
                aggiornati = []
                if geometria is not None:
                    cv["geometria"].update({k: float(v) for k, v in geometria.items()
                                            if k in ("lunghezza", "base", "altezza")})
                    aggiornati.append("geometria")
                if caratteristiche is not None:
                    cv["caratteristiche"].update(
                        {k: float(v) for k, v in caratteristiche.items()}
                    )
                    aggiornati.append("caratteristiche")
                if posizione is not None:
                    cv["posizione"] = list(posizione)
                    aggiornati.append("posizione")
                if rotazione is not None:
                    cv["rotazione"] = list(rotazione)
                    aggiornati.append("rotazione")
                if not aggiornati:
                    return False, "Nessun campo da aggiornare specificato.", None
                cv_data[chiave] = lista
                self._salva_cv(cv_data)
                self._ricarica_cv()
                return True, (
                    f"Carico/vincolo '{nome_cv}' aggiornato: {', '.join(aggiornati)}."
                ), cv
        return False, f"Carico/vincolo '{nome_cv}' non trovato in '{nome_elemento}'.", None

    def elimina_carico_vincolo(
        self, nome_elemento: str, nome_cv: str,
    ) -> tuple[bool, str, Any]:
        ok, msg, el = self._trova_elemento_per_nome(nome_elemento)
        if not ok:
            return False, msg, None
        cv_data = self._cv_data()
        chiave = str(el.id)
        lista = cv_data.get(chiave, [])
        prima = len(lista)
        lista = [cv for cv in lista if cv.get("nome") != nome_cv]
        if len(lista) == prima:
            return False, f"Carico/vincolo '{nome_cv}' non trovato in '{nome_elemento}'.", None
        cv_data[chiave] = lista
        self._salva_cv(cv_data)
        self._ricarica_cv()
        return True, f"Carico/vincolo '{nome_cv}' rimosso da '{nome_elemento}'.", None

    # ------------------------------------------------------------------
    #  MANIPOLAZIONE VERTICI
    # ------------------------------------------------------------------

    def _trova_obj_o_cv(self, nome_elemento: str, nome_oggetto: str, target: str = "oggetto"):
        """
        Find an Oggetto3D or CaricoVincolo by name.
        Returns (ok, msg, obj_or_cv, is_cv, el_data_or_cv_data, extra_key).
        """
        if not self._main.ha_progetto():
            return False, "Nessun progetto aperto.", None, False, None, None

        target = (target or "oggetto").lower().strip()

        if target == "carico_vincolo":
            ok, msg, el = self._trova_elemento_per_nome(nome_elemento)
            if not ok:
                return False, msg, None, False, None, None
            cv_data = self._cv_data()
            chiave = str(el.id)
            lista = cv_data.get(chiave, [])
            for cv in lista:
                if cv.get("nome") == nome_oggetto:
                    return True, "", cv, True, cv_data, chiave
            return False, f"Carico/vincolo '{nome_oggetto}' non trovato in '{nome_elemento}'.", None, False, None, None
        else:
            # Search among element objects
            ok, msg, el, el_data, _ = self._trova_elem_modificabile(nome_elemento)
            if not ok:
                return False, msg, None, False, None, None
            for obj in el.get("oggetti", []):
                if obj.get("nome") == nome_oggetto:
                    return True, "", obj, False, el_data, None
            return False, f"Oggetto '{nome_oggetto}' non trovato in '{nome_elemento}'.", None, False, None, None

    def get_vertici_oggetto(
        self, nome_elemento: str, nome_oggetto: str,
        target: str = "oggetto",
    ) -> tuple[bool, str, Any]:
        ok, msg, obj, is_cv, _, _ = self._trova_obj_o_cv(nome_elemento, nome_oggetto, target)
        if not ok:
            return False, msg, None

        # Compute vertices
        if is_cv:
            from elementi.modello_carichi_vincoli import CaricoVincolo
            cv_obj = CaricoVincolo.from_dict(obj)
            verts = cv_obj.get_vertices_local()
        else:
            from elementi.modello_3d import Oggetto3D
            o = Oggetto3D.from_dict(obj)
            verts = o.get_vertices_local()

        lines = [f"Vertici di '{nome_oggetto}' ({len(verts)} vertici):"]
        for i, v in enumerate(verts):
            lines.append(f"  v{i}: [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]")
        return True, "\n".join(lines), verts

    def modifica_vertici_oggetto(
        self, nome_elemento: str, nome_oggetto: str,
        target: str = "oggetto",
        vertici: list = None,
        modifiche: dict = None,
    ) -> tuple[bool, str, Any]:
        if vertici is None and modifiche is None:
            return False, "Specificare 'vertici' (lista completa) o 'modifiche' (dict indice→[x,y,z]).", None

        ok, msg, obj, is_cv, data, extra_key = self._trova_obj_o_cv(
            nome_elemento, nome_oggetto, target
        )
        if not ok:
            return False, msg, None

        # Get current vertices to use as base for partial updates
        if is_cv:
            from elementi.modello_carichi_vincoli import CaricoVincolo
            cv_obj = CaricoVincolo.from_dict(obj)
            current_verts = cv_obj.get_vertices_local()
        else:
            from elementi.modello_3d import Oggetto3D
            o = Oggetto3D.from_dict(obj)
            current_verts = o.get_vertices_local()

        if vertici is not None:
            # Full replacement
            new_verts = [[float(c) for c in v] for v in vertici]
        else:
            # Partial update via modifiche dict
            new_verts = [list(v) for v in current_verts]
            for idx_str, coords in modifiche.items():
                idx = int(idx_str)
                if 0 <= idx < len(new_verts):
                    new_verts[idx] = [float(c) for c in coords]
                else:
                    return False, f"Indice vertice {idx} fuori range (0-{len(new_verts)-1}).", None

        # Apply custom geometry
        obj["custom_geometry"] = True
        obj["geometria"]["vertici_custom"] = new_verts

        # Save
        if is_cv:
            self._salva_cv(data)
            self._ricarica_cv()
        else:
            self._salva_el(data)
            self._ricarica_elementi()

        n_mod = len(new_verts) if vertici is not None else len(modifiche)
        return True, (
            f"Vertici di '{nome_oggetto}' aggiornati ({n_mod} "
            f"{'vertici totali' if vertici is not None else 'vertici modificati'})."
        ), new_verts
