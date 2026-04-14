"""
gestione_elementi.py – Coordinatore del modulo Elementi.

Collega:
  • ControllerListaElementi    (index 5 – lista bottoni elemento)
  • ControllerElementi         (index 6 – spazio 3D e pannello proprietà)
  • ControllerExtraElemento    (index 7 – workspace carichi/vincoli)

Responsabilità proprie:
  • Persistenza (_salva / ricarica_da_progetto)
  • Cattura preview dopo ogni modifica 3D (senza repaint visibile = no glitch)
"""

import math
import numpy as np

from PyQt5.QtCore import QTimer

from .modello_3d                  import Elemento, Oggetto3D
from .controller_lista_elementi   import ControllerListaElementi
from .controller_elementi         import ControllerElementi
from .controller_extra_elemento   import ControllerExtraElemento


class GestioneElementi:

    def __init__(self, ui, main_window):
        self._ui   = ui
        self._main = main_window

        self._lista_ctrl = ControllerListaElementi(ui, main_window)
        self._elem_ctrl  = ControllerElementi(ui, main_window)
        self._extra_ctrl = ControllerExtraElemento(ui, main_window)

        self._coda_preview: list = []

        # ── Connessioni tra i controller ──────────────────────────────
        self._lista_ctrl.elemento_selezionato.connect(self._on_elemento_selezionato)
        self._lista_ctrl.richiedi_salvataggio.connect(self._salva)
        self._lista_ctrl.apri_extra_elemento.connect(self._on_apri_extra_elemento)
        self._elem_ctrl.richiedi_salvataggio.connect(self._salva)
        self._elem_ctrl.richiedi_preview.connect(self._schedula_preview)
        self._extra_ctrl.richiedi_salvataggio.connect(self._salva)
        self._extra_ctrl.richiedi_preview_cv.connect(self._schedula_cv_preview)

        # Genera preview elementi standard al primo avvio (widget GL non ancora visibile)
        QTimer.singleShot(800, self._schedula_previews_tutti)

    # ================================================================
    #  COLLEGAMENTO TRA I CONTROLLER
    # ================================================================

    def _on_apri_extra_elemento(self, el: Elemento):
        """
        L'utente clicca il pulsante C/V accanto all'elemento nella lista (index 5).
        Carica l'elemento di riferimento e i suoi carichi/vincoli in index 7.
        """
        if el is not None:
            self._extra_ctrl.carica_elemento(el)
        print(f">> Workspace C/V aperto per: {el.nome if el else '—'}")

    def _on_elemento_selezionato(self, el: Elemento | None):
        """
        L'utente clicca un bottone nella lista (index 5).
        Le preview (elemento + CV) vengono catturate PRIMA di passare a index 6,
        così il widget GL è ancora nascosto → nessun glitch visivo.
        """
        if el is not None:
            spazio = self._elem_ctrl.get_spazio()
            if el.oggetti:
                self._render_preview_per(spazio, el.id, el.oggetti)
            else:
                btn = self._lista_ctrl.get_bottoni().get(el.id)
                if btn:
                    btn.set_preview(None)
                    btn.update()
            # Cattura anche la preview C/V per questo elemento
            self._render_cv_preview_for_element(el)

        self._elem_ctrl.carica_elemento(el)

    # ================================================================
    #  PREVIEW CARICHI/VINCOLI
    # ================================================================

    def _schedula_cv_preview(self):
        """Schedula la cattura preview C/V con lo stesso delay degli elementi base."""
        QTimer.singleShot(150, self._cattura_cv_preview)

    def _cattura_cv_preview(self):
        """
        Cattura la preview C/V per l'elemento attualmente caricato.
        Usa _render_cv_preview_per con render isometrico (come gli elementi base).
        """
        el_rif = self._extra_ctrl._elem_rif
        if el_rif is None:
            return
        self._render_cv_preview_for_element(el_rif)

    def _render_cv_preview_for_element(self, el):
        """
        Render isometrico della preview C/V per un elemento.
        Usa ExtraSpazio3D in preview_mode con camera fissa a 30°/-45°,
        indipendentemente dalla visibilità del widget (stessa tecnica degli elementi base).
        """
        from .modello_carichi_vincoli import CaricoVincolo

        pair = self._lista_ctrl._pairs.get(el.id)
        if pair is None:
            return

        # Recupera i CV dal progetto
        cv_data = self._main.get_sezione("carichi") or {}
        cv_list_raw = cv_data.get(str(el.id), [])
        if not cv_list_raw and not el.oggetti:
            # Nessun oggetto e nessun CV: nessuna preview
            pair.set_cv_preview(None)
            pair.btn_cv.update()
            return

        cv_objects = [CaricoVincolo.from_dict(d) for d in cv_list_raw]

        spazio = self._extra_ctrl.get_spazio()

        # Salva stato
        old_rot_x    = spazio.rot_x
        old_rot_y    = spazio.rot_y
        old_pan_x    = spazio.pan_x
        old_pan_y    = spazio.pan_y
        old_dist     = spazio.cam_dist
        old_ortho    = spazio._ortho
        old_oggetti  = spazio._oggetti
        old_rif      = spazio._oggetti_rif

        px = None

        try:
            spazio._preview_mode = True
            spazio._ortho        = False
            spazio.rot_x         = 30.0
            spazio.rot_y         = -45.0
            spazio._oggetti_rif  = el.oggetti
            spazio._oggetti      = cv_objects

            # Centra su tutti gli oggetti (rif + CV)
            tutti = list(el.oggetti) + list(cv_objects)
            self._centra_per_preview(spazio, tutti)

            if spazio.isValid():
                spazio.makeCurrent()
                spazio.paintGL()
                img = spazio.grabFramebuffer()
                if not img.isNull():
                    from PyQt5.QtGui import QPixmap
                    px = QPixmap.fromImage(img)

        except Exception as e:
            print(f"Errore render preview CV: {e}")

        finally:
            spazio._preview_mode = False
            spazio.rot_x         = old_rot_x
            spazio.rot_y         = old_rot_y
            spazio.pan_x         = old_pan_x
            spazio.pan_y         = old_pan_y
            spazio.cam_dist      = old_dist
            spazio._ortho        = old_ortho
            spazio._oggetti      = old_oggetti
            spazio._oggetti_rif  = old_rif

            if spazio.isValid():
                spazio.update()

        if px is not None and not px.isNull():
            self._lista_ctrl.aggiorna_cv_preview(el.id, px)

    # ================================================================
    #  PREVIEW  (nessun repaint sincrono visibile → no glitch)
    # ================================================================

    def _schedula_preview(self):
        """Cattura il framebuffer corrente 150 ms dopo la modifica."""
        QTimer.singleShot(150, self._cattura_preview)

    def _cattura_preview(self):
        """
        Cattura la preview con render isometrico (30°/-45°) senza griglia/assi.
        Prospettiva fissa indipendente dalla visuale corrente dell'utente.
        """
        el = self._elem_ctrl.get_elem_corrente()
        if el is None:
            return
        btn = self._lista_ctrl.get_bottoni().get(el.id)
        if btn is None:
            return

        if not el.oggetti:
            btn.set_preview(None)
            btn.update()
            return

        spazio = self._elem_ctrl.get_spazio()
        self._render_preview_per(spazio, el.id, el.oggetti)

    # ----------------------------------------------------------------
    #  Centering helper per preview isometrica
    # ----------------------------------------------------------------

    @staticmethod
    def _centra_per_preview(spazio, oggetti):
        """
        Imposta pan e cam_dist in modo che tutti gli oggetti siano
        perfettamente centrati e inquadrati nella vista isometrica corrente.

        Correzione rotazione: il centro del bbox viene trasformato con la
        stessa catena di rotazioni GL (Rx(rot_x) × Ry(rot_y) × Rx(−90))
        così pan_x/pan_y annullano esattamente lo spostamento in eye-space.
        """
        pts = []
        for o in oggetti:
            if o.visibile:
                pts.extend(o.get_vertices_world())
        if not pts:
            spazio.cam_dist = 12.0
            spazio.pan_x = spazio.pan_y = 0.0
            return

        arr = np.array(pts, dtype=float)
        mn, mx = arr.min(axis=0), arr.max(axis=0)
        ctr = (mn + mx) / 2.0

        # Rotation matrix matching GL chain:
        # glRotatef(rot_x,1,0,0) → glRotatef(rot_y,0,1,0) → glRotatef(-90,1,0,0)
        def Rx(deg):
            a = math.radians(deg)
            c, s = math.cos(a), math.sin(a)
            return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)
        def Ry(deg):
            a = math.radians(deg)
            c, s = math.cos(a), math.sin(a)
            return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)

        R = Rx(spazio.rot_x) @ Ry(spazio.rot_y) @ Rx(-90.0)
        ctr_eye = R @ ctr
        spazio.pan_x = float(-ctr_eye[0])
        spazio.pan_y = float(-ctr_eye[1])

        # cam_dist: frame the bounding sphere with FOV=45° (half=22.5°)
        # radius = diag/2  →  cam_dist = radius / tan(22.5°) × margin
        diag = float(np.linalg.norm(mx - mn))
        spazio.cam_dist = max(diag / (2.0 * math.tan(math.radians(22.5))) * 1.25, 0.5)

    # ----------------------------------------------------------------

    def _render_preview_per(self, spazio, el_id: int, oggetti: list):
        """
        Render isometrico centrato sull'oggetto.
        """
        # Salva stato
        old_rot_x   = spazio.rot_x
        old_rot_y   = spazio.rot_y
        old_pan_x   = spazio.pan_x
        old_pan_y   = spazio.pan_y
        old_dist    = spazio.cam_dist
        old_ortho   = spazio._ortho
        old_oggetti = spazio._oggetti
        
        px = None # Inizializza la QPixmap a None

        try:
            spazio._preview_mode = True
            spazio._ortho        = False
            spazio.rot_x         = 30.0
            spazio.rot_y         = -45.0
            spazio._oggetti      = oggetti
            self._centra_per_preview(spazio, oggetti)

            # Controlla che il widget OpenGL sia inizializzato
            if spazio.isValid():
                spazio.makeCurrent()
                spazio.paintGL()
                img = spazio.grabFramebuffer() # Restituisce QImage
                if not img.isNull():
                    # CONVERSIONE SALVAVITA
                    from PyQt5.QtGui import QPixmap
                    px = QPixmap.fromImage(img)

        except Exception as e:
            print(f"Errore render preview per (batch): {e}")
            
        finally:
            spazio._preview_mode = False
            spazio.rot_x         = old_rot_x
            spazio.rot_y         = old_rot_y
            spazio.pan_x         = old_pan_x
            spazio.pan_y         = old_pan_y
            spazio.cam_dist      = old_dist
            spazio._ortho        = old_ortho
            spazio._oggetti      = old_oggetti
            
            # Update asincrono sicuro
            if spazio.isValid():
                spazio.update() 

        if px is not None and not px.isNull():
            self._lista_ctrl.aggiorna_preview(el_id, px)

    # ── preview batch al caricamento progetto ─────────────────────────

    def _schedula_previews_tutti(self):
        """Accoda tutti gli elementi con oggetti per la cattura preview (elem + CV)."""
        self._coda_preview = [
            el
            for lista in self._lista_ctrl.get_elementi().values()
            for el in lista if el.oggetti
        ]
        if self._coda_preview:
            QTimer.singleShot(250, self._processa_prossima_preview)

    def _processa_prossima_preview(self):
        if not self._coda_preview:
            return
        el     = self._coda_preview.pop(0)
        spazio = self._elem_ctrl.get_spazio()
        # Preview elemento base
        self._render_preview_per(spazio, el.id, el.oggetti)
        # Preview C/V (usa ExtraSpazio3D in preview_mode)
        self._render_cv_preview_for_element(el)
        if self._coda_preview:
            QTimer.singleShot(80, self._processa_prossima_preview)

    # ================================================================
    #  PERSISTENZA
    # ================================================================

    def _salva(self):
        if not self._main.ha_progetto():
            return
        elementi = self._lista_ctrl.get_elementi()
        dati = {tipo: [el.to_dict() for el in lista if not getattr(el, "standard", False)]
                for tipo, lista in elementi.items()}
        self._main.push_undo("Modifica elementi")
        self._main.set_sezione("elementi", dati)

    def ricarica_da_progetto(self):
        sezione = self._main.get_sezione("elementi")
        if not sezione:
            return

        Elemento._id_counter  = 0;  Elemento._nome_count  = {}
        Oggetto3D._id_counter = 0;  Oggetto3D._nome_count = {}

        self._lista_ctrl.ricarica_da_progetto(sezione)
        self._elem_ctrl.svuota()
        print(">> Modulo Elementi: progetto ricaricato.")

        # Sync ID counters to the max loaded IDs to prevent collisions with
        # objects created after loading (add / duplicate).
        elementi_flat = [el for lista in self._lista_ctrl.get_elementi().values()
                         for el in lista]
        if elementi_flat:
            Elemento._id_counter = max(el.id for el in elementi_flat)
            oggetti_flat = [o for el in elementi_flat for o in el.oggetti]
            if oggetti_flat:
                Oggetto3D._id_counter = max(o.id for o in oggetti_flat)

        # Reload carichi/vincoli
        dati_carichi = self._main.get_sezione("carichi")
        self._extra_ctrl.ricarica_da_progetto(dati_carichi)

        # Cattura preview per tutti gli elementi con oggetti
        self._schedula_previews_tutti()

    def ripristina_contesto(self, elemento_id: int | None, pagina: int = 5):
        """
        Chiamata da MainWindow dopo undo/redo per riaprire l'elemento
        che era in editing al momento dello snapshot.
        Se l'elemento non esiste più (es. è stato rimosso dall'undo),
        torna alla lista (pagina 5).
        """
        if elemento_id is not None:
            # Cerca l'elemento per ID nella lista ricaricata
            for lista in self._lista_ctrl.get_elementi().values():
                for el in lista:
                    if el.id == elemento_id:
                        self._on_elemento_selezionato(el)
                        return
        # Elemento non trovato o None: vai alla lista
        if pagina in (6, 7):
            self._ui.stackedWidget_main.setCurrentIndex(5)

    def ripristina_contesto_extra(self, elemento_id: int | None):
        """
        Chiamata da MainWindow dopo undo/redo nello scope 'extra_elemento'
        (pagina 7 – carichi/vincoli).  Riapre l'elemento nel workspace C/V.
        """
        if elemento_id is not None:
            for lista in self._lista_ctrl.get_elementi().values():
                for el in lista:
                    if el.id == elemento_id:
                        self._extra_ctrl.carica_elemento(el)
                        return
        # Se non trovato, l'extra controller è già in stato svuotato
