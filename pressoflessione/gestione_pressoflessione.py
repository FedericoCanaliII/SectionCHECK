"""
gestione_pressoflessione.py  –  v2
====================================
Gestore dell'interfaccia utente per il modulo Pressoflessione / Flessione semplice.

Nuove funzionalità rispetto alla v1:
  • Calcolo in QThread → progressBar_pressoflessione aggiornata in tempo reale
  • btn_pressoflessione_normale   → set_display_mode('normale')
  • btn_pressoflessione_gradiente → set_display_mode('gradiente')  (esclusivi)

Oggetti UI attesi:
  combobox_pressoflessione_sezioni   – selezione sezione
  pressoflessione_precisione         – passo griglia [mm]
  pressoflessione_rotazione          – angolo NA [°]
  pressoflessione_sle / _slu         – QCheckBox tipo analisi
  pressoflessione_N / _M             – valori sollecitazioni
  btn_pressoflessione_verifica       – avvia analisi
  pressoflessione_testo_punto        – label risultato
  progressBar_pressoflessione        – barra progresso
  btn_pressoflessione_normale        – modo colore normale
  btn_pressoflessione_gradiente      – modo colore gradiente
  widget_pressoflessione_sezione     – contenitore OpenGL sezione
  widget_pressoflessione             – contenitore OpenGL diagrammi
"""
from __future__ import annotations

import traceback
from typing import Dict, List, Optional

from PyQt5.QtCore    import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QButtonGroup, QComboBox, QVBoxLayout

from output.verifica import Verifica
from output.calcolo  import SezioneRinforzata, Materiale

from pressoflessione.calcolo_pressoflessione         import CalcoloPressoflessione
from pressoflessione.disegno_pressoflessione_sezione import OpenGLPressoflessioneSezioneWidget
from pressoflessione.disegno_pressoflessione         import OpenGLPressoflessioneWidget

# ─────────────────────────────────────────────────────────────────────────────
# STILI FEEDBACK
# ─────────────────────────────────────────────────────────────────────────────
_BASE = ("background-color: rgb(55,55,55); "
         "font: 10pt 'Segoe UI'; padding-left: 6px; border-radius: 4px; ")
_OK   = _BASE + "border: 1px solid #00CC44; color: #00CC44; font-weight: bold;"
_FAIL = _BASE + "border: 1px solid #FF4444; color: #FF4444; font-weight: bold;"
_INFO = _BASE + "border: 1px solid #888888; color: #AAAAAA;"


# ═════════════════════════════════════════════════════════════════════════════
# THREAD DI CALCOLO
# ═════════════════════════════════════════════════════════════════════════════
class _CalcoloThread(QThread):
    """
    Esegue CalcoloPressoflessione in background.
    Emette progress (0–100) e done (dict risultati) / error (str).
    """
    progress = pyqtSignal(int)
    done     = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, elementi, materiali_dict, grid_step,
                 N_Ed, M_Ed, theta_deg, usa_sle, parent=None):
        super().__init__(parent)
        self._elementi      = elementi
        self._mat_dict      = materiali_dict
        self._grid_step     = grid_step
        self._N_Ed          = N_Ed
        self._M_Ed          = M_Ed
        self._theta_deg     = theta_deg
        self._usa_sle       = usa_sle

    def run(self) -> None:
        try:
            self.progress.emit(5)

            # 1. Costruzione sezione
            sezione_rc = SezioneRinforzata(self._elementi, self._mat_dict)
            self.progress.emit(15)

            # 2. Preparazione punti di integrazione (passo più lento)
            calc = CalcoloPressoflessione(sezione_rc, grid_step=self._grid_step)
            self.progress.emit(45)

            # 3. Analisi
            if self._usa_sle:
                res = calc.analisi_sle(self._N_Ed, self._M_Ed, self._theta_deg)
            else:
                res = calc.analisi_slu(self._N_Ed, self._M_Ed, self._theta_deg)

            self.progress.emit(95)
            self.done.emit(res)
            self.progress.emit(100)

        except Exception as exc:
            traceback.print_exc()
            self.error.emit(str(exc))


# ═════════════════════════════════════════════════════════════════════════════
# CONTROLLER PRINCIPALE
# ═════════════════════════════════════════════════════════════════════════════
class GestionePressoflessione:
    """
    Controller del pannello Pressoflessione.

    Parametri
    ---------
    ui                 : oggetto UI generato da Qt Designer
    gestione_sezioni   : istanza GestioneSezioni (espone section_manager)
    gestione_materiali : istanza GestioneMateriali
    """

    def __init__(self, ui, gestione_sezioni, gestione_materiali) -> None:
        self.ui                 = ui
        self.gestione_sezioni   = gestione_sezioni
        self.gestione_materiali = gestione_materiali

        self.verifica = Verifica(gestione_sezioni, gestione_materiali)

        self._section_map: List[int] = []
        self.selected_section_index: Optional[int] = None
        self._last_results: Optional[Dict] = None
        self._calc_thread: Optional[_CalcoloThread] = None

        # ── Widget OpenGL sezione (geometria) ─────────────────────────────────
        self.widget_sezione: Optional[OpenGLPressoflessioneSezioneWidget] = None
        try:
            cont = getattr(self.ui, 'widget_pressoflessione_sezione', None)
            if cont is not None:
                self.widget_sezione = OpenGLPressoflessioneSezioneWidget(
                    self.ui, parent=cont)
                lay = QVBoxLayout(cont)
                lay.setContentsMargins(1, 1, 1, 1)
                lay.addWidget(self.widget_sezione)
        except Exception:
            traceback.print_exc()

        # ── Widget OpenGL diagrammi ───────────────────────────────────────────
        self.widget_diagrammi: Optional[OpenGLPressoflessioneWidget] = None
        try:
            cont = getattr(self.ui, 'widget_pressoflessione', None)
            if cont is not None:
                self.widget_diagrammi = OpenGLPressoflessioneWidget(
                    self.ui, parent=cont)
                lay = QVBoxLayout(cont)
                lay.setContentsMargins(1, 1, 1, 1)
                lay.addWidget(self.widget_diagrammi)
        except Exception:
            traceback.print_exc()

        # ── Connessioni ───────────────────────────────────────────────────────
        self._connetti()

    # ──────────────────────────────────────────────────────────────────────────
    # CONNESSIONI
    # ──────────────────────────────────────────────────────────────────────────

    def _connetti(self) -> None:
        ui = self.ui

        # Pulsante avvia
        btn = getattr(ui, 'btn_pressoflessione_verifica', None)
        if btn is not None:
            try: btn.clicked.connect(self.esegui_verifica)
            except Exception: traceback.print_exc()

        # Combobox
        cb = getattr(ui, 'combobox_pressoflessione_sezioni', None)
        if cb is not None and isinstance(cb, QComboBox):
            try: cb.currentIndexChanged.connect(self._on_combo)
            except Exception: traceback.print_exc()

        # Checkbox SLE / SLU – mutualmente esclusivi
        chk_sle = getattr(ui, 'pressoflessione_sle', None)
        chk_slu = getattr(ui, 'pressoflessione_slu', None)
        if chk_sle and chk_slu:
            try:
                chk_sle.stateChanged.connect(
                    lambda s: chk_slu.setChecked(False) if s else None)
                chk_slu.stateChanged.connect(
                    lambda s: chk_sle.setChecked(False) if s else None)
                QTimer.singleShot(0, lambda: chk_slu.setChecked(True))
            except Exception: traceback.print_exc()

        # Bottoni modalità colore (esclusivi)
        btn_n = getattr(ui, 'btn_pressoflessione_normale', None)
        btn_g = getattr(ui, 'btn_pressoflessione_gradiente', None)
        if btn_n or btn_g:
            try:
                grp = QButtonGroup()
                grp.setExclusive(True)
                if btn_n:
                    btn_n.setCheckable(True)
                    grp.addButton(btn_n)
                    btn_n.clicked.connect(lambda: self._set_mode('normale'))
                if btn_g:
                    btn_g.setCheckable(True)
                    grp.addButton(btn_g)
                    btn_g.clicked.connect(lambda: self._set_mode('gradiente'))
                # Mantieni riferimento per non essere garbage-collected
                self._btn_mode_grp = grp
                # Default: normale attivo
                if btn_n:
                    QTimer.singleShot(0, lambda: btn_n.setChecked(True))
            except Exception: traceback.print_exc()

        # Bottone populate combobox dal menu principale
        btn_m = getattr(ui, 'btn_main_pressoflessione', None)
        if btn_m:
            try: btn_m.clicked.connect(self.popola_combobox)
            except Exception: traceback.print_exc()

    def _set_mode(self, mode: str) -> None:
        if self.widget_diagrammi is not None:
            try: self.widget_diagrammi.set_display_mode(mode)
            except Exception: traceback.print_exc()

    # ──────────────────────────────────────────────────────────────────────────
    # COMBOBOX
    # ──────────────────────────────────────────────────────────────────────────

    def popola_combobox(self) -> None:
        cb = getattr(self.ui, 'combobox_pressoflessione_sezioni', None)
        if cb is None: return
        try:
            prev = cb.itemData(cb.currentIndex(), Qt.UserRole) if cb.count() else None
            cb.blockSignals(True); cb.clear()
            self._section_map = []; self.selected_section_index = None

            sm = self._get_sm()
            items: List[tuple] = []
            if sm is not None and hasattr(sm, 'sections'):
                secs = getattr(sm, 'sections') or []
                if not secs:
                    cb.addItem("Nessuna sezione"); cb.setEnabled(False)
                    cb.blockSignals(False); self._aggiorna_viewer(); return
                cb.setEnabled(True)
                for i, sec in enumerate(secs):
                    n = (getattr(sec, 'name', None)
                         or (sec.get('name') if isinstance(sec, dict) else None)
                         or f"Sezione {i+1}")
                    items.append((str(n), i))
            else:
                cb.addItem("Nessuna sezione"); cb.setEnabled(False)
                cb.blockSignals(False); self._aggiorna_viewer(); return

            target = 0
            for i, (name, ri) in enumerate(items):
                cb.addItem(name); cb.setItemData(i, ri, Qt.UserRole)
                self._section_map.append(ri)
                if prev is not None and ri == prev: target = i

            cb.setCurrentIndex(target)
            if 0 <= target < len(self._section_map):
                self.selected_section_index = self._section_map[target]
            cb.blockSignals(False)
            self._aggiorna_viewer()
        except Exception:
            traceback.print_exc()
            try: cb.blockSignals(False)
            except Exception: pass

    def _on_combo(self, idx: int) -> None:
        cb = getattr(self.ui, 'combobox_pressoflessione_sezioni', None)
        if cb is None: self.selected_section_index = None; return
        try:
            d = cb.itemData(idx, Qt.UserRole)
            self.selected_section_index = (
                self._section_map[idx] if d is None and 0 <= idx < len(self._section_map)
                else int(d) if d is not None else idx)
        except Exception:
            self.selected_section_index = idx
        self._aggiorna_viewer()

    def _get_sm(self):
        gs = self.gestione_sezioni
        if hasattr(gs, 'section_manager'): return getattr(gs, 'section_manager')
        if hasattr(gs, 'sections'): return gs
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # VIEWER GEOMETRIA
    # ──────────────────────────────────────────────────────────────────────────

    def _aggiorna_viewer(self) -> None:
        if self.widget_sezione is None: return
        idx = self.selected_section_index
        if idx is None:
            self.widget_sezione.set_section_data(None); return
        try:
            ts = self.verifica.get_tutte_matrici_sezioni()
            if 0 <= idx < len(ts):
                self.widget_sezione.set_section_data(ts[idx])
            else:
                self.widget_sezione.set_section_data(None)
        except Exception:
            traceback.print_exc()
            self.widget_sezione.set_section_data(None)

    # ──────────────────────────────────────────────────────────────────────────
    # LETTURA VALORI UI
    # ──────────────────────────────────────────────────────────────────────────

    def _float(self, attr: str, default: float) -> float:
        w = getattr(self.ui, attr, None)
        if w is None: return default
        try:
            if hasattr(w, 'text'):  return float(w.text().replace(',', '.') or str(default))
            if hasattr(w, 'value'): return float(w.value())
        except (ValueError, AttributeError): pass
        return default

    def _bool(self, attr: str) -> bool:
        w = getattr(self.ui, attr, None)
        if w is None: return False
        try: return bool(w.isChecked())
        except Exception: return False

    # ──────────────────────────────────────────────────────────────────────────
    # ESEGUI VERIFICA (asincrona)
    # ──────────────────────────────────────────────────────────────────────────

    def esegui_verifica(self) -> None:
        """
        Avvia il calcolo in un thread separato.
        La progress bar viene aggiornata via segnale.
        """
        self._set_txt("Calcolo in corso…", _INFO)
        self._set_progress(0)

        try:
            gs        = max(self._float('pressoflessione_precisione', 10.0), 0.1)
            theta_deg = self._float('pressoflessione_rotazione', 0.0)
            N_Ed      = self._float('pressoflessione_N', 0.0)
            M_Ed      = self._float('pressoflessione_M', 0.0)
            usa_sle   = self._bool('pressoflessione_sle')
            usa_slu   = self._bool('pressoflessione_slu')
            if not usa_sle and not usa_slu: usa_slu = True

            # Recupera dati
            ts = self.verifica.get_tutte_matrici_sezioni()
            tm = self.verifica.get_tutte_matrici_materiali()
            if not ts:
                self._set_txt("Nessuna sezione definita.", _FAIL); return
            if not tm:
                self._set_txt("Nessun materiale definito.", _FAIL); return

            sel = self.selected_section_index or 0
            sel = max(0, min(sel, len(ts)-1))
            pagina = ts[sel]

            mat_dict: Dict[str, Materiale] = {}
            for nome, matr in tm.items():
                try: mat_dict[nome] = Materiale(matr, str(nome))
                except Exception: traceback.print_exc()
            if not mat_dict:
                self._set_txt("Errore nei materiali.", _FAIL); return

            # Ferma eventuale thread precedente
            if self._calc_thread is not None and self._calc_thread.isRunning():
                try:
                    self._calc_thread.requestInterruption()
                    self._calc_thread.quit()
                except Exception: pass

            # Lancia thread
            self._calc_thread = _CalcoloThread(
                elementi    = pagina['elementi'],
                materiali_dict = mat_dict,
                grid_step   = gs,
                N_Ed        = N_Ed,
                M_Ed        = M_Ed,
                theta_deg   = theta_deg,
                usa_sle     = usa_sle,
            )
            self._calc_thread.progress.connect(self._set_progress)
            self._calc_thread.done.connect(self._on_calc_done)
            self._calc_thread.error.connect(self._on_calc_error)
            self._calc_thread.start()

        except Exception:
            traceback.print_exc()
            self._set_txt("Errore interno nel calcolo.", _FAIL)

    # ── Callback thread ───────────────────────────────────────────────────────

    def _on_calc_done(self, res: Dict) -> None:
        self._last_results = res
        if self.widget_diagrammi is not None:
            try: self.widget_diagrammi.set_results(res)
            except Exception: traceback.print_exc()
        self._aggiorna_label(res)

    def _on_calc_error(self, msg: str) -> None:
        self._set_txt(f"Errore: {msg}", _FAIL)
        self._set_progress(0)

    # ──────────────────────────────────────────────────────────────────────────
    # AGGIORNAMENTO LABEL RISULTATO
    # ──────────────────────────────────────────────────────────────────────────

    def _aggiorna_label(self, res: Dict) -> None:
        tipo = res.get('tipo', '')
        ok   = res.get('verificata', False)

        if tipo == 'SLU':
            M   = abs(res.get('M_Ed',  0.0))
            Mr  = res.get('M_Rd',  0.0)
            N   = res.get('N_Ed',  0.0)
            rr  = res.get('rapporto_MEd_MRd', 0.0)
            if res.get('fuori_dominio', False):
                self._set_txt(
                    f"SLU ✗  Fuori dominio  |  N={N:.1f} kN  "
                    f"M_Ed={M:.1f} kNm  M_Rd={Mr:.1f} kNm", _FAIL)
            elif ok:
                self._set_txt(
                    f"SLU ✓  Verifica soddisfatta  |  "
                    f"M_Ed={M:.1f} ≤ M_Rd={Mr:.1f} kNm  (ratio={rr:.3f})", _OK)
            else:
                self._set_txt(
                    f"SLU ✗  Non verificato  |  "
                    f"M_Ed={M:.1f} > M_Rd={Mr:.1f} kNm  (ratio={rr:.3f})", _FAIL)

        elif tipo == 'SLE':
            sc = res.get('sigma_c_compr_max', 0.0)
            ss = res.get('sigma_s_traz_max',  0.0)
            note_str = "  |  " + "  ;  ".join(res.get('note', [])) if res.get('note') else ""
            if ok:
                self._set_txt(
                    f"SLE ✓  Verifica soddisfatta  |  "
                    f"σ_c={sc:.1f} MPa  σ_s={ss:.1f} MPa{note_str}", _OK)
            else:
                self._set_txt(
                    f"SLE ✗  Non verificato  |  "
                    f"σ_c={sc:.1f} MPa  σ_s={ss:.1f} MPa{note_str}", _FAIL)
        else:
            self._set_txt("Analisi completata.", _INFO)

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS UI
    # ──────────────────────────────────────────────────────────────────────────

    def _set_txt(self, txt: str, style: str) -> None:
        w = getattr(self.ui, 'pressoflessione_testo_punto', None)
        if w is None: return
        try: w.setText(txt); w.setStyleSheet(style)
        except Exception: pass

    def _set_progress(self, val: int) -> None:
        pb = getattr(self.ui, 'progressBar_pressoflessione', None)
        if pb is None: return
        try: pb.setValue(max(0, min(val, 100)))
        except Exception: pass
