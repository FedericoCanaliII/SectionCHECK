from typing import Any, List, Optional, Tuple
import traceback
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QTableView, QComboBox, QHeaderView, QMenu, QAction, QShortcut
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence, QStandardItemModel, QStandardItem


class InstantCombo(QComboBox):
    """Combo che apre subito la tendina al click (single click)."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setEditable(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        QTimer.singleShot(0, self.showPopup)


class Valori:
    """
    Valori per QTableView:
    - gestisce rects, circles, polys, bars, staffe e reinforcements
    - usa QStandardItemModel + view.setIndexWidget per mettere QComboBox sempre visibili
    - salva direttamente in sec[list_name][index]['material']
    """

    def __init__(self, section_manager: Any, ui: Optional[Any] = None, gestione_materiali: Optional[Any] = None, print_button: Optional[Any] = None):
        self.section_manager = section_manager
        self.ui = ui
        self.gestione_materiali = gestione_materiali or (getattr(ui, 'gestione_materiali', None) if ui else None)

        # modelli per le tableview
        self.model_sezioni = QStandardItemModel(0, 2)
        self.model_sezioni.setHorizontalHeaderLabels(["Nome", "Materiale"])
        self.model_barre = QStandardItemModel(0, 2)
        self.model_barre.setHorizontalHeaderLabels(["Nome", "Materiale"])
        self.model_rinforzi = QStandardItemModel(0, 2)
        self.model_rinforzi.setHorizontalHeaderLabels(["Nome", "Materiale"])
        self.model_staffe = QStandardItemModel(0, 2)
        self.model_staffe.setHorizontalHeaderLabels(["Nome", "Materiale"])

        # keep last built entries
        self._last_sec_entries = []
        self._last_bars_entries = []
        self._last_reinf_entries = []
        self._last_staffe_entries = []

        # install context menus
        for attr, handler in [('tableView_sezione', self._on_context_menu_section),
                              ('tableView_barre', self._on_context_menu_bar),
                              ('tableView_rinforzi', self._on_context_menu_reinf),
                              ('tableView_staffe', self._on_context_menu_staffe)]:
            try:
                view = getattr(self.ui, attr, None)
                if isinstance(view, QTableView):
                    view.setContextMenuPolicy(Qt.CustomContextMenu)
                    view.customContextMenuRequested.connect(handler)
            except Exception:
                pass

        # shortcut Enter per aggiornamento
        try:
            if isinstance(self.ui, QWidget):
                QShortcut(QKeySequence(Qt.Key_Return), self.ui).activated.connect(self.update_tables)
                QShortcut(QKeySequence(Qt.Key_Enter), self.ui).activated.connect(self.update_tables)
        except Exception:
            pass

        # bottone stampa se fornito
        if print_button is not None:
            try:
                print_button.clicked.connect(self.print_valori)
            except Exception:
                try:
                    print_button.clicked.connect(lambda: self.print_valori())
                except Exception:
                    pass

    # ---------- materiali ----------
    def get_available_materials(self) -> List[str]:
        mats: List[str] = []
        try:
            gm = self.gestione_materiali or (getattr(self.ui, 'gestione_materiali', None) if self.ui else None)
            if gm and hasattr(gm, 'get_material_names') and callable(getattr(gm, 'get_material_names')):
                names = gm.get_material_names()
                if names:
                    return list(names)
            buttons = getattr(gm, 'btn_group_materiali', None)
            if buttons:
                for b in buttons.buttons():
                    try: mats.append(b.text().strip())
                    except Exception: mats.append("")
                if mats: return mats
            cb = getattr(self.ui, 'combobox_materiali', None)
            if cb and isinstance(cb, QComboBox):
                for i in range(cb.count()):
                    mats.append(cb.itemText(i).strip())
                if mats: return mats
            mats = ["Calcestruzzo", "Acciaio Barre", "Acciaio Profili", "FRP"]
        except Exception:
            pass
        return mats

    # ---------- helper combo ----------
    def _create_and_attach_combo(self, view: QTableView, model_index, materials: List[str], current: Optional[str], on_change_cb):
        cb = InstantCombo(view)
        for m in materials:
            cb.addItem(m)
        if current:
            i = cb.findText(current)
            if i >= 0:
                cb.setCurrentIndex(i)
        if on_change_cb:
            cb.currentTextChanged.connect(on_change_cb)
        view.setIndexWidget(model_index, cb)
        return cb

    # ---------- aggiornamento tabelle ----------
    def update_tables(self) -> None:
        try:
            mgr = self.section_manager
            idx = getattr(mgr, 'current_index', None)
            if idx is None: return
            sec = mgr.get_section(idx)
            if sec is None: return
            materials = self.get_available_materials()

            # --- entries ---
            sec_entries = [{'list': 'rects', 'index': i, 'name': r.get('name') or f"RETTANGOLO {i+1}"} for i, r in enumerate(getattr(sec, 'rects', []))]
            sec_entries += [{'list': 'circles', 'index': i, 'name': c.get('name') or f"CERCHIO {i+1}"} for i, c in enumerate(getattr(sec, 'circles', []))]
            sec_entries += [{'list': 'polys', 'index': i, 'name': p.get('name') or f"POLIGONO {i+1}"} for i, p in enumerate(getattr(sec, 'polys', []))]

            bars_entries = [{'list': 'bars', 'index': i, 'name': b.get('name') or f"BARRA {i+1}"} for i, b in enumerate(getattr(sec, 'bars', []))]
            reinf_entries = [{'list': 'reinforcements', 'index': i, 'name': r.get('name') or f"RINFORZO {i+1}"} for i, r in enumerate(getattr(sec, 'reinforcements', []))]
            staffe_entries = [{'list': 'staffe', 'index': i, 'name': s.get('name') or f"STAFFA {i+1}"} for i, s in enumerate(getattr(sec, 'staffe', []))]

            # --- view sezione ---
            self._update_view(getattr(self.ui, 'tableView_sezione', None), self.model_sezioni, sec_entries, sec, materials)
            self._update_view(getattr(self.ui, 'tableView_barre', None), self.model_barre, bars_entries, sec, materials, default_index=1)
            self._update_view(getattr(self.ui, 'tableView_rinforzi', None), self.model_rinforzi, reinf_entries, sec, materials, default_index=3)
            self._update_view(getattr(self.ui, 'tableView_staffe', None), self.model_staffe, staffe_entries, sec, materials, default_index=1)

            self._last_sec_entries = sec_entries
            self._last_bars_entries = bars_entries
            self._last_reinf_entries = reinf_entries
            self._last_staffe_entries = staffe_entries

        except Exception:
            traceback.print_exc()

    def _update_view(self, view, model, entries, sec, materials, default_index=0):
        if not isinstance(view, QTableView): return
        view.setModel(model)
        model.removeRows(0, model.rowCount())
        for ent in entries:
            item0 = QStandardItem(str(ent['name']))
            item0.setEditable(False)
            item1 = QStandardItem("")
            model.appendRow([item0, item1])
            cur_mat = None
            try:
                lst = getattr(sec, ent['list'], None)
                if lst and ent['index'] < len(lst):
                    cur_mat = lst[ent['index']].get('material')
            except Exception:
                pass

            def on_changed_factory(list_name, index):
                return lambda mat, ln=list_name, ix=index: self._on_material_changed(ln, ix, mat)

            mdl_index = model.index(model.rowCount() - 1, 1)
            self._create_and_attach_combo(view, mdl_index, materials, cur_mat or (materials[default_index] if len(materials) > default_index else None), on_changed_factory(ent['list'], ent['index']))

        try:
            hdr = view.horizontalHeader()
            hdr.repaint()
            view.repaint()
            view.updateGeometry()
            hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        except Exception:
            pass

    # ---------- context menu staffe ----------
    def _on_context_menu_staffe(self, pos):
        self._generic_context_menu(pos, 'tableView_staffe', self._last_staffe_entries)

    # ---------- generico context menu ----------
    def _generic_context_menu(self, pos, table_attr, entries):
        try:
            view = getattr(self.ui, table_attr, None)
            if view is None: return
            idx = view.indexAt(pos)
            if not idx.isValid(): return
            row = idx.row()
            if row < 0 or row >= len(entries): return
            ent = entries[row]
            menu = QMenu(view)
            action_remove = QAction("Rimuovi", view)
            menu.addAction(action_remove)
            def do_remove():
                mgr = self.section_manager
                sec_idx = getattr(mgr, 'current_index', None)
                if sec_idx is None: return
                mgr.remove_item_from_section(sec_idx, ent['list'], ent['index'])
                try: self.update_tables()
                except Exception: pass
            action_remove.triggered.connect(do_remove)
            menu.exec_(view.mapToGlobal(pos))
        except Exception:
            pass

    # alias per compatibilità
    _on_context_menu_section = lambda self, pos: self._generic_context_menu(pos, 'tableView_sezione', self._last_sec_entries)
    _on_context_menu_bar = lambda self, pos: self._generic_context_menu(pos, 'tableView_barre', self._last_bars_entries)
    _on_context_menu_reinf = lambda self, pos: self._generic_context_menu(pos, 'tableView_rinforzi', self._last_reinf_entries)

    # ---------- salva materiali ----------
    def _on_material_changed(self, list_name: str, idx: int, material: str):
        try:
            mgr = self.section_manager
            sec = mgr.get_section(getattr(mgr, 'current_index', 0))
            if sec is None: return
            lst = getattr(sec, list_name, None)
            if lst is None or idx < 0 or idx >= len(lst): return
            if isinstance(lst[idx], dict):
                lst[idx]['material'] = material
        except Exception:
            pass

    # ---------- helper stampa ----------
    def _norm_number(self, x):
        if isinstance(x, (np.floating, np.integer)):
            x = x.item()
        if isinstance(x, float) and abs(x - int(round(x))) < 1e-9:
            return int(round(x))
        return x

    def _format_point(self, p) -> str:
        try:
            if p is None: return "(None)"
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                x = self._norm_number(p[0])
                y = self._norm_number(p[1])
                return f"({x}, {y})"
            if hasattr(p, 'x') and hasattr(p, 'y'):
                x = self._norm_number(p.x())
                y = self._norm_number(p.y())
                return f"({x}, {y})"
            return str(p)
        except Exception:
            return str(p)

    def _find_thickness(self, dct: dict):
        for key in ('offset', 'thickness', 't', 'spessore', 'diam', 'diametro', 'diameter'):
            if key in dct:
                return self._norm_number(dct.get(key))
        return None

    def _format_shape_row(self, name: str, points: List[Tuple], extra: Optional[float] = None) -> str:
        parts = [name]
        for p in points:
            parts.append(self._format_point(p))
        if extra is not None:
            parts.append(str(self._norm_number(extra)))
        return "[" + ", ".join(parts) + "]"

    # ---------- stampa valori ----------

    def print_valori(self):
        try:
            mgr = self.section_manager
            idx = getattr(mgr, 'current_index', None)
            if idx is None: 
                print("Nessuna sezione attiva.")
                return
            sec = mgr.get_section(idx)
            if sec is None:
                print("Sezione non trovata.")
                return

            print("\n-- RECTS --")
            for r in getattr(sec, 'rects', []):
                print(self._format_shape_row(r.get('name', "RETTANGOLO"), [r.get('v1'), r.get('v2')]))

            print("\n-- CIRCLES --")
            for c in getattr(sec, 'circles', []):
                print(self._format_shape_row(c.get('name', "CERCHIO"), [c.get('center'), c.get('radius_point')]))

            print("\n-- POLYS --")
            for p in getattr(sec, 'polys', []):
                print(self._format_shape_row(p.get('name', "POLIGONO"), p.get('vertices', [])))

            print("\n-- BARS --")
            for b in getattr(sec, 'bars', []):
                print(self._format_shape_row(b.get('name', "BARRA"), [b.get('center')], b.get('diam')))

            print("\n-- STAFFE --")
            for s in getattr(sec, 'staffe', []):
                # CORREZIONE: usare 'points' invece di 'vertices'
                points = s.get('points', [])
                print(self._format_shape_row(s.get('name', "STAFFA"), points, s.get('diam')))

            print("\n-- REINFORCEMENTS --")
            for r in getattr(sec, 'reinforcements', []):
                thickness = self._find_thickness(r)
                pts = r.get('poly') or []
                if pts:
                    print(self._format_shape_row(r.get('name', "RINFORZO"), pts, thickness))
                else:
                    bp1 = r.get('base_p1')
                    bp2 = r.get('base_p2')
                    if bp1 and bp2:
                        print(self._format_shape_row(r.get('name', "RINFORZO"), [bp1, bp2], thickness))
            print("\n=== FINE ===")
            try: 
                self.update_tables()
            except Exception: 
                pass
        except Exception:
            traceback.print_exc()