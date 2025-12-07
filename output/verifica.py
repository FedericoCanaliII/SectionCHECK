# output/verifica.py
from typing import Any, Dict, List, Tuple, Optional
import math
import traceback

# PyQt imports usati solo se l'app gira con PyQt attivo (sono opzionali ma presenti nell'app)
try:
    from PyQt5.QtWidgets import QTableView, QComboBox
    from PyQt5.QtCore import Qt
except Exception:
    QTableView = None
    QComboBox = None
    Qt = None


class Verifica:
    """
    Verifica riprogettata:
    - estrae forme/barre/rinforzi dalla SectionManager (o dallo stackedWidget),
    - recupera i materiali dalle strutture e dalle QTableView (indexWidget),
    - restituisce tuple pronte per SezioneRinforzata:
       ('shape','rect'|'poly'|'circle', label, material_name, p1, p2|verts|radius)
       ('bar', name, material_name, diam, (x,y))
       ('reinf','rect'|'poly'|'circular', name, material_name, params..., th)
    """
    def __init__(self, gestione_sezioni: Any, gestione_materiali: Any):
        self.gestione_sezioni = gestione_sezioni
        self.gestione_materiali = gestione_materiali

    # ------------------ UTIL ------------------
    def _norm_point(self, p) -> Optional[Tuple[float, float]]:
        try:
            if p is None:
                return None
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                return (float(p[0]), float(p[1]))
            # oggetto PyQt (QPoint / QPointF)
            if hasattr(p, 'x') and hasattr(p, 'y'):
                try:
                    return (float(p.x()), float(p.y()))
                except Exception:
                    try:
                        return (float(p.x), float(p.y))
                    except Exception:
                        return None
            return None
        except Exception:
            return None

    def _find_thickness(self, dct: dict) -> Optional[float]:
        if not isinstance(dct, dict):
            return None
        for key in ('th', 't', 'thickness', 'offset', 'spessore', 'diam', 'diametro', 'diameter'):
            if key in dct:
                try:
                    return float(dct.get(key) or 0.0)
                except Exception:
                    return None
        return None

    def _try_get_material_from_obj(self, obj) -> str:
        if not isinstance(obj, dict):
            return ''
        for k in ('material', 'mat', 'material_name', 'materiale', 'mat_name'):
            if k in obj:
                v = obj.get(k) or ''
                return str(v)
        return ''

    def _read_materials_from_view(self, view) -> List[str]:
        """
        Legge le combobox posizionate nella colonna materiali (colonna 1)
        di un QTableView impostato con setIndexWidget. Restituisce la lista dei testi
        (uno per riga). Se non trova widget per una riga ritorna '' per quella riga.
        """
        mats: List[str] = []
        try:
            if view is None or QTableView is None:
                return mats
            if not isinstance(view, QTableView):
                return mats
            model = view.model()
            if model is None:
                return mats
            rows = model.rowCount()
            for row in range(rows):
                try:
                    idx = model.index(row, 1)
                    w = view.indexWidget(idx)
                    if w is not None and QComboBox is not None and isinstance(w, QComboBox):
                        mats.append(str(w.currentText()))
                    else:
                        # fallback: prova a leggere dal model stesso
                        data = model.data(idx)
                        mats.append(str(data) if data is not None else '')
                except Exception:
                    mats.append('')
        except Exception:
            traceback.print_exc()
        return mats

    # ------------------ ESTRAZIONE DA SectionModel ------------------
    def _extract_from_section_model(self, sec, ui_for_tables=None) -> List[Tuple]:
        """
        Estrae elementi dalla SectionModel (SectionManager.sections element),
        usando i campi presenti e integrando i materiali letti dalle tabelle UI
        se il dict non contiene 'material'.
        """
        elems = []
        try:
            # Proviamo prima a leggere le liste canoniche
            rects = getattr(sec, 'rects', []) or []
            circles = getattr(sec, 'circles', []) or []
            polys = getattr(sec, 'polys', []) or []
            bars = getattr(sec, 'bars', []) or []
            reinfs = getattr(sec, 'reinforcements', []) or []

            # Leggi materiali dalle tabelle (se disponibili nell'UI)
            sec_table = None
            bar_table = None
            reinf_table = None
            if ui_for_tables is not None:
                sec_table = getattr(ui_for_tables, 'tableView_sezione', None)
                bar_table = getattr(ui_for_tables, 'tableView_barre', None)
                reinf_table = getattr(ui_for_tables, 'tableView_rinforzi', None)

            sec_table_mats = self._read_materials_from_view(sec_table) if sec_table is not None else []
            bar_table_mats = self._read_materials_from_view(bar_table) if bar_table is not None else []
            reinf_table_mats = self._read_materials_from_view(reinf_table) if reinf_table is not None else []

            # ---- SHAPES (rects, circles, polys) in that order ----
            # NOTE: Valori.update_tables costruisce la stessa sequenza (rects->circles->polys)
            # quindi possiamo consumare sec_table_mats nell'ordine.
            sec_mat_iter = iter(sec_table_mats)

            for i, r in enumerate(rects, start=1):
                name = (r.get('name') if isinstance(r, dict) else None) or f"RETTANGOLO {i}"
                mat = self._try_get_material_from_obj(r) or (next(sec_mat_iter, '') if sec_table_mats else '')
                p1 = self._norm_point(r.get('v1') or r.get('p1') or r.get('p0'))
                p2 = self._norm_point(r.get('v2') or r.get('p2') or r.get('p1'))
                elems.append(('shape', 'rect', name, mat, p1, p2))

            for i, c in enumerate(circles, start=1):
                name = (c.get('name') if isinstance(c, dict) else None) or f"CERCHIO {i}"
                mat = self._try_get_material_from_obj(c) or (next(sec_mat_iter, '') if sec_table_mats else '')
                center = self._norm_point(c.get('center') or c.get('centroid'))
                radius = None
                if isinstance(c, dict):
                    if 'radius' in c and isinstance(c.get('radius'), (int, float)):
                        radius = float(c.get('radius'))
                    else:
                        rp = self._norm_point(c.get('radius_point') or c.get('rpoint') or c.get('perimeter_point'))
                        if center and rp:
                            radius = math.hypot(rp[0] - center[0], rp[1] - center[1])
                elems.append(('shape', 'circle', name, mat, center, radius))

            for i, p in enumerate(polys, start=1):
                name = (p.get('name') if isinstance(p, dict) else None) or f"POLIGONO {i}"
                mat = self._try_get_material_from_obj(p) or (next(sec_mat_iter, '') if sec_table_mats else '')
                raw = p.get('vertices') or p.get('verts') or p.get('pts') or p.get('poly') or p.get('points') or []
                verts = [self._norm_point(v) for v in raw if self._norm_point(v) is not None]
                elems.append(('shape', 'poly', name, mat, verts))

            # ---- BARS ----
            for i, b in enumerate(bars, start=1):
                name = (b.get('name') if isinstance(b, dict) else None) or f"BARRA {i}"
                mat = self._try_get_material_from_obj(b)
                if not mat:
                    # try table list by index (bars table usually aligned 1:1)
                    try:
                        mat = bar_table_mats[i - 1] if i - 1 < len(bar_table_mats) else ''
                    except Exception:
                        mat = ''
                diam = b.get('diam') or b.get('d') or b.get('diameter') or 0.0
                try:
                    diam = float(diam)
                except Exception:
                    diam = 0.0
                center = self._norm_point(b.get('center') or b.get('p') or b.get('pos')) or (0.0, 0.0)
                elems.append(('bar', name, mat, diam, center))

            # ---- REINFORCEMENTS ----
            for i, r in enumerate(reinfs, start=1):
                name = (r.get('name') if isinstance(r, dict) else None) or f"RINFORZO {i}"
                mat = self._try_get_material_from_obj(r)
                if not mat:
                    try:
                        mat = reinf_table_mats[i - 1] if i - 1 < len(reinf_table_mats) else ''
                    except Exception:
                        mat = ''
                th = self._find_thickness(r) or 0.0

                # rect-like
                if isinstance(r, dict) and ('base_p1' in r and 'base_p2' in r or 'p0' in r and 'p1' in r):
                    p0 = self._norm_point(r.get('base_p1') or r.get('p0'))
                    p1 = self._norm_point(r.get('base_p2') or r.get('p1'))
                    elems.append(('reinf', 'rect', name, mat, p0, p1, th))
                    continue

                # poly inside reinf
                if isinstance(r, dict):
                    raw = r.get('poly') or r.get('pts') or r.get('points')
                    if isinstance(raw, (list, tuple)) and raw:
                        pts = [self._norm_point(v) for v in raw if self._norm_point(v) is not None]
                        elems.append(('reinf', 'poly', name, mat, pts, th))
                        continue

                # parent circle
                parent = r.get('parent') if isinstance(r, dict) else None
                if isinstance(parent, dict) and (('center' in parent) or ('radius_point' in parent) or ('radius' in parent)):
                    center = self._norm_point(parent.get('center') or parent.get('centroid'))
                    elems.append(('reinf', 'circular', name, mat, center, th))
                    continue

                # fallback on points/pts
                fallback = []
                if isinstance(r, dict):
                    for key in ('points', 'pts', 'poly'):
                        raw = r.get(key)
                        if isinstance(raw, (list, tuple)) and raw:
                            fallback = [self._norm_point(v) for v in raw if self._norm_point(v) is not None]
                            break
                if fallback:
                    elems.append(('reinf', 'poly', name, mat, fallback, th))
                    continue

                # else ignore (no geometry)
                pass

        except Exception:
            traceback.print_exc()
        return elems

    # ------------------ estrazione da pagina/stored widget ------------------
    def _extract_from_page_widget(self, page, ui_for_tables=None) -> List[Tuple]:
        elems = []
        try:
            # use legacy drawers if present
            sd = getattr(page, 'section_drawer', None)
            if sd is not None:
                shapes = getattr(sd, 'shapes', []) or []
                for i, s in enumerate(shapes, start=1):
                    tipo = s.get('type')
                    label = s.get('name') or s.get('label') or f"Forma {i}"
                    mat = s.get('material') or s.get('mat') or ''
                    if tipo == 'rect':
                        pts = s.get('pts', []) or []
                        p1 = self._norm_point(pts[0]) if len(pts) > 0 else None
                        p2 = self._norm_point(pts[2]) if len(pts) > 2 else (self._norm_point(pts[1]) if len(pts) > 1 else None)
                        elems.append(('shape', 'rect', label, mat, p1, p2))
                    elif tipo == 'poly':
                        pts = s.get('pts', []) or []
                        verts = [self._norm_point(v) for v in pts if self._norm_point(v) is not None]
                        elems.append(('shape', 'poly', label, mat, verts))
                    elif tipo == 'circle':
                        center = self._norm_point(s.get('center'))
                        radius = s.get('radius') or None
                        elems.append(('shape', 'circle', label, mat, center, radius))

            # bars drawer
            bd = getattr(page, 'bar_drawer', None)
            if bd is not None:
                confirmed = getattr(bd, 'confirmed_bars', []) or []
                for i, b in enumerate(confirmed, start=1):
                    name = b.get('name') or f"Barra {i}"
                    mat = b.get('material') or ''
                    diam = b.get('diam') or b.get('d') or 0.0
                    try:
                        diam = float(diam)
                    except Exception:
                        diam = 0.0
                    center = self._norm_point(b.get('center') or b.get('p')) or (0.0, 0.0)
                    elems.append(('bar', name, mat, diam, center))

            # reinf drawer
            rd = getattr(page, 'reinf_drawer', None)
            if rd is not None:
                confirmed = getattr(rd, 'confirmed', []) or []
                for i, r in enumerate(confirmed, start=1):
                    name = r.get('name') or f"Rinforzo {i}"
                    mat = r.get('material') or ''
                    th = r.get('th') or r.get('t') or 0.0
                    if r.get('type') == 'rect':
                        p0 = self._norm_point(r.get('p0'))
                        p1 = self._norm_point(r.get('p1'))
                        elems.append(('reinf', 'rect', name, mat, p0, p1, th))
                    elif r.get('type') == 'poly':
                        pts = [self._norm_point(v) for v in r.get('pts', []) if self._norm_point(v) is not None]
                        elems.append(('reinf', 'poly', name, mat, pts, th))
                    else:
                        center = self._norm_point(r.get('center'))
                        elems.append(('reinf', 'circular', name, mat, center, th))

            # fallback: if page has rects/circles/polys attributes
            if any(hasattr(page, a) for a in ('rects', 'circles', 'polys', 'bars', 'reinforcements')):
                elems = self._extract_from_section_model(page, ui_for_tables=ui_for_tables)

        except Exception:
            traceback.print_exc()
        return elems

    # ------------------ API pubbliche ------------------
    def get_tutte_matrici_sezioni(self) -> List[Dict[str, Any]]:
        risultato = []
        try:
            # 1) preferisci SectionManager.sections se presente
            sm = None
            if hasattr(self.gestione_sezioni, 'section_manager'):
                sm = getattr(self.gestione_sezioni, 'section_manager')
            elif hasattr(self.gestione_sezioni, 'sections'):
                sm = self.gestione_sezioni

            ui = getattr(self.gestione_sezioni, 'ui', None)

            if sm is not None and hasattr(sm, 'sections'):
                secs = getattr(sm, 'sections', []) or []
                for idx, sec in enumerate(secs):
                    elems = self._extract_from_section_model(sec, ui_for_tables=ui)
                    risultato.append({'pagina': idx, 'elementi': elems})
                return risultato

            # 2) fallback a stackedWidget_sezioni-like
            if ui is not None:
                stack = getattr(ui, 'stackedWidget_sezioni', None) or getattr(ui, 'stackedWidget', None) or getattr(ui, 'stackedWidget_main', None)
                if stack is not None:
                    for idx in range(stack.count()):
                        page = stack.widget(idx)
                        elems = self._extract_from_page_widget(page, ui_for_tables=ui)
                        risultato.append({'pagina': idx, 'elementi': elems})
                    return risultato

            print("[Verifica] Nessuna sezione trovata (ni SectionManager.sections né stackedWidget_sezioni).")
            return risultato
        except Exception:
            traceback.print_exc()
            return risultato

    def get_tutte_matrici_materiali(self) -> Dict[str, Any]:
        materiali_dict: Dict[str, Any] = {}
        try:
            gm = self.gestione_materiali
            ui = getattr(gm, 'ui', None)

            # mapping noto (manteniamo compatibilità)
            if ui is not None:
                known = [
                    ('Calcestruzzo', getattr(ui, 'calcestruzzo', None), 'generatore_matrice_calcestruzzo'),
                    ('Acciaio per barre', getattr(ui, 'acciaio_barre', None), 'generatore_matrice_acciaiobarre'),
                    ('Acciaio per profili', getattr(ui, 'acciaio_profili', None), 'generatore_matrice_acciaioprofili'),
                    ('FRP', getattr(ui, 'frp', None), 'generatore_matrice_frp'),
                ]
                for nome, obj, meth in known:
                    if obj is not None and hasattr(obj, meth):
                        try:
                            matrice = getattr(obj, meth)()
                            if matrice is not None:
                                materiali_dict[nome] = matrice
                        except Exception:
                            traceback.print_exc()

            # stackedWidget_materiali
            if ui is not None:
                stack = getattr(ui, 'stackedWidget_materiali', None)
                if stack is not None:
                    for idx in range(stack.count()):
                        page = stack.widget(idx)
                        nome_materiale = None
                        matrice = None

                        for meth in (
                            'generatore_matrice_calcestruzzo',
                            'generatore_matrice_acciaiobarre',
                            'generatore_matrice_acciaioprofili',
                            'generatore_matrice_frp',
                            'generatore_matrice_diagramma',
                            'generatore_matrice',
                            'generatore_matrice_diagram'
                        ):
                            if hasattr(page, meth) and callable(getattr(page, meth)):
                                try:
                                    matrice = getattr(page, meth)()
                                    break
                                except Exception:
                                    traceback.print_exc()
                                    matrice = None

                        if matrice is None:
                            for attr in ('matrice', 'matrix', 'data', 'diagram_matrix'):
                                if hasattr(page, attr):
                                    try:
                                        candidate = getattr(page, attr)
                                        matrice = candidate() if callable(candidate) else candidate
                                        break
                                    except Exception:
                                        traceback.print_exc()
                                        matrice = None

                        if hasattr(page, 'nome_materiale'):
                            nome_materiale = getattr(page, 'nome_materiale')
                        elif hasattr(page, 'title'):
                            nome_materiale = getattr(page, 'title')

                        if not nome_materiale:
                            try:
                                btns = getattr(gm, 'btn_group_materiali', None)
                                if btns is not None:
                                    b = btns.buttons()
                                    if idx < len(b):
                                        nome_materiale = b[idx].text()
                                    elif idx - 1 >= 0 and idx - 1 < len(b):
                                        nome_materiale = b[idx - 1].text()
                            except Exception:
                                pass

                        if not nome_materiale:
                            mapping = {1: 'Calcestruzzo', 2: 'Acciaio per barre', 3: 'Acciaio per profili', 4: 'FRP'}
                            nome_materiale = mapping.get(idx, None)

                        if nome_materiale and matrice is not None:
                            materiali_dict[str(nome_materiale)] = matrice

            if not materiali_dict:
                print("[Verifica] Trovati 0 materiali nello stackedWidget_materiali")

            return materiali_dict
        except Exception:
            traceback.print_exc()
            return materiali_dict

    # ------------------ DEBUG ------------------
    def debug_scan(self):
        print("--- DEBUG SCAN Verifica ---")
        try:
            # sections
            sm = None
            if hasattr(self.gestione_sezioni, 'section_manager'):
                sm = getattr(self.gestione_sezioni, 'section_manager')
            elif hasattr(self.gestione_sezioni, 'sections'):
                sm = self.gestione_sezioni
            if sm is None:
                print("SectionManager: NON trovato su gestione_sezioni (attributo 'section_manager').")
            else:
                secs = getattr(sm, 'sections', None)
                print(f"SectionManager: trovato, sections count = {len(secs) if secs is not None else 'None'}")

            ui_s = getattr(self.gestione_sezioni, 'ui', None)
            if ui_s is None:
                print("gestione_sezioni.ui: NON trovato")
            else:
                has_stack = any(hasattr(ui_s, n) for n in ('stackedWidget_sezioni', 'stackedWidget', 'stackedWidget_main'))
                print(f"gestione_sezioni.ui: found, has stackedWidget_sezioni-like = {has_stack}")
                if has_stack:
                    stack = getattr(ui_s, 'stackedWidget_sezioni', None) or getattr(ui_s, 'stackedWidget', None) or getattr(ui_s, 'stackedWidget_main', None)
                    if stack is not None:
                        print(f"  stackedWidget pages = {stack.count()}")
                        # info sulle tabelle
                        tvs = ('tableView_sezione','tableView_barre','tableView_rinforzi')
                        for tv in tvs:
                            print(f"   has {tv} = {hasattr(ui_s, tv)}")

            # materials
            gm = self.gestione_materiali
            ui_m = getattr(gm, 'ui', None)
            if ui_m is None:
                print("gestione_materiali.ui: NON trovato")
            else:
                stack_m = getattr(ui_m, 'stackedWidget_materiali', None)
                print(f"stackedWidget_materiali: {'trovato, pages=' + str(stack_m.count()) if stack_m is not None else 'NON trovato'}")
                btns = getattr(gm, 'btn_group_materiali', None)
                if btns is None:
                    print("btn_group_materiali: NON trovato su gestione_materiali")
                else:
                    try:
                        b = btns.buttons()
                        print(f"btn_group_materiali: trovato, bottoni = {len(b)}; texts = {[bb.text() for bb in b]}")
                    except Exception:
                        print("btn_group_materiali: trovato ma errore nel leggere i bottoni")
        except Exception:
            traceback.print_exc()
        print("--- END DEBUG ---")
