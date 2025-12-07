from typing import Any, List, Dict, Tuple, Optional
import pprint
import math

# Manteniamo la classe Material invariata (non è la causa del problema)
Point = Tuple[float, float]

class Material:
    def __init__(self, matrice: List[Tuple[str, float, float]], nome: str = "") -> None:
        self.nome = nome
        self.intervalli: List[Dict[str, Any]] = []
        self.ult_compression = 0.0
        self.ult_tension = 0.0

        all_strains: List[float] = []
        for i, intervallo in enumerate(matrice):
            if isinstance(intervallo, dict):
                expr_raw = intervallo.get('expr') or intervallo.get('f') or ''
                a = intervallo.get('a') or intervallo.get('low') or intervallo.get('start') or 0.0
                b = intervallo.get('b') or intervallo.get('high') or intervallo.get('end') or 0.0
            else:
                expr_raw, a, b = intervallo

            low = float(min(a, b))
            high = float(max(a, b))
            expr = str(expr_raw).replace('^', '**')
            compiled = compile(expr, f'<material_expr_{i}>', 'eval')

            self.intervalli.append({'expr': compiled, 'low': low, 'high': high})
            all_strains.extend([low, high])

        if all_strains:
            self.ult_compression = min(all_strains)
            self.ult_tension = max(all_strains)

        self._safe_globals = {'__builtins__': None}
        math_funcs = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        self._safe_locals = {'math': math, **math_funcs}

    def sigma(self, eps: float) -> float:
        for intervallo in self.intervalli:
            if intervallo['low'] <= eps <= intervallo['high']:
                try:
                    return float(eval(intervallo['expr'], self._safe_globals, {'x': eps, **self._safe_locals}))
                except Exception:
                    return 0.0
        return 0.0


# -------------------- BeamValori Corretta --------------------
class BeamValori:
    def __init__(self, section_manager: Any, ui: Optional[Any] = None, gestione_materiali: Optional[Any] = None, print_to_terminal: bool = True):
        self.section_manager = section_manager
        self.ui = ui
        self.print_to_terminal = print_to_terminal
        self.gestione_materiali = gestione_materiali

    # -------------------- PUBLIC --------------------
    def generate_matrices(self, section_index: Optional[int] = None):
        sm = self.section_manager
        if sm is None:
            raise RuntimeError("Section manager non fornito")

        if section_index is None:
            section_index = getattr(sm, 'current_index', None)

        sec = sm.get_section(section_index) if section_index is not None else None
        if sec is None:
            if self.print_to_terminal:
                print("Nessuna sezione attiva o sezione non trovata")
            return [], []

        objs = self._collect_objects_from_section(sec)
        # Costruiamo prima i materiali per averli pronti
        materials_matrix = self._build_materials_matrix(objs)
        objects_matrix = self._build_objects_matrix(objs)

        if self.print_to_terminal:
            print(f"\n--- MATRICE MATERIALI GENERATA ({len(materials_matrix)}) ---")
            pprint.pprint(materials_matrix)
            print("---------------------------------------------------\n")

        if self.print_to_terminal:
            print (f"\n---MATRICE OGGETTI GENERATA ({len(objects_matrix)}) ---")
            pprint.pprint(objects_matrix)
            print("---------------------------------------------------\n")

        return materials_matrix, objects_matrix

    def attach_to_button(self, button, section_index_getter: Optional[Any] = None):
        try:
            if section_index_getter is None:
                button.clicked.connect(lambda: self.generate_matrices())
            else:
                button.clicked.connect(lambda: self.generate_matrices(section_index_getter()))
        except Exception:
            try:
                button.connect(lambda: self.generate_matrices())
            except Exception:
                pass

    # -------------------- MATERIAL RESOLVER --------------------
    def _resolve_material(self, raw):
        if raw is None: return ""
        if isinstance(raw, str): return raw.strip()
        if hasattr(raw, "currentText"):
            try:
                txt = raw.currentText()
                if txt: return str(txt).strip()
            except: pass
        if hasattr(raw, "text"):
            try:
                txt = raw.text()
                if txt: return str(txt).strip()
            except: pass
        for attr in ("name", "nome", "material_name", "text", "label"):
            if hasattr(raw, attr):
                try:
                    v = getattr(raw, attr)
                    v = v() if callable(v) else v
                    if v: return str(v).strip()
                except: pass
        if isinstance(raw, (tuple, list)) and raw:
            first = raw[0]
            if isinstance(first, str): return first.strip()
            return self._resolve_material(first)
        try:
            return str(raw).strip()
        except:
            return ""

    # -------------------- HELPER: Estrazione Nome da Pagina --------------------
    def _extract_page_name(self, page, idx, gm):
        """
        Cerca il nome del materiale in modo robusto:
        1. Attributi diretti della pagina.
        2. Cerca una QLineEdit figlia chiamata '...nome...' o '...name...'.
        3. Cerca nel ButtonGroup mappando ID o Indice (con correzione shift).
        """
        # 1. Attributi diretti
        if hasattr(page, 'nome_materiale') and page.nome_materiale:
            return str(page.nome_materiale)
        if hasattr(page, 'title') and page.title:
            return str(page.title)

        # 2. Scansione Widget Figli (QLineEdit)
        # Questo risolve il problema alla radice leggendo cosa c'è scritto nella casella di testo
        if hasattr(page, 'findChildren'):
            # Nota: Non importiamo QLineEdit per evitare dipendenze, controlliamo duck typing
            children = page.children() # Ottieni figli diretti o usa findChildren se importassi Qt
            # Se findChildren è disponibile e usabile senza tipo (PyQt a volte lo permette, altrimenti iteriamo)
            # Iteriamo genericamente
            try:
                # Metodo euristico: cerchiamo figli con metodo text() e nome simile a 'nome'
                all_children = page.findChildren(object) # Cerca tutti ricorsivamente
                for child in all_children:
                    if hasattr(child, 'text') and hasattr(child, 'objectName'):
                        name = child.objectName().lower()
                        # Se è un campo di input (esclude label se possibile, o controlliamo se è editabile)
                        if ('nome' in name or 'name' in name) and not 'label' in name:
                            val = child.text()
                            if val: return str(val).strip()
            except Exception:
                pass

        # 3. Fallback Pulsanti (Gestione Shift)
        if gm:
            btns = getattr(gm, 'btn_group_materiali', None)
            if btns:
                # A. Tentativo per ID (Il più preciso se settato)
                if hasattr(btns, 'button'):
                    btn = btns.button(idx)
                    if btn and btn.text():
                        return btn.text()
                
                # B. Tentativo per Lista (con correzione Shift)
                if hasattr(btns, 'buttons'):
                    b_list = btns.buttons()
                    # Il problema utente: Stack[i] prendeva Nome[i+1] (Lista[i]).
                    # Quindi proviamo prima i-1 se esiste.
                    if 0 <= idx - 1 < len(b_list):
                        t = b_list[idx - 1].text()
                        if t: return t
                    
                    # Poi proviamo i (ma spesso i è quello "dopo" se c'è offset)
                    if 0 <= idx < len(b_list):
                        t = b_list[idx].text()
                        if t: return t
        
        return None

    # -------------------- BUILD OBJECTS MATRIX --------------------
    def _default_material_for_type(self, t: str) -> str:
        if t in ("rect", "poly", "circle"): return "Calcestruzzo"
        if t in ("bar", "staffe"): return "Acciaio per barre"
        return ""

    def _build_objects_matrix(self, objs):
        rows = []
        for i, ent in enumerate(objs):
            t = ent["type"]
            d = ent["data"]
            name = d.get("name") or self._default_name(t, i)
            mat_raw = d.get("material") or d.get("mat")
            mat = self._resolve_material(mat_raw) or self._default_material_for_type(t)

            if t == "rect":
                v1, v2 = self._rect_points(d)
                rows.append([name, v1, v2, mat])
            elif t == "circle":
                center = self._circle_center(d)
                rp = self._circle_radius_point(d)
                rows.append([name, center, rp, mat])
            elif t == "poly":
                verts = self._poly_vertices(d)
                rows.append([name] + verts + [mat])
            elif t == "bar":
                center = self._bar_center(d)
                diam = d.get("diam") or d.get("diameter") or d.get("t") or d.get("spessore")
                rows.append([name, center, diam, mat])
            elif t == "staffe":
                pts = self._staffe_points(d)
                diam = d.get("diam") or d.get("diametro") or d.get("t")
                rows.append([name] + pts + [diam, mat])
            elif t == "reinforcement":
                pts = d.get("poly") or d.get("pts") or d.get("points")
                thickness = self._find_thickness(d)
                if pts:
                    rows.append([name] + pts + [thickness, mat])
                else:
                    bp1 = d.get("base_p1")
                    bp2 = d.get("base_p2")
                    rows.append([name, bp1, bp2, thickness, mat])
            else:
                rows.append([name, mat])
        return rows

    # -------------------- MATERIALS MATRIX (CORRETTA) --------------------
    def _build_materials_matrix(self, objs):
        materiali_dict: Dict[str, Any] = {}

        gm = getattr(self, 'gestione_materiali', None)
        ui = getattr(gm, 'ui', None) if gm else None

        # --- 1. generatori standard ---
        # Questi funzionavano bene, li manteniamo.
        if ui:
            known = [
                ('Calcestruzzo', getattr(ui, 'calcestruzzo', None), 'generatore_matrice_calcestruzzo'),
                ('Acciaio per barre', getattr(ui, 'acciaio_barre', None), 'generatore_matrice_acciaiobarre'),
                ('Acciaio per profili', getattr(ui, 'acciaio_profili', None), 'generatore_matrice_acciaioprofili'),
                ('FRP', getattr(ui, 'frp', None), 'generatore_matrice_frp'),
            ]
            for nome, obj, meth in known:
                if obj and hasattr(obj, meth):
                    try:
                        matrice = getattr(obj, meth)()
                        if matrice:
                            materiali_dict[nome] = matrice
                    except Exception:
                        pass

        # --- 2. stackedWidget_materiali (materiali creati dall’utente) ---
        if ui:
            stack = getattr(ui, 'stackedWidget_materiali', None)
            if stack:
                for idx in range(stack.count()):
                    page = stack.widget(idx)
                    matrice = None

                    # A. Troviamo la matrice (Logica invariata, funzionava)
                    for meth in (
                        'generatore_matrice_calcestruzzo', 'generatore_matrice_acciaiobarre',
                        'generatore_matrice_acciaioprofili', 'generatore_matrice_frp',
                        'generatore_matrice_diagramma', 'generatore_matrice', 'generatore_matrice_diagram'
                    ):
                        if hasattr(page, meth) and callable(getattr(page, meth)):
                            try:
                                matrice = getattr(page, meth)()
                                if matrice: break
                            except Exception: pass
                    
                    if matrice is None:
                        for attr in ('matrice', 'matrix', 'data', 'diagram_matrix'):
                            if hasattr(page, attr):
                                try:
                                    candidate = getattr(page, attr)
                                    matrice = candidate() if callable(candidate) else candidate
                                    if matrice: break
                                except Exception: pass

                    # B. Troviamo il nome (LOGICA CORRETTA QUI)
                    # Solo se abbiamo trovato una matrice valida procediamo a cercare il nome
                    if matrice is not None:
                        nome_materiale = self._extract_page_name(page, idx, gm)
                        
                        # Fallback numerico se proprio non troviamo il nome
                        if not nome_materiale:
                            mapping = {0: 'Calcestruzzo', 1: 'Acciaio per barre', 2: 'Acciaio per profili', 3: 'FRP'}
                            # Usiamo un nome univoco per evitare sovrascritture
                            nome_materiale = mapping.get(idx, f"Materiale_Custom_{idx}")

                        # Salvataggio
                        # Se il nome esiste già (es. Calcestruzzo), sovrascriviamo o ignoriamo?
                        # I custom dovrebbero avere nomi diversi. Se collidono con standard, meglio verificare.
                        if nome_materiale:
                            materiali_dict[str(nome_materiale)] = matrice

        # --- 3. Normalizzazione ---
        result: List[List[Any]] = []
        for nome, mat in materiali_dict.items():
            result.append([nome] + list(mat))

        return result

    # ---------------------------------------------------------------------
    # SHAPE HELPERS (INVARIATI)
    # ---------------------------------------------------------------------

    def _default_name(self, t: str, i: int) -> str:
        mapping = {
            "rect": f"RETTANGOLO {i+1}", "circle": f"CERCHIO {i+1}", "poly": f"POLIGONO {i+1}",
            "bar": f"B{i+1}", "staffe": f"S{i+1}", "reinforcement": f"RINFORZO {i+1}",
        }
        return mapping.get(t, f"OBJ {i+1}")

    def _rect_points(self, d):
        if "v1" in d and "v2" in d: return tuple(d["v1"]), tuple(d["v2"])
        if all(k in d for k in ("x1", "y1", "x2", "y2")): return (d["x1"], d["y1"]), (d["x2"], d["y2"])
        if all(k in d for k in ("cx", "cy", "w", "h")):
            cx, cy, w, h = d["cx"], d["cy"], d["w"], d["h"]
            return (cx - w / 2, cy - h / 2), (cx + w / 2, cy + h / 2)
        pts = d.get("pts") or d.get("points")
        if pts: return tuple(pts[0]), tuple(pts[1])
        return (0.0, 0.0), (0.0, 0.0)

    def _circle_center(self, d):
        if "center" in d: return tuple(d["center"])
        if "cx" in d and "cy" in d: return (d["cx"], d["cy"])
        return None

    def _circle_radius_point(self, d):
        if "radius_point" in d: return tuple(d["radius_point"])
        r = d.get("r") or d.get("radius")
        c = self._circle_center(d)
        if r is not None and c is not None: return (c[0] + r, c[1])
        return None

    def _poly_vertices(self, d):
        pts = d.get("vertices") or d.get("points") or d.get("pts") or d.get("poly")
        return [tuple(p) for p in pts] if pts else []

    def _bar_center(self, d):
        if "center" in d: return tuple(d["center"])
        if "cx" in d and "cy" in d: return (d["cx"], d["cy"])
        if "p1" in d and "p2" in d:
            p1 = tuple(d["p1"])
            p2 = tuple(d["p2"])
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        return None

    def _staffe_points(self, d):
        pts = d.get("points") or d.get("vertices") or d.get("pts")
        return [tuple(p) for p in pts] if pts else []

    def _find_thickness(self, d):
        for key in ("offset", "thickness", "t", "spessore", "diam", "diametro", "diameter"):
            if key in d: return d[key]
        return None

    def _collect_objects_from_section(self, sec: Any) -> List[Dict[str, Any]]:
        out = []
        def add(t, o):
            if isinstance(o, dict):
                data = dict(o)
                out.append({"type": t, "data": data})

        for r in getattr(sec, "rects", []) or []: add("rect", r)
        for c in getattr(sec, "circles", []) or []: add("circle", c)
        for p in getattr(sec, "polys", []) or []: add("poly", p)
        for b in getattr(sec, "bars", []) or []: add("bar", b)
        for s in getattr(sec, "staffe", []) or []: add("staffe", s)
        for r in getattr(sec, "reinforcements", []) or []: add("reinforcement", r)
        return out