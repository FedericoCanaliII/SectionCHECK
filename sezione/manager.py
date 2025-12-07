from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

# PyQt imports (per widget)
from PyQt5.QtWidgets import QLineEdit, QShortcut
from PyQt5.QtGui import QPainter, QColor, QKeySequence
from PyQt5.QtCore import Qt, QRect

# OpenGL imports (opzionale)
try:
    from OpenGL.GL import *
    _HAS_GL = True
except Exception:
    _HAS_GL = False


@dataclass
class SectionModel:
    id: int
    name: str
    rects: List[Dict[str, Any]] = field(default_factory=list)
    circles: List[Dict[str, Any]] = field(default_factory=list)
    polys: List[Dict[str, Any]] = field(default_factory=list)
    bars: List[Dict[str, Any]] = field(default_factory=list)
    staffe: List[Dict[str, Any]] = field(default_factory=list)  # <-- lista staffe
    reinforcements: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class SectionManager:
    """Gestisce più SectionModel e fornisce API per tool/widget."""

    def __init__(self):
        self.sections: List[SectionModel] = []
        self.current_index: Optional[int] = None
        self.widget = None
        self.tools: Dict[str, Any] = {}
        self.on_update = None

    # ---------- Sections CRUD ----------
    def create_section(self, name: str = None) -> int:
        idx = len(self.sections) + 1
        if name is None:
            name = f"Sezione {idx}"
        s = SectionModel(id=idx, name=name)
        self.sections.append(s)
        if self.current_index is None:
            self.switch_section(0)
        return len(self.sections) - 1

    def remove_section(self, index: int):
        if not (0 <= index < len(self.sections)):
            return
        self.sections.pop(index)
        for i, s in enumerate(self.sections):
            s.id = i + 1
        if len(self.sections) == 0:
            self.current_index = None
        else:
            self.current_index = min(index, len(self.sections) - 1)
            self.switch_section(self.current_index)

    def rename_section(self, index: int, name: str):
        if 0 <= index < len(self.sections):
            self.sections[index].name = name

    def get_section(self, index: int) -> Optional[SectionModel]:
        if 0 <= index < len(self.sections):
            return self.sections[index]
        return None

    # ---------- Widget/Tool Binding ----------
    def bind_widget(self, widget):
        self.widget = widget

    def bind_tools(self, tools: Dict[str, Any]):
        """Collega i tool (devono avere set_confirmed_list o simili)."""
        self.tools = tools
        if self.current_index is not None:
            self._link_tools_to_section(self.current_index)

    def _link_tools_to_section(self, index: int):
        """Aggancia le liste della sezione corrente ai tool corrispondenti."""
        if not (0 <= index < len(self.sections)):
            return
        sec = self.sections[index]

        mapping = {
            'rect': 'rects',
            'circle': 'circles',
            'poly': 'polys',
            'bar': 'bars',
            'staffe': 'staffe',               # <-- aggiunto supporto staffe
            'reinforcement': 'reinforcements'
        }

        for name, tool in self.tools.items():
            key = mapping.get(name)
            if key is None:
                continue

            # assegna la lista alla proprietà confermata del tool
            lst = getattr(sec, key, [])
            if hasattr(tool, 'set_confirmed_list'):
                try:
                    tool.set_confirmed_list(lst)
                except Exception:
                    pass
            else:
                try:
                    setattr(tool, 'confirmed_' + key, lst)
                except Exception:
                    pass

            # fallback per tool che hanno metodi dedicati
            try:
                if key == 'rects' and hasattr(tool, 'set_rects'):
                    tool.set_rects(lst)
                if key == 'polys' and hasattr(tool, 'set_polys'):
                    tool.set_polys(lst)
                if key == 'circles' and hasattr(tool, 'set_circles'):
                    tool.set_circles(lst)
                if key == 'bars' and hasattr(tool, 'set_bars'):
                    tool.set_bars(lst)
                if key == 'staffe' and hasattr(tool, 'set_staffes'):
                    tool.set_staffes(lst)
                if key == 'reinforcements' and hasattr(tool, 'set_reinforcements'):
                    tool.set_reinforcements(lst)
            except Exception:
                pass

        # aggiorna widget con strumenti persistenti
        if self.widget is not None:
            persistent_tools = [self.tools.get(k) for k in ['rect','circle','poly','bar','staffe','reinforcement','select','move'] if k in self.tools]
            self.widget._tools = persistent_tools
            try:
                self.widget.update()
            except Exception:
                pass

    # ---------- Switching Sections ----------
    def switch_section(self, index: int):
        if not (0 <= index < len(self.sections)):
            return
        self.current_index = index
        self._link_tools_to_section(index)

    # ---------- Adding Shapes ----------
    def add_rect_to_section(self, index: int, rect_entry: Dict[str, Any]):
        if not (0 <= index < len(self.sections)):
            return
        s = self.sections[index]
        s.rects.append(rect_entry)
        if index == self.current_index and self.widget is not None:
            self.widget.update()
        self._trigger_update()

    def attach_rects_bulk(self, index: int, rects: List[Dict[str, Any]]):
        if not (0 <= index < len(self.sections)):
            return
        s = self.sections[index]
        s.rects.extend(rects)
        if index == self.current_index and self.widget is not None:
            self.widget.update()
        self._trigger_update()

    # Rimuove un elemento generico da una lista della sezione
    def remove_item_from_section(self, section_index: int, list_name: str, item_index: int):
        if section_index is None or not (0 <= section_index < len(self.sections)):
            return
        sec = self.sections[section_index]
        lst = getattr(sec, list_name, None)
        if not isinstance(lst, list) or not (0 <= item_index < len(lst)):
            return
        lst.pop(item_index)
        if section_index == self.current_index and self.widget is not None:
            try: self.widget.update()
            except Exception: pass
        self._trigger_update()

    # ---------- Serialization ----------
    def save(self, path: str):
        serial = []
        for s in self.sections:
            serial.append({
                'id': s.id,
                'name': s.name,
                'rects': s.rects,
                'circles': s.circles,
                'polys': s.polys,
                'bars': s.bars,
                'staffe': s.staffe,
                'reinforcements': s.reinforcements,
                'meta': s.meta
            })
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serial, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.sections = []
        for s in data:
            sec = SectionModel(
                id=s.get('id', len(self.sections)+1),
                name=s.get('name', f"Sezione {len(self.sections)+1}"),
                rects=s.get('rects', []),
                circles=s.get('circles', []),
                polys=s.get('polys', []),
                bars=s.get('bars', []),
                staffe=s.get('staffe', []),
                reinforcements=s.get('reinforcements', []),
                meta=s.get('meta', {})
            )
            self.sections.append(sec)
        if len(self.sections) > 0:
            self.switch_section(0)

    # ---------- Utility ----------
    def _trigger_update(self):
        if callable(self.on_update):
            try:
                self.on_update()
            except Exception:
                pass


# ---------------- Utility: canonicalizer rettangoli ----------------
def to_canonical_rect(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    if 'v1' in obj and 'v2' in obj:
        try:
            v1 = tuple(obj['v1'])
            v2 = tuple(obj['v2'])
            return {'v1':(float(v1[0]),float(v1[1])),'v2':(float(v2[0]),float(v2[1])),'name':obj.get('name','')}
        except Exception: return None
    if all(k in obj for k in ('x1','y1','x2','y2')):
        try:
            return {'v1':(float(obj['x1']),float(obj['y1'])),'v2':(float(obj['x2']),float(obj['y2'])),'name':obj.get('name','')}
        except Exception: return None
    if all(k in obj for k in ('cx','cy','w','h')):
        try:
            cx,cy,w,h = float(obj['cx']),float(obj['cy']),float(obj['w']),float(obj['h'])
            return {'v1':(cx - w/2, cy - h/2),'v2':(cx + w/2, cy + h/2),'name':obj.get('name','')}
        except Exception: return None
    pts = obj.get('pts') or obj.get('points') or obj.get('vertices')
    if pts and len(pts) >= 2:
        try:
            v1 = tuple(pts[0])
            v2 = tuple(pts[1])
            return {'v1':(float(v1[0]),float(v1[1])),'v2':(float(v2[0]),float(v2[1])),'name':obj.get('name','')}
        except Exception: return None
    return None

def convert_bulk_rects(raw_list: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out = []
    for o in raw_list:
        try:
            c = to_canonical_rect(o)
            if c is not None:
                out.append(c)
        except Exception:
            pass
    return out
