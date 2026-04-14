"""
tool_base.py – Basic selection tool (no gizmo).

Left-click on object → select it
Left-click on empty  → deselect object
"""

from PyQt5.QtCore import Qt
from .base_tool_3d import BaseTool3D


class ToolBase(BaseTool3D):
    name = "base"

    def on_activate(self, spazio):
        spazio.setCursor(Qt.ArrowCursor)

    def on_mouse_press(self, spazio, event) -> bool:
        if event.button() != Qt.LeftButton:
            return False
            
        picked = spazio._pick_at(event.x(), event.y())
        
        # IL FIX È QUI: Applichiamo il risultato qualunque esso sia!
        # Se 'picked' è un ID seleziona l'oggetto. Se è -1, lo disseleziona.
        spazio.set_id_selezionato(picked)
        
        return True

    def reset(self):
        pass