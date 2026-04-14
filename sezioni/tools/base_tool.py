"""
base_tool.py – Classe base con supporto proprietà.
"""
from PyQt5.QtCore import Qt, QRectF


class BaseTool:
    def on_activate(self, widget):
        pass
    def on_deactivate(self, widget):
        widget.setCursor(Qt.ArrowCursor)
        self.reset()

    def on_mouse_press(self, widget, event) -> bool: return False
    def on_mouse_move(self, widget, event) -> bool:  return False
    def on_mouse_release(self, widget, event) -> bool: return False
    def on_wheel(self, widget, event) -> bool: return False

    def on_key_press(self, widget, event) -> bool:
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter): return self.confirm(widget)
        if key == Qt.Key_Escape: return self.cancel(widget)
        return False

    @property
    def is_pending(self) -> bool: return False

    def confirm(self, widget) -> bool: return False
    def cancel(self, widget) -> bool:
        self.reset(); widget.update(); return True

    # --- Proprietà per lineEdit esterno ---
    def get_properties_text(self) -> str:
        return ""

    def apply_properties_text(self, text: str):
        pass

    def draw_gl(self, widget): pass
    def draw_painter(self, widget, painter): pass
    def reset(self): pass
