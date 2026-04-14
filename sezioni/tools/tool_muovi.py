"""tool_muovi.py – Tool PAN (muovi la vista)."""
from PyQt5.QtCore import Qt
from .base_tool import BaseTool

class ToolMuovi(BaseTool):
    def __init__(self):
        self._dragging = False; self._start_pos = None
        self._start_px = 0.0;  self._start_py = 0.0

    def on_activate(self, widget):  widget.setCursor(Qt.OpenHandCursor)
    def on_deactivate(self, widget): widget.setCursor(Qt.ArrowCursor); self._dragging = False

    def on_mouse_press(self, widget, event) -> bool:
        if event.button() == Qt.LeftButton:
            self._dragging = True; self._start_pos = event.pos()
            self._start_px = widget.pan_x; self._start_py = widget.pan_y
            widget.setCursor(Qt.ClosedHandCursor); return True
        return False

    def on_mouse_move(self, widget, event) -> bool:
        if self._dragging and self._start_pos:
            dx = event.x() - self._start_pos.x(); dy = event.y() - self._start_pos.y()
            w, h = max(1, widget.width()), max(1, widget.height())
            mn_x, mx_x, mn_y, mx_y = widget._world_bounds()
            widget.pan_x = self._start_px - dx * (mx_x - mn_x) / w
            widget.pan_y = self._start_py + dy * (mx_y - mn_y) / h
            return True
        return False

    def on_mouse_release(self, widget, event) -> bool:
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False; widget.setCursor(Qt.OpenHandCursor); return True
        return False
