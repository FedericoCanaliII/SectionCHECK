from PyQt5.QtCore import Qt

# ---------------- MoveTool ----------------
class MoveTool():
    """Tool per la traslazione (pan) della vista:
       quando attivo, trascinando con il tasto sinistro si trasla lo spazio."""
    def __init__(self):
        self.dragging = False
        self.last_pos = None

    def on_activate(self, widget):
        try:
            widget.setCursor(Qt.OpenHandCursor)
        except Exception:
            pass

    def on_deactivate(self, widget):
        try:
            widget.unsetCursor()
        except Exception:
            pass
        self.dragging = False
        self.last_pos = None

    def on_mouse_press(self, widget, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()
            event.accept()
            return True
        return False

    def on_mouse_move(self, widget, event):
        if self.dragging and self.last_pos is not None:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            widget.pan_x -= dx * (2 * widget.data_range_x * widget.zoom) / max(1, widget.width())
            widget.pan_y += dy * (2 * widget.data_range_y * widget.zoom) / max(1, widget.height())
            self.last_pos = event.pos()
            widget.update()
            event.accept()
            return True
        widget.cursor_pos = event.pos()
        return False

    def on_mouse_release(self, widget, event):
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.last_pos = None
            event.accept()
            return True
        return False