from PyQt5.QtCore import Qt

# ---------------- tools.py (updated) ----------------
class Tool:
    """Classe base per i tool. Override i metodi che ti servono."""
    def on_activate(self, widget):
        """Chiamato quando il tool viene attivato."""
        pass

    def on_deactivate(self, widget):
        """Chiamato quando il tool viene disattivato."""
        pass

    def on_mouse_press(self, widget, event):
        """Return True se l'evento è stato consumato."""
        return False

    def on_mouse_move(self, widget, event):
        return False

    def on_mouse_release(self, widget, event):
        return False

    def on_wheel(self, widget, event):
        return False

    # optional drawing hooks
    def draw_gl(self, widget):
        """Draw persistent world-space geometry using OpenGL.
        Override in concrete tools."""
        pass

    def draw_painter(self, widget, painter):
        """Draw screen-space overlays (preview) using QPainter."""
        pass

# ---------------- Skeletons for other tools ----------------
class SelectionTool(Tool):
    def on_activate(self, widget):
        widget.setCursor(Qt.ArrowCursor)
