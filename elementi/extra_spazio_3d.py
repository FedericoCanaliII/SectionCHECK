"""
extra_spazio_3d.py – Extended 3D OpenGL widget for the carichi/vincoli workspace.

Extends ElementiSpazio3D to:
  • Display the reference structural element (read-only, super-transparent gray)
  • Render vincoli as green parallelepipedi
  • Render carichi as purple parallelepipedi
  • Reference objects are never selectable
"""

from OpenGL.GL import (
    glColor4f, glDepthMask, GL_FALSE, GL_TRUE,
    glEnable, glDisable, GL_DEPTH_TEST,
    glLineWidth, glPushMatrix, glPopMatrix,
    glTranslatef, glRotatef,
)

from .elementi_spazio_3d import ElementiSpazio3D

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# Vincolo (green)
_VINCOLO_FILL = (0.20, 0.75, 0.30, 0.22)
_VINCOLO_EDGE = (0.30, 0.95, 0.45, 0.85)
_VINCOLO_GLOW = (0.30, 0.95, 0.45, 0.35)

# Carico (purple)
_CARICO_FILL  = (0.55, 0.15, 0.85, 0.22)
_CARICO_EDGE  = (0.75, 0.30, 1.00, 0.85)
_CARICO_GLOW  = (0.75, 0.30, 1.00, 0.35)

# Selected edge (bright yellow, same for both types)
_SEL_EDGE_CV  = (1.00, 0.90, 0.20, 1.00)

# Reference element (very transparent gray – read-only ghost)
_RIF_FILL     = (0.55, 0.55, 0.55, 0.10)
_RIF_EDGE     = (0.70, 0.70, 0.70, 0.30)


# ---------------------------------------------------------------------------
# ExtraSpazio3D
# ---------------------------------------------------------------------------

class ExtraSpazio3D(ElementiSpazio3D):
    """
    3D workspace for designing carichi/vincoli on top of a reference element.

    _oggetti      → carichi/vincoli (CaricoVincolo), fully interactive
    _oggetti_rif  → reference Elemento's Oggetto3D list, displayed ghost-only
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._oggetti_rif: list = []

    # ------------------------------------------------------------------ public

    def aggiorna_rif(self, lista: list):
        """Set the reference element objects (read-only ghost display)."""
        self._oggetti_rif = lista
        self.update()

    # ------------------------------------------------------------------ render

    def _disegna_oggetti(self):
        """Draw reference objects first (ghost), then carichi/vincoli."""
        for obj in self._oggetti_rif:
            if obj.visibile:
                self._disegna_oggetto_rif(obj)

        # carichi/vincoli with normal selection support
        super()._disegna_oggetti()

    def _disegna_oggetto_rif(self, obj):
        """Render a reference object as a ghost (non-interactive)."""
        glPushMatrix()
        glTranslatef(*obj.posizione)
        glRotatef(obj.rotazione[0], 1, 0, 0)
        glRotatef(obj.rotazione[1], 0, 1, 0)
        glRotatef(obj.rotazione[2], 0, 0, 1)

        if obj.tipo == "parallelepipedo":
            glDepthMask(GL_FALSE)
            self._draw_box_fill(obj, _RIF_FILL)
            glDepthMask(GL_TRUE)
            glDisable(GL_DEPTH_TEST)
            glColor4f(*_RIF_EDGE)
            glLineWidth(1.0)
            self._draw_box_edges(obj)
            glEnable(GL_DEPTH_TEST)

        elif obj.tipo == "cilindro":
            glDepthMask(GL_FALSE)
            self._draw_cyl_fill(obj, _RIF_FILL)
            glDepthMask(GL_TRUE)
            glDisable(GL_DEPTH_TEST)
            glColor4f(*_RIF_EDGE)
            glLineWidth(1.0)
            self._draw_cyl_edges(obj)
            glEnable(GL_DEPTH_TEST)

        elif obj.tipo == "sfera":
            glDepthMask(GL_FALSE)
            self._draw_sph_fill(obj, _RIF_FILL)
            glDepthMask(GL_TRUE)
            glDisable(GL_DEPTH_TEST)
            glColor4f(*_RIF_EDGE)
            glLineWidth(1.0)
            self._draw_sph_edges(obj)
            glEnable(GL_DEPTH_TEST)

        glLineWidth(1.0)
        glPopMatrix()

    def _draw_structural(self, obj, is_sel: bool, glow_pass: bool):
        """
        Override: apply type-specific colors for vincoli (green) and carichi (purple).
        Falls back to parent behaviour for non-CV objects (shouldn't happen here,
        but kept for safety).
        """
        sottotipo = getattr(obj, "sottotipo", None)
        if sottotipo is None:
            super()._draw_structural(obj, is_sel, glow_pass)
            return

        if sottotipo == "vincolo":
            fill = _VINCOLO_GLOW if glow_pass else _VINCOLO_FILL
            edge = _SEL_EDGE_CV  if is_sel   else _VINCOLO_EDGE
        else:  # carico
            fill = _CARICO_GLOW  if glow_pass else _CARICO_FILL
            edge = _SEL_EDGE_CV  if is_sel   else _CARICO_EDGE

        line_w = 3.0 if is_sel else 2.0

        # Filled body
        glDepthMask(GL_FALSE)
        self._draw_box_fill(obj, fill)
        glDepthMask(GL_TRUE)

        # Wireframe edges
        if not glow_pass:
            glDisable(GL_DEPTH_TEST)
            glColor4f(*edge)
            glLineWidth(line_w)
            self._draw_box_edges(obj)
            glLineWidth(1.0)
            glEnable(GL_DEPTH_TEST)
