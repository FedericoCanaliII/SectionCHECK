"""
base_tool_3d.py – Abstract base class for all 3D element tools.
"""
from PyQt5.QtCore import Qt


class BaseTool3D:
    """
    Abstract base for tools operating in the 3D element workspace.

    Sub-classes override the event handlers they care about and return True
    to signal that the event was consumed (suppressing default behaviour).
    """

    name: str = "base"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_activate(self, spazio):
        """Called when this tool becomes the active tool."""

    def on_deactivate(self, spazio):
        """Called when another tool replaces this one."""
        spazio.setCursor(Qt.ArrowCursor)
        self.reset()

    # ------------------------------------------------------------------
    # Mouse / keyboard events
    # ------------------------------------------------------------------

    def on_mouse_press(self, spazio, event) -> bool:
        """Return True if the event was consumed."""
        return False

    def on_mouse_move(self, spazio, event) -> bool:
        return False

    def on_mouse_release(self, spazio, event) -> bool:
        return False

    def on_wheel(self, spazio, event) -> bool:
        return False

    # ------------------------------------------------------------------
    # GL overlay (called inside paintGL after the scene is drawn)
    # ------------------------------------------------------------------

    def draw_overlay(self, spazio):
        """Draw tool-specific OpenGL overlays (gizmos, vertex handles …)."""

    def on_hover(self, spazio, event):
        """Called on every mouse move (no button pressed or during drag).
        Update highlight state and call spazio.update() if visual changes."""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self):
        """Reset transient state (called on deactivate and on cancel)."""

    # ------------------------------------------------------------------
    # Gizmo helpers (shared across move/rotate tools)
    # ------------------------------------------------------------------

    @staticmethod
    def _world_to_screen(wx, wy, wz, spazio):
        """Project a world-space point to screen-space (x, y, depth) using Numpy."""
        if getattr(spazio, '_gl_ready', False) is False:
            return None, None, None
            
        try:
            import numpy as np
            
            # 1. Estraiamo le matrici come abbiamo fatto per il raycasting
            mm = np.array(spazio._gl_model_mat, dtype=float).reshape(4, 4).T
            pm = np.array(spazio._gl_proj_mat, dtype=float).reshape(4, 4).T
            vp = spazio._gl_viewport
            
            # 2. Punto in coordinate omogenee 4D
            p_world = np.array([wx, wy, wz, 1.0])
            
            # 3. Moltiplichiamo per ModelView e Projection
            p_clip = pm @ (mm @ p_world)
            
            if p_clip[3] == 0.0:
                return None, None, None
                
            # 4. Spazio Normalizzato (NDC) [-1, 1]
            p_ndc = p_clip[:3] / p_clip[3]
            
            # 5. Mappatura al Viewport (Pixel Fisici)
            win_x = vp[0] + (p_ndc[0] + 1.0) * (vp[2] / 2.0)
            win_y = vp[1] + (p_ndc[1] + 1.0) * (vp[3] / 2.0)
            win_z = (p_ndc[2] + 1.0) / 2.0
            
            # 6. Conversione alle coordinate Qt (Pixel Logici)
            dpr = spazio.devicePixelRatioF()
            sx = win_x / dpr
            
            # OpenGL ha lo zero in basso a sinistra, Qt in alto a sinistra. Invertiamo la Y!
            sy = (vp[3] - win_y) / dpr 
            
            return sx, sy, win_z
            
        except Exception as e:
            print(f"\n--- ERRORE IN WORLD_TO_SCREEN ---\n{e}\n")
            return None, None, None

    @staticmethod
    def _axis_drag_delta(screen_dx, screen_dy,
                         axis_x, axis_y, axis_z,
                         ref_x, ref_y, ref_z, spazio) -> float:
        """
        Convert a screen-space mouse delta (pixels) into a world-space
        movement amount along the given axis direction.
        """
        if getattr(spazio, '_gl_ready', False) is False:
            return 0.0

        try:
            import numpy as np

            # 1. Estraiamo le matrici (stesso metodo solido usato altrove)
            mm = np.array(spazio._gl_model_mat, dtype=float).reshape(4, 4).T
            pm = np.array(spazio._gl_proj_mat, dtype=float).reshape(4, 4).T
            vp = spazio._gl_viewport
            dpr = spazio.devicePixelRatioF()

            def project_point(wx, wy, wz):
                p_world = np.array([wx, wy, wz, 1.0])
                p_clip = pm @ (mm @ p_world)
                if p_clip[3] == 0.0: return None, None
                
                p_ndc = p_clip[:3] / p_clip[3]
                win_x = vp[0] + (p_ndc[0] + 1.0) * (vp[2] / 2.0)
                win_y = vp[1] + (p_ndc[1] + 1.0) * (vp[3] / 2.0)
                
                sx = win_x / dpr
                sy = (vp[3] - win_y) / dpr 
                return sx, sy

            # 2. Proiettiamo la base del gizmo e la punta della freccia a schermo
            sx0, sy0 = project_point(ref_x, ref_y, ref_z)
            sx1, sy1 = project_point(ref_x + axis_x, ref_y + axis_y, ref_z + axis_z)

            if sx0 is None or sx1 is None:
                return 0.0

            # 3. Vettore direzionale della freccia sullo schermo (in pixel)
            ax = sx1 - sx0
            ay = sy1 - sy0
            screen_len_sq = ax**2 + ay**2
            
            if screen_len_sq < 1.0:
                return 0.0

            # 4. Proiezione del movimento del mouse lungo il vettore della freccia
            dot = screen_dx * ax + screen_dy * ay
            
            # IL FIX MATEMATICO: 
            # Dividendo per la lunghezza al quadrato, convertiamo i pixel dello 
            # schermo in proporzioni esatte dello spazio 3D reale!
            return dot / screen_len_sq
            
        except Exception as e:
            print(f"\n--- ERRORE NEL DRAG GIZMO ---\n{e}\n")
            return 0.0
