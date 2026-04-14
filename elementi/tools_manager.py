"""
tools_manager.py – Binds UI buttons to tool instances for the Elements module.
"""
from PyQt5.QtWidgets import QButtonGroup

from .tools import ToolBase, ToolMuovi, ToolRuota, ToolModifica


class ToolsManager:
    """
    Wires the tool buttons in the Elements panel to the four tool instances
    and keeps the QButtonGroup exclusive/checkable contract.
    """

    def __init__(self, ui, spazio):
        self._ui     = ui
        self._spazio = spazio

        self._tools = {
            "base":     ToolBase(),
            "muovi":    ToolMuovi(),
            "ruota":    ToolRuota(),
            "modifica": ToolModifica(),
        }

        self._setup_tool_buttons()
        self._setup_view_buttons()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_tool_buttons(self):
        """Bind base / muovi / ruota / modifica as exclusive checkable buttons."""
        mappa = {
            self._ui.elemento_btn_base:     "base",
            self._ui.elemento_btn_muovi:    "muovi",
            self._ui.elemento_btn_ruota:    "ruota",
            self._ui.elemento_btn_modifica: "modifica",
        }
        group = QButtonGroup(self._spazio)
        group.setExclusive(True)

        for btn, nome in mappa.items():
            btn.setCheckable(True)
            group.addButton(btn)
            btn.clicked.connect(lambda _c, n=nome: self._attiva_tool(n))

        # Activate base tool by default
        self._ui.elemento_btn_base.setChecked(True)
        self._attiva_tool("base")

        self._btn_group_tools = group

    def _setup_view_buttons(self):
        """Bind the four orthogonal/perspective view buttons as exclusive."""
        mappa = {
            self._ui.elemento_btn_vista_3d: "3d",
            self._ui.elemento_btn_vista_x:  "x",
            self._ui.elemento_btn_vista_y:  "y",
            self._ui.elemento_btn_vista_z:  "z",
        }
        group = QButtonGroup(self._spazio)
        group.setExclusive(True)

        for btn, preset in mappa.items():
            btn.setCheckable(True)
            group.addButton(btn)
            btn.clicked.connect(lambda _c, p=preset: self._spazio.imposta_vista(p))

        self._ui.elemento_btn_vista_3d.setChecked(True)
        self._btn_group_views = group

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _attiva_tool(self, nome: str):
        tool = self._tools.get(nome)
        if tool:
            self._spazio.set_active_tool(tool)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_tool(self, nome: str):
        return self._tools.get(nome)

    def deseleziona_tool_esclusivi(self):
        """Deselect all exclusive tool buttons (e.g. when switching element)."""
        for btn in self._btn_group_tools.buttons():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self._spazio.set_active_tool(None)
