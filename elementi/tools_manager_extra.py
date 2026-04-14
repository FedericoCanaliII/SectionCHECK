"""
tools_manager_extra.py – Binds the extra-panel tool buttons to tool instances.

Mirrors ToolsManager but uses extra_elemento_btn_* UI widgets so that
the extra 3D workspace (ExtraSpazio3D) has its own independent tool state.
"""

from PyQt5.QtWidgets import QButtonGroup

from .tools import ToolBase, ToolMuovi, ToolRuota, ToolModifica


class ToolsManagerExtra:
    """
    Wires extra_elemento_btn_{base,muovi,ruota,modifica} and
    extra_elemento_btn_vista_{3d,x,y,z} to the ExtraSpazio3D instance.
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
        mappa = {
            self._ui.extra_elemento_btn_base:     "base",
            self._ui.extra_elemento_btn_muovi:    "muovi",
            self._ui.extra_elemento_btn_ruota:    "ruota",
            self._ui.extra_elemento_btn_modifica: "modifica",
        }
        group = QButtonGroup(self._spazio)
        group.setExclusive(True)

        for btn, nome in mappa.items():
            btn.setCheckable(True)
            group.addButton(btn)
            btn.clicked.connect(lambda _c, n=nome: self._attiva_tool(n))

        self._ui.extra_elemento_btn_base.setChecked(True)
        self._attiva_tool("base")
        self._btn_group_tools = group

    def _setup_view_buttons(self):
        mappa = {
            self._ui.extra_elemento_btn_vista_3d: "3d",
            self._ui.extra_elemento_btn_vista_x:  "x",
            self._ui.extra_elemento_btn_vista_y:  "y",
            self._ui.extra_elemento_btn_vista_z:  "z",
        }
        group = QButtonGroup(self._spazio)
        group.setExclusive(True)

        for btn, preset in mappa.items():
            btn.setCheckable(True)
            group.addButton(btn)
            btn.clicked.connect(lambda _c, p=preset: self._spazio.imposta_vista(p))

        self._ui.extra_elemento_btn_vista_3d.setChecked(True)
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

    def reset_to_base(self):
        """Deselect all tool buttons and re-activate the base tool."""
        for btn in self._btn_group_tools.buttons():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self._ui.extra_elemento_btn_base.blockSignals(True)
        self._ui.extra_elemento_btn_base.setChecked(True)
        self._ui.extra_elemento_btn_base.blockSignals(False)
        self._attiva_tool("base")
