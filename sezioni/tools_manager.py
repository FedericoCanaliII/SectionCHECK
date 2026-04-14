"""
tools_manager.py – Binding pulsanti tool ↔ istanze tool + griglia/snap.
"""
from PyQt5.QtWidgets import QButtonGroup
from .tools import (ToolMuovi, ToolModifica, ToolRettangolo, ToolPoligono,
                     ToolCerchio, ToolBarra, ToolStaffa,
                     ToolForoRettangolo, ToolForoPoligono, ToolForoCerchio)


class ToolsManager:
    def __init__(self, ui, spazio):
        self._ui = ui; self._spazio = spazio
        self._tools = {
            "muovi": ToolMuovi(), "modifica": ToolModifica(),
            "rettangolo": ToolRettangolo(), "poligono": ToolPoligono(),
            "cerchio": ToolCerchio(), "barra": ToolBarra(diametro=16.0),
            "staffa": ToolStaffa(diametro=8.0),
            "foro_rettangolo": ToolForoRettangolo(),
            "foro_poligono": ToolForoPoligono(), "foro_cerchio": ToolForoCerchio(),
        }
        self._setup_tool_buttons(); self._setup_griglia()

    def _setup_tool_buttons(self):
        mappa = {
            self._ui.sezione_btn_muovi: "muovi", self._ui.sezione_btn_modifica: "modifica",
            self._ui.sezione_btn_rettangolo: "rettangolo", self._ui.sezione_btn_poligono: "poligono",
            self._ui.sezione_btn_cerchio: "cerchio", self._ui.sezione_btn_barra: "barra",
            self._ui.sezione_btn_staffa: "staffa",
            self._ui.sezione_btn_foro_rettangolo: "foro_rettangolo",
            self._ui.sezione_btn_foro_poligono: "foro_poligono",
            self._ui.sezione_btn_foro_cerchio: "foro_cerchio",
        }
        gruppo = QButtonGroup(self._spazio); gruppo.setExclusive(True)
        for btn, nome in mappa.items():
            btn.setCheckable(True); gruppo.addButton(btn)
            btn.clicked.connect(lambda _c, n=nome: self._attiva_tool(n))
        self._btn_group = gruppo

    def _attiva_tool(self, nome):
        tool = self._tools.get(nome)
        if tool: self._spazio.set_active_tool(tool)

    def _setup_griglia(self):
        self._ui.sezione_btn_griglia.setCheckable(True)
        self._ui.sezione_btn_griglia.setChecked(self._spazio.show_grid)
        self._ui.sezione_btn_griglia.toggled.connect(self._spazio.set_show_grid)
        self._ui.sezione_btn_snap.setCheckable(True)
        self._ui.sezione_btn_snap.setChecked(self._spazio.snap_to_grid)
        self._ui.sezione_btn_snap.toggled.connect(self._spazio.set_snap_to_grid)
        self._ui.sezione_dimensione_griglia.setText(str(self._spazio.grid_spacing))
        self._ui.sezione_dimensione_griglia.textChanged.connect(self._on_sp)

    def _on_sp(self, t):
        try:
            v = float(t.replace(",",".").strip())
            if v > 0: self._spazio.set_grid_spacing(v)
        except ValueError: pass

    def get_tool(self, nome): return self._tools.get(nome)
    def set_diametro_barra(self, d): self._tools["barra"].diametro = d
    def set_diametro_staffa(self, d): self._tools["staffa"].diametro = d
