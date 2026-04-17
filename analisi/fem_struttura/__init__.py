"""
fem_struttura – Modulo di analisi FEM per strutture a telaio con OpenSees.

Sottosistemi:
  - raccolta_dati_struttura     : Raccolta dati dalla struttura selezionata
  - generatore_dati_materiali   : Risoluzione materiali (riferimento → dati)
  - generatore_dati_sezioni     : Discretizzazione e omogenizzazione sezioni
  - generatore_mesh_struttura   : Discretizzazione beam e shell
  - analisi_fem_struttura       : Motore di analisi OpenSees
  - risultati_struttura         : Classi dati per i risultati
  - disegno_fem_struttura       : Visualizzazione OpenGL 3D dei risultati
  - gestione_fem_struttura      : Controller principale (connessione UI)
"""
from .gestione_fem_struttura import GestioneFemStruttura

__all__ = ["GestioneFemStruttura"]
