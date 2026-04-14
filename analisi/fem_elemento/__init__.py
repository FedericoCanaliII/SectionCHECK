"""
fem_elemento – Modulo FEM per analisi elementi strutturali.

Sottosistemi:
  - generatore_mesh: generazione mesh esaedriche e beam
  - scrittore_inp: scrittura file .inp per CalculiX
  - disegno_fem_elemento: visualizzazione 3D OpenGL
  - gestione_mesh_elemento: controller e interfaccia
"""
from .gestione_mesh_elemento import GestioneMeshElemento
