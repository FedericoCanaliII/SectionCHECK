"""
setup.py – Build dell'eseguibile SectionCHECK con cx_Freeze.

Uso:
    python setup.py build

L'eseguibile viene generato nella cartella build/.
"""

import sys
import os
from pathlib import Path
from cx_Freeze import setup, Executable
import scipy

# -----------------------------------------------------------------------
# 1. FIX CRITICI
# -----------------------------------------------------------------------
sys.setrecursionlimit(10000)

# -----------------------------------------------------------------------
# 2. CONFIGURAZIONE PROGETTO
# -----------------------------------------------------------------------
PROJECT_NAME = "SectionCHECK"
VERSION      = "1.0.0"
DESCRIPTION  = "Software per la verifica strutturale di sezioni in c.a."
AUTHOR       = "SectionCHECK Team"
MAIN_SCRIPT  = "main.py"
TARGET_NAME  = "SectionCHECK.exe"
ICON_PATH    = Path("interfaccia") / "icone" / "logo.ico"

# -----------------------------------------------------------------------
# 3. RISORSE DA INCLUDERE NEL BUNDLE
# -----------------------------------------------------------------------
include_files: list[tuple[str, str]] = []

# Icone e immagini dell'interfaccia
for cartella in ("icone", "immagini"):
    src = Path("interfaccia") / cartella
    if src.exists():
        include_files.append((str(src), str(Path("interfaccia") / cartella)))

# Database JSON (materiali, sezioni, elementi)
_db_files = [
    Path("materiali") / "database_materiali.json",
    Path("sezioni")   / "database_sezioni.json",
    Path("elementi")  / "database_elementi.json",
]
for db in _db_files:
    if db.exists():
        include_files.append((str(db), str(db)))

# File UI per la finestra AI
_ai_ui = Path("ai") / "ai_interfaccia.ui"
if _ai_ui.exists():
    include_files.append((str(_ai_ui), str(_ai_ui)))

# Licenza
if Path("LICENSE").exists():
    include_files.append(("LICENSE", "LICENSE"))

# FIX Scipy – le DLL native devono essere copiate esplicitamente
scipy_libs = Path(scipy.__file__).parent / ".libs"
if scipy_libs.exists():
    include_files.append((str(scipy_libs), "lib"))

# -----------------------------------------------------------------------
# 4. OPZIONI DI BUILD
# -----------------------------------------------------------------------
build_exe_options = {
    "packages": [
        # Stdlib richieste esplicitamente
        "os", "sys", "unittest",
        # Dipendenze runtime di terze parti
        "numpy",
        "scipy",
        "shapely",
        "PIL",          # Pillow
        "requests",
        "PyQt5",
        "OpenGL",       # PyOpenGL
        "openseespy",   # <-- AGGIUNTO OPENSEES QUI
        # Moduli dell'applicazione
        "interfaccia",
        "materiali",
        "sezioni",
        "elementi",
        "struttura",
        "analisi",
        "ai",
    ],
    "excludes": [
        "tkinter", "email", "xmlrpc", "pydoc",
        "test", "distutils",
    ],
    "include_files": include_files,
    "include_msvcr":  True,
}

# -----------------------------------------------------------------------
# 5. BASE (Gestione della console su Windows)
# -----------------------------------------------------------------------
# FIX: Attivando Win32GUI, il terminale nero in background non apparirà più.
base = "Win32GUI" if sys.platform == "win32" else None

# -----------------------------------------------------------------------
# 6. ESECUZIONE
# -----------------------------------------------------------------------
setup(
    name        = PROJECT_NAME,
    version     = VERSION,
    description = DESCRIPTION,
    author      = AUTHOR,
    license     = "AGPL-3.0",
    options     = {"build_exe": build_exe_options},
    executables = [
        Executable(
            script      = MAIN_SCRIPT,
            base        = base,
            target_name = TARGET_NAME,
            icon        = str(ICON_PATH) if ICON_PATH.exists() else None,
        )
    ],
)