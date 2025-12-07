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
VERSION = "1.0"
DESCRIPTION = "Software verifica sezioni"
MAIN_SCRIPT = "main.py"
TARGET_NAME = "SectionCHECK.exe"
ICON_PATH = Path("interfaccia") / "icone" / "logo.ico"

# -----------------------------------------------------------------------
# 3. GESTIONE DIPENDENZE
# -----------------------------------------------------------------------
# Lista dei tuoi pacchetti (cartelle con codice)
my_packages = [
    "interfaccia",
    "materiali",
    "sezione",
    "tools",
    "beam",
    "output"
]

include_files = []

# Aggiunta Icona
if ICON_PATH.exists():
    include_files.append((str(ICON_PATH), "logo.ico"))

# Aggiunta cartella icone completa se necessario per la GUI
path_icone = Path("interfaccia") / "icone"
if path_icone.exists():
    # Mantiene la struttura cartelle: interfaccia/icone
    include_files.append((str(path_icone), str(Path("interfaccia") / "icone")))

# FIX SCIPY (DLLs)
scipy_path = Path(scipy.__file__).parent
scipy_libs = scipy_path / ".libs"
if scipy_libs.exists():
    include_files.append((str(scipy_libs), "lib"))

# -----------------------------------------------------------------------
# 4. OPZIONI DI BUILD
# -----------------------------------------------------------------------
build_exe_options = {
    # IMPORTANTE: Ho aggiunto 'unittest' qui perché NumPy lo richiede
    "packages": [
        "os", 
        "sys", 
        "numpy", 
        "scipy", 
        "PyQt5", 
        "OpenGL",
        "unittest" 
    ] + my_packages,

    # IMPORTANTE: Ho rimosso 'unittest' da questa lista di esclusione
    "excludes": ["tkinter", "email", "http", "xmlrpc", "pydoc"],
    
    "include_files": include_files,
    "include_msvcr": True,
}

# -----------------------------------------------------------------------
# 5. BASE
# -----------------------------------------------------------------------
base = None
if sys.platform == "win32":
    base = "Win32GUI" 

# -----------------------------------------------------------------------
# 6. ESECUZIONE
# -----------------------------------------------------------------------
setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            script=MAIN_SCRIPT,
            base=base,
            target_name=TARGET_NAME,
            icon=str(ICON_PATH) if ICON_PATH.exists() else None,
        )
    ],
)