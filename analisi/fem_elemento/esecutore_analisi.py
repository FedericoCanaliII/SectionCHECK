"""
esecutore_analisi.py – Lancia CalculiX e gestisce il processo di analisi.

Responsabilita':
  - Prepara i file .inp (lineare e/o nonlineare) nella cartella analisi_svolte
  - Lancia ccx_dynamic.exe come sottoprocesso
  - Monitora il progresso del solutore
  - Legge i risultati .frd al termine
  - Supporta analisi lineare, nonlineare, peso proprio, gravita', contatto
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Optional

from PyQt5 import QtCore

from .generatore_mesh import RisultatoMesh
from .scrittore_inp import ScrittoreINP
from .lettore_frd import leggi_frd, leggi_dat, RisultatiFRD


# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------

_FEM_DIR = os.path.dirname(os.path.abspath(__file__))
_SOLVER_DIR = os.path.join(_FEM_DIR, "Calculix_Solver")
_SOLVER_EXE = os.path.join(_SOLVER_DIR, "ccx_dynamic.exe")
_ANALISI_DIR = os.path.join(_FEM_DIR, "analisi_svolte")


# ---------------------------------------------------------------------------
# Parametri Analisi
# ---------------------------------------------------------------------------

class ParametriAnalisi:
    """Raccoglie tutti i parametri per un'analisi FEM."""

    def __init__(self):
        self.gravita: float = 9.81          # m/s^2
        self.peso_proprio: bool = True
        self.collisioni: bool = False
        self.analisi_lineare: bool = True
        self.analisi_nonlineare: bool = True
        self.nome_elemento: str = "elemento"


# ---------------------------------------------------------------------------
# Thread Analisi
# ---------------------------------------------------------------------------

class AnalisiThread(QtCore.QThread):
    """
    Esegue una o due analisi CalculiX (lineare + nonlineare) in background.

    Segnali:
      avanzamento(int)       : percentuale 0-100
      log_message(str)       : messaggio di log per il terminale
      completato(dict)       : risultati {"lineare": RisultatiFRD|None,
                                          "nonlineare": RisultatiFRD|None}
      errore(str)            : messaggio di errore
    """

    avanzamento  = QtCore.pyqtSignal(int)
    log_message  = QtCore.pyqtSignal(str)
    completato   = QtCore.pyqtSignal(dict)
    errore       = QtCore.pyqtSignal(str)

    def __init__(self, mesh: RisultatoMesh, materiali: dict,
                 parametri: ParametriAnalisi, parent=None):
        super().__init__(parent)
        self._mesh = mesh
        self._materiali = materiali
        self._params = parametri
        self._aborted = False

    def abort(self):
        self._aborted = True

    def run(self):
        try:
            risultati = {}
            analisi_da_fare = []

            if self._params.analisi_lineare:
                analisi_da_fare.append("lineare")
            if self._params.analisi_nonlineare:
                analisi_da_fare.append("nonlineare")

            if not analisi_da_fare:
                self.errore.emit("Nessuna analisi selezionata.")
                return

            totale_peso = len(analisi_da_fare)
            for idx, tipo_analisi in enumerate(analisi_da_fare):
                if self._aborted:
                    return

                base_pct = int(idx / totale_peso * 100)
                step_pct = int(100 / totale_peso)

                self.log_message.emit(f">> Preparazione analisi {tipo_analisi}...")
                self.avanzamento.emit(base_pct + 5)

                # 1. Scrivi .inp
                nonlineare = (tipo_analisi == "nonlineare")
                inp_path = self._prepara_inp(tipo_analisi, nonlineare)
                if inp_path is None:
                    self.errore.emit(f"Errore preparazione .inp per analisi {tipo_analisi}")
                    return

                self.avanzamento.emit(base_pct + 10)

                # 2. Lancia solver
                self.log_message.emit(f">> Lancio CalculiX ({tipo_analisi})...")
                successo = self._lancia_solver(inp_path, base_pct + 10, step_pct - 20)
                if not successo:
                    if self._aborted:
                        return
                    self.errore.emit(f"CalculiX ha fallito per analisi {tipo_analisi}.")
                    return

                self.avanzamento.emit(base_pct + step_pct - 10)

                # 3. Leggi risultati
                self.log_message.emit(f">> Lettura risultati {tipo_analisi}...")
                frd_path = inp_path.replace(".inp", ".frd")
                dat_path = inp_path.replace(".inp", ".dat")

                frd = leggi_frd(frd_path)
                if frd is None or not frd.steps:
                    # Prova con .dat
                    dat = leggi_dat(dat_path)
                    if dat and dat.get("spostamenti"):
                        from .lettore_frd import StepFRD, RisultatiFRD as RFRD
                        frd = RFRD()
                        step = StepFRD(1, 1.0)
                        for nid, (ux, uy, uz) in dat["spostamenti"].items():
                            step.spostamenti[nid] = (ux, uy, uz)
                        frd.steps.append(step)

                risultati[tipo_analisi] = frd
                self.avanzamento.emit(base_pct + step_pct)
                self.log_message.emit(
                    f">> Analisi {tipo_analisi} completata"
                    f" ({frd.n_steps if frd else 0} step)."
                )

            self.avanzamento.emit(100)
            self.completato.emit(risultati)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.errore.emit(str(e))

    # ------------------------------------------------------------------
    # PREPARAZIONE .INP
    # ------------------------------------------------------------------

    def _prepara_inp(self, suffisso: str, nonlineare: bool) -> Optional[str]:
        """Prepara il file .inp nella cartella analisi_svolte."""
        nome_safe = self._params.nome_elemento.replace(" ", "_").replace(".", "_")
        nome_file = f"{nome_safe}_{suffisso}"

        os.makedirs(_ANALISI_DIR, exist_ok=True)
        percorso = os.path.join(_ANALISI_DIR, f"{nome_file}.inp")

        scrittore = ScrittoreINPAnalisi(
            risultato=self._mesh,
            materiali=self._materiali,
            nome_elemento=self._params.nome_elemento,
            parametri=self._params,
            nonlineare=nonlineare,
        )
        if scrittore.scrivi(percorso):
            return percorso
        return None

    # ------------------------------------------------------------------
    # LANCIO SOLVER
    # ------------------------------------------------------------------

    def _lancia_solver(self, inp_path: str, pct_base: int, pct_range: int) -> bool:
        """
        Lancia ccx_dynamic.exe e monitora il progresso.
        Il solver viene lanciato nella directory del file .inp.
        """
        if not os.path.exists(_SOLVER_EXE):
            self.log_message.emit(f"ERR  Solver non trovato: {_SOLVER_EXE}")
            return False

        # CalculiX vuole il nome del job senza estensione
        job_dir = os.path.dirname(inp_path)
        job_name = os.path.splitext(os.path.basename(inp_path))[0]

        # Pulisce eventuali file di output residui da run precedenti.
        # CCX in alcuni casi non riesce ad aprire/cancellare un .dat lasciato
        # bloccato da un editor o da un processo figlio: meglio rimuoverli
        # noi prima di partire (errore tipico: "could not delete file ...dat",
        # exit code 201).
        for ext in (".dat", ".frd", ".cvg", ".sta", ".12d", ".out"):
            stale = os.path.join(job_dir, job_name + ext)
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass

        # Copia le DLL del solver nella directory del job (se necessario)
        for dll in ("libiomp5md.dll", "mkl_core.2.dll", "mkl_def.2.dll",
                     "mkl_intel_thread.2.dll", "mkl_rt.2.dll"):
            src = os.path.join(_SOLVER_DIR, dll)
            dst = os.path.join(job_dir, dll)
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

        cmd = [_SOLVER_EXE, "-i", job_name]

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=job_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Monitora output per progresso
            step_count = 0
            while True:
                if self._aborted:
                    proc.terminate()
                    return False

                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    continue

                line = line.strip()
                if line:
                    # Cerca indicazioni di progresso nelle righe di output
                    if "increment" in line.lower() or "step" in line.lower():
                        step_count += 1
                        pct = min(pct_base + int(step_count * 2), pct_base + pct_range)
                        self.avanzamento.emit(pct)

                    # Log selettivo (non tutto l'output)
                    if any(k in line.lower() for k in
                           ("error", "warning", "convergence", "total", "step",
                            "section", "surface", "truss", "fatal")):
                        self.log_message.emit(f"   CCX: {line}")

            ret = proc.returncode
            if ret != 0:
                self.log_message.emit(f"WARN  CalculiX uscito con codice {ret}")
                # Anche con codice != 0 potrebbe aver prodotto risultati parziali
                # Controlliamo se il .frd esiste
                frd_path = os.path.join(job_dir, f"{job_name}.frd")
                return os.path.exists(frd_path)

            return True

        except FileNotFoundError:
            self.log_message.emit(f"ERR  Impossibile avviare: {_SOLVER_EXE}")
            return False
        except Exception as e:
            self.log_message.emit(f"ERR  Errore esecuzione solver: {e}")
            return False


# ==============================================================================
# SCRITTORE INP ANALISI (estende ScrittoreINP con supporto avanzato)
# ==============================================================================

class ScrittoreINPAnalisi:
    """
    Scrive .inp con supporto completo per:
    - Gravita' e peso proprio (*DLOAD, *GRAVITY)
    - Analisi nonlineare (*STEP, NLGEOM, materiali nonlineari)
    - Contatto tra superfici (*CONTACT PAIR)
    - Incrementi multipli per nonlineare
    """

    def __init__(self, risultato: RisultatoMesh, materiali: dict,
                 nome_elemento: str, parametri: ParametriAnalisi,
                 nonlineare: bool = False):
        self._mesh = risultato
        self._materiali = materiali
        self._nome = nome_elemento
        self._params = parametri
        self._nonlineare = nonlineare

    def scrivi(self, percorso_file: str) -> bool:
        try:
            # Set di nodi usati dagli elementi hex
            self._nodi_hex = set()
            for nodi in self._mesh.elementi_hex.values():
                self._nodi_hex.update(nodi)
            # Set di nodi usati dagli elementi truss (armatura)
            self._nodi_truss = set()
            for nodi in self._mesh.elementi_beam.values():
                self._nodi_truss.update(nodi)
            # Tutti i nodi che partecipano all'analisi
            self._nodi_analisi = self._nodi_hex | self._nodi_truss

            os.makedirs(os.path.dirname(percorso_file), exist_ok=True)
            with open(percorso_file, "w", encoding="utf-8") as f:
                self._scrivi_header(f)
                self._scrivi_nodi(f)
                self._scrivi_elementi_hex(f)
                self._scrivi_elementi_truss(f)
                self._scrivi_nset_elset(f)
                self._scrivi_materiali(f)
                self._scrivi_sezioni(f)
                self._scrivi_tie(f)
                self._scrivi_tie_armatura(f)
                if self._params.collisioni:
                    self._scrivi_contatto(f)
                self._scrivi_step(f)
            return True
        except Exception as e:
            print(f"ERR  Scrittura .inp fallita: {e}")
            import traceback; traceback.print_exc()
            return False

    def _scrivi_header(self, f):
        tipo = "NONLINEARE" if self._nonlineare else "LINEARE"
        f.write(f"** CalculiX input file – SectionCHECK ({tipo})\n")
        f.write(f"** Elemento: {self._nome}\n")
        f.write(f"** Nodi: {self._mesh.n_nodi}  Elementi: {self._mesh.n_elementi}\n")
        f.write(f"**\n")

    def _scrivi_nodi(self, f):
        f.write("*NODE, NSET=NALL\n")
        for nid in sorted(self._nodi_analisi):
            x, y, z = self._mesh.nodi[nid]
            f.write(f"{nid}, {x:.8e}, {y:.8e}, {z:.8e}\n")

    def _scrivi_elementi_hex(self, f):
        if not self._mesh.elementi_hex:
            return
        f.write("*ELEMENT, TYPE=C3D8, ELSET=EALL_HEX\n")
        for eid in sorted(self._mesh.elementi_hex.keys()):
            nodi = self._mesh.elementi_hex[eid]
            f.write(f"{eid}, {', '.join(str(n) for n in nodi)}\n")

    def _scrivi_elementi_truss(self, f):
        """Scrive elementi T3D2 per barre e staffe."""
        if not self._mesh.elementi_beam:
            return
        f.write("*ELEMENT, TYPE=T3D2, ELSET=EALL_TRUSS\n")
        for eid in sorted(self._mesh.elementi_beam.keys()):
            nodi = self._mesh.elementi_beam[eid]
            f.write(f"{eid}, {nodi[0]}, {nodi[1]}\n")

    def _scrivi_nset_elset(self, f):
        for obj_id, nodi_set in self._mesh.nodi_per_oggetto.items():
            tipo = self._mesh.tipo_oggetto.get(obj_id, "unknown")
            nome = f"OBJ_{obj_id}_{tipo.upper()}"

            if tipo == "carpenteria":
                nodi_filtrati = sorted(nodi_set & self._nodi_hex)
                if nodi_filtrati:
                    f.write(f"*NSET, NSET=N_{nome}\n")
                    self._scrivi_ids(f, nodi_filtrati)
                elems = self._mesh.elementi_per_oggetto.get(obj_id, set())
                elems_hex = sorted(e for e in elems if e in self._mesh.elementi_hex)
                if elems_hex:
                    f.write(f"*ELSET, ELSET=E_{nome}\n")
                    self._scrivi_ids(f, elems_hex)
            elif tipo in ("barra", "staffa"):
                nodi_filtrati = sorted(nodi_set & self._nodi_truss)
                if nodi_filtrati:
                    f.write(f"*NSET, NSET=N_{nome}\n")
                    self._scrivi_ids(f, nodi_filtrati)
                elems = self._mesh.elementi_per_oggetto.get(obj_id, set())
                elems_truss = sorted(e for e in elems if e in self._mesh.elementi_beam)
                if elems_truss:
                    f.write(f"*ELSET, ELSET=E_{nome}\n")
                    self._scrivi_ids(f, elems_truss)

        # Vincoli e carichi: tutti i nodi dell'analisi
        vincolati = {k: v for k, v in self._mesh.nodi_vincolati.items()
                     if k in self._nodi_analisi}
        caricati = {k: v for k, v in self._mesh.nodi_caricati.items()
                    if k in self._nodi_analisi}
        if vincolati:
            f.write("*NSET, NSET=N_VINCOLATI\n")
            self._scrivi_ids(f, sorted(vincolati.keys()))
        if caricati:
            f.write("*NSET, NSET=N_CARICATI\n")
            self._scrivi_ids(f, sorted(caricati.keys()))

    def _scrivi_materiali(self, f):
        materiali_scritti = set()

        for obj_id, nome_mat in self._mesh.materiale_oggetto.items():
            if not nome_mat or nome_mat in materiali_scritti:
                continue
            materiali_scritti.add(nome_mat)
            dati = self._materiali.get(nome_mat, {})
            nome_safe = nome_mat.replace(" ", "_").replace(".", "_").replace("/", "_")

            f.write(f"**\n")
            f.write(f"*MATERIAL, NAME=MAT_{nome_safe}\n")

            # Modulo elastico (database in MPa -> CalculiX in Pa)
            E_mpa = float(dati.get("m_elastico",
                          dati.get("E", dati.get("Ecm", dati.get("Es", 30000.0)))))
            E_pa = E_mpa * 1e6
            nu = float(dati.get("poisson", 0.2))

            f.write(f"*ELASTIC\n")
            f.write(f"{E_pa:.4e}, {nu:.4f}\n")

            # Densita' (kg/m^3 – coerente con sistema SI: m, N, Pa, kg)
            rho = dati.get("densita", dati.get("rho"))
            if rho is not None:
                f.write(f"*DENSITY\n")
                f.write(f"{float(rho):.4e}\n")

            # Plasticita' per analisi nonlineare
            if self._nonlineare:
                self._scrivi_plasticita(f, dati, nome_mat)

        # Materiale default (calcestruzzo C25/30)
        f.write(f"**\n*MATERIAL, NAME=MAT_DEFAULT\n*ELASTIC\n3.0000e+10, 0.2000\n")
        f.write(f"*DENSITY\n2500.0\n")

    def _scrivi_plasticita(self, f, dati: dict, nome_mat: str):
        """
        Genera curve di plasticita' a partire dai dati SLU dei materiali.
        Per calcestruzzo: comportamento parabola-rettangolo in compressione.
        Per acciaio/barre: bilineare con incrudimento.
        """
        slu = dati.get("slu")
        if not slu:
            return

        E = float(dati.get("m_elastico",
                  dati.get("E", dati.get("Ecm", dati.get("Es", 30000.0))))) * 1e6  # MPa -> Pa
        gamma = float(dati.get("gamma", 1.0))

        # Determina se e' calcestruzzo o acciaio
        fck = dati.get("fck")
        fyk = dati.get("fyk")

        if fck is not None:
            # Calcestruzzo – comportamento compressivo (MPa -> Pa)
            fcd_mpa = float(fck) * float(dati.get("alpha", 0.85)) / gamma
            fcd = fcd_mpa * 1e6  # Pa
            eps_y = fcd / E  # deformazione elastica (E gia' in Pa)
            f.write(f"*PLASTIC\n")
            f.write(f"{fcd * 0.4:.4e}, 0.0000\n")
            f.write(f"{fcd:.4e}, {eps_y:.6e}\n")
            f.write(f"{fcd:.4e}, {3.5e-3 - eps_y:.6e}\n")

        elif fyk is not None:
            # Acciaio barre – bilineare (MPa -> Pa)
            fyd = float(fyk) / gamma * 1e6  # Pa
            ftk_pa = float(dati.get("ftk", fyk * 1.08)) / gamma * 1e6
            eps_uk = float(dati.get("eps_uk", 0.05))
            eps_y = fyd / E  # E gia' in Pa

            f.write(f"*PLASTIC\n")
            f.write(f"{fyd:.4e}, 0.0000\n")
            f.write(f"{ftk_pa:.4e}, {eps_uk - eps_y:.6e}\n")

    def _scrivi_sezioni(self, f):
        import math as _math
        for obj_id in self._mesh.elementi_per_oggetto:
            tipo = self._mesh.tipo_oggetto.get(obj_id, "carpenteria")
            nome_set = f"OBJ_{obj_id}_{tipo.upper()}"
            nome_mat = self._mesh.materiale_oggetto.get(obj_id, "")
            mat_safe = f"MAT_{nome_mat.replace(' ', '_').replace('.', '_').replace('/', '_')}" if nome_mat else "MAT_DEFAULT"

            if tipo == "carpenteria":
                f.write(f"*SOLID SECTION, ELSET=E_{nome_set}, MATERIAL={mat_safe}\n\n")
            elif tipo in ("barra", "staffa"):
                # T3D2: *SOLID SECTION con area della sezione trasversale
                diametro = self._mesh.diametro_oggetto.get(obj_id, 0.016)
                area = _math.pi * diametro**2 / 4.0  # m^2
                f.write(f"*SOLID SECTION, ELSET=E_{nome_set}, MATERIAL={mat_safe}\n")
                f.write(f"{area:.10f}\n")

    def _scrivi_tie(self, f):
        """
        Scrive i vincoli *TIE tra le superfici di oggetti di carpenteria contigui.
        Permette l'incollaggio cinematico di mesh non conformi (surface-to-surface).
        """
        if not self._mesh.tie_constraints:
            return

        f.write(f"**\n** TIE CONSTRAINTS - Superficie-Superficie tra carpenteria\n")
        
        for idx, tie in enumerate(self._mesh.tie_constraints):
            master_obj = tie.get("master_obj")
            slave_obj = tie.get("slave_obj")
            master_faces = tie.get("master_faces", [])
            slave_faces = tie.get("slave_faces", [])

            if not master_faces or not slave_faces:
                continue
            
            # Crea nomi univoci per le superfici di questo accoppiamento
            surf_master = f"SURF_TIE_M_{master_obj}_{idx}"
            surf_slave = f"SURF_TIE_S_{slave_obj}_{idx}"
            tie_name = f"TIE_CARP_{idx}"

            # 1. Definizione Superficie Master (Element-based)
            f.write(f"*SURFACE, NAME={surf_master}, TYPE=ELEMENT\n")
            for eid, sname in master_faces:
                f.write(f"{eid}, {sname}\n")

            # 2. Definizione Superficie Slave (Element-based)
            f.write(f"*SURFACE, NAME={surf_slave}, TYPE=ELEMENT\n")
            for eid, sname in slave_faces:
                f.write(f"{eid}, {sname}\n")

            # 3. Dichiarazione del TIE (Sintassi: Slave, Master)
            # Usiamo una piccola position tolerance (1 mm) per catturare i nodi
            # in caso di arrotondamenti o disallineamenti minimi.
            f.write(f"*TIE, NAME={tie_name}, POSITION TOLERANCE=0.001\n")
            f.write(f"{surf_slave}, {surf_master}\n")

    def _scrivi_tie_armatura(self, f):
        """
        Scrive i TIE constraints per collegare armatura (T3D2) alla
        carpenteria (C3D8). Usa POSITION TOLERANCE per embedded rebar.
        Master = element-based surface costruita elencando tutte le facce
        S1..S6 dell'ELSET dell'oggetto carpenteria (CCX scarta le facce
        interne). Slave = node-based surface (nodi armatura).
        """
        if not self._mesh.tie_armatura:
            return
        f.write(f"**\n** TIE ARMATURA-CARPENTERIA (embedded rebar)\n")
        offset = len(self._mesh.tie_constraints)

        master_objs_scritti: dict[int, str] = {}

        for idx, tie in enumerate(self._mesh.tie_armatura):
            tie_idx = offset + idx
            slave_nodi = sorted(tie["slave_nodes"] & self._nodi_truss)
            master_obj = tie["master_obj"]
            if not slave_nodi:
                continue

            if master_obj not in master_objs_scritti:
                tipo_m = self._mesh.tipo_oggetto.get(master_obj, "carpenteria")
                nome_elset = f"E_OBJ_{master_obj}_{tipo_m.upper()}"
                surf_name = f"SURF_ARM_M_{master_obj}"
                f.write(f"*SURFACE, NAME={surf_name}, TYPE=ELEMENT\n")
                for sface in ("S1", "S2", "S3", "S4", "S5", "S6"):
                    f.write(f"{nome_elset}, {sface}\n")
                master_objs_scritti[master_obj] = surf_name

            surf_master = master_objs_scritti[master_obj]

            f.write(f"*NSET, NSET=TIE_{tie_idx}_SLAVE\n")
            self._scrivi_ids(f, slave_nodi)
            f.write(f"*SURFACE, NAME=SURF_TIE_{tie_idx}_S, TYPE=NODE\n")
            f.write(f"TIE_{tie_idx}_SLAVE\n")
            f.write(f"*TIE, NAME=TIE_{tie_idx}, POSITION TOLERANCE=0.15\n")
            f.write(f"SURF_TIE_{tie_idx}_S, {surf_master}\n")

    def _scrivi_contatto(self, f):
        """
        Genera coppie di contatto tra le superfici esterne degli oggetti
        di carpenteria che NON sono gia' collegati da TIE. Usa il modello
        HARD pressure-overclosure (formulazione standard CCX, tutti i nodi
        in penetrazione vengono spinti fuori senza compliance artificiale).
        Le superfici sono costruite con TUTTE le S1..S6 dell'ELSET
        dell'oggetto: CCX riconosce automaticamente le facce interne
        (condivise tra elementi adiacenti) e le scarta dalla superficie.
        """
        obj_ids = [oid for oid, t in self._mesh.tipo_oggetto.items()
                   if t == "carpenteria"]
        if len(obj_ids) < 2:
            return

        # Coppie gia' collegate da TIE (incollate): non serve contatto
        tie_set = set()
        for tie in self._mesh.tie_constraints:
            tie_set.add(frozenset({tie["master_obj"], tie["slave_obj"]}))

        pairs = []
        for i in range(len(obj_ids)):
            for j in range(i + 1, len(obj_ids)):
                if frozenset({obj_ids[i], obj_ids[j]}) not in tie_set:
                    pairs.append((obj_ids[i], obj_ids[j]))
        if not pairs:
            return

        f.write(f"**\n** CONTATTO TRA OGGETTI\n")
        f.write(f"*SURFACE INTERACTION, NAME=CONTACT_PROP\n")
        f.write(f"*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD\n")

        surfs_scritte: dict[int, str] = {}
        for oid_a, oid_b in pairs:
            for oid in (oid_a, oid_b):
                if oid in surfs_scritte:
                    continue
                tipo = self._mesh.tipo_oggetto.get(oid, "carpenteria")
                nome_elset = f"E_OBJ_{oid}_{tipo.upper()}"
                surf_name = f"CONT_SURF_{oid}"
                f.write(f"*SURFACE, NAME={surf_name}, TYPE=ELEMENT\n")
                for sface in ("S1", "S2", "S3", "S4", "S5", "S6"):
                    f.write(f"{nome_elset}, {sface}\n")
                surfs_scritte[oid] = surf_name

            # Slave = oggetto con piu' elementi (mesh piu' fine)
            n_a = len(self._mesh.elementi_per_oggetto.get(oid_a, set()))
            n_b = len(self._mesh.elementi_per_oggetto.get(oid_b, set()))
            slave, master = (oid_a, oid_b) if n_a >= n_b else (oid_b, oid_a)
            f.write(f"*CONTACT PAIR, INTERACTION=CONTACT_PROP, "
                    f"TYPE=SURFACE TO SURFACE\n")
            f.write(f"{surfs_scritte[slave]}, {surfs_scritte[master]}\n")

    def _scrivi_step(self, f):
        f.write(f"**\n** ============================================\n")
        f.write(f"** STEP DI ANALISI\n")
        f.write(f"** ============================================\n")

        # Il contatto richiede sempre NLGEOM + time increments in CalculiX
        bisogna_nlgeom = self._nonlineare or self._params.collisioni
        if bisogna_nlgeom:
            f.write(f"*STEP, NLGEOM, INC=200\n")
            f.write(f"*STATIC\n")
            # initial_dt, total_t, min_dt, max_dt
            f.write(f"0.05, 1.0, 1e-8, 0.15\n")
        else:
            f.write(f"*STEP\n")
            f.write(f"*STATIC\n")

        # Vincoli (tutti i nodi dell'analisi)
        vincolati = {k: v for k, v in self._mesh.nodi_vincolati.items()
                     if k in self._nodi_analisi}
        if vincolati:
            f.write(f"**\n** VINCOLI\n")
            f.write(f"*BOUNDARY\n")
            for nid, ced in vincolati.items():
                sx = ced.get("sx", 0.0)
                sy = ced.get("sy", 0.0)
                sz = ced.get("sz", 0.0)
                f.write(f"{nid}, 1, 1, {sx:.8e}\n")
                f.write(f"{nid}, 2, 2, {sy:.8e}\n")
                f.write(f"{nid}, 3, 3, {sz:.8e}\n")

        # Peso proprio + gravita' (hex + truss)
        if self._params.peso_proprio:
            g = self._params.gravita
            f.write(f"**\n** PESO PROPRIO (gravita' = {g} m/s^2)\n")
            f.write(f"*DLOAD\n")
            if self._mesh.elementi_hex:
                f.write(f"EALL_HEX, GRAV, {g:.4e}, 0.0, 0.0, -1.0\n")
            if self._mesh.elementi_beam:
                f.write(f"EALL_TRUSS, GRAV, {g:.4e}, 0.0, 0.0, -1.0\n")

        # Carichi concentrati (tutti i nodi analisi, kN -> N)
        caricati = {k: v for k, v in self._mesh.nodi_caricati.items()
                    if k in self._nodi_analisi}
        if caricati:
            f.write(f"**\n** CARICHI (convertiti da kN a N)\n")
            f.write(f"*CLOAD\n")
            for nid, forze in caricati.items():
                fx = forze.get("fx", 0.0) * 1e3  # kN -> N
                fy = forze.get("fy", 0.0) * 1e3
                fz = forze.get("fz", 0.0) * 1e3
                if abs(fx) > 1e-9:
                    f.write(f"{nid}, 1, {fx:.8e}\n")
                if abs(fy) > 1e-9:
                    f.write(f"{nid}, 2, {fy:.8e}\n")
                if abs(fz) > 1e-9:
                    f.write(f"{nid}, 3, {fz:.8e}\n")

        # Output richiesti
        f.write(f"**\n*NODE FILE\nU\n*EL FILE\nS, E\n")
        if self._nonlineare:
            f.write(f"*NODE PRINT, NSET=NALL, FREQUENCY=1\nU\n")
        f.write(f"*END STEP\n")

    @staticmethod
    def _scrivi_ids(f, ids: list[int], per_riga: int = 16):
        for i in range(0, len(ids), per_riga):
            chunk = ids[i:i + per_riga]
            f.write(", ".join(str(x) for x in chunk))
            f.write(",\n" if i + per_riga < len(ids) else "\n")
