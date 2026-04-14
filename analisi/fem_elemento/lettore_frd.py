"""
lettore_frd.py – Parser per file risultati CalculiX (.frd).

Il formato .frd ASCII usa colonne fisse:
  - Record key: colonne 1-3  (" -1" = dato, " -3" = fine blocco, " -4" = header)
  - Node ID:    colonne 4-13 (10 caratteri)
  - Valori:     colonne 14-25, 26-37, 38-49, ... (12 caratteri ciascuno)

Estrae:
  - Spostamenti nodali (DISP: ux, uy, uz)
  - Tensioni nodali (STRESS: sxx, syy, szz, sxy, syz, sxz -> von Mises)
  - Supporto multi-step per analisi nonlineare incrementale
"""
from __future__ import annotations

import os
from typing import Optional


class StepFRD:
    """Dati risultanti per un singolo step/incremento."""

    def __init__(self, step_number: int, step_time: float = 0.0):
        self.step_number = step_number
        self.step_time = step_time
        # Spostamenti: node_id -> (ux, uy, uz)
        self.spostamenti: dict[int, tuple[float, float, float]] = {}
        # Tensioni von Mises per nodo: node_id -> sigma_vm
        self.stress_vm: dict[int, float] = {}
        # Componenti tensione: node_id -> (sxx, syy, szz, sxy, syz, sxz)
        self.stress_comp: dict[int, tuple] = {}


class RisultatiFRD:
    """Container completo dei risultati letti da un file .frd."""

    def __init__(self):
        self.steps: list[StepFRD] = []
        self.nodi: dict[int, tuple[float, float, float]] = {}

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def get_step(self, idx: int) -> Optional[StepFRD]:
        if 0 <= idx < len(self.steps):
            return self.steps[idx]
        return None

    def get_ultimo_step(self) -> Optional[StepFRD]:
        return self.steps[-1] if self.steps else None

    def max_spostamento(self, step_idx: int = -1) -> float:
        """Ritorna il massimo spostamento (norma) per lo step indicato."""
        step = self.steps[step_idx] if self.steps else None
        if not step or not step.spostamenti:
            return 0.0
        max_u = 0.0
        for ux, uy, uz in step.spostamenti.values():
            u = (ux**2 + uy**2 + uz**2) ** 0.5
            if u > max_u:
                max_u = u
        return max_u

    def max_stress_vm(self, step_idx: int = -1) -> float:
        step = self.steps[step_idx] if self.steps else None
        if not step or not step.stress_vm:
            return 0.0
        return max(step.stress_vm.values())


# ── Utilita' per parsing colonne fisse ──────────────────────────────

def _parse_frd_floats(line: str, start: int, width: int, count: int) -> list[float]:
    """Legge 'count' valori float da colonne fisse (larghezza 'width')."""
    vals = []
    for i in range(count):
        col_start = start + i * width
        col_end = col_start + width
        campo = line[col_start:col_end]
        if not campo.strip():
            break
        try:
            vals.append(float(campo))
        except ValueError:
            break
    return vals


def _parse_frd_node_id(line: str) -> int:
    """Legge il node ID dalle colonne 3-13 di una riga ' -1'."""
    return int(line[3:13])


# ══════════════════════════════════════════════════════════════════════
# PARSER .FRD
# ══════════════════════════════════════════════════════════════════════

def leggi_frd(percorso: str) -> Optional[RisultatiFRD]:
    """
    Legge un file .frd di CalculiX (formato ASCII a colonne fisse).
    Ritorna None se il file non esiste o non e' parsabile.
    """
    if not os.path.exists(percorso):
        print(f"WARN  File .frd non trovato: {percorso}")
        return None

    try:
        with open(percorso, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"ERR  Lettura .frd fallita: {e}")
        return None

    risultati = RisultatiFRD()
    i = 0
    n = len(lines)

    current_step: StepFRD | None = None
    current_block_type: str | None = None  # "DISP", "STRESS", "STRAIN"
    pending_step_time: float = 0.0
    steps_by_time: dict[float, StepFRD] = {}

    while i < n:
        line = lines[i]

        # ── Blocco nodi (coordinate) ──
        # "    2C" header, poi righe " -1" con node_id x y z
        if line.startswith("    2C"):
            i += 1
            while i < n and lines[i].startswith(" -1"):
                row = lines[i]
                if len(row) >= 49:
                    try:
                        nid = _parse_frd_node_id(row)
                        vals = _parse_frd_floats(row, 13, 12, 3)
                        if len(vals) == 3:
                            risultati.nodi[nid] = (vals[0], vals[1], vals[2])
                    except (ValueError, IndexError):
                        pass
                i += 1
            continue

        # ── Inizio di un blocco risultati ──
        # " -4  DISP" / " -4  STRESS" / " -4  TOSTRAIN"
        if line.startswith(" -4"):
            block_name = line[4:].strip().split()[0] if len(line) > 4 else ""
            if "DISP" in block_name:
                current_block_type = "DISP"
            elif "STRESS" in block_name:
                current_block_type = "STRESS"
            elif "STRAIN" in block_name or "TOSTRAIN" in block_name:
                current_block_type = "STRAIN"
            else:
                current_block_type = None
            i += 1
            continue

        # ── Linee di step / time ──
        # "    1PSTEP" segna inizio info step
        # " 100CL" contiene il tempo dello step
        # Entrambi precedono ogni blocco risultati, ma piu' blocchi
        # (DISP, STRESS, STRAIN, ERROR) appartengono allo stesso step.
        # Creiamo uno step solo se il tempo e' nuovo.
        if "1PSTEP" in line[:12]:
            # Solo segnalazione, il tempo e' nella riga 100CL successiva
            i += 1
            continue

        if "100CL" in line[:10]:
            step_time = 0.0
            try:
                # Il tempo e' nelle colonne 12-24 circa
                parts = line.split()
                for p in parts:
                    try:
                        val = float(p)
                        if val > 0:
                            step_time = val
                            break
                    except ValueError:
                        continue
            except Exception:
                pass

            # Riusa lo step se ha lo stesso tempo, altrimenti crea nuovo
            t_key = round(step_time, 10)
            if t_key in steps_by_time:
                current_step = steps_by_time[t_key]
            else:
                step_num = len(risultati.steps) + 1
                current_step = StepFRD(step_num, step_time)
                risultati.steps.append(current_step)
                steps_by_time[t_key] = current_step
            i += 1
            continue

        # ── Dati risultati (" -1" righe dentro un blocco) ──
        if line.startswith(" -1") and current_step is not None and current_block_type:
            if len(line) >= 25:
                try:
                    nid = _parse_frd_node_id(line)
                except (ValueError, IndexError):
                    i += 1
                    continue

                if current_block_type == "DISP":
                    vals = _parse_frd_floats(line, 13, 12, 3)
                    if len(vals) == 3:
                        current_step.spostamenti[nid] = (vals[0], vals[1], vals[2])

                elif current_block_type == "STRESS":
                    vals = _parse_frd_floats(line, 13, 12, 6)
                    if len(vals) >= 6:
                        sxx, syy, szz, sxy, syz, sxz = vals[:6]
                        current_step.stress_comp[nid] = (sxx, syy, szz, sxy, syz, sxz)
                        # Von Mises
                        vm = ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2
                              + 6.0*(sxy**2 + syz**2 + sxz**2)) ** 0.5 / (2.0**0.5)
                        current_step.stress_vm[nid] = vm
                    elif len(vals) >= 1:
                        current_step.stress_vm[nid] = abs(vals[0])

            i += 1
            continue

        # ── Fine blocco (" -3") ──
        if line.startswith(" -3"):
            current_block_type = None

        i += 1

    # Se nessuno step e' stato creato esplicitamente ma ci sono dati,
    # crea uno step di default
    if not risultati.steps:
        risultati.steps.append(StepFRD(1, 1.0))

    return risultati


# ══════════════════════════════════════════════════════════════════════
# PARSER .DAT (fallback)
# ══════════════════════════════════════════════════════════════════════

def leggi_dat(percorso: str) -> Optional[dict]:
    """
    Legge un file .dat (output tabulare CalculiX) come fallback.
    Estrae spostamenti e reazioni.
    """
    if not os.path.exists(percorso):
        return None

    risultato = {"spostamenti": {}, "reazioni": {}}

    try:
        with open(percorso, "r", encoding="utf-8", errors="replace") as f:
            sezione = None
            for line in f:
                line = line.strip()
                if "displacements" in line.lower():
                    sezione = "spostamenti"
                    continue
                elif "forces" in line.lower() or "reaction" in line.lower():
                    sezione = "reazioni"
                    continue
                elif not line or line.startswith("*"):
                    continue

                if sezione:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            nid = int(parts[0])
                            vx = float(parts[1])
                            vy = float(parts[2])
                            vz = float(parts[3])
                            risultato[sezione][nid] = (vx, vy, vz)
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        print(f"ERR  Lettura .dat fallita: {e}")
        return None

    return risultato
