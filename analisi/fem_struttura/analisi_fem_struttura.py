"""
analisi_fem_struttura.py
------------------------
Motore di analisi lineare per strutture a telaio tramite OpenSeesPy.

Gestisce:
  - Creazione del modello 3D (ndm=3, ndf=6)
  - Materiali elastici isotropi
  - Elementi elasticBeamColumn per le aste
  - Elementi ShellMITC4 / ShellDKGT per le shell
  - Vincoli, carichi nodali, distribuiti sulle aste e sulle shell
  - Peso proprio (opzionale)
  - Estrazione dei risultati: sforzi interni, spostamenti, tensioni shell
"""
from __future__ import annotations

import math
import os
import json
from datetime import datetime
from typing import Optional

import numpy as np

from .generatore_mesh_struttura import MeshStruttura, ElementoBeamMesh
from .risultati_struttura import (
    RisultatiFEMStruttura, SforziAsta, SpostamentoNodo, TensioniShell
)


# ==============================================================================
#  UTILITY
# ==============================================================================

def _vettore_trasformazione(xi, yi, zi, xj, yj, zj):
    """
    Calcola il vettore vecxz per la trasformazione geometrica di un beam 3D.
    Ritorna anche il sistema di riferimento locale (e_x, e_y, e_z).
    """
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    L = math.sqrt(dx * dx + dy * dy + dz * dz)
    if L < 1e-12:
        return (0, 0, 1), np.eye(3)

    e_x = np.array([dx, dy, dz]) / L

    # Scelta del vettore ausiliario
    if abs(e_x[2]) > 0.95:
        # Asta quasi verticale -> usa asse X globale
        v_aux = np.array([1.0, 0.0, 0.0])
    else:
        v_aux = np.array([0.0, 0.0, 1.0])

    e_y = np.cross(v_aux, e_x)
    norm_ey = np.linalg.norm(e_y)
    if norm_ey < 1e-12:
        v_aux = np.array([1.0, 0.0, 0.0])
        e_y = np.cross(v_aux, e_x)
        norm_ey = np.linalg.norm(e_y)
    e_y /= norm_ey
    e_z = np.cross(e_x, e_y)

    # vecxz: vettore nel piano xz locale (usato da OpenSees)
    vecxz = tuple(float(c) for c in e_z)
    R = np.vstack([e_x, e_y, e_z])  # Matrice rotazione 3x3

    return vecxz, R


def _trasforma_carico_globale_a_locale(R, wx, wy, wz):
    """Trasforma un carico distribuito da globale a coordinate locali."""
    w_global = np.array([wx, wy, wz])
    w_local = R @ w_global
    return float(w_local[0]), float(w_local[1]), float(w_local[2])


# ==============================================================================
#  ANALISI
# ==============================================================================

class AnalisiStruttura:
    """Esegue l'analisi lineare elastica della struttura con OpenSees."""

    def __init__(self) -> None:
        self._ops = None  # modulo openseespy.opensees

    def esegui(self, mesh: MeshStruttura,
               materiali: dict[int, dict],
               sezioni: dict[int, dict],
               dati_struttura: dict,
               gravita: float = 9.81,
               peso_proprio: bool = True,
               progress_cb=None) -> RisultatiFEMStruttura:
        """
        Esegue l'analisi lineare.

        Parametri
        ---------
        mesh : MeshStruttura
            Mesh discretizzata.
        materiali : dict
            {mid: {"nome", "densita", "E", "G", "J"}} risolti.
        sezioni : dict
            {sid: {"nome", "Area", "Iy", "Iz", "J_torsione",
                   "E_ref", "G_ref", "materiale_ref"}} risolte.
        dati_struttura : dict
            Dati parsati originali (per carichi e aste).
        gravita : float
            Accelerazione di gravita' [m/s^2].
        peso_proprio : bool
            Se True include il peso proprio.
        progress_cb : callable, optional
            Callback(int) avanzamento 0-100.

        Ritorna
        -------
        RisultatiFEMStruttura
        """
        risultati = RisultatiFEMStruttura()

        try:
            import openseespy.opensees as ops
            self._ops = ops
        except ImportError:
            risultati.messaggio = (
                "OpenSeesPy non installato. Installa con: pip install openseespy"
            )
            print(f"ERR  {risultati.messaggio}")
            return risultati

        if progress_cb:
            progress_cb(5)

        try:
            self._costruisci_modello(mesh, materiali, sezioni,
                                     dati_struttura, gravita, peso_proprio,
                                     progress_cb)

            if progress_cb:
                progress_cb(60)

            # Analisi
            self._esegui_analisi()

            if progress_cb:
                progress_cb(75)

            # Estrazione risultati
            self._estrai_risultati(mesh, materiali, sezioni,
                                   dati_struttura, risultati)

            risultati.successo = True
            risultati.messaggio = "Analisi completata con successo."

            if progress_cb:
                progress_cb(95)

        except Exception as e:
            risultati.messaggio = f"Errore durante l'analisi: {e}"
            print(f"ERR  FEM Struttura: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                self._ops.wipe()
            except Exception:
                pass

        if progress_cb:
            progress_cb(100)

        return risultati

    # ------------------------------------------------------------------
    #  COSTRUZIONE MODELLO
    # ------------------------------------------------------------------

    def _costruisci_modello(self, mesh, materiali, sezioni,
                            dati_struttura, gravita, peso_proprio,
                            progress_cb):
        ops = self._ops
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        # ---- Nodi ----
        for tag, nodo in mesh.nodi.items():
            ops.node(tag, nodo.x, nodo.y, nodo.z)

        # ---- Vincoli ----
        for tag, vals in mesh.vincoli.items():
            ops.fix(tag, *[int(v) for v in vals])

        if progress_cb:
            progress_cb(15)

        # ---- Trasformazioni geometriche (una per asta originale) ----
        aste_orig = dati_struttura.get("aste", {})
        transf_tags: dict[int, int] = {}  # asta_id -> transf_tag
        transf_rotazioni: dict[int, np.ndarray] = {}
        transf_counter = 1

        for bid in sorted(aste_orig.keys()):
            nodi_asta = mesh.mappa_nodi_aste.get(bid, [])
            if len(nodi_asta) < 2:
                continue
            n_i = mesh.nodi[nodi_asta[0]]
            n_j = mesh.nodi[nodi_asta[-1]]
            vecxz, R = _vettore_trasformazione(n_i.x, n_i.y, n_i.z,
                                               n_j.x, n_j.y, n_j.z)
            ops.geomTransf('Linear', transf_counter, *vecxz)
            transf_tags[bid] = transf_counter
            transf_rotazioni[bid] = R
            transf_counter += 1

        if progress_cb:
            progress_cb(25)

        # ---- Mappa sezione -> proprieta' ----
        # Le sezioni nella struttura sono riferite per nome o id
        sez_per_nome: dict[str, dict] = {}
        for sid, sd in sezioni.items():
            sez_per_nome[sd["nome"]] = sd
            sez_per_nome[str(sid)] = sd

        # ---- Elementi beam ----
        for elem in mesh.elementi_beam:
            sez_ref = elem.sezione
            sez_dati = sez_per_nome.get(sez_ref, {})

            A = float(sez_dati.get("Area", 0.01))
            E = float(sez_dati.get("E_ref", 30000.0)) * 1e6  # MPa -> Pa
            G = float(sez_dati.get("G_ref", 12500.0)) * 1e6  # MPa -> Pa
            Iy = float(sez_dati.get("Iy", 1e-4))
            Iz = float(sez_dati.get("Iz", 1e-4))
            J = float(sez_dati.get("J_torsione", 1e-4))

            bid = elem.asta_originale
            tt = transf_tags.get(bid, 1)

            # elasticBeamColumn eleTag iNode jNode A E G J Iy Iz transfTag
            ops.element('elasticBeamColumn', elem.tag,
                        elem.nodo_i, elem.nodo_j,
                        A, E, G, J, Iy, Iz, tt)

        if progress_cb:
            progress_cb(35)

        # ---- Materiali e sezioni shell ----
        mat_per_nome: dict[str, dict] = {}
        for mid, md in materiali.items():
            mat_per_nome[md["nome"]] = md
            mat_per_nome[str(mid)] = md

        shell_sec_map: dict[str, int] = {}  # chiave -> secTag
        shell_sec_counter = 1000

        for elem_shell in mesh.elementi_shell:
            mat_nome = elem_shell.materiale
            sp = elem_shell.spessore

            sec_key = f"{mat_nome}_{sp:.6f}"
            if sec_key not in shell_sec_map:
                md = mat_per_nome.get(mat_nome, {})
                E_sh = float(md.get("E", 30000.0)) * 1e6  # Pa
                poisson = 0.2
                rho_sh = float(md.get("densita", 2500.0))

                # nDMaterial
                mat_tag_nd = shell_sec_counter + 5000
                try:
                    ops.nDMaterial('ElasticIsotropic', mat_tag_nd,
                                  E_sh, poisson, rho_sh)
                except Exception:
                    pass

                # Shell section
                ops.section('ElasticMembranePlateSection', shell_sec_counter,
                            E_sh, poisson, sp, rho_sh)
                shell_sec_map[sec_key] = shell_sec_counter
                shell_sec_counter += 1

            sec_tag = shell_sec_map[sec_key]

            n_nodi = len(elem_shell.nodi)
            if n_nodi == 4:
                ops.element('ShellMITC4', elem_shell.tag,
                            *elem_shell.nodi, sec_tag)
            elif n_nodi == 3:
                try:
                    ops.element('ShellDKGT', elem_shell.tag,
                                *elem_shell.nodi, sec_tag)
                except Exception:
                    # Fallback: tri3 non supportato in tutte le versioni
                    ops.element('ShellNLDKGT', elem_shell.tag,
                                *elem_shell.nodi, sec_tag)

        if progress_cb:
            progress_cb(45)

        # ---- Carichi ----
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)

        # Carichi nodali
        for carico in mesh.carichi_nodali:
            tag, fx, fy, fz = carico
            # Converti kN -> N (OpenSees lavora con unita' SI base)
            ops.load(tag, fx * 1e3, fy * 1e3, fz * 1e3, 0.0, 0.0, 0.0)

        # Carichi distribuiti sulle aste
        for carico in mesh.carichi_distribuiti:
            bid, wx, wy, wz = carico
            # wx, wy, wz in kN/m -> N/m
            wx_N = wx * 1e3
            wy_N = wy * 1e3
            wz_N = wz * 1e3

            elem_tags = mesh.mappa_aste.get(bid, [])
            R = transf_rotazioni.get(bid, np.eye(3))

            for etag in elem_tags:
                # Trasforma da globale a locale
                w_loc_x, w_loc_y, w_loc_z = _trasforma_carico_globale_a_locale(
                    R, wx_N, wy_N, wz_N
                )
                try:
                    ops.eleLoad('-ele', etag, '-type', '-beamUniform',
                                w_loc_y, w_loc_z, w_loc_x)
                except Exception as e:
                    # Fallback: applica come forze nodali equivalenti
                    self._applica_carico_nodale_equivalente(
                        mesh, etag, wx_N, wy_N, wz_N)

        # Carichi sulle shell (convertiti a forze nodali equivalenti)
        shell_orig = dati_struttura.get("shell", {})
        for carico in mesh.carichi_shell:
            sid, qx, qy, qz = carico
            # qx, qy, qz in kN/m^2 -> N/m^2
            for elem_shell in mesh.elementi_shell:
                if elem_shell.shell_originale == sid:
                    self._applica_carico_shell_nodale(
                        mesh, elem_shell, qx * 1e3, qy * 1e3, qz * 1e3)

        # Peso proprio
        if peso_proprio:
            self._applica_peso_proprio(mesh, materiali, sezioni,
                                       dati_struttura, gravita,
                                       sez_per_nome, mat_per_nome,
                                       transf_rotazioni)

        if progress_cb:
            progress_cb(55)

    def _applica_carico_nodale_equivalente(self, mesh, etag, wx, wy, wz):
        """Applica carico distribuito come forze nodali equivalenti."""
        ops = self._ops
        for elem in mesh.elementi_beam:
            if elem.tag == etag:
                L = elem.lunghezza
                F = 0.5 * L
                ops.load(elem.nodo_i, wx * F, wy * F, wz * F, 0, 0, 0)
                ops.load(elem.nodo_j, wx * F, wy * F, wz * F, 0, 0, 0)
                break

    def _applica_carico_shell_nodale(self, mesh, elem_shell, qx, qy, qz):
        """Converte carico superficiale shell in forze nodali equivalenti."""
        ops = self._ops
        nodi = elem_shell.nodi
        n = len(nodi)
        if n < 3:
            return

        # Calcola area approssimata dell'elemento
        coords = [mesh.nodi[t] for t in nodi]
        if n == 3:
            v1 = np.array([coords[1].x - coords[0].x,
                           coords[1].y - coords[0].y,
                           coords[1].z - coords[0].z])
            v2 = np.array([coords[2].x - coords[0].x,
                           coords[2].y - coords[0].y,
                           coords[2].z - coords[0].z])
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        else:
            v1 = np.array([coords[2].x - coords[0].x,
                           coords[2].y - coords[0].y,
                           coords[2].z - coords[0].z])
            v2 = np.array([coords[3].x - coords[0].x,
                           coords[3].y - coords[0].y,
                           coords[3].z - coords[0].z])
            v3 = np.array([coords[1].x - coords[0].x,
                           coords[1].y - coords[0].y,
                           coords[1].z - coords[0].z])
            area = 0.5 * (np.linalg.norm(np.cross(v1, v3)) +
                          np.linalg.norm(np.cross(v1, v2)))

        F_per_nodo = area / n
        for nodo_tag in nodi:
            ops.load(nodo_tag, qx * F_per_nodo, qy * F_per_nodo,
                     qz * F_per_nodo, 0, 0, 0)

    def _applica_peso_proprio(self, mesh, materiali, sezioni,
                              dati_struttura, gravita,
                              sez_per_nome, mat_per_nome,
                              transf_rotazioni):
        """Applica il peso proprio come carico distribuito sulle aste
        e come forze nodali sulle shell."""
        ops = self._ops
        aste_orig = dati_struttura.get("aste", {})

        # ---- Peso proprio aste ----
        for bid, asta in aste_orig.items():
            sez_ref = asta.get("sezione", "")
            sez_dati = sez_per_nome.get(sez_ref, {})
            A = float(sez_dati.get("Area", 0.01))
            mat_ref = sez_dati.get("materiale_ref", "")
            md = mat_per_nome.get(mat_ref, {})
            rho = float(md.get("densita", 2500.0))

            # Peso per unita' di lunghezza [N/m]
            w_self = rho * A * gravita

            # Applicato in direzione -Z globale
            elem_tags = mesh.mappa_aste.get(bid, [])
            R = transf_rotazioni.get(bid, np.eye(3))

            for etag in elem_tags:
                _, w_loc_y, w_loc_z = _trasforma_carico_globale_a_locale(
                    R, 0.0, 0.0, -w_self
                )
                w_loc_x = 0.0
                try:
                    ops.eleLoad('-ele', etag, '-type', '-beamUniform',
                                w_loc_y, w_loc_z, w_loc_x)
                except Exception:
                    self._applica_carico_nodale_equivalente(
                        mesh, etag, 0.0, 0.0, -w_self)

        # ---- Peso proprio shell ----
        for elem_shell in mesh.elementi_shell:
            md = mat_per_nome.get(elem_shell.materiale, {})
            rho = float(md.get("densita", 2500.0))
            sp = elem_shell.spessore
            w_shell = rho * sp * gravita  # N/m^2
            self._applica_carico_shell_nodale(
                mesh, elem_shell, 0.0, 0.0, -w_shell)

    # ------------------------------------------------------------------
    #  ANALISI
    # ------------------------------------------------------------------

    def _esegui_analisi(self):
        ops = self._ops
        ops.system('BandSPD')
        ops.numberer('RCM')
        ops.constraints('Plain')
        ops.integrator('LoadControl', 1.0)
        ops.algorithm('Linear')
        ops.analysis('Static')

        result = ops.analyze(1)
        if result != 0:
            # Prova con un solver diverso
            ops.system('UmfPack')
            ops.analysis('Static')
            result = ops.analyze(1)
            if result != 0:
                raise RuntimeError(
                    f"L'analisi non e' convergita (codice {result}). "
                    "Verificare vincoli e carichi."
                )

    # ------------------------------------------------------------------
    #  ESTRAZIONE RISULTATI
    # ------------------------------------------------------------------

    def _estrai_risultati(self, mesh, materiali, sezioni,
                          dati_struttura, risultati):
        ops = self._ops

        # ---- Spostamenti ----
        for tag in mesh.nodi:
            try:
                disp = ops.nodeDisp(tag)
                risultati.spostamenti[tag] = SpostamentoNodo(
                    nodo_tag=tag,
                    dx=disp[0], dy=disp[1], dz=disp[2],
                    rx=disp[3], ry=disp[4], rz=disp[5],
                )
            except Exception:
                pass

        # ---- Reazioni vincolari ----
        # OpenSees calcola le reazioni solo se richiesto esplicitamente:
        # senza questa chiamata, nodeReaction(tag) restituisce zeri.
        try:
            ops.reactions()
        except Exception:
            pass
        for tag in mesh.vincoli:
            try:
                react = ops.nodeReaction(tag)
                risultati.reazioni[tag] = tuple(react[:6])
            except Exception:
                pass

        # ---- Sforzi interni aste ----
        aste_orig = dati_struttura.get("aste", {})
        # Mappa sezione per nome/id
        sez_per_nome: dict[str, dict] = {}
        for sid, sd in sezioni.items():
            sez_per_nome[sd["nome"]] = sd
            sez_per_nome[str(sid)] = sd

        for bid in sorted(aste_orig.keys()):
            elem_tags = mesh.mappa_aste.get(bid, [])
            nodi_asta = mesh.mappa_nodi_aste.get(bid, [])
            if not elem_tags:
                continue

            # Proprieta' sezione per il calcolo della tensione equivalente
            asta = aste_orig.get(bid, {})
            sez_dati = sez_per_nome.get(asta.get("sezione", ""), {})
            A_sez = float(sez_dati.get("Area", 0.0))
            Iy_sez = float(sez_dati.get("Iy", 0.0))
            Iz_sez = float(sez_dati.get("Iz", 0.0))

            sf = SforziAsta(asta_id=bid)

            # Calcola posizioni lungo l'asta
            pos_corrente = 0.0

            for idx, etag in enumerate(elem_tags):
                try:
                    forces = ops.eleForce(etag)
                except Exception:
                    continue

                # OpenSees elasticBeamColumn restituisce 12 valori:
                # [N_i, Vy_i, Vz_i, T_i, My_i, Mz_i,
                #  N_j, Vy_j, Vz_j, T_j, My_j, Mz_j]
                # Nota: le forze al nodo i hanno segno opposto alla
                # convenzione di Ingegneria Civile per N, V, M
                if len(forces) < 12:
                    continue

                # Trova lunghezza sotto-elemento
                for e in mesh.elementi_beam:
                    if e.tag == etag:
                        L_sub = e.lunghezza
                        break
                else:
                    L_sub = 1.0

                if idx == 0:
                    # Primo sotto-elemento: aggiungi nodo iniziale
                    sf.posizioni.append(pos_corrente)
                    # Forze al nodo i (segno: N positivo = trazione)
                    # Converti da N a kN e da Nm a kNm
                    sf.N.append(-forces[0] / 1e3)
                    sf.Vy.append(forces[1] / 1e3)
                    sf.Vz.append(forces[2] / 1e3)
                    sf.T.append(forces[3] / 1e3)
                    sf.My.append(forces[4] / 1e3)
                    sf.Mz.append(forces[5] / 1e3)

                # Aggiungi nodo finale del sotto-elemento
                pos_corrente += L_sub
                sf.posizioni.append(pos_corrente)
                # Forze al nodo j (cambio segno per convenzione)
                sf.N.append(forces[6] / 1e3)
                sf.Vy.append(-forces[7] / 1e3)
                sf.Vz.append(-forces[8] / 1e3)
                sf.T.append(-forces[9] / 1e3)
                sf.My.append(-forces[10] / 1e3)
                sf.Mz.append(-forces[11] / 1e3)

            # Calcola tensione equivalente [MPa] in ogni nodo
            # sigma_eq = |N|/A + sqrt((My/sqrt(Iy*A))^2 + (Mz/sqrt(Iz*A))^2)
            # (riferita al raggio d'inerzia, come stima della massima fibra)
            if A_sez > 1e-12 and len(sf.N) > 0:
                sqrt_Iy_A = math.sqrt(max(Iy_sez, 1e-12) * A_sez)
                sqrt_Iz_A = math.sqrt(max(Iz_sez, 1e-12) * A_sez)
                for i in range(len(sf.N)):
                    # Conversione a Pa: kN->N (*1e3), kNm->Nm (*1e3)
                    s_N  = abs(sf.N[i])  * 1e3 / A_sez
                    s_My = abs(sf.My[i]) * 1e3 / sqrt_Iy_A if Iy_sez > 0 else 0.0
                    s_Mz = abs(sf.Mz[i]) * 1e3 / sqrt_Iz_A if Iz_sez > 0 else 0.0
                    sigma_pa = s_N + math.sqrt(s_My * s_My + s_Mz * s_Mz)
                    sf.sigma_eq.append(sigma_pa / 1e6)   # Pa -> MPa
            else:
                sf.sigma_eq = [0.0] * len(sf.N)

            risultati.sforzi_aste[bid] = sf

        # ---- Tensioni shell ----
        # OpenSees ShellMITC4 / ShellDKGT restituiscono per 'stresses' i
        # risultanti di sforzo AI PUNTI DI INTEGRAZIONE (NON per nodo):
        #   per IP: [Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Vxz, Vyz]  (8 componenti)
        # Procedura:
        #   1. leggi 'stresses' (con fallback 'force'/'forces')
        #   2. media Nxx/Nyy/Nxy sugli IP -> risultanti membranali [N/m]
        #   3. converti a tensione dividendo per lo spessore [Pa] -> [MPa]
        #   4. assegna la stessa sigma_vm a tutti i nodi dell'elemento
        n_shell_ok = 0
        n_shell_ko = 0
        for elem_shell in mesh.elementi_shell:
            stresses = None
            for resp_name in ('stresses', 'force', 'forces'):
                try:
                    r = ops.eleResponse(elem_shell.tag, resp_name)
                    if r and len(r) >= 3:
                        stresses = list(r)
                        break
                except Exception:
                    continue

            if stresses is None:
                n_shell_ko += 1
                continue

            n_tot = len(stresses)
            # ElasticMembranePlateSection fornisce 8 componenti per IP.
            # Fallback su 6 (no taglio trasverso) o 3 (solo membrana).
            if n_tot % 8 == 0 and n_tot >= 8:
                n_comp = 8
            elif n_tot % 6 == 0 and n_tot >= 6:
                n_comp = 6
            elif n_tot % 3 == 0 and n_tot >= 3:
                n_comp = 3
            else:
                n_shell_ko += 1
                continue
            n_ip = max(1, n_tot // n_comp)

            nxx_sum = nyy_sum = nxy_sum = 0.0
            for ip in range(n_ip):
                off = ip * n_comp
                nxx_sum += stresses[off + 0]
                nyy_sum += stresses[off + 1]
                nxy_sum += stresses[off + 2]
            nxx = nxx_sum / n_ip
            nyy = nyy_sum / n_ip
            nxy = nxy_sum / n_ip

            # Converti N/m -> Pa -> MPa
            sp = max(float(elem_shell.spessore), 1e-9)
            sx  = (nxx / sp) / 1e6
            sy  = (nyy / sp) / 1e6
            sxy = (nxy / sp) / 1e6

            sigma_vm = math.sqrt(sx * sx + sy * sy - sx * sy + 3.0 * sxy * sxy)

            ts = TensioniShell(
                shell_tag=elem_shell.tag,
                shell_originale=elem_shell.shell_originale,
            )
            for nodo_tag in elem_shell.nodi:
                ts.tensioni_nodi[nodo_tag] = (sx, sy, sxy, sigma_vm)
            risultati.tensioni_shell.append(ts)
            n_shell_ok += 1

        if mesh.elementi_shell:
            print(f">> FEM Struttura: tensioni shell estratte su "
                  f"{n_shell_ok}/{len(mesh.elementi_shell)} elementi "
                  f"(falliti: {n_shell_ko})")

    # ------------------------------------------------------------------
    #  SALVATAGGIO RISULTATI
    # ------------------------------------------------------------------

    def salva_risultati(self, risultati: RisultatiFEMStruttura,
                        nome_struttura: str,
                        cartella: str) -> str:
        """
        Salva i risultati in formato JSON nella cartella specificata.
        Ritorna il percorso del file salvato.
        """
        os.makedirs(cartella, exist_ok=True)

        nome_safe = nome_struttura.replace(" ", "_").replace(".", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{nome_safe}_{ts}.json"
        filepath = os.path.join(cartella, filename)

        dati_export = {
            "metadata": {
                "nome_struttura": nome_struttura,
                "data_analisi": datetime.now().isoformat(),
                "successo": risultati.successo,
                "messaggio": risultati.messaggio,
                "max_spostamento": risultati.max_spostamento,
            },
            "sforzi_aste": {},
            "spostamenti": {},
            "reazioni": {},
        }

        for bid, sf in risultati.sforzi_aste.items():
            dati_export["sforzi_aste"][str(bid)] = {
                "posizioni": sf.posizioni,
                "N": sf.N, "Vy": sf.Vy, "Vz": sf.Vz,
                "My": sf.My, "Mz": sf.Mz, "T": sf.T,
                "sigma_eq": sf.sigma_eq,
            }

        for tag, sp in risultati.spostamenti.items():
            dati_export["spostamenti"][str(tag)] = {
                "dx": sp.dx, "dy": sp.dy, "dz": sp.dz,
                "rx": sp.rx, "ry": sp.ry, "rz": sp.rz,
            }

        for tag, react in risultati.reazioni.items():
            dati_export["reazioni"][str(tag)] = list(react)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(dati_export, f, indent=2, ensure_ascii=False)
            print(f">> FEM Struttura: risultati salvati -> {filepath}")
        except Exception as e:
            print(f"ERR  Salvataggio risultati: {e}")
            filepath = ""

        return filepath
