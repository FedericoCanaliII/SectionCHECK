"""
fem_concrete_3d_opengl_with_rebars_and_stirrups.py
Versione estesa del solver FEM non-lineare con visualizzazione OpenGL.
Aggiunta: visualizzazione delle fessure della trave (bianco) e rottura delle barre
quando vengono superati gli stati limite ultimo definiti dall'utente.

Regole implementate per la fessurazione / rottura (semplificate, didattiche):
 - Calcestruzzo: si considera fessurato in un elemento se in uno qualsiasi dei
   punti di integrazione risulta in trazione (eps_m > 0) oppure in compressione
   con deformazione compressiva maggiore di eps_u (es: 0.0035). Tutti i nodi
   dell'elemento verranno marcati come fessurati e colorati di bianco.
 - Acciaio (barre): un elemento trave viene considerato rotto se lo sforzo
   assiale produce una deformazione assoluta maggiore di `eps_max` (es: 0.7).
   Le barre rotte vengono disegnate in bianco.

Nota: questo meccanismo e' puramente illustrativo. Per analisi realistiche
servono modelli di danno/fessurazione più complessi (smeared cracks, cohesive
zone, discrete crack tracking, ecc.).

Controlli UI:
 - rotazione/pan/zoom come prima
 - W/F, SPACE per deformata
 - B per mostrare/nascondere barre longitudinali
 - V per mostrare/nascondere staffe (stirrups)
 - C per mostrare/nascondere la visualizzazione delle fessure/rotture

Prerequisiti: numpy, scipy, pygame, PyOpenGL
pip install numpy scipy pygame PyOpenGL PyOpenGL_accelerate
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import sys

# -----------------------------
# Geometria / mesh (parallelepipedo)
# -----------------------------
Lx, Ly, Lz = 4.0, 0.3, 0.4  # m
nx, ny, nz = 40, 9, 12        # elementi (attenzione: aumentano i tempi)

# -----------------------------
# USER CONFIG: vincoli e carichi per facce (piu' di una)
# -----------------------------
constrained_faces = ['X0', 'Xl']  # vincolo su piu' facce
load_faces = ['zL']
load_direction = np.array([0.0, 0.0, 1.0])

# -----------------------------
# Materiale calcestruzzo (parabola-rettangolo)
# -----------------------------
f_ck = 25e6
f_cm = f_ck + 8e6
eps0 = 0.002
eps_u = 0.0035
E_c = 22e9 * (f_cm/10e6)**0.3
E_c = float(E_c)
nu_c = 0.2
mu_c = E_c / (2*(1+nu_c))
K_linear = E_c / (3*(1-2*nu_c))

# -----------------------------
# Materiale acciaio B450C (non-lineare perfettamente plastico)
# -----------------------------
E_s = 210e9
f_y = 450e6
eps_y = 0.002
eps_max = 0.7

# -----------------------------
# Carico e solver
# -----------------------------
force_total = 500000.0
apply_distributed = True
n_steps = 1
tol_res = 1e-6
max_iter = 1

# barra params (utente puo' cambiare)
rebar_offset = 0.03   # distanza dal bordo verso l'interno (m)
rebar_diameter = None   # m (es.: 0.012)
rebar_area_default = 1e-4  # m^2

# staffe (stirrups) params
stirrup_diameter = 0.008   # m (8 mm)
stirrup_spacing = 0.10     # passo lungo l'asse (m)

# penalty spring (bar <-> solid) stiffness (N/m)
K_penalty = 1e12

# visual
exaggeration = 5e2

# -----------------------------
# Mesh solido (hex8)
# -----------------------------
def make_mesh(Lx, Ly, Lz, nx, ny, nz):
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    zs = np.linspace(0, Lz, nz + 1)
    coords = []
    idx = {}
    c = 0
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                coords.append([x, y, z])
                idx[(i, j, k)] = c
                c += 1
    coords = np.array(coords, dtype=float)
    elems = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = idx[(i, j, k)]
                n1 = idx[(i+1, j, k)]
                n2 = idx[(i+1, j+1, k)]
                n3 = idx[(i, j+1, k)]
                n4 = idx[(i, j, k+1)]
                n5 = idx[(i+1, j, k+1)]
                n6 = idx[(i+1, j+1, k+1)]
                n7 = idx[(i, j+1, k+1)]
                elems.append([n0,n1,n2,n3,n4,n5,n6,n7])
    return coords, np.array(elems, dtype=int)

coords_solid, elems = make_mesh(Lx, Ly, Lz, nx, ny, nz)

# -----------------------------
# Aggiungo nodi e elementi per le barre longitudinali (4 barre nei 4 angoli della sezione)
# -----------------------------
xs = np.linspace(0, Lx, nx + 1)
bar_positions_yz = [ (rebar_offset, rebar_offset),
                     (Ly - rebar_offset, rebar_offset),
                     (rebar_offset, Lz - rebar_offset),
                     (Ly - rebar_offset, Lz - rebar_offset) ]

coords = coords_solid.tolist()

# compute area for longitudinal bars
if rebar_diameter is not None and rebar_diameter > 0:
    rebar_area = math.pi * (rebar_diameter/2.0)**2
else:
    rebar_area = rebar_area_default

bar_nodes_indices = []
bar_elements = []

for b_id, (yb, zb) in enumerate(bar_positions_yz):
    nodes_line = []
    for xi in xs:
        coords.append([float(xi), float(yb), float(zb)])
        nodes_line.append(len(coords)-1)
    bar_nodes_indices.append(nodes_line)
    for i in range(len(nodes_line)-1):
        n1 = nodes_line[i]
        n2 = nodes_line[i+1]
        bar_elements.append((n1, n2, rebar_area, b_id, 'long'))

coords = np.array(coords, dtype=float)

# -----------------------------
# Aggiungo le staffe (stirrups)
# -----------------------------
if stirrup_diameter is not None and stirrup_diameter > 0:
    stirrup_area = math.pi * (stirrup_diameter/2.0)**2
else:
    stirrup_area = 5e-5

if stirrup_spacing <= 0 or stirrup_spacing >= Lx:
    stirrup_levels = list(range(len(xs)))
else:
    desired = np.arange(0, Lx+1e-12, stirrup_spacing)
    stirrup_levels = []
    for xv in desired:
        idx_closest = int(np.argmin(np.abs(xs - xv)))
        if idx_closest not in stirrup_levels:
            stirrup_levels.append(idx_closest)
    if 0 not in stirrup_levels:
        stirrup_levels.insert(0, 0)
    if len(xs)-1 not in stirrup_levels:
        stirrup_levels.append(len(xs)-1)

loop_order = [0, 1, 3, 2]
stirrup_id_base = 1000
for i_level in stirrup_levels:
    loop_nodes = [bar_nodes_indices[idx][i_level] for idx in loop_order]
    for j in range(4):
        n1 = loop_nodes[j]
        n2 = loop_nodes[(j+1)%4]
        bar_elements.append((n1, n2, stirrup_area, stirrup_id_base + i_level, 'stirrup'))

n_nodes = coords.shape[0]

# -----------------------------
# helper per faces: normalizzazione nomi e ricerca nodi
# -----------------------------

def normalize_face_name(face):
    s = str(face).strip()
    mapping = {
        'x0': ('x', 0.0), 'left': ('x', 0.0), 'x=0': ('x', 0.0),
        'xL': ('x', Lx), 'right': ('x', Lx), f'x={Lx}': ('x', Lx),
        'y0': ('y', 0.0), 'front': ('y', 0.0), 'y=0': ('y', 0.0),
        'yL': ('y', Ly), 'back': ('y', Ly), f'y={Ly}': ('y', Ly),
        'z0': ('z', 0.0), 'bottom': ('z', 0.0), 'z=0': ('z', 0.0),
        'zL': ('z', Lz), 'top': ('z', Lz), f'z={Lz}': ('z', Lz),
    }
    sl = s.lower()
    if sl in mapping:
        return mapping[sl]
    if sl.startswith('x'):
        if '0' in sl:
            return ('x', 0.0)
        else:
            return ('x', Lx)
    if sl.startswith('y'):
        if '0' in sl:
            return ('y', 0.0)
        else:
            return ('y', Ly)
    if sl.startswith('z'):
        if '0' in sl:
            return ('z', 0.0)
        else:
            return ('z', Lz)
    raise ValueError(f"Nome faccia non riconosciuto: {face}")


def get_face_nodes_solid(face_name, tol=1e-8):
    axis, val = normalize_face_name(face_name)
    if axis == 'x':
        idx = np.where(np.abs(coords_solid[:,0] - val) < tol)[0]
    elif axis == 'y':
        idx = np.where(np.abs(coords_solid[:,1] - val) < tol)[0]
    else:
        idx = np.where(np.abs(coords_solid[:,2] - val) < tol)[0]
    return idx


def get_face_nodes_all(face_name, tol=1e-8):
    axis, val = normalize_face_name(face_name)
    if axis == 'x':
        idx = np.where(np.abs(coords[:,0] - val) < tol)[0]
    elif axis == 'y':
        idx = np.where(np.abs(coords[:,1] - val) < tol)[0]
    else:
        idx = np.where(np.abs(coords[:,2] - val) < tol)[0]
    return idx

# -----------------------------
# Shape functions e precomputazioni per elementi hexa
# -----------------------------

def hexa_shape_and_derivs(xi, eta, zeta):
    s = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    N = 1/8.0 * np.array([(1 + si*xi)*(1 + sj*eta)*(1 + sk*zeta) for si,sj,sk in s])
    dN_dxi = 1/8.0 * np.array([ [si*(1+sj*eta)*(1+sk*zeta) for si,sj,sk in s] ])[0]
    dN_deta = 1/8.0 * np.array([ [(1+si*xi)*sj*(1+sk*zeta) for si,sj,sk in s] ])[0]
    dN_dzeta = 1/8.0 * np.array([ [(1+si*xi)*(1+sj*eta)*sk for si,sj,sk in s] ])[0]
    return N, dN_dxi, dN_deta, dN_dzeta


def build_B(dNdx, dNdy, dNdz):
    B = np.zeros((6, 24))
    for i in range(8):
        ix = 3*i
        B[0, ix    ] = dNdx[i]
        B[1, ix+1  ] = dNdy[i]
        B[2, ix+2  ] = dNdz[i]
        B[3, ix    ] = dNdy[i]
        B[3, ix+1  ] = dNdx[i]
        B[4, ix+1  ] = dNdz[i]
        B[4, ix+2  ] = dNdy[i]
        B[5, ix    ] = dNdz[i]
        B[5, ix+2  ] = dNdx[i]
    return B

# Gauss points 2x2x2
gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
weights = [1.0, 1.0]

# precompute B, vol per gp per elemento solido
elem_gp_data = []
for e, conn in enumerate(elems):
    xe = coords[conn]
    gp_list = []
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                N, dNdxi, dNdeta, dNdzet = hexa_shape_and_derivs(xi, eta, zeta)
                dN_dxi = np.vstack([dNdxi, dNdeta, dNdzet])
                J = dN_dxi @ xe
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise RuntimeError("Jacobian non positivo")
                invJ = np.linalg.inv(J)
                grads = invJ @ dN_dxi
                dNdx = grads[0,:]
                dNdy = grads[1,:]
                dNdz = grads[2,:]
                B = build_B(dNdx, dNdy, dNdz)
                w = 1.0
                vol = detJ * w
                gp_list.append((B, vol))
    elem_gp_data.append(gp_list)

# -----------------------------
# Vincoli e caricamento (ora multi-faccia)
# -----------------------------

n_dof = n_nodes * 3
F_ext_total = np.zeros(n_dof)

if isinstance(load_faces, dict):
    load_map = load_faces.copy()
elif isinstance(load_faces, (list, tuple)):
    if len(load_faces) == 0:
        load_map = {}
    else:
        share = force_total / float(len(load_faces))
        load_map = {face: share for face in load_faces}
else:
    load_map = {load_faces: force_total}

if apply_distributed:
    for face, fval in load_map.items():
        nodes_on_face = get_face_nodes_solid(face)
        if nodes_on_face.size == 0:
            print(f"Warning: nessun nodo solido trovato sulla faccia '{face}'")
            continue
        f_each = fval / float(nodes_on_face.size)
        for n in nodes_on_face:
            F_ext_total[3*n:3*n+3] += f_each * load_direction
else:
    for face, fval in load_map.items():
        nodes_on_face = get_face_nodes_solid(face)
        if nodes_on_face.size == 0:
            print(f"Warning: nessun nodo solido trovato sulla faccia '{face}'")
            continue
        pts = coords_solid[nodes_on_face]
        center = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        chosen = nodes_on_face[np.argmin(dists)]
        F_ext_total[3*chosen:3*chosen+3] += fval * load_direction

fixed_nodes_solid = np.array([], dtype=int)
for face in constrained_faces:
    nodes = get_face_nodes_solid(face)
    fixed_nodes_solid = np.union1d(fixed_nodes_solid, nodes)

fixed_dofs = []
for n in fixed_nodes_solid:
    fixed_dofs += [3*n, 3*n+1, 3*n+2]

# blocco anche le estremita' delle barre che cadono su una delle facce vincolate
tol_fix = 1e-8
for b_nodes in bar_nodes_indices:
    for bn in b_nodes:
        Xb = coords[bn]
        for face in constrained_faces:
            axis, val = normalize_face_name(face)
            if axis == 'x' and abs(Xb[0] - val) < tol_fix:
                fixed_dofs += [3*bn, 3*bn+1, 3*bn+2]
                break
            if axis == 'y' and abs(Xb[1] - val) < tol_fix:
                fixed_dofs += [3*bn, 3*bn+1, 3*bn+2]
                break
            if axis == 'z' and abs(Xb[2] - val) < tol_fix:
                fixed_dofs += [3*bn, 3*bn+1, 3*bn+2]
                break

fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)
free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

# -----------------------------
# helper costitutivi
# -----------------------------

def concrete_parabola_rect(eps):
    if eps <= 0.0:
        return 0.0, 0.0
    if eps <= eps0:
        r = eps/eps0
        sigma = f_cm * (2*r - r*r)
        dsigma = f_cm * (2/eps0 - 2*eps/(eps0*eps0))
        return float(sigma), float(dsigma)
    elif eps <= eps_u:
        sigma = f_cm
        dsigma = 0.0
        return float(sigma), float(dsigma)
    else:
        return 0.0, 0.0


def make_D_from_K_mu(Kt, mu):
    lam = Kt - 2.0*mu/3.0
    D = np.zeros((6,6))
    D[0,0] = lam + 2*mu; D[0,1] = lam;       D[0,2] = lam
    D[1,0] = lam;       D[1,1] = lam + 2*mu; D[1,2] = lam
    D[2,0] = lam;       D[2,1] = lam;       D[2,2] = lam + 2*mu
    D[3,3] = mu
    D[4,4] = mu
    D[5,5] = mu
    return D


def steel_b450c(eps):
    if abs(eps) <= eps_y:
        sigma = E_s * eps
        ds = E_s
    elif abs(eps) <= eps_max:
        sigma = math.copysign(f_y, eps)
        ds = 0.0
    else:
        sigma = math.copysign(f_y, eps)
        ds = 0.0
    return float(sigma), float(ds)

# -----------------------------
# Inizializzo spostamenti e iterativa non-lineare
# -----------------------------

u = np.zeros(n_dof)

print("Starting nonlinear solve with rebars and stirrups...")
for step in range(1, n_steps+1):
    F_target = F_ext_total * (step / n_steps)
    it = 0
    converged = False
    while it < max_iter:
        it += 1
        row = []; col = []; data = []
        F_int = np.zeros(n_dof)

        # 1) elementi solido (hexa)
        for e, conn in enumerate(elems):
            dof_idx = np.zeros(24, dtype=int)
            for i_node, node in enumerate(conn):
                dof_idx[3*i_node:3*i_node+3] = [3*node, 3*node+1, 3*node+2]
            ue = u[dof_idx]
            ke_e = np.zeros((24,24))
            fe_e = np.zeros(24)
            for (B, vol) in elem_gp_data[e]:
                strain = B @ ue
                exx, eyy, ezz = strain[0], strain[1], strain[2]
                eps_m = (exx + eyy + ezz) / 3.0
                strain_dev = strain.copy()
                strain_dev[0] -= eps_m
                strain_dev[1] -= eps_m
                strain_dev[2] -= eps_m
                if eps_m < 0:
                    eps_c = -eps_m
                    sigma_c, dsigma_deps = concrete_parabola_rect(eps_c)
                    p = -sigma_c
                    if eps_c < 1e-12:
                        Kt = K_linear
                    else:
                        Kt = dsigma_deps / 3.0
                        if Kt < 1e-9:
                            Kt = 0.0
                    D_tan = make_D_from_K_mu(Kt, mu_c)
                    stress = np.zeros(6)
                    stress[0:3] = p
                    stress += 2.0 * mu_c * strain_dev
                else:
                    D_tan = make_D_from_K_mu(K_linear, mu_c)
                    stress = D_tan @ strain
                fe_e += B.T @ stress * vol
                ke_e += B.T @ (D_tan @ B) * vol
            for i_local in range(24):
                I = dof_idx[i_local]
                F_int[I] += fe_e[i_local]
                for j_local in range(24):
                    row.append(dof_idx[i_local])
                    col.append(dof_idx[j_local])
                    data.append(ke_e[i_local, j_local])

        # 2) elementi truss (barre e staffe)
        for elem in bar_elements:
            n1, n2, A, b_id, etype = elem
            X1 = coords[n1]
            X2 = coords[n2]
            L0 = np.linalg.norm(X2 - X1)
            if L0 <= 0:
                continue
            nvec = (X2 - X1) / L0
            dof1 = np.array([3*n1, 3*n1+1, 3*n1+2], dtype=int)
            dof2 = np.array([3*n2, 3*n2+1, 3*n2+2], dtype=int)
            u1 = u[dof1]; u2 = u[dof2]
            du = u2 - u1
            axial_elong = np.dot(nvec, du)
            eps_axial = axial_elong / L0
            sigma_axial, dsig_de = steel_b450c(eps_axial)
            axial_force = sigma_axial * A
            fe_local = np.zeros(6)
            fe_local[0:3] = -axial_force * nvec
            fe_local[3:6] = axial_force * nvec
            k_axial = A * dsig_de / L0
            Kloc = np.zeros((6,6))
            K_block = k_axial * np.outer(nvec, nvec)
            Kloc[0:3,0:3] = K_block
            Kloc[0:3,3:6] = -K_block
            Kloc[3:6,0:3] = -K_block
            Kloc[3:6,3:6] = K_block
            dof_idx = np.hstack([dof1, dof2])
            for i_local in range(6):
                I = dof_idx[i_local]
                F_int[I] += fe_local[i_local]
                for j_local in range(6):
                    row.append(dof_idx[i_local])
                    col.append(dof_idx[j_local])
                    data.append(Kloc[i_local, j_local])

        # 3) collegamenti penalty barra->trave (penalty spring)
        for b_nodes in bar_nodes_indices:
            for bn in b_nodes:
                if bn < len(coords_solid):
                    continue
                Xb = coords[bn]
                tolx = 1e-6
                candidates = np.where(np.abs(coords_solid[:,0] - Xb[0]) < tolx)[0]
                if candidates.size == 0:
                    dists = np.linalg.norm(coords_solid - Xb, axis=1)
                    near = np.argmin(dists)
                else:
                    dists = np.linalg.norm(coords_solid[candidates] - Xb, axis=1)
                    near = candidates[np.argmin(dists)]
                node_s = int(near)
                dof_b = np.array([3*bn, 3*bn+1, 3*bn+2], dtype=int)
                dof_s = np.array([3*node_s, 3*node_s+1, 3*node_s+2], dtype=int)
                ub = u[dof_b]; us = u[dof_s]
                fb = K_penalty * (ub - us)
                fs = -fb
                for i_local in range(3):
                    I = dof_b[i_local]
                    F_int[I] += fb[i_local]
                    J = dof_s[i_local]
                    F_int[J] += fs[i_local]
                Kbb = K_penalty * np.eye(3)
                Kbs = -K_penalty * np.eye(3)
                Ksb = -K_penalty * np.eye(3)
                Kss = K_penalty * np.eye(3)
                dof_idx = np.hstack([dof_b, dof_s])
                Kloc = np.block([[Kbb, Kbs],[Ksb, Kss]])
                for i_local in range(6):
                    for j_local in range(6):
                        row.append(dof_idx[i_local])
                        col.append(dof_idx[j_local])
                        data.append(Kloc[i_local, j_local])

        K_tan = sp.coo_matrix((data, (row, col)), shape=(n_dof, n_dof)).tocsr()
        R = F_target - F_int
        Rf = R[free_dofs]
        norm_Rf = np.linalg.norm(Rf)
        norm_F = np.linalg.norm(F_target[free_dofs]) + 1e-12
        print(f"Step {step}/{n_steps} iter {it} ||R||={norm_Rf:.3e}")
        if norm_Rf < tol_res * norm_F:
            converged = True
            break
        Kff = K_tan[free_dofs,:][:,free_dofs]
        du_free = spla.spsolve(Kff.tocsr(), Rf)
        u[free_dofs] += du_free

    if not converged:
        print(f"WARNING: no convergence at step {step}")
    else:
        print(f"Step {step} converged in {it} iterations")

print("Nonlinear solve finished")

# -----------------------------
# Post-processing: tensioni vonMises per solido e sforzo assiale per barre
# Detect cracks (concrete) and broken bars (steel) here
# -----------------------------

nodal_stress_sum = np.zeros(len(coords_solid))
nodal_count = np.zeros(len(coords_solid))

# cracking flags for solid nodes
nodal_cracked = np.zeros(len(coords_solid), dtype=bool)

# For bars we want to know which bar elements broke
bar_elem_broken = [False]*len(bar_elements)

# helper
def voigt_to_tensor(s):
    sxx, syy, szz, sxy2, syz2, sxz2 = s
    sxy = sxy2
    syz = syz2
    sxz = sxz2
    S = np.array([[sxx, sxy, sxz],[sxy, syy, syz],[sxz, syz, szz]])
    return S

# loop sul solido: calcolo stress medio e verifico fessurazione
for e, conn in enumerate(elems):
    s_elem = np.zeros((3,3))
    count = 0
    cracked_this_elem = False
    dof_idx = np.zeros(24, dtype=int)
    for i_node, node in enumerate(conn):
        dof_idx[3*i_node:3*i_node+3] = [3*node, 3*node+1, 3*node+2]
    ue = u[dof_idx]
    for (B, vol) in elem_gp_data[e]:
        strain = B @ ue
        exx, eyy, ezz = strain[0], strain[1], strain[2]
        eps_m = (exx + eyy + ezz) / 3.0
        # cracking criteria (semplificata): tensione (eps_m>0) oppure compressione oltre eps_u
        if eps_m > 0 or (-eps_m) > eps_u:
            cracked_this_elem = True
        strain_dev = strain.copy()
        strain_dev[0] -= eps_m
        strain_dev[1] -= eps_m
        strain_dev[2] -= eps_m
        if eps_m < 0:
            eps_c = -eps_m
            sigma_c, dsig = concrete_parabola_rect(eps_c)
            p = -sigma_c
            stress = np.zeros(6)
            stress[0:3] = p
            stress += 2.0 * mu_c * strain_dev
        else:
            D_tan = make_D_from_K_mu(K_linear, mu_c)
            stress = D_tan @ strain
        S = voigt_to_tensor(stress)
        s_elem += S
        count += 1
    S_avg = s_elem / count
    sxx = S_avg[0,0]; syy = S_avg[1,1]; szz = S_avg[2,2]
    sxy = S_avg[0,1]; syz = S_avg[1,2]; sxz = S_avg[0,2]
    vm = math.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3*(sxy**2 + syz**2 + sxz**2))
    for node in conn:
        if node < len(coords_solid):
            nodal_stress_sum[node] += vm
            nodal_count[node] += 1
            if cracked_this_elem:
                nodal_cracked[node] = True

nodal_vm = nodal_stress_sum / (nodal_count + 1e-12)
vmin = float(np.nanmin(nodal_vm)); vmax = float(np.nanmax(nodal_vm))
print(f"Solid VM range: {vmin:.3e} to {vmax:.3e} Pa")

# barra: sforzi assiali medi per nodo e check rottura
bar_nodal_axial_sum = np.zeros(len(coords))
bar_nodal_count = np.zeros(len(coords))
for i_elem, elem in enumerate(bar_elements):
    n1, n2, A, b_id, etype = elem
    X1 = coords[n1]; X2 = coords[n2]
    L0 = np.linalg.norm(X2 - X1)
    if L0 <= 0:
        continue
    nvec = (X2 - X1) / L0
    dof1 = np.array([3*n1, 3*n1+1, 3*n1+2], dtype=int)
    dof2 = np.array([3*n2, 3*n2+1, 3*n2+2], dtype=int)
    u1 = u[dof1]; u2 = u[dof2]
    axial_elong = np.dot(nvec, (u2 - u1))
    eps_axial = axial_elong / L0
    sigma_axial, _ = steel_b450c(eps_axial)
    # if steel exceeded eps_max -> broken
    if abs(eps_axial) > eps_max:
        bar_elem_broken[i_elem] = True
    bar_nodal_axial_sum[n1] += abs(sigma_axial)
    bar_nodal_count[n1] += 1
    bar_nodal_axial_sum[n2] += abs(sigma_axial)
    bar_nodal_count[n2] += 1

bar_node_axial = bar_nodal_axial_sum / (bar_nodal_count + 1e-12)
bar_vmin = float(np.nanmin(bar_node_axial)); bar_vmax = float(np.nanmax(bar_node_axial))
print(f"Bar axial stress range: {bar_vmin:.3e} to {bar_vmax:.3e} Pa")

n_cracked = np.count_nonzero(nodal_cracked)
n_broken_bars = sum(1 for b in bar_elem_broken if b)
print(f"Cracked solid nodes: {n_cracked} / {len(nodal_cracked)}")
print(f"Broken bar elements: {n_broken_bars} / {len(bar_elements)}")

# -----------------------------
# OpenGL visualization: trave colorata (VM) e barre colorate (axial)
# Fessure mostrata come bianco sui nodi/triangoli dove nodal_cracked True
# Barre rotte disegnate in bianco
# -----------------------------
hex_faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]

def colormap(val, vmin, vmax):
    if vmax <= vmin:
        return (1.0, 1.0, 1.0)
    t = (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        u = t/0.25
        r = 0.0; g = u; b = 1.0
    elif t < 0.5:
        u = (t-0.25)/0.25
        r = 0.0; g = 1.0; b = 1.0 - u
    elif t < 0.75:
        u = (t-0.5)/0.25
        r = u; g = 1.0; b = 0.0
    else:
        u = (t-0.75)/0.25
        r = 1.0; g = 1.0 - u; b = 0.0
    return (r, g, b)

# build vertex colors for solid nodes: if cracked -> white
solid_colors = []
for i in range(len(coords_solid)):
    if nodal_cracked[i]:
        solid_colors.append((1.0,1.0,1.0))
    else:
        val = nodal_vm[i]
        solid_colors.append(colormap(val, vmin, vmax))
for i in range(len(coords_solid), len(coords)):
    solid_colors.append((0.7,0.7,0.7))

# bar colors per node (kept per node), but also maintain per-element broken flag
bar_colors = []
for i in range(len(coords)):
    val = bar_node_axial[i] if bar_nodal_count[i] > 0 else bar_vmin
    bar_colors.append(colormap(val, bar_vmin, bar_vmax))

# build renderable vertex arrays for solid (with deformation option)
def build_render_data(show_deformed=True):
    verts = coords.copy()
    if show_deformed:
        disp = u.reshape((-1,3))
        verts = verts + disp * exaggeration
    return verts

# OpenGL init and draw

def init_opengl(width, height):
    glClearColor(0.12, 0.12, 0.12, 1.0)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width/float(height), 0.001, 100.0)
    glMatrixMode(GL_MODELVIEW)


def draw_scene(verts, show_faces=True, show_wire=True, show_longbars=True, show_stirrups=True, show_cracks=True):
    center = np.mean(verts, axis=0)
    glPushMatrix()
    glTranslatef(-center[0], -center[1], -center[2])
    if show_faces:
        glBegin(GL_TRIANGLES)
        for e, conn in enumerate(elems):
            for face in hex_faces:
                a,b,c,d = face
                pa = verts[conn[a]]; pb = verts[conn[b]]; pc = verts[conn[c]]; pd = verts[conn[d]]
                ca = solid_colors[conn[a]]; cb = solid_colors[conn[b]]; cc = solid_colors[conn[c]]; cd = solid_colors[conn[d]]
                # if any of the face's nodes are cracked and show_cracks True -> override color to white
                if show_cracks and (nodal_cracked[conn[a]] or nodal_cracked[conn[b]] or nodal_cracked[conn[c]] or nodal_cracked[conn[d]]):
                    ca = cb = cc = cd = (1.0, 1.0, 1.0)
                # tri 1
                glColor3f(*ca); glVertex3f(pa[0], pa[1], pa[2])
                glColor3f(*cb); glVertex3f(pb[0], pb[1], pb[2])
                glColor3f(*cc); glVertex3f(pc[0], pc[1], pc[2])
                # tri 2
                glColor3f(*ca); glVertex3f(pa[0], pa[1], pa[2])
                glColor3f(*cc); glVertex3f(pc[0], pc[1], pc[2])
                glColor3f(*cd); glVertex3f(pd[0], pd[1], pd[2])
        glEnd()
    if show_wire:
        glColor3f(0.05, 0.05, 0.05)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for conn in elems:
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for (i,j) in edges:
                p1 = verts[conn[i]]; p2 = verts[conn[j]]
                glVertex3f(p1[0], p1[1], p1[2])
                glVertex3f(p2[0], p2[1], p2[2])
        glEnd()
    # barre e staffe
    glLineWidth(4.0)
    glBegin(GL_LINES)
    for i_elem, elem in enumerate(bar_elements):
        n1, n2, A, b_id, etype = elem
        p1 = verts[n1]; p2 = verts[n2]
        c1 = bar_colors[n1]; c2 = bar_colors[n2]
        broken = bar_elem_broken[i_elem]
        if etype == 'long' and show_longbars:
            if broken:
                glColor3f(1.0,1.0,1.0)
                glVertex3f(p1[0], p1[1], p1[2])
                glVertex3f(p2[0], p2[1], p2[2])
            else:
                glColor3f(*c1); glVertex3f(p1[0], p1[1], p1[2])
                glColor3f(*c2); glVertex3f(p2[0], p2[1], p2[2])
        if etype == 'stirrup' and show_stirrups:
            if broken:
                glColor3f(1.0,1.0,1.0)
                glVertex3f(p1[0], p1[1], p1[2])
                glVertex3f(p2[0], p2[1], p2[2])
            else:
                glColor3f(*c1); glVertex3f(p1[0], p1[1], p1[2])
                glColor3f(*c2); glVertex3f(p2[0], p2[1], p2[2])
    glEnd()
    glPopMatrix()

# main UI

def main():
    pygame.init()
    width, height = 1200, 800
    screen = pygame.display.set_mode((width,height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption('FEM Concrete + Rebars + Stirrups 3D (with cracking)')
    init_opengl(width, height)

    rot_x = -20.0; rot_y = -20.0; dist = 2.0; pan_x = 0.0; pan_y = 0.0
    mouse_down = False; right_down = False; last_pos = None
    show_faces = True; show_wire = True; show_longbars = True; show_stirrups = True; show_deformed = True; show_cracks = True

    verts = build_render_data(show_deformed)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_w:
                    show_wire = not show_wire
                elif event.key == pygame.K_f:
                    show_faces = not show_faces
                elif event.key == pygame.K_b:
                    show_longbars = not show_longbars
                elif event.key == pygame.K_v:
                    show_stirrups = not show_stirrups
                elif event.key == pygame.K_c:
                    show_cracks = not show_cracks
                elif event.key == pygame.K_SPACE:
                    show_deformed = not show_deformed
                    verts = build_render_data(show_deformed)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down = True; last_pos = event.pos
                elif event.button == 3:
                    right_down = True; last_pos = event.pos
                elif event.button == 4:
                    dist *= 0.9
                elif event.button == 5:
                    dist *= 1.1
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: mouse_down = False
                if event.button == 3: right_down = False
            elif event.type == pygame.MOUSEMOTION:
                if last_pos is not None:
                    x,y = event.pos; lx,ly = last_pos; dx = x-lx; dy = y-ly
                    if mouse_down: rot_x += dy * 0.3; rot_y += dx * 0.3
                    if right_down: pan_x += dx * 0.001 * dist; pan_y -= dy * 0.001 * dist
                    last_pos = event.pos

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(pan_x, pan_y, -dist)
        glRotatef(rot_x, 1, 0, 0); glRotatef(rot_y, 0, 1, 0)

        draw_scene(verts, show_faces=show_faces, show_wire=show_wire, show_longbars=show_longbars, show_stirrups=show_stirrups, show_cracks=show_cracks)

        # overlay: colorbar per solid
        glMatrixMode(GL_PROJECTION)
        glPushMatrix(); glLoadIdentity(); glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        bar_x, bar_y, bar_w, bar_h = width-140, 80, 20, 400
        steps = 128
        for i in range(steps):
            val = vmin + (i/(steps-1))*(vmax-vmin)
            r,g,b = colormap(val, vmin, vmax)
            glColor3f(r,g,b)
            y0 = bar_y + int(i*(bar_h/steps)); y1 = bar_y + int((i+1)*(bar_h/steps))
            glBegin(GL_QUADS)
            glVertex2f(bar_x, y0); glVertex2f(bar_x+bar_w, y0); glVertex2f(bar_x+bar_w, y1); glVertex2f(bar_x, y1)
            glEnd()
        try:
            font = pygame.font.SysFont('Arial', 14)
            tmax = font.render(f"{vmax/1e6:.2f} MPa", True, (255,255,255))
            tmin = font.render(f"{vmin/1e6:.2f} MPa", True, (255,255,255))
            glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
            screen.blit(tmax, (bar_x+bar_w+6, bar_y+bar_h-12))
            screen.blit(tmin, (bar_x+bar_w+6, bar_y))
        except Exception:
            glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

        try:
            font = pygame.font.SysFont('Arial', 12)
            lab = font.render('B: toggle long bars | V: toggle stirrups | C: toggle cracks | SPACE: deformata', True, (240,240,240))
            screen.blit(lab, (10,10))
        except Exception:
            pass

        pygame.display.flip(); clock.tick(30)

    pygame.quit(); sys.exit()

if __name__ == '__main__':
    main()
