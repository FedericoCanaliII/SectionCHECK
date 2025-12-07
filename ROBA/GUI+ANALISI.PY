"""
fem_gui_with_draw_preview.py

Interfaccia grafica PyQt5 per costruire una trave 3D, aggiungere barre d'armatura, gestire materiali
(eleggendo leggi costitutive) e lanciare l'analisi non-lineare.

Aggiornamento richiesto: possibilità di scegliere
 - su quale faccia applicare il carico (x=0, x=Lx, y=0, y=Ly, z=0, z=Lz)
 - quali facce fissare (lista di facce selezionabili)

Caratteristiche principali:
 - pannello Geometry: inserisci dimensioni trave (Lx, Ly, Lz) e mesh (nx,ny,nz)
 - preview 2D per posizionare barre
 - Material manager: definisci materiali
 - Lista barre: area, materiale
 - Pannello Load & BC: scegli faccia di carico e facce da vincolare
 - Pulsante "Generate mesh" crea la mesh hexa8
 - Pulsante "Run analysis" lancia il solver non-lineare
 - 3D preview OpenGL interattiva mentre si disegna/mostra risultati

Prerequisiti:
    pip install PyQt5 PyOpenGL numpy scipy

Nota: progetto didattico. Per mesh grandi il calcolo è lento — usare mesh moderate.
"""

import sys
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QFormLayout, QSpinBox, QDoubleSpinBox, QListWidget, QMessageBox, QTextEdit,
                             QComboBox, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtOpenGL import QGLWidget
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except Exception as e:
    print("ERROR: PyOpenGL not available. Install with: pip install PyOpenGL")
    raise

# -----------------------------
# Material models
# -----------------------------
class Material:
    def __init__(self, name, kind, params):
        self.name = name
        self.kind = kind
        self.params = params
    def __repr__(self):
        return f"Material({self.name},{self.kind},{self.params})"

# concrete parabola-rect
def concrete_parabola_rect(eps, params):
    eps0 = params.get('eps0', 0.002)
    eps_u = params.get('eps_u', 0.0035)
    f_cm = params.get('f_cm', 33e6)
    if eps <= 0.0:
        return 0.0, 0.0
    if eps <= eps0:
        r = eps/eps0
        sigma = f_cm * (2*r - r*r)
        dsigma = f_cm * (2/eps0 - 2*eps/(eps0*eps0))
        return float(sigma), float(dsigma)
    elif eps <= eps_u:
        return float(f_cm), 0.0
    else:
        return 0.0, 0.0

# steel B450C
def steel_b450c(eps, params):
    E = params.get('E', 210e9)
    f_y = params.get('f_y', 450e6)
    eps_y = params.get('eps_y', 0.002)
    eps_max = params.get('eps_max', 0.7)
    if abs(eps) <= eps_y:
        return E*eps, E
    elif abs(eps) <= eps_max:
        return math.copysign(f_y, eps), 0.0
    else:
        return math.copysign(f_y, eps), 0.0

# -----------------------------
# FEM solver helper
# -----------------------------
class FEMSolver:
    def __init__(self):
        self.coords = None
        self.elems = None
        self.n_nodes = 0
        self.n_dof = 0
        self.rebars = []
        self.materials = {}
        self.u = None
        self.nodal_vm = None
        self.bar_node_axial = None

    def set_mesh(self, coords, elems):
        self.coords = coords.copy()
        self.elems = elems.copy()
        self.n_nodes = coords.shape[0]
        self.n_dof = self.n_nodes * 3
        self.u = np.zeros(self.n_dof)

    def add_material(self, mat: Material):
        self.materials[mat.name] = mat

    def add_rebar_line(self, node_indices, area, mat_name):
        mat = self.materials.get(mat_name)
        if mat is None:
            raise RuntimeError('Material not found')
        self.rebars.append({'nodes': node_indices, 'area': area, 'mat': mat})

    def make_regular_hex_mesh(self, Lx, Ly, Lz, nx, ny, nz):
        xs = np.linspace(0, Lx, nx+1)
        ys = np.linspace(0, Ly, ny+1)
        zs = np.linspace(0, Lz, nz+1)
        coords = []
        idx = {}
        c=0
        for k,z in enumerate(zs):
            for j,y in enumerate(ys):
                for i,x in enumerate(xs):
                    coords.append([x,y,z])
                    idx[(i,j,k)] = c; c+=1
        coords = np.array(coords, dtype=float)
        elems = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n0 = idx[(i,j,k)]; n1 = idx[(i+1,j,k)]; n2 = idx[(i+1,j+1,k)]; n3 = idx[(i,j+1,k)]
                    n4 = idx[(i,j,k+1)]; n5 = idx[(i+1,j,k+1)]; n6 = idx[(i+1,j+1,k+1)]; n7 = idx[(i,j+1,k+1)]
                    elems.append([n0,n1,n2,n3,n4,n5,n6,n7])
        self.set_mesh(coords, np.array(elems,dtype=int))
        return self.coords, self.elems

    # helper face selector
    def face_nodes(self, face, Lx, Ly, Lz, tol=1e-8):
        # face: one of 'x0','xL','y0','yL','z0','zL'
        if self.coords is None:
            return np.array([], dtype=int)
        if face == 'x0': return np.where(np.abs(self.coords[:,0] - 0.0) < tol)[0]
        if face == 'xL': return np.where(np.abs(self.coords[:,0] - Lx) < tol)[0]
        if face == 'y0': return np.where(np.abs(self.coords[:,1] - 0.0) < tol)[0]
        if face == 'yL': return np.where(np.abs(self.coords[:,1] - Ly) < tol)[0]
        if face == 'z0': return np.where(np.abs(self.coords[:,2] - 0.0) < tol)[0]
        if face == 'zL': return np.where(np.abs(self.coords[:,2] - Lz) < tol)[0]
        return np.array([], dtype=int)

    def make_D_from_K_mu(self, Kt, mu):
        lam = Kt - 2.0*mu/3.0
        D = np.zeros((6,6))
        D[0,0] = lam + 2*mu; D[0,1] = lam; D[0,2] = lam
        D[1,0] = lam; D[1,1] = lam + 2*mu; D[1,2] = lam
        D[2,0] = lam; D[2,1] = lam; D[2,2] = lam + 2*mu
        D[3,3] = mu; D[4,4] = mu; D[5,5] = mu
        return D

    # main solver: params must include Lx,Ly,Lz, load_face string and fixed_faces list
    def solve_nonlinear(self, params):
        if self.coords is None or self.elems is None:
            raise RuntimeError('Mesh not defined')
        Lx = params.get('Lx', 1.0); Ly = params.get('Ly', 0.2); Lz = params.get('Lz', 0.2)
        load_face = params.get('load_face', 'xL')  # default x=Lx
        load_value = params.get('load_value', -10000.0)
        fixed_faces = params.get('fixed_faces', ['x0'])  # list like ['x0','y0']

        n_nodes = self.coords.shape[0]
        n_dof = n_nodes * 3
        u = np.zeros(n_dof)

        # precompute gauss B and volumes
        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        def hexa_shape_and_derivs(xi,eta,zeta):
            s = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
            N = 1/8.0 * np.array([(1+si*xi)*(1+sj*eta)*(1+sk*zeta) for si,sj,sk in s])
            dN_dxi = 1/8.0 * np.array([[si*(1+sj*eta)*(1+sk*zeta) for si,sj,sk in s]])[0]
            dN_deta = 1/8.0 * np.array([[(1+si*xi)*sj*(1+sk*zeta) for si,sj,sk in s]])[0]
            dN_dzeta = 1/8.0 * np.array([[(1+si*xi)*(1+sj*eta)*sk for si,sj,sk in s]])[0]
            return N, dN_dxi, dN_deta, dN_dzeta
        def build_B(dNdx,dNdy,dNdz):
            B = np.zeros((6,24))
            for i in range(8):
                ix = 3*i
                B[0,ix] = dNdx[i]; B[1,ix+1]=dNdy[i]; B[2,ix+2]=dNdz[i]
                B[3,ix]=dNdy[i]; B[3,ix+1]=dNdx[i]
                B[4,ix+1]=dNdz[i]; B[4,ix+2]=dNdy[i]
                B[5,ix]=dNdz[i]; B[5,ix+2]=dNdx[i]
            return B

        elem_gp = []
        for e,conn in enumerate(self.elems):
            xe = self.coords[conn]
            gp_list = []
            for xi in gp:
                for eta in gp:
                    for zeta in gp:
                        N, dNdxi, dNdeta, dNdzet = hexa_shape_and_derivs(xi,eta,zeta)
                        dN_dxi = np.vstack([dNdxi, dNdeta, dNdzet])
                        J = dN_dxi @ xe
                        detJ = np.linalg.det(J)
                        if detJ <= 0:
                            raise RuntimeError('Jacobian non positivo')
                        invJ = np.linalg.inv(J)
                        grads = invJ @ dN_dxi
                        dNdx = grads[0,:]; dNdy = grads[1,:]; dNdz = grads[2,:]
                        B = build_B(dNdx,dNdy,dNdz)
                        vol = detJ
                        gp_list.append((B,vol))
            elem_gp.append(gp_list)

        # external load assembly on chosen face
        F_ext = np.zeros(n_dof)
        face_nodes = self.face_nodes(load_face, Lx, Ly, Lz)
        if face_nodes.size > 0:
            f_each = load_value / face_nodes.size
            # assume load direction is z (default). For extension, allow choosing direction.
            for n in face_nodes:
                F_ext[3*n + 2] += f_each

        # fixed DOFs from selected faces
        fixed_dofs = []
        for face in fixed_faces:
            nodes = self.face_nodes(face, Lx, Ly, Lz)
            for n in nodes:
                fixed_dofs += [3*n, 3*n+1, 3*n+2]
        # also fix first nodes of rebars (if they overlap solid) to avoid rigid body
        for rb in self.rebars:
            nodes = rb['nodes']
            if len(nodes) > 0 and nodes[0] < self.n_nodes:
                fixed_dofs += [3*nodes[0], 3*nodes[0]+1, 3*nodes[0]+2]
        fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

        # solver parameters
        n_steps = params.get('n_steps', 8)
        tol_res = params.get('tol_res', 1e-6)
        max_iter = params.get('max_iter', 20)
        K_penalty = params.get('K_penalty', 1e12)

        # material helpers
        def get_concrete_material():
            return self.materials.get('Concrete', Material('Concrete','concrete',{'f_cm':33e6,'eps0':0.002,'eps_u':0.0035,'E':22e9,'nu':0.2}))

        u = np.zeros(n_dof)
        for step in range(1, n_steps+1):
            F_target = F_ext * (step / n_steps)
            for it in range(1, max_iter+1):
                row=[]; col=[]; data=[]
                F_int = np.zeros(n_dof)
                # solid elements
                for e,conn in enumerate(self.elems):
                    dof_idx = np.zeros(24, dtype=int)
                    for i_node,node in enumerate(conn):
                        dof_idx[3*i_node:3*i_node+3] = [3*node,3*node+1,3*node+2]
                    ue = u[dof_idx]
                    ke_e = np.zeros((24,24)); fe_e = np.zeros(24)
                    for (B,vol) in elem_gp[e]:
                        strain = B @ ue
                        exx,eyy,ezz = strain[0],strain[1],strain[2]
                        eps_m = (exx+eyy+ezz)/3.0
                        strain_dev = strain.copy(); strain_dev[0]-=eps_m; strain_dev[1]-=eps_m; strain_dev[2]-=eps_m
                        if eps_m < 0:
                            eps_c = -eps_m
                            mat = get_concrete_material()
                            sigma_c, dsigma = concrete_parabola_rect(eps_c, mat.params)
                            p = -sigma_c
                            if eps_c < 1e-12:
                                Kt = params.get('K_linear', 1e10)
                            else:
                                Kt = dsigma / 3.0
                                if Kt < 1e-9: Kt = 0.0
                            mu = mat.params.get('E') / (2*(1+mat.params.get('nu',0.2)))
                            D_tan = self.make_D_from_K_mu(Kt, mu)
                            stress = np.zeros(6); stress[0:3]=p; stress += 2.0*mu*strain_dev
                        else:
                            mat = get_concrete_material()
                            mu = mat.params.get('E') / (2*(1+mat.params.get('nu',0.2)))
                            D_tan = self.make_D_from_K_mu(params.get('K_linear',1e10), mu)
                            stress = D_tan @ strain
                        fe_e += B.T @ stress * vol
                        ke_e += B.T @ (D_tan @ B) * vol
                    for i_local in range(24):
                        I = dof_idx[i_local]
                        F_int[I] += fe_e[i_local]
                        for j_local in range(24):
                            row.append(dof_idx[i_local]); col.append(dof_idx[j_local]); data.append(ke_e[i_local,j_local])

                # truss bars
                for rb in self.rebars:
                    nodes = rb['nodes']; A = rb['area']; mat = rb['mat']
                    for i in range(len(nodes)-1):
                        n1 = nodes[i]; n2 = nodes[i+1]
                        X1 = self.coords[n1]; X2 = self.coords[n2]
                        L0 = np.linalg.norm(X2 - X1)
                        if L0<=0: continue
                        nvec = (X2 - X1) / L0
                        dof1 = np.array([3*n1,3*n1+1,3*n1+2],dtype=int); dof2 = np.array([3*n2,3*n2+1,3*n2+2],dtype=int)
                        u1 = u[dof1]; u2 = u[dof2]
                        du = u2 - u1
                        axial_elong = np.dot(nvec, du)
                        eps_axial = axial_elong / L0
                        if mat.kind == 'steel':
                            sigma_axial, ds = steel_b450c(eps_axial, mat.params)
                        else:
                            sigma_axial, ds = steel_b450c(eps_axial, {'E':210e9,'f_y':450e6,'eps_y':0.002,'eps_max':0.7})
                        axial_force = sigma_axial * A
                        fe_local = np.zeros(6); fe_local[0:3] = -axial_force * nvec; fe_local[3:6] = axial_force * nvec
                        k_axial = A * ds / L0
                        Kloc = np.zeros((6,6)); K_block = k_axial * np.outer(nvec, nvec)
                        Kloc[0:3,0:3] = K_block; Kloc[0:3,3:6] = -K_block; Kloc[3:6,0:3] = -K_block; Kloc[3:6,3:6] = K_block
                        dof_idx = np.hstack([dof1,dof2])
                        for i_local in range(6):
                            I = dof_idx[i_local]
                            F_int[I] += fe_local[i_local]
                            for j_local in range(6):
                                row.append(dof_idx[i_local]); col.append(dof_idx[j_local]); data.append(Kloc[i_local,j_local])

                # penalty coupling: nearest solid node at same x-level
                for rb in self.rebars:
                    for bn in rb['nodes']:
                        if bn < self.n_nodes:
                            continue
                        Xb = self.coords[bn]
                        candidates = np.where(np.abs(self.coords[:self.n_nodes,0] - Xb[0])<1e-8)[0]
                        if candidates.size == 0:
                            dists = np.linalg.norm(self.coords[:self.n_nodes] - Xb, axis=1)
                            near = np.argmin(dists)
                        else:
                            dists = np.linalg.norm(self.coords[candidates] - Xb, axis=1)
                            near = candidates[np.argmin(dists)]
                        node_s = int(near)
                        dof_b = np.array([3*bn,3*bn+1,3*bn+2],dtype=int); dof_s = np.array([3*node_s,3*node_s+1,3*node_s+2],dtype=int)
                        ub = u[dof_b]; us = u[dof_s]
                        fb = K_penalty * (ub - us); fs = -fb
                        for i_local in range(3):
                            F_int[dof_b[i_local]] += fb[i_local]
                            F_int[dof_s[i_local]] += fs[i_local]
                        Kbb = K_penalty * np.eye(3); Kbs = -K_penalty * np.eye(3); Ksb = -K_penalty * np.eye(3); Kss = K_penalty * np.eye(3)
                        dof_idx = np.hstack([dof_b,dof_s])
                        Kloc = np.block([[Kbb,Kbs],[Ksb,Kss]])
                        for i_local in range(6):
                            for j_local in range(6):
                                row.append(dof_idx[i_local]); col.append(dof_idx[j_local]); data.append(Kloc[i_local,j_local])

                K_tan = sp.coo_matrix((data,(row,col)), shape=(n_dof,n_dof)).tocsr()
                R = F_target - F_int
                Rf = R[free_dofs]
                norm_Rf = np.linalg.norm(Rf); norm_F = np.linalg.norm(F_target[free_dofs]) + 1e-12
                print(f"step {step} it {it} ||R||={norm_Rf:.3e}")
                if norm_Rf < tol_res * norm_F:
                    break
                Kff = K_tan[free_dofs,:][:,free_dofs]
                du_free = spla.spsolve(Kff.tocsr(), Rf)
                u[free_dofs] += du_free

        self.u = u
        # postprocess nodal VM (solid)
        nodal_vm = np.zeros(self.n_nodes); nodal_cnt = np.zeros(self.n_nodes)
        for e,conn in enumerate(self.elems):
            dof_idx = np.zeros(24,dtype=int)
            for i_node,node in enumerate(conn):
                dof_idx[3*i_node:3*i_node+3] = [3*node,3*node+1,3*node+2]
            ue = u[dof_idx]
            Savg = np.zeros((3,3)); count = 0
            for (B,vol) in elem_gp[e]:
                strain = B @ ue
                exx,eyy,ezz = strain[0],strain[1],strain[2]
                eps_m = (exx+eyy+ezz)/3.0
                strain_dev = strain.copy(); strain_dev[0]-=eps_m; strain_dev[1]-=eps_m; strain_dev[2]-=eps_m
                if eps_m < 0:
                    sigma_c, ds = concrete_parabola_rect(-eps_m, get_concrete_material().params)
                    p = -sigma_c
                    mu = get_concrete_material().params.get('E')/(2*(1+get_concrete_material().params.get('nu',0.2)))
                    stress = np.zeros(6); stress[0:3]=p; stress += 2.0*mu*strain_dev
                else:
                    mu = get_concrete_material().params.get('E')/(2*(1+get_concrete_material().params.get('nu',0.2)))
                    D = self.make_D_from_K_mu(params.get('K_linear',1e10), mu)
                    stress = D @ strain
                S = np.array([[stress[0], stress[3], stress[5]],[stress[3], stress[1], stress[4]],[stress[5], stress[4], stress[2]]])
                Savg += S; count += 1
            Savg /= max(1,count)
            sxx = Savg[0,0]; syy = Savg[1,1]; szz = Savg[2,2]
            sxy = Savg[0,1]; syz = Savg[1,2]; sxz = Savg[0,2]
            vm = math.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3*(sxy**2 + syz**2 + sxz**2))
            for node in conn:
                nodal_vm[node] += vm; nodal_cnt[node] += 1
        nodal_vm = nodal_vm / (nodal_cnt + 1e-12)
        self.nodal_vm = nodal_vm
        # bar axial
        bar_node_axial = np.zeros(self.coords.shape[0]); bar_cnt = np.zeros(self.coords.shape[0])
        for rb in self.rebars:
            mat = rb['mat']
            for i in range(len(rb['nodes'])-1):
                n1 = rb['nodes'][i]; n2 = rb['nodes'][i+1]
                X1 = self.coords[n1]; X2 = self.coords[n2]
                L0 = np.linalg.norm(X2 - X1); nvec = (X2 - X1) / L0
                dof1 = np.array([3*n1,3*n1+1,3*n1+2],dtype=int); dof2 = np.array([3*n2,3*n2+1,3*n2+2],dtype=int)
                u1 = self.u[dof1]; u2 = self.u[dof2]
                eps_axial = np.dot(nvec, u2-u1) / L0
                sigma, ds = steel_b450c(eps_axial, mat.params) if mat.kind == 'steel' else steel_b450c(eps_axial, {'E':210e9,'f_y':450e6,'eps_y':0.002,'eps_max':0.7})
                bar_node_axial[n1] += abs(sigma); bar_cnt[n1] += 1
                bar_node_axial[n2] += abs(sigma); bar_cnt[n2] += 1
        bar_node_axial = bar_node_axial / (bar_cnt + 1e-12)
        self.bar_node_axial = bar_node_axial
        return True

# -----------------------------
# OpenGL Widget for preview (unchanged logic)
# -----------------------------
class GLPreview(QGLWidget):
    def __init__(self, femsolver: FEMSolver, parent=None):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(600,400)
        self.solver = femsolver
        self.rot_x = -25; self.rot_y = -30; self.dist = 2.5
        self.pan_x = 0.0; self.pan_y = 0.0; self.last_pos = None
        self.show_faces = True; self.show_wire = True; self.show_bars = True; self.show_deformed = True
    def initializeGL(self):
        glClearColor(0.12,0.12,0.12,1.0); glEnable(GL_DEPTH_TEST); glShadeModel(GL_SMOOTH)
    def resizeGL(self, w, h):
        glViewport(0,0,w,h); glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluPerspective(45.0, w/float(h if h else 1), 0.001, 100.0); glMatrixMode(GL_MODELVIEW)
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); glLoadIdentity()
        glTranslatef(self.pan_x, self.pan_y, -self.dist); glRotatef(self.rot_x,1,0,0); glRotatef(self.rot_y,0,1,0)
        solver = self.solver
        if solver.coords is None: return
        coords = solver.coords.copy()
        if self.show_deformed and solver.u is not None:
            disp = solver.u.reshape((-1,3)); coords = coords + disp * 500.0
        # faces
        if solver.elems is not None and self.show_faces:
            glBegin(GL_TRIANGLES)
            for e,conn in enumerate(solver.elems):
                for face in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]:
                    a,b,c,d = face
                    for idx in (a,b,c):
                        node = conn[idx]; col = (0.7,0.7,0.7)
                        if solver.nodal_vm is not None and node < solver.nodal_vm.shape[0]: col = self.colormap(solver.nodal_vm[node], solver.nodal_vm.min(), solver.nodal_vm.max())
                        glColor3f(*col); p = coords[node]; glVertex3f(p[0],p[1],p[2])
                    for idx in (a,c,d):
                        node = conn[idx]; col = (0.7,0.7,0.7)
                        if solver.nodal_vm is not None and node < solver.nodal_vm.shape[0]: col = self.colormap(solver.nodal_vm[node], solver.nodal_vm.min(), solver.nodal_vm.max())
                        glColor3f(*col); p = coords[node]; glVertex3f(p[0],p[1],p[2])
            glEnd()
        # wireframe
        if solver.elems is not None and self.show_wire:
            glColor3f(0.05,0.05,0.05); glLineWidth(1.0); glBegin(GL_LINES)
            for conn in solver.elems:
                edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                for (i,j) in edges:
                    p1 = coords[conn[i]]; p2 = coords[conn[j]]; glVertex3f(p1[0],p1[1],p1[2]); glVertex3f(p2[0],p2[1],p2[2])
            glEnd()
        # bars
        if self.show_bars and len(solver.rebars)>0:
            glLineWidth(4.0); glBegin(GL_LINES)
            for rb in solver.rebars:
                for i in range(len(rb['nodes'])-1):
                    n1 = rb['nodes'][i]; n2 = rb['nodes'][i+1]; p1 = coords[n1]; p2 = coords[n2]
                    c1=(0.8,0.2,0.2); c2=(0.8,0.2,0.2)
                    if solver.bar_node_axial is not None:
                        c1 = self.colormap(solver.bar_node_axial[n1], solver.bar_node_axial.min(), solver.bar_node_axial.max())
                        c2 = self.colormap(solver.bar_node_axial[n2], solver.bar_node_axial.min(), solver.bar_node_axial.max())
                    glColor3f(*c1); glVertex3f(p1[0],p1[1],p1[2]); glColor3f(*c2); glVertex3f(p2[0],p2[1],p2[2])
            glEnd()
    def colormap(self,val,vmin,vmax):
        if vmax<=vmin: return (1.0,1.0,1.0)
        t = (val - vmin)/(vmax-vmin); t = max(0.0,min(1.0,t))
        if t<0.25: u=t/0.25; return (0.0,u,1.0)
        elif t<0.5: u=(t-0.25)/0.25; return (0.0,1.0,1.0-u)
        elif t<0.75: u=(t-0.5)/0.25; return (u,1.0,0.0)
        else: u=(t-0.75)/0.25; return (1.0,1.0-u,0.0)
    def mousePressEvent(self,event): self.last_pos = event.pos();
    def mouseReleaseEvent(self,event): self.last_pos=None
    def mouseMoveEvent(self,event):
        if self.last_pos is None: return
        dx = event.x() - self.last_pos.x(); dy = event.y() - self.last_pos.y()
        buttons = event.buttons()
        if buttons & Qt.LeftButton: self.rot_x += dy*0.3; self.rot_y += dx*0.3
        if buttons & Qt.RightButton: self.pan_x += dx*0.001*self.dist; self.pan_y -= dy*0.001*self.dist
        self.last_pos = event.pos(); self.update()
    def wheelEvent(self,event): delta = event.angleDelta().y(); self.dist *= 0.9 if delta>0 else 1.1; self.update()

# -----------------------------
# Section preview widget (2D)
# -----------------------------
class SectionPreview(QWidget):
    def __init__(self, L_y=0.2, L_z=0.2, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200,200)
        self.Ly = L_y; self.Lz = L_z
        self.rebar_positions = []
        self.on_add = None
    def set_dims(self, Ly, Lz): self.Ly = Ly; self.Lz = Lz; self.update()
    def mousePressEvent(self,event):
        x = event.x(); y = event.y(); w=self.width(); h=self.height(); margin=10
        # map click to (y,z) inside rect
        yy = ((x - margin) / (w - 2*margin)) * self.Ly if w>2*margin else 0
        zz = ((h - y - margin) / (h - 2*margin)) * self.Lz if h>2*margin else 0
        yy = max(0.0, min(self.Ly, yy)); zz = max(0.0, min(self.Lz, zz))
        self.rebar_positions.append((yy, zz))
        if self.on_add: self.on_add(yy, zz)
        self.update()
    def paintEvent(self,event):
        p = QtGui.QPainter(self); p.fillRect(self.rect(), QtGui.QColor(40,40,40)); pen = QtGui.QPen(QtGui.QColor(200,200,200)); pen.setWidth(2); p.setPen(pen)
        margin=10; w=self.width()-2*margin; h=self.height()-2*margin; p.drawRect(margin,margin,w,h)
        pen = QtGui.QPen(QtGui.QColor(200,80,80)); pen.setWidth(8); p.setPen(pen)
        for (yy,zz) in self.rebar_positions:
            x = margin + (yy / self.Ly) * w if self.Ly>0 else margin
            y = margin + h - (zz / self.Lz) * h if self.Lz>0 else margin
            p.drawPoint(int(x), int(y))
        p.end()

# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle('FEM Concrete + Rebars GUI'); self.resize(1400,900)
        self.fem = FEMSolver(); self.init_default_materials(); self.build_ui()

    def init_default_materials(self):
        concrete_params = {'f_cm':33e6, 'eps0':0.002, 'eps_u':0.0035, 'E':22e9, 'nu':0.2}
        steel_params = {'E':210e9, 'f_y':450e6, 'eps_y':0.002, 'eps_max':0.7}
        matc = Material('Concrete','concrete',concrete_params); mats = Material('B450C','steel',steel_params)
        self.fem.add_material(matc); self.fem.add_material(mats)

    def build_ui(self):
        central = QWidget(); self.setCentralWidget(central); mainlay = QHBoxLayout(); central.setLayout(mainlay)
        left = QVBoxLayout(); mainlay.addLayout(left,0)
        geom_box = QtWidgets.QGroupBox('Geometry & Mesh'); gfl = QFormLayout(); geom_box.setLayout(gfl)
        self.spin_Lx = QDoubleSpinBox(); self.spin_Lx.setRange(0.01,10.0); self.spin_Lx.setValue(1.0)
        self.spin_Ly = QDoubleSpinBox(); self.spin_Ly.setRange(0.01,5.0); self.spin_Ly.setValue(0.2)
        self.spin_Lz = QDoubleSpinBox(); self.spin_Lz.setRange(0.01,5.0); self.spin_Lz.setValue(0.2)
        self.spin_nx = QSpinBox(); self.spin_nx.setRange(1,40); self.spin_nx.setValue(8)
        self.spin_ny = QSpinBox(); self.spin_ny.setRange(1,20); self.spin_ny.setValue(4)
        self.spin_nz = QSpinBox(); self.spin_nz.setRange(1,20); self.spin_nz.setValue(4)
        gfl.addRow('Lx (m):', self.spin_Lx); gfl.addRow('Ly (m):', self.spin_Ly); gfl.addRow('Lz (m):', self.spin_Lz)
        gfl.addRow('nx:', self.spin_nx); gfl.addRow('ny:', self.spin_ny); gfl.addRow('nz:', self.spin_nz)
        btn_gen = QPushButton('Generate Mesh'); btn_gen.clicked.connect(self.on_generate_mesh); gfl.addRow(btn_gen)
        left.addWidget(geom_box)

        sec_box = QtWidgets.QGroupBox('Section Preview & Rebars'); svl = QVBoxLayout(); sec_box.setLayout(svl)
        self.section_preview = SectionPreview(self.spin_Ly.value(), self.spin_Lz.value()); self.section_preview.on_add = self.on_section_add; svl.addWidget(self.section_preview)
        hb = QHBoxLayout(); self.spin_offset = QDoubleSpinBox(); self.spin_offset.setRange(0.0,0.2); self.spin_offset.setValue(0.03)
        self.spin_area = QDoubleSpinBox(); self.spin_area.setRange(1e-6, 1e-2); self.spin_area.setDecimals(6); self.spin_area.setValue(1e-4)
        hb.addWidget(QLabel('offset:')); hb.addWidget(self.spin_offset); hb.addWidget(QLabel('area (m2):')); hb.addWidget(self.spin_area)
        svl.addLayout(hb)
        btn_add4 = QPushButton('Add 4 corner rebars'); btn_add4.clicked.connect(self.add_4_rebars); svl.addWidget(btn_add4)
        left.addWidget(sec_box)

        load_box = QtWidgets.QGroupBox('Loads & Boundary Conditions'); llay = QFormLayout(); load_box.setLayout(llay)
        self.cmb_load_face = QComboBox(); self.cmb_load_face.addItems(['x=Lx','x=0','y=Ly','y=0','z=Lz','z=0']); self.cmb_load_face.setCurrentIndex(0)
        self.spin_load_value = QDoubleSpinBox(); self.spin_load_value.setRange(-1e9,1e9); self.spin_load_value.setValue(-10000.0); self.spin_load_value.setDecimals(2)
        llay.addRow('Load face:', self.cmb_load_face); llay.addRow('Load (N, total):', self.spin_load_value)
        # fixed faces checkboxes
        self.chk_x0 = QCheckBox('x=0'); self.chk_xL = QCheckBox('x=Lx'); self.chk_y0 = QCheckBox('y=0'); self.chk_yL = QCheckBox('y=Ly'); self.chk_z0 = QCheckBox('z=0'); self.chk_zL = QCheckBox('z=Lz')
        # default fix x=0
        self.chk_x0.setChecked(True)
        grid = QHBoxLayout(); vcol = QVBoxLayout(); vcol.addWidget(self.chk_x0); vcol.addWidget(self.chk_xL); grid.addLayout(vcol); vcol2 = QVBoxLayout(); vcol2.addWidget(self.chk_y0); vcol2.addWidget(self.chk_yL); grid.addLayout(vcol2); vcol3 = QVBoxLayout(); vcol3.addWidget(self.chk_z0); vcol3.addWidget(self.chk_zL); grid.addLayout(vcol3)
        llay.addRow(QLabel('Fix faces:'), grid)
        left.addWidget(load_box)

        mat_box = QtWidgets.QGroupBox('Materials'); mlay = QVBoxLayout(); mat_box.setLayout(mlay)
        self.mat_table = QtWidgets.QTableWidget(0,3); self.mat_table.setHorizontalHeaderLabels(['Name','Type','Params']); mlay.addWidget(self.mat_table)
        btn_add_mat = QPushButton('Add Material'); btn_add_mat.clicked.connect(self.add_material_dialog); mlay.addWidget(btn_add_mat); left.addWidget(mat_box)
        self.refresh_mat_table()

        bar_box = QtWidgets.QGroupBox('Rebars'); blay = QVBoxLayout(); bar_box.setLayout(blay); self.bar_list = QListWidget(); blay.addWidget(self.bar_list)
        btn_del_bar = QPushButton('Remove Selected Bar'); btn_del_bar.clicked.connect(self.remove_selected_bar); blay.addWidget(btn_del_bar); left.addWidget(bar_box)

        run_box = QtWidgets.QGroupBox('Analysis'); rlay = QVBoxLayout(); run_box.setLayout(rlay)
        btn_run = QPushButton('Run Nonlinear Analysis'); btn_run.clicked.connect(self.run_analysis); rlay.addWidget(btn_run)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(120); rlay.addWidget(self.log); left.addWidget(run_box)

        # center 3D preview
        self.gl = GLPreview(self.fem); mainlay.addWidget(self.gl,1)
        right = QVBoxLayout(); mainlay.addLayout(right,0); right.addWidget(QLabel('Elements / Info'))
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); right.addWidget(self.info_text)
        self.spin_Ly.valueChanged.connect(lambda v: self.section_preview.set_dims(v, self.spin_Lz.value()))
        self.spin_Lz.valueChanged.connect(lambda v: self.section_preview.set_dims(self.spin_Ly.value(), v))

    def on_generate_mesh(self):
        Lx = self.spin_Lx.value(); Ly = self.spin_Ly.value(); Lz = self.spin_Lz.value()
        nx = self.spin_nx.value(); ny = self.spin_ny.value(); nz = self.spin_nz.value()
        coords, elems = self.fem.make_regular_hex_mesh(Lx, Ly, Lz, nx, ny, nz)
        self.log.append(f"Mesh generated: {coords.shape[0]} nodes, {elems.shape[0]} elems")
        self.section_preview.set_dims(Ly, Lz)
        # update GL widget
        self.gl.update()

    def on_section_add(self, y, z):
        if self.fem.coords is None:
            QMessageBox.warning(self, 'Warning','Generate mesh first')
            return
        coords = self.fem.coords
        xs_unique = np.unique(coords[:,0])
        nodes_line = []
        for x in xs_unique:
            dists = np.linalg.norm(coords - np.array([x,y,z]), axis=1)
            node = np.argmin(dists)
            nodes_line.append(int(node))
        area = self.spin_area.value(); mat_name = 'B450C'
        try:
            self.fem.add_rebar_line(nodes_line, area, mat_name)
        except Exception as e:
            QMessageBox.warning(self,'Error',str(e)); return
        self.bar_list.addItem(f"Bar {len(self.fem.rebars)}: A={area:.3e} m2, mat={mat_name}")
        self.log.append(f"Added rebar at y={y:.3f}, z={z:.3f}")
        self.gl.update()

    def add_4_rebars(self):
        if self.fem.coords is None:
            QMessageBox.warning(self,'Warning','Generate mesh first'); return
        Ly=self.spin_Ly.value(); Lz=self.spin_Lz.value(); off=self.spin_offset.value(); area=self.spin_area.value()
        poss = [(off,off),(Ly-off,off),(off,Lz-off),(Ly-off,Lz-off)]
        for (y,z) in poss:
            self.section_preview.rebar_positions.append((y,z)); self.on_section_add(y,z)
        self.section_preview.update()

    def add_material_dialog(self):
        text, ok = QtWidgets.QInputDialog.getText(self,'Add Material','Enter name,type (concrete/steel),params key=val;...')
        if not ok or not text: return
        try:
            parts = [p.strip() for p in text.split(',')]
            name = parts[0]; typ = parts[1]
            params = {}
            for kv in parts[2:]:
                if '=' in kv:
                    k,v = kv.split('=')
                    params[k.strip()] = float(v)
            mat = Material(name, typ, params); self.fem.add_material(mat); self.refresh_mat_table(); self.log.append(f"Added material {name}")
        except Exception as e:
            QMessageBox.warning(self,'Error','Could not parse input: ' + str(e))

    def refresh_mat_table(self):
        mats = list(self.fem.materials.values())
        self.mat_table.setRowCount(len(mats))
        for i,m in enumerate(mats):
            self.mat_table.setItem(i,0,QtWidgets.QTableWidgetItem(m.name)); self.mat_table.setItem(i,1,QtWidgets.QTableWidgetItem(m.kind)); self.mat_table.setItem(i,2,QtWidgets.QTableWidgetItem(str(m.params)))

    def remove_selected_bar(self):
        sel = self.bar_list.currentRow()
        if sel < 0: return
        self.bar_list.takeItem(sel); self.fem.rebars.pop(sel); self.log.append(f"Removed bar {sel+1}"); self.gl.update()

    def run_analysis(self):
        if self.fem.coords is None:
            QMessageBox.warning(self,'Warning','Generate mesh first'); return
        # collect load face and fixed faces
        face_map = {'x=Lx':'xL','x=0':'x0','y=Ly':'yL','y=0':'y0','z=Lz':'zL','z=0':'z0'}
        load_face = face_map.get(self.cmb_load_face.currentText(), 'xL')
        fixed_faces = []
        if self.chk_x0.isChecked(): fixed_faces.append('x0')
        if self.chk_xL.isChecked(): fixed_faces.append('xL')
        if self.chk_y0.isChecked(): fixed_faces.append('y0')
        if self.chk_yL.isChecked(): fixed_faces.append('yL')
        if self.chk_z0.isChecked(): fixed_faces.append('z0')
        if self.chk_zL.isChecked(): fixed_faces.append('zL')
        params = {'Lx':self.spin_Lx.value(), 'Ly':self.spin_Ly.value(), 'Lz':self.spin_Lz.value(), 'load_face':load_face, 'load_value':self.spin_load_value.value(), 'fixed_faces':fixed_faces, 'n_steps':6}
        self.log.append(f"Run: load on {self.cmb_load_face.currentText()} = {self.spin_load_value.value()} N; fixed faces: {fixed_faces}")
        try:
            success = self.fem.solve_nonlinear(params)
            if success:
                self.log.append('Analysis finished')
                self.gl.update()
                max_disp = np.max(np.linalg.norm(self.fem.u.reshape((-1,3)),axis=1))
                vmmin = float(np.nanmin(self.fem.nodal_vm)); vmmax = float(np.nanmax(self.fem.nodal_vm))
                self.info_text.setPlainText(f"Max disp: {max_disp:.3e} m\nSolid VM range: {vmmin:.3e} - {vmmax:.3e} Pa")
        except Exception as e:
            QMessageBox.warning(self,'Error','Analysis failed: '+str(e)); self.log.append('Analysis failed: '+str(e))

# -----------------------------
# run
# -----------------------------
def main():
    app = QtWidgets.QApplication(sys.argv); win = MainWindow(); win.show(); sys.exit(app.exec_())

if __name__ == '__main__':
    main()
