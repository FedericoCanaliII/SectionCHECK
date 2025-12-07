import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math
import traceback
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QMessageBox

# Importiamo la logica base geometrica per riusare il parsing delle sezioni
# Assicurati che beam.mesh sia accessibile
from beam.mesh import BeamMeshCore

class MaterialParser:
    def __init__(self, definition):
        # Gestisce sia definizioni complesse (liste di segmenti) sia semplici dizionari/stringhe
        self.segments = []
        
        # Se definition è solo un nome o None, usa un default elastico lineare
        if not definition or isinstance(definition, str):
            self.name = str(definition)
            # Default Cemento-like: parabola fittizia
            self.segments.append({'func': '30000e6*x', 'min': -1.0, 'max': 1.0})
            return

        # Se è la lista strutturata come nell'esempio FEM
        try:
            self.name = definition[0]
            if len(definition) > 1 and isinstance(definition[1], (list, tuple)):
                for segment in definition[1:]:
                    if len(segment) >= 3:
                        formula = segment[0]
                        start = float(segment[1])
                        end = float(segment[2])
                        self.segments.append({'func': formula, 'min': start, 'max': end})
            else:
                # Fallback se non ci sono segmenti espliciti
                self.segments.append({'func': '210000e6*x', 'min': -1.0, 'max': 1.0})
        except Exception:
            self.name = "Unknown"
            self.segments.append({'func': '210000e6*x', 'min': -1.0, 'max': 1.0})

    def evaluate(self, strain):
        h = 1e-7
        active_seg = None
        for seg in self.segments:
            if seg['min'] <= strain <= seg['max']:
                active_seg = seg
                break
        
        if active_seg is None:
            # Fuori range: assume snervamento o rottura (rigidezza residua piccola)
            return 0.0, 1e-3 
            
        def eval_str(f_str, x_val):
            try:
                # Safe eval environment
                return eval(f_str, {"__builtins__": None}, {"x": x_val, "abs": abs, "math": math})
            except:
                return 0.0

        sigma = eval_str(active_seg['func'], strain)
        sigma_h = eval_str(active_seg['func'], strain + h)
        tangent = (sigma_h - sigma) / h
        if tangent < 100.0: tangent = 100.0 
        return float(sigma), float(tangent)

class FemWorker(QThread):
    # Segnali per comunicare con la GUI
    # Aggiunti: stress_history (lista np.array), max_stress (float)
    finished_computation = pyqtSignal(object, object, object, object, object, object, float) 
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)

    def __init__(self, section_data, materials_db, params):
        super().__init__()
        self.section = section_data
        self.materials_db = materials_db
        self.params = params
        self.core = BeamMeshCore()

    def run(self):
        try:
            self.progress_update.emit("Generazione Mesh FEM...")
            coords, solid_elements, bar_elements, penalty_links, n_solid_nodes = self.generate_fem_mesh()
            
            self.progress_update.emit("Assemblaggio Matrici e Solver...")
            # run_solver ora ritorna sia u_history che stress_history
            u_history, stress_history = self.run_solver(coords, solid_elements, bar_elements, penalty_links)
            
            # Calcola max displacement per la scala
            max_disp = 0.0
            if u_history:
                d_mag = np.linalg.norm(u_history[-1].reshape(-1, 3), axis=1)
                max_disp = d_mag.max()

            # Calcola max stress per la scala colori
            max_stress = 0.0
            if stress_history:
                # Trova il massimo valore assoluto nella storia finale o globale
                # Usiamo l'ultimo step per riferimento
                s_final = np.abs(stress_history[-1])
                max_stress = np.max(s_final) if len(s_final) > 0 else 1.0

            self.finished_computation.emit(u_history, coords, solid_elements, bar_elements, max_disp, stress_history, max_stress)

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def generate_fem_mesh(self):
        # Estrai parametri
        L_beam = self.params['L']
        nx = self.params['nx']
        ny = self.params['ny']
        nz = self.params['nz']
        
        # 1. Bounding box sezione
        min_x, max_x, min_y, max_y = self.core._get_section_bounding_box(self.section)
        margin = 0.05
        min_x = self.core._mm_to_m(min_x) - margin
        max_x = self.core._mm_to_m(max_x) + margin
        min_y = self.core._mm_to_m(min_y) - margin
        max_y = self.core._mm_to_m(max_y) + margin
        
        Lx_bbox = max_x - min_x
        Ly_bbox = max_y - min_y
        dx = Lx_bbox / max(1, nx)
        dy = Ly_bbox / max(1, ny)
        dz = L_beam / max(1, nz)

        # 2. Identificazione voxels attivi (Solidi)
        voxel_mask = {} # (ix, iy) -> material_name
        xs = np.linspace(min_x + dx/2, max_x - dx/2, nx)
        ys = np.linspace(min_y + dy/2, max_y - dy/2, ny)
        
        active_indices = []

        # Usiamo la logica di mesh.py ma adattata per ottenere solo il materiale
        for iy in range(ny):
            for ix in range(nx):
                xc_mm = xs[ix] * 1000.0
                yc_mm = ys[iy] * 1000.0
                is_inside, mat = self.core._is_point_in_section(xc_mm, yc_mm, self.section)
                if is_inside:
                    voxel_mask[(ix, iy)] = mat
                    active_indices.append((ix, iy))

        # 3. Generazione Nodi Solidi
        coords = []
        node_map = {} # (ix, iy, iz) -> node_index
        node_counter = 0
        
        zs = np.linspace(0, L_beam, nz + 1)
        
        for iz in range(nz + 1):
            z = zs[iz]
            for (ix, iy) in active_indices:
                # Crea nodi per ogni angolo del voxel se non esistono
                for dx_i in [0, 1]:
                    for dy_i in [0, 1]:
                        key = (ix + dx_i, iy + dy_i, iz)
                        if key not in node_map:
                            rx = min_x + (key[0]) * dx
                            ry = min_y + (key[1]) * dy
                            coords.append([rx, ry, z])
                            node_map[key] = node_counter
                            node_counter += 1
                            
        coords = np.array(coords)
        
        # 4. Generazione Elementi Hex8
        elements = []
        for iz in range(nz):
            for (ix, iy) in active_indices:
                mat = voxel_mask[(ix, iy)]
                # Ordine nodi standard Hex8
                try:
                    n_indices = [
                        node_map[(ix, iy, iz)], node_map[(ix+1, iy, iz)], 
                        node_map[(ix+1, iy+1, iz)], node_map[(ix, iy+1, iz)],
                        node_map[(ix, iy, iz+1)], node_map[(ix+1, iy, iz+1)], 
                        node_map[(ix+1, iy+1, iz+1)], node_map[(ix, iy+1, iz+1)]
                    ]
                    elements.append({'nodes': n_indices, 'mat': mat, 'type': 'HEX8'})
                except KeyError:
                    pass 

        solid_node_count = len(coords)
        bar_elements = []

        # 5. Barre Longitudinali
        for bar in self.section.get('bars', []):
            bx = self.core._mm_to_m(bar['center'][0])
            by = self.core._mm_to_m(bar['center'][1])
            diam = self.core._mm_to_m(bar['diam'])
            area = math.pi * (diam/2)**2
            mat = bar.get('material')
            
            prev_node = -1
            for iz in range(nz + 1):
                z = zs[iz]
                coords = np.vstack([coords, [bx, by, z]])
                curr_node = len(coords) - 1
                if prev_node != -1:
                    bar_elements.append({'nodes': [prev_node, curr_node], 'area': area, 'mat': mat, 'type': 'TRUSS_LONG'})
                prev_node = curr_node

        # Staffe
        passo_staffe = self.params.get('stirrup_step', 0.0)
        if passo_staffe > 0:
            z_stirrups = np.arange(passo_staffe, L_beam, passo_staffe)
            for s in self.section.get('staffe', []):
                pts_mm = s.get('points', [])
                diam = self.core._mm_to_m(s.get('diam', 8))
                area = math.pi * (diam/2)**2
                mat = s.get('material')
                
                if len(pts_mm) < 2: continue
                
                pts_m = [(self.core._mm_to_m(p[0]), self.core._mm_to_m(p[1])) for p in pts_mm]
                
                for z_s in z_stirrups:
                    if z_s > L_beam: break
                    first = -1
                    prev = -1
                    for i, p in enumerate(pts_m):
                        coords = np.vstack([coords, [p[0], p[1], z_s]])
                        curr = len(coords) - 1
                        if i == 0: first = curr
                        else:
                            bar_elements.append({'nodes': [prev, curr], 'area': area, 'mat': mat, 'type': 'TRUSS_STIR'})
                        prev = curr
                    # Chiudi staffa se necessario
                    if len(pts_m) > 2 and pts_m[0] != pts_m[-1]:
                         bar_elements.append({'nodes': [prev, first], 'area': area, 'mat': mat, 'type': 'TRUSS_STIR'})

        # 6. Penalty Links (Collega barre al calcestruzzo)
        penalty_pairs = []
        solid_coords = coords[:solid_node_count]
        if len(coords) > solid_node_count:
            for i in range(solid_node_count, len(coords)):
                dists = np.linalg.norm(solid_coords - coords[i], axis=1)
                nearest_idx = np.argmin(dists)
                if dists[nearest_idx] < 0.2: # soglia tolleranza
                    penalty_pairs.append((i, nearest_idx))

        return coords, elements, bar_elements, penalty_pairs, solid_node_count

    def run_solver(self, coords, solid_elems, bar_elems, penalty_links):
        n_dof = len(coords) * 3
        u = np.zeros(n_dof)
        
        # --- Boundary Conditions ---
        fixed_dofs = []
        L = self.params['L']
        tol_geom = 1e-4

        def get_face_nodes(fname):
            indices = []
            for i, c in enumerate(coords):
                match = False
                if fname == 'x0' and abs(c[0] - np.min(coords[:,0])) < tol_geom: match = True
                elif fname == 'xL' and abs(c[0] - np.max(coords[:,0])) < tol_geom: match = True
                elif fname == 'y0' and abs(c[1] - np.min(coords[:,1])) < tol_geom: match = True
                elif fname == 'yL' and abs(c[1] - np.max(coords[:,1])) < tol_geom: match = True
                elif fname == 'z0' and abs(c[2] - 0.0) < tol_geom: match = True
                elif fname == 'zL' and abs(c[2] - L) < tol_geom: match = True
                
                if match: indices.append(i)
            return indices

        constraints = self.params['constraints'] 
        for c_name in constraints:
            nodes = get_face_nodes(c_name)
            for n in nodes:
                fixed_dofs.extend([3*n, 3*n+1, 3*n+2])
        
        fixed_dofs = np.unique(fixed_dofs)
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

        # --- Loads ---
        F_ext_total = np.zeros(n_dof)
        load_val = self.params['load_value']
        load_dir_str = self.params['load_dir']
        load_dir = 0 if load_dir_str == 'x' else (1 if load_dir_str == 'y' else 2)
        
        target_nodes = []
        for loc in self.params['load_locations']:
            target_nodes.extend(get_face_nodes(loc))
        target_nodes = np.unique(target_nodes)
        
        if len(target_nodes) > 0:
            force_per_node = load_val / len(target_nodes)
            for n in target_nodes:
                F_ext_total[3*n + load_dir] += force_per_node
        
        # --- Precompute Gauss (Hex8) ---
        def get_hex8_shape(xi, eta, zeta):
            pts = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
            N = np.zeros(8); dN = np.zeros((3,8))
            for i in range(8):
                px, py, pz = pts[i]
                factor = 0.125
                N[i] = factor * (1 + px*xi) * (1 + py*eta) * (1 + pz*zeta)
                dN[0,i] = factor * px * (1 + py*eta) * (1 + pz*zeta)
                dN[1,i] = factor * py * (1 + px*xi) * (1 + pz*zeta)
                dN[2,i] = factor * pz * (1 + px*xi) * (1 + py*eta)
            return N, dN

        gauss_pts = [-0.57735, 0.57735]
        solid_predata = []
        for el in solid_elems:
            el_nodes = el['nodes']
            el_coords = coords[el_nodes]
            gps = []
            for gz in gauss_pts:
                for gy in gauss_pts:
                    for gx in gauss_pts:
                        N, local_grad = get_hex8_shape(gx, gy, gz)
                        J = local_grad @ el_coords
                        detJ = np.linalg.det(J)
                        if detJ <= 0: detJ = 1e-12
                        invJ = np.linalg.inv(J)
                        dN_dX = invJ @ local_grad
                        B = np.zeros((6, 24))
                        for i in range(8):
                            idx = 3*i
                            B[0, idx] = dN_dX[0,i]
                            B[1, idx+1] = dN_dX[1,i]
                            B[2, idx+2] = dN_dX[2,i]
                            B[3, idx] = dN_dX[1,i]; B[3, idx+1] = dN_dX[0,i]
                            B[4, idx+1] = dN_dX[2,i]; B[4, idx+2] = dN_dX[1,i]
                            B[5, idx] = dN_dX[2,i]; B[5, idx+2] = dN_dX[0,i]
                        gps.append({'B': B, 'detJ': detJ})
            solid_predata.append({'gps': gps, 'nodes': el_nodes, 'mat': el['mat']})

        # --- Loop ---
        steps = self.params['steps']
        iters = self.params['iters']
        tol = self.params['tol']
        K_penalty = 1e13
        
        u_history = [np.zeros(n_dof)]
        stress_history = [np.zeros(len(coords))] # Stress nodale iniziale (0)

        default_mat = MaterialParser("Default")

        for step in range(1, steps + 1):
            load_factor = step / steps
            F_target = F_ext_total * load_factor
            self.progress_update.emit(f"Step {step}/{steps} (Load: {load_factor:.2f})")
            
            # --- Newton-Raphson Loop ---
            for it in range(iters):
                rows, cols, data = [], [], []
                R_int = np.zeros(n_dof)
                
                # Solids Assembly
                for el_dat in solid_predata:
                    mat_obj = self.materials_db.get(el_dat['mat'], default_mat)
                    dof_ind = []
                    for n in el_dat['nodes']: dof_ind.extend([3*n, 3*n+1, 3*n+2])
                    u_el = u[dof_ind]
                    k_el = np.zeros((24, 24))
                    r_el = np.zeros(24)
                    
                    for gp in el_dat['gps']:
                        B, vol = gp['B'], gp['detJ']
                        eps = B @ u_el
                        eps_vm = math.sqrt(0.5*((eps[0]-eps[1])**2+(eps[1]-eps[2])**2+(eps[2]-eps[0])**2)+3*(eps[3]**2+eps[4]**2+eps[5]**2))
                        sign = -1.0 if np.sum(eps[:3]) < 0 else 1.0
                        _, E_tan = mat_obj.evaluate(eps_vm * sign)
                        
                        nu = 0.2
                        f = E_tan / ((1+nu)*(1-2*nu))
                        D = np.zeros((6,6))
                        D[:3,:3] = f*(1-nu); D[:3,:3] -= f*nu; np.fill_diagonal(D[:3,:3], f*(1-nu))
                        D[0,1]=D[0,2]=D[1,0]=D[1,2]=D[2,0]=D[2,1] = f*nu
                        sh = E_tan/(2*(1+nu)); D[3,3]=D[4,4]=D[5,5] = sh
                        
                        stress = D @ eps
                        r_el += (B.T @ stress) * vol
                        k_el += (B.T @ D @ B) * vol
                    
                    for r in range(24):
                        R_int[dof_ind[r]] += r_el[r]
                        for c in range(24):
                            val = k_el[r,c]
                            if abs(val) > 1e-9:
                                rows.append(dof_ind[r]); cols.append(dof_ind[c]); data.append(val)

                # Bars Assembly
                for bel in bar_elems:
                    mat_obj = self.materials_db.get(bel['mat'], default_mat)
                    n1, n2 = bel['nodes']
                    idx1, idx2 = [3*n1, 3*n1+1, 3*n1+2], [3*n2, 3*n2+1, 3*n2+2]
                    p1, p2 = coords[n1], coords[n2]
                    L0 = np.linalg.norm(p2 - p1)
                    if L0 < 1e-9: continue
                    direction = (p2 - p1) / L0
                    u1, u2 = u[idx1], u[idx2]
                    curr_len = np.linalg.norm((p2+u2) - (p1+u1))
                    strain = (curr_len - L0) / L0
                    stress, Et = mat_obj.evaluate(strain)
                    
                    force = stress * bel['area']
                    stiff = Et * bel['area'] / L0
                    f_vec = force * direction
                    
                    R_int[idx1] -= f_vec; R_int[idx2] += f_vec
                    K_loc = np.outer(direction, direction) * stiff
                    
                    for i in range(3):
                        for j in range(3):
                            v = K_loc[i,j]
                            rows.append(idx1[i]); cols.append(idx1[j]); data.append(v)
                            rows.append(idx1[i]); cols.append(idx2[j]); data.append(-v)
                            rows.append(idx2[i]); cols.append(idx1[j]); data.append(-v)
                            rows.append(idx2[i]); cols.append(idx2[j]); data.append(v)

                # Penalty Assembly
                for (bn, sn) in penalty_links:
                    bi, si = [3*bn, 3*bn+1, 3*bn+2], [3*sn, 3*sn+1, 3*sn+2]
                    ub, us = u[bi], u[si]
                    f_pen = K_penalty * (ub - us)
                    R_int[bi] += f_pen; R_int[si] -= f_pen
                    for k in range(3):
                        rows.extend([bi[k], si[k], bi[k], si[k]])
                        cols.extend([bi[k], si[k], si[k], bi[k]])
                        data.extend([K_penalty, K_penalty, -K_penalty, -K_penalty])
                
                res = F_target - R_int
                norm = np.linalg.norm(res[free_dofs])
                
                if norm < tol:
                    break
                
                K_global = sp.coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
                K_free = K_global[free_dofs, :][:, free_dofs]
                
                try:
                    du = spla.spsolve(K_free, res[free_dofs])
                    u[free_dofs] += du
                except:
                    break # Singolarità

            # --- POST-PROCESSING STRESS FOR VISUALIZATION ---
            # Alla fine dello step, calcoliamo le tensioni nodali da passare alla GUI
            step_node_stress = np.zeros(len(coords))
            step_node_count = np.zeros(len(coords)) # Per fare la media sui nodi condivisi

            # 1. Stress Solidi (Von Mises)
            for el_dat in solid_predata:
                mat_obj = self.materials_db.get(el_dat['mat'], default_mat)
                dof_ind = []
                for n in el_dat['nodes']: dof_ind.extend([3*n, 3*n+1, 3*n+2])
                u_el = u[dof_ind]
                
                # Calcoliamo Von Mises medio sugli Integration Points
                vm_avg = 0.0
                for gp in el_dat['gps']:
                    B = gp['B']
                    eps = B @ u_el
                    eps_vm = math.sqrt(0.5*((eps[0]-eps[1])**2+(eps[1]-eps[2])**2+(eps[2]-eps[0])**2)+3*(eps[3]**2+eps[4]**2+eps[5]**2))
                    sign = -1.0 if np.sum(eps[:3]) < 0 else 1.0
                    sigma_scalar, _ = mat_obj.evaluate(eps_vm * sign)
                    vm_avg += abs(sigma_scalar) # Visualizziamo modulo della tensione
                
                vm_avg /= len(el_dat['gps'])
                
                # Distribuiamo ai nodi dell'elemento
                for n in el_dat['nodes']:
                    step_node_stress[n] += vm_avg
                    step_node_count[n] += 1.0

            # 2. Stress Barre (Assiale)
            for bel in bar_elems:
                mat_obj = self.materials_db.get(bel['mat'], default_mat)
                n1, n2 = bel['nodes']
                idx1, idx2 = [3*n1, 3*n1+1, 3*n1+2], [3*n2, 3*n2+1, 3*n2+2]
                p1, p2 = coords[n1], coords[n2]
                L0 = np.linalg.norm(p2 - p1)
                if L0 < 1e-9: continue
                u1, u2 = u[idx1], u[idx2]
                curr_len = np.linalg.norm((p2+u2) - (p1+u1))
                strain = (curr_len - L0) / L0
                sigma_scalar, _ = mat_obj.evaluate(strain)
                
                # Assegniamo ai nodi
                val = abs(sigma_scalar)
                step_node_stress[n1] += val; step_node_count[n1] += 1.0
                step_node_stress[n2] += val; step_node_count[n2] += 1.0

            # Media sui nodi
            # Evita divisione per zero per nodi isolati (se esistono)
            avg_stress = np.divide(step_node_stress, step_node_count, where=step_node_count!=0)
            
            u_history.append(u.copy())
            stress_history.append(avg_stress)
            
        return u_history, stress_history


class BeamCalcolo(QObject):
    def __init__(self, parent, ui, mesh_generator):
        super().__init__(parent)
        self.ui = ui
        self.mesh_generator = mesh_generator
        self.worker = None

    def start_fem_analysis(self):
        # 1. Recupera Dati dalla UI
        try:
            # Parametri Geometrici e Mesh
            try: L = float(self.ui.beam_lunghezza.text())
            except: L = 3.0
            
            try: nx = int(self.ui.beam_definizione_x.text())
            except: nx = 12
            try: ny = int(self.ui.beam_definizione_y.text())
            except: ny = 12
            try: nz = int(self.ui.beam_definizione_z.text())
            except: nz = 10
            
            try: stirrup_step = float(self.ui.beam_passo.text())
            except: stirrup_step = 0.0

            # Carico
            try: load_val = float(self.ui.beam_carico.text())
            except: load_val = -1000.0
            
            if self.ui.beam_carico_direzione_x.isChecked(): load_dir = 'x'
            elif self.ui.beam_carico_direzione_y.isChecked(): load_dir = 'y'
            elif self.ui.beam_carico_direzione_z.isChecked(): load_dir = 'z'
            
            load_locs = []
            if getattr(self.ui, 'beam_carico_x0', None) and self.ui.beam_carico_x0.isChecked(): load_locs.append('x0')
            if getattr(self.ui, 'beam_carico_xL', None) and self.ui.beam_carico_xL.isChecked(): load_locs.append('xL')
            if getattr(self.ui, 'beam_carico_y0', None) and self.ui.beam_carico_y0.isChecked(): load_locs.append('y0')
            if getattr(self.ui, 'beam_carico_yL', None) and self.ui.beam_carico_yL.isChecked(): load_locs.append('yL')
            if getattr(self.ui, 'beam_carico_z0', None) and self.ui.beam_carico_z0.isChecked(): load_locs.append('z0')
            if getattr(self.ui, 'beam_carico_zL', None) and self.ui.beam_carico_zL.isChecked(): load_locs.append('zL')

            # Vincoli
            constraints = []
            if getattr(self.ui, 'beam_vincolo_x0', None) and self.ui.beam_vincolo_x0.isChecked(): constraints.append('x0')
            if getattr(self.ui, 'beam_vincolo_xL', None) and self.ui.beam_vincolo_xL.isChecked(): constraints.append('xL')
            if getattr(self.ui, 'beam_vincolo_y0', None) and self.ui.beam_vincolo_y0.isChecked(): constraints.append('y0')
            if getattr(self.ui, 'beam_vincolo_yL', None) and self.ui.beam_vincolo_yL.isChecked(): constraints.append('yL')
            if getattr(self.ui, 'beam_vincolo_z0', None) and self.ui.beam_vincolo_z0.isChecked(): constraints.append('z0')
            if getattr(self.ui, 'beam_vincolo_zL', None) and self.ui.beam_vincolo_zL.isChecked(): constraints.append('zL')

            # Parametri Solver
            try: steps = int(self.ui.beam_steps.text())
            except: steps = 5
            try: iters = int(self.ui.beam_iterazioni.text())
            except: iters = 10
            try: tol = float(self.ui.beam_tolleranza.text())
            except: tol = 1e-2
            try: scale_def = float(self.ui.beam_scala_deformazione.text())
            except: scale_def = 1.0

        except Exception as e:
            QMessageBox.critical(self.ui, "Errore Input", f"Errore lettura parametri GUI: {e}")
            return

        # 2. Genera Dati Sezione e Materiali
        self.mesh_generator.generate_mesh() 
        sel_idx = self.mesh_generator.selected_section_index
        if sel_idx is None: sel_idx = 0
        
        try:
            mats, objs = self.mesh_generator.beam_valori.generate_matrices(sel_idx)
            materials_db = {}
            for m_def in mats:
                parser = MaterialParser(m_def)
                materials_db[parser.name] = parser
            
            section = self.mesh_generator._build_section_from_matrices(mats, objs)

        except Exception as e:
             QMessageBox.critical(self.ui, "Errore Dati", f"Errore recupero dati sezione: {e}")
             return

        # 3. Avvia Worker
        params = {
            'L': L, 'nx': nx, 'ny': ny, 'nz': nz, 'stirrup_step': stirrup_step,
            'load_value': load_val, 'load_dir': load_dir, 'load_locations': load_locs,
            'constraints': constraints,
            'steps': steps, 'iters': iters, 'tol': tol
        }

        self.worker = FemWorker(section, materials_db, params)
        self.worker.progress_update.connect(lambda s: print(f"[FEM] {s}")) 
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished_computation.connect(self._on_success)
        
        self.target_scale = scale_def
        self.worker.start()

    def _on_error(self, msg):
        QMessageBox.critical(self.ui, "Errore FEM", msg)

    def _on_success(self, history, coords, solid_elems, bar_elems, max_disp, stress_history, max_stress):
        print(f"[FEM] Analisi completata. Max disp: {max_disp:.4f}, Max Stress: {max_stress:.2f}")
        
        gl_widget = self.mesh_generator._ensure_gl_widget_in_ui()
        if gl_widget:
            # Passiamo ANCHE stress_history e max_stress
            gl_widget.set_fem_results(history, coords, solid_elems, bar_elems, max_disp, stress_history, max_stress)
            gl_widget.deformation_scale = self.target_scale
            gl_widget.start_animation()
        else:
            QMessageBox.warning(self.ui, "Attenzione", "Widget 3D non trovato per visualizzare i risultati.")