import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math
import traceback
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QMessageBox

# Importiamo la logica base geometrica
from beam.mesh import BeamMeshCore

class MaterialParser:
    def __init__(self, definition):
        self.segments = []
        self.name = "Unknown"
        self.tensile_strength = 0.0  # Magnitudo massima in trazione
        self.comp_strength = 0.0     # Magnitudo massima in compressione
        
        # --- PARSING DELLA DEFINIZIONE UTENTE ---
        # Nessun materiale di default. Se la definizione è vuota, crea un materiale dummy debole
        # per evitare crash, ma non assume che sia acciaio o cls.
        
        if not definition:
             # Fallback estremo solo se arriva None, per stabilità software
            self.name = "Void"
            self.segments.append({'func': '1e-9*x', 'min': -1.0, 'max': 1.0})
            return

        self.name = definition[0]
        
        if len(definition) > 1 and isinstance(definition[1], (list, tuple)):
            for segment in definition[1:]:
                if len(segment) >= 3:
                    try:
                        formula = segment[0]
                        start = float(segment[1])
                        end = float(segment[2])
                        self.segments.append({'func': formula, 'min': start, 'max': end})
                    except:
                        pass
        
        if not self.segments:
            # Se l'utente ha creato un materiale vuoto
            self.segments.append({'func': '0*x', 'min': -1.0, 'max': 1.0})

        # --- CALCOLO AUTOMATICO DEI LIMITI BASATO SULLA FUNZIONE ---
        self._compute_limits_from_segments()
        print (self.tensile_strength, self.comp_strength)

    def _compute_limits_from_segments(self):
        """
        Scansiona i segmenti definiti dall'utente per trovare i valori massimi
        di resistenza in trazione (strain < 0) e compressione (strain > 0).
        """
        max_t = 0.0 # Trazione (magnitudo)
        max_c = 0.0 # Compressione (magnitudo)
        
        for seg in self.segments:
            # Creiamo dei punti di test all'interno del segmento
            # Evitiamo divisioni per zero o range nulli
            if seg['max'] <= seg['min']: continue
            
            steps = np.linspace(seg['min'], seg['max'], 20)
            
            for s in steps:
                try:
                    # Valutazione sicura
                    val = eval(seg['func'], {"__builtins__": None}, {"x": s, "abs": abs, "math": math})
                    val = float(val)
                    
                    # Convenzione: s < 0 è Trazione, s > 0 è Compressione
                    if s < 0:
                        # In trazione lo stress dovrebbe essere negativo (o positivo se l'utente ha invertito la formula)
                        # Prendiamo il valore assoluto come capacità resistente
                        if abs(val) > max_t: max_t = abs(val)
                    else:
                        if abs(val) > max_c: max_c = abs(val)
                except:
                    pass
        
        self.tensile_strength = max_t
        self.comp_strength = max_c * 1.7
        
        # Assicuriamo valori minimi per evitare divisioni per zero nel solver
        if self.tensile_strength < 1e-3: self.tensile_strength = 1e-3
        if self.comp_strength < 1e-3: self.comp_strength = 1e-3

    def evaluate(self, strain):
        """
        Valuta stress e rigidezza tangente.
        Strain input: Positivo = Compressione, Negativo = Trazione.
        """
        active_seg = None
        
        # Cerca il segmento attivo
        for seg in self.segments:
            if seg['min'] <= strain <= seg['max']:
                active_seg = seg
                break
        
        # --- FALLBACK PER STRAIN FUORI RANGE ---
        if active_seg is None:
            # Se siamo fuori dai limiti definiti dall'utente:
            # Assumiamo che il materiale abbia ceduto o non offra resistenza.
            # Ritorniamo una rigidezza residua minima per la stabilità numerica.
            return 0.0, 1.0 
            
        # Funzione helper per eval
        def eval_str(f_str, x_val):
            try:
                return float(eval(f_str, {"__builtins__": None}, {"x": x_val, "abs": abs, "math": math}))
            except:
                return 0.0

        # Calcolo numerico della tangente
        h = 1e-7
        sigma = eval_str(active_seg['func'], strain)
        sigma_h = eval_str(active_seg['func'], strain + h)
        
        tangent = (sigma_h - sigma) / h
        
        # Clamp valori numerici per sicurezza (NaN o Inf)
        if math.isnan(sigma) or math.isinf(sigma): sigma = 0.0
        if math.isnan(tangent) or math.isinf(tangent): tangent = 1.0
        
        # Evita rigidezza nulla esatta (singolarità)
        if abs(tangent) < 1.0: tangent = 1.0
        
        return sigma, tangent

    def get_tensile_limit(self):
        """Ritorna la resistenza massima a trazione rilevata dalle funzioni utente"""
        return self.tensile_strength

class FemWorkerDG(QThread):
    finished_computation = pyqtSignal(object, object, object, object, object, object, float) 
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    progress_percent = pyqtSignal(int)

    def __init__(self, section_data, materials_db, params):
        super().__init__()
        self.section = section_data
        self.materials_db = materials_db
        self.params = params
        self.core = BeamMeshCore()

    def run(self):
        try:
            self.progress_percent.emit(0)
            self.progress_update.emit("Generazione Mesh DG (Discontinuous)...")
            
            # 1. Mesh Generation (Nodes exploded)
            mesh_data = self.generate_dg_mesh()
            self.progress_percent.emit(15)

            # 2. Solver Loop
            self.progress_update.emit("Solver Non-Lineare DG...")
            results = self.run_solver_dg(mesh_data)
            
            self.progress_percent.emit(100)
            self.finished_computation.emit(*results)

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def generate_dg_mesh(self):
        # ... (Logica di mesh identica alla precedente, omessa per brevità ma inclusa funzionalmente)
        # La logica di generazione mesh non cambia: crea nodi duplicati per ogni elemento Hex8.
        # Qui riporto il codice essenziale per garantire il funzionamento.
        L_beam = self.params['L']
        nx, ny, nz = self.params['nx'], self.params['ny'], self.params['nz']
        
        min_x, max_x, min_y, max_y = self.core._get_section_bounding_box(self.section)
        min_x, max_x = self.core._mm_to_m(min_x), self.core._mm_to_m(max_x)
        min_y, max_y = self.core._mm_to_m(min_y), self.core._mm_to_m(max_y)
        
        Lx_bbox = max_x - min_x
        Ly_bbox = max_y - min_y
        
        dx = Lx_bbox / max(1, nx) if Lx_bbox > 1e-9 else 1.0
        dy = Ly_bbox / max(1, ny) if Ly_bbox > 1e-9 else 1.0
        dz = L_beam / max(1, nz)

        xs = np.linspace(min_x + dx/2, max_x - dx/2, nx)
        ys = np.linspace(min_y + dy/2, max_y - dy/2, ny)
        
        active_voxels = {}
        for iy in range(ny):
            for ix in range(nx):
                xc_mm = xs[ix] * 1000.0; yc_mm = ys[iy] * 1000.0
                is_inside, mat = self.core._is_point_in_section(xc_mm, yc_mm, self.section)
                if is_inside:
                    for iz in range(nz): active_voxels[(ix, iy, iz)] = mat

        coords = []
        solid_elements = []
        element_map = {}
        node_cursor = 0

        for key, mat in active_voxels.items():
            ix, iy, iz = key
            x0 = min_x + ix * dx; x1 = x0 + dx
            y0 = min_y + iy * dy; y1 = y0 + dy
            z0 = iz * dz;         z1 = z0 + dz
            
            corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                       (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
            
            el_nodes_indices = []
            for c in corners:
                coords.append(list(c))
                el_nodes_indices.append(node_cursor)
                node_cursor += 1
            
            el_idx = len(solid_elements)
            solid_elements.append({'nodes': el_nodes_indices, 'mat': mat, 'type': 'HEX8'})
            element_map[key] = el_idx

        # Interfacce
        interfaces = []
        neighbor_offsets = [(1, 0, 0, 0), (0, 1, 0, 1), (0, 0, 1, 2)] # dx, dy, dz, axis
        face_map = {
            0: {'A': [1, 2, 6, 5], 'B': [0, 3, 7, 4], 'n': [1, 0, 0]},
            1: {'A': [2, 3, 7, 6], 'B': [1, 0, 4, 5], 'n': [0, 1, 0]},
            2: {'A': [4, 5, 6, 7], 'B': [0, 1, 2, 3], 'n': [0, 0, 1]}
        }

        for key, el_idx_A in element_map.items():
            ix, iy, iz = key
            for dix, diy, diz, axis in neighbor_offsets:
                neighbor_key = (ix + dix, iy + diy, iz + diz)
                if neighbor_key in element_map:
                    el_idx_B = element_map[neighbor_key]
                    nodes_A = solid_elements[el_idx_A]['nodes']
                    nodes_B = solid_elements[el_idx_B]['nodes']
                    
                    indices_A = [nodes_A[i] for i in face_map[axis]['A']]
                    indices_B = [nodes_B[i] for i in face_map[axis]['B']]
                    
                    area = (dy*dz if axis==0 else (dx*dz if axis==1 else dx*dy))
                    interfaces.append({
                        'nodes_A': indices_A, 'nodes_B': indices_B,
                        'normal': np.array(face_map[axis]['n']), 'area': area,
                        'mat_A': solid_elements[el_idx_A]['mat'],
                        'mat_B': solid_elements[el_idx_B]['mat']
                    })

        solid_node_count = len(coords)
        coords = np.array(coords)
        
        # Barre e Penalty (Codice condensato, logica identica)
        bar_elements = []
        temp_coords = coords.tolist()
        zs = np.linspace(0, L_beam, nz + 1)
        
        for bar in self.section.get('bars', []):
            bx, by = self.core._mm_to_m(bar['center'][0]), self.core._mm_to_m(bar['center'][1])
            diam = self.core._mm_to_m(bar['diam'])
            area = math.pi * (diam/2)**2
            prev = -1
            for iz in range(nz+1):
                curr = len(temp_coords)
                temp_coords.append([bx, by, zs[iz]])
                if prev != -1:
                    bar_elements.append({'nodes': [prev, curr], 'area': area, 'mat': bar.get('material'), 'type': 'TRUSS_LONG'})
                prev = curr
        
        # Staffe
        passo = self.params.get('stirrup_step', 0.0)
        if passo > 0:
            num = max(1, int(math.ceil(L_beam / passo)))
            for s in self.section.get('staffe', []):
                diam = self.core._mm_to_m(s.get('diam', 8))
                area = math.pi * (diam/2)**2
                pts = [(self.core._mm_to_m(p[0]), self.core._mm_to_m(p[1])) for p in s.get('points', [])]
                if len(pts)<2: continue
                for i in range(num):
                    z = (i+0.5)*passo
                    if z > L_beam: continue
                    first = -1; prev = -1
                    for k, p in enumerate(pts):
                        curr = len(temp_coords)
                        temp_coords.append([p[0], p[1], z])
                        if k==0: first=curr
                        else: bar_elements.append({'nodes':[prev,curr], 'area':area, 'mat':s.get('material'), 'type':'TRUSS_STIR'})
                        prev=curr
                    if len(pts)>2 and pts[0]!=pts[-1]:
                         bar_elements.append({'nodes':[prev,first], 'area':area, 'mat':s.get('material'), 'type':'TRUSS_STIR'})

        coords = np.array(temp_coords)
        penalty_links = []
        solid_coords = coords[:solid_node_count]
        search_rad = max(dx, dy) * 1.5
        for i in range(solid_node_count, len(coords)):
            dists = np.linalg.norm(solid_coords - coords[i], axis=1)
            near = np.argmin(dists)
            if dists[near] < search_rad: penalty_links.append((i, near))

        return {'coords': coords, 'solid_elems': solid_elements, 'interfaces': interfaces,
                'bar_elems': bar_elements, 'penalty_links': penalty_links}

    def run_solver_dg(self, mesh_data):
        coords = mesh_data['coords']
        solid_elems = mesh_data['solid_elems']
        interfaces = mesh_data['interfaces']
        bar_elems = mesh_data['bar_elems']
        penalty_links = mesh_data['penalty_links']
        
        n_dof = len(coords) * 3
        u = np.zeros(n_dof)
        
        solid_predata = self._precompute_solids(coords, solid_elems)
        
        interface_damage = np.zeros(len(interfaces), dtype=int)
        bar_damage = np.zeros(len(bar_elems), dtype=int)
        
        steps = self.params['steps']
        iters = self.params['iters']
        tol = self.params['tol']
        
        fixed_dofs = self._get_constraints(coords, self.params['constraints'], self.params['L'])
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)
        F_ext_base = self._get_loads(coords, n_dof)
        
        u_history = [np.zeros(n_dof)]
        stress_history = [np.zeros(len(coords))]
        
        K_int_elastic = 1e13 
        K_penalty = 1e13
        default_mat = MaterialParser(None) # Materiale vuoto di sicurezza

        for step in range(1, steps + 1):
            load_factor = step / steps
            F_target = F_ext_base * load_factor
            
            prog = 10 + int((step/steps)*90)
            self.progress_percent.emit(prog)
            self.progress_update.emit(f"Step {step}/{steps} (Carico {load_factor*100:.0f}%)")
            
            for it in range(iters):
                rows, cols, data = [], [], []
                R_int = np.zeros(n_dof)
                
                # --- SOLIDI ---
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
                        
                        # Calcolo Strain Scalare (Segno Positivo = Compressione, Negativo = Trazione)
                        # FEM Eps > 0 -> Estensione. FEM Eps < 0 -> Compressione.
                        eps_vol = np.sum(eps[:3])
                        eps_vm = math.sqrt(0.5*((eps[0]-eps[1])**2+(eps[1]-eps[2])**2+(eps[2]-eps[0])**2)+3*(eps[3]**2+eps[4]**2+eps[5]**2))
                        
                        # Mapping convenzione: 
                        # Se eps_vol < 0 (FEM Comp), vogliamo strain_scalar > 0 (Civil Comp).
                        # Se eps_vol > 0 (FEM Ext), vogliamo strain_scalar < 0 (Civil Tens).
                        sign = 1.0 if eps_vol < 0 else -1.0
                        strain_scalar = eps_vm * sign
                        
                        # Evaluate dal materiale reale
                        sigma_val, E_tan = mat_obj.evaluate(strain_scalar)
                        
                        # Modulo secante per forze interne corrette
                        if abs(strain_scalar) < 1e-12: E_sec = E_tan
                        else: E_sec = sigma_val / strain_scalar
                        
                        # Stiffness Matrix (Tangente)
                        nu = 0.2
                        f = E_tan/((1+nu)*(1-2*nu))
                        D_tan = np.zeros((6,6))
                        D_tan[:3,:3]=f*(1-nu); D_tan[:3,:3]-=f*nu; np.fill_diagonal(D_tan[:3,:3],f*(1-nu))
                        D_tan[0,1]=D_tan[0,2]=D_tan[1,0]=D_tan[1,2]=D_tan[2,0]=D_tan[2,1]=f*nu
                        D_tan[3,3]=D_tan[4,4]=D_tan[5,5]=E_tan/(2*(1+nu))

                        # Stress calculation (Secante)
                        f_s = E_sec/((1+nu)*(1-2*nu))
                        D_sec = np.zeros((6,6))
                        D_sec[:3,:3]=f_s*(1-nu); D_sec[:3,:3]-=f_s*nu; np.fill_diagonal(D_sec[:3,:3],f_s*(1-nu))
                        D_sec[0,1]=D_sec[0,2]=D_sec[1,0]=D_sec[1,2]=D_sec[2,0]=D_sec[2,1]=f_s*nu
                        D_sec[3,3]=D_sec[4,4]=D_sec[5,5]=E_sec/(2*(1+nu))
                        
                        r_el += (B.T @ (D_sec @ eps)) * vol
                        k_el += (B.T @ D_tan @ B) * vol
                    
                    for r in range(24):
                        R_int[dof_ind[r]] += r_el[r]
                        for c in range(24):
                            if abs(k_el[r,c]) > 1e-9:
                                rows.append(dof_ind[r]); cols.append(dof_ind[c]); data.append(k_el[r,c])

                # --- INTERFACCE ---
                broken_count = 0
                TRIGGER_FACTOR = 0.75

                for i_idx, iface in enumerate(interfaces):
                    if interface_damage[i_idx] == 1:
                        broken_count += 1; continue
                    
                    nA, nB = iface['nodes_A'], iface['nodes_B']
                    norm = iface['normal']
                    area = iface['area']
                    
                    # Calcolo limite di rottura dai materiali
                    matA = self.materials_db.get(iface['mat_A'], default_mat)
                    matB = self.materials_db.get(iface['mat_B'], default_mat)
                    # La resistenza è il minimo tra i due materiali a contatto
                    ft_real = min(matA.get_tensile_limit(), matB.get_tensile_limit())
                    ft_trigger = ft_real * TRIGGER_FACTOR
                    
                    disp_A = np.mean([u[3*n:3*n+3] for n in nA], axis=0)
                    disp_B = np.mean([u[3*n:3*n+3] for n in nB], axis=0)
                    gap_n = np.dot(disp_B - disp_A, norm) # Positivo = Apertura = Trazione
                    
                    # Check Rottura: Se Trazione > Limite Trazione Materiale
                    traction_stress = K_int_elastic * gap_n # Stress approssimato interfaccia
                    
                    # NOTA: ft_lim è una magnitudo positiva. traction_stress > ft_lim significa rottura.
                    if gap_n > 0 and traction_stress > ft_trigger:
                        interface_damage[i_idx] = 1
                        broken_count += 1
                        continue
                    
                    k_node = (K_int_elastic * area) / 4.0
                    for k in range(4):
                        idxA = [3*nA[k], 3*nA[k]+1, 3*nA[k]+2]
                        idxB = [3*nB[k], 3*nB[k]+1, 3*nB[k]+2]
                        f_vec = k_node * (u[idxB] - u[idxA])
                        R_int[idxA] -= f_vec; R_int[idxB] += f_vec
                        for d in range(3):
                            rows.extend([idxA[d], idxB[d], idxA[d], idxB[d]])
                            cols.extend([idxA[d], idxB[d], idxB[d], idxA[d]])
                            data.extend([k_node, k_node, -k_node, -k_node])

                # --- BARRE ---
                for bel in bar_elems:
                    mat_obj = self.materials_db.get(bel['mat'], default_mat)
                    n1, n2 = bel['nodes']
                    idx1, idx2 = [3*n1, 3*n1+1, 3*n1+2], [3*n2, 3*n2+1, 3*n2+2]
                    p1, p2 = coords[n1], coords[n2]
                    L0 = np.linalg.norm(p2-p1)
                    if L0 < 1e-9: continue
                    
                    u1, u2 = u[idx1], u[idx2]
                    curr_len = np.linalg.norm((p2+u2)-(p1+u1))
                    strain = (curr_len - L0) / L0
                    
                    # Strain geometrico pos = allungamento. Materiale input: pos = compressione.
                    strain_input = -strain
                    sigma_val, Et = mat_obj.evaluate(strain_input)
                    
                    if abs(strain) < 1e-12: E_sec = Et
                    else: E_sec = abs(sigma_val / strain) # Modulo secante
                    
                    # Forza: se allungamento (strain > 0), E_sec * strain -> Forza positiva (trazione)
                    # Corretto per FEM
                    force = E_sec * strain * bel['area']
                    dir_vec = (p2-p1)/L0
                    f_vec = force * dir_vec
                    
                    R_int[idx1] -= f_vec; R_int[idx2] += f_vec
                    
                    stiff = Et * bel['area'] / L0
                    K_loc = np.outer(dir_vec, dir_vec) * stiff
                    for r in range(3):
                        for c in range(3):
                            val = K_loc[r,c]
                            rows.append(idx1[r]); cols.append(idx1[c]); data.append(val)
                            rows.append(idx1[r]); cols.append(idx2[c]); data.append(-val)
                            rows.append(idx2[r]); cols.append(idx1[c]); data.append(-val)
                            rows.append(idx2[r]); cols.append(idx2[c]); data.append(val)

                # --- PENALTY ---
                for (bn, sn) in penalty_links:
                    bi, si = [3*bn, 3*bn+1, 3*bn+2], [3*sn, 3*sn+1, 3*sn+2]
                    f_pen = K_penalty * (u[bi] - u[si])
                    R_int[bi] += f_pen; R_int[si] -= f_pen
                    for k in range(3):
                        rows.extend([bi[k], si[k], bi[k], si[k]])
                        cols.extend([bi[k], si[k], si[k], bi[k]])
                        data.extend([K_penalty, K_penalty, -K_penalty, -K_penalty])
                
                # SOLVE
                res = F_target - R_int
                res_norm = np.linalg.norm(res[free_dofs])

                print(f"Cracks: {broken_count}/{len(interfaces)}")

                if res_norm < tol: break
                
                K_global = sp.coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
                K_free = K_global[free_dofs, :][:, free_dofs]
                try:
                    du = spla.spsolve(K_free, res[free_dofs])
                    u[free_dofs] += du
                except:
                    break
            
            u_history.append(u.copy())
            
            # Post-Processing Stress
            node_stress = np.zeros(len(coords))
            node_counts = np.zeros(len(coords))
            
            for el_dat in solid_predata:
                mat_obj = self.materials_db.get(el_dat['mat'], default_mat)
                u_el = u[[i for n in el_dat['nodes'] for i in (3*n, 3*n+1, 3*n+2)]]
                vm_sum = 0
                for gp in el_dat['gps']:
                    eps = gp['B'] @ u_el
                    sign = 1.0 if np.sum(eps[:3]) < 0 else -1.0
                    eps_vm = math.sqrt(0.5*((eps[0]-eps[1])**2+(eps[1]-eps[2])**2+(eps[2]-eps[0])**2)+3*(eps[3]**2+eps[4]**2+eps[5]**2))
                    sig, _ = mat_obj.evaluate(eps_vm * sign)
                    vm_sum += sig
                vm_avg = vm_sum / len(el_dat['gps'])
                for n in el_dat['nodes']:
                    node_stress[n] += vm_avg; node_counts[n] += 1
            
            for bel in bar_elems:
                mat_obj = self.materials_db.get(bel['mat'], default_mat)
                n1, n2 = bel['nodes']
                p1, p2 = coords[n1], coords[n2]
                u1, u2 = u[[3*n1, 3*n1+1, 3*n1+2]], u[[3*n2, 3*n2+1, 3*n2+2]]
                L0 = np.linalg.norm(p2-p1)
                curr_len = np.linalg.norm((p2+u2)-(p1+u1))
                strain = (curr_len - L0) / L0
                sig, _ = mat_obj.evaluate(-strain)
                node_stress[n1]+=sig; node_counts[n1]+=1
                node_stress[n2]+=sig; node_counts[n2]+=1
            
            avg_stress = np.divide(node_stress, node_counts, where=node_counts!=0)
            stress_history.append(avg_stress)

        max_disp = np.max(np.linalg.norm(u.reshape(-1, 3), axis=1))
        max_stress = np.max(stress_history[-1]) if len(stress_history) > 0 else 0
        return u_history, coords, solid_elems, bar_elems, max_disp, stress_history, max_stress

    def _precompute_solids(self, coords, solid_elems):
        # (Stessa implementazione helper del codice precedente)
        def get_hex8_shape(xi, eta, zeta):
            pts = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
            N = np.zeros(8); dN = np.zeros((3,8))
            for i in range(8):
                px, py, pz = pts[i]
                f = 0.125
                N[i] = f*(1+px*xi)*(1+py*eta)*(1+pz*zeta)
                dN[0,i]=f*px*(1+py*eta)*(1+pz*zeta)
                dN[1,i]=f*py*(1+px*xi)*(1+pz*zeta)
                dN[2,i]=f*pz*(1+px*xi)*(1+py*eta)
            return N, dN
        
        gauss_pts = [-0.57735, 0.57735]
        data = []
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
                            idx=3*i
                            B[0,idx]=dN_dX[0,i]; B[1,idx+1]=dN_dX[1,i]; B[2,idx+2]=dN_dX[2,i]
                            B[3,idx]=dN_dX[1,i]; B[3,idx+1]=dN_dX[0,i]
                            B[4,idx+1]=dN_dX[2,i]; B[4,idx+2]=dN_dX[1,i]
                            B[5,idx]=dN_dX[2,i]; B[5,idx+2]=dN_dX[0,i]
                        gps.append({'B': B, 'detJ': detJ})
            data.append({'gps': gps, 'nodes': el_nodes, 'mat': el['mat']})
        return data

    def _get_constraints(self, coords, constraints, L):
        fixed = []
        tol = 1e-4
        for i, c in enumerate(coords):
            x,y,z = c
            if 'x0' in constraints and abs(x-np.min(coords[:,0]))<tol: fixed.extend([3*i,3*i+1,3*i+2])
            elif 'xL' in constraints and abs(x-np.max(coords[:,0]))<tol: fixed.extend([3*i,3*i+1,3*i+2])
            elif 'y0' in constraints and abs(y-np.min(coords[:,1]))<tol: fixed.extend([3*i,3*i+1,3*i+2])
            elif 'yL' in constraints and abs(y-np.max(coords[:,1]))<tol: fixed.extend([3*i,3*i+1,3*i+2])
            elif 'z0' in constraints and abs(z)<tol: fixed.extend([3*i,3*i+1,3*i+2])
            elif 'zL' in constraints and abs(z-L)<tol: fixed.extend([3*i,3*i+1,3*i+2])
        return np.unique(fixed)

    def _get_loads(self, coords, n_dof):
        F = np.zeros(n_dof)
        val = self.params['load_value']
        ldir = 0 if self.params['load_dir']=='x' else (1 if self.params['load_dir']=='y' else 2)
        locs = self.params['load_locations']
        target_nodes = []
        tol=1e-4; L=self.params['L']
        for i,c in enumerate(coords):
            x,y,z=c
            match=False
            if 'x0' in locs and abs(x-np.min(coords[:,0]))<tol: match=True
            elif 'xL' in locs and abs(x-np.max(coords[:,0]))<tol: match=True
            elif 'y0' in locs and abs(y-np.min(coords[:,1]))<tol: match=True
            elif 'yL' in locs and abs(y-np.max(coords[:,1]))<tol: match=True
            elif 'z0' in locs and abs(z)<tol: match=True
            elif 'zL' in locs and abs(z-L)<tol: match=True
            if match: target_nodes.append(i)
        if target_nodes:
            f_node = val/len(target_nodes)
            for n in target_nodes: F[3*n+ldir]+=f_node
        return F

class BeamCalcolo_dg(QObject):
    def __init__(self, parent, ui, mesh_generator):
        super().__init__(parent)
        self.ui = ui
        self.mesh_generator = mesh_generator
        self.worker = None

    def start_fem_analysis(self):
        try:
            try: L = float(self.ui.beam_lunghezza.text())
            except: L = 3.0
            
            try: nx = int(self.ui.beam_definizione_x.text())
            except: nx = 8
            try: ny = int(self.ui.beam_definizione_y.text())
            except: ny = 8
            try: nz = int(self.ui.beam_definizione_z.text())
            except: nz = 10
            
            try: stirrup_step = float(self.ui.beam_passo.text())
            except: stirrup_step = 0.0

            try: load_val = float(self.ui.beam_carico.text())/1000
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

            constraints = []
            if getattr(self.ui, 'beam_vincolo_x0', None) and self.ui.beam_vincolo_x0.isChecked(): constraints.append('x0')
            if getattr(self.ui, 'beam_vincolo_xL', None) and self.ui.beam_vincolo_xL.isChecked(): constraints.append('xL')
            if getattr(self.ui, 'beam_vincolo_y0', None) and self.ui.beam_vincolo_y0.isChecked(): constraints.append('y0')
            if getattr(self.ui, 'beam_vincolo_yL', None) and self.ui.beam_vincolo_yL.isChecked(): constraints.append('yL')
            if getattr(self.ui, 'beam_vincolo_z0', None) and self.ui.beam_vincolo_z0.isChecked(): constraints.append('z0')
            if getattr(self.ui, 'beam_vincolo_zL', None) and self.ui.beam_vincolo_zL.isChecked(): constraints.append('zL')

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

        self.mesh_generator.generate_mesh() 
        sel_idx = self.mesh_generator.selected_section_index
        if sel_idx is None: sel_idx = 0
        
        try:
            # Recupero esatto dei materiali e della matrice come nel codice originale
            mats, objs = self.mesh_generator.beam_valori.generate_matrices(sel_idx)
            materials_db = {}
            for m_def in mats:
                # MaterialParser ora è pulito e usa solo la definizione
                parser = MaterialParser(m_def)
                materials_db[parser.name] = parser
            
            section = self.mesh_generator._build_section_from_matrices(mats, objs)

        except Exception as e:
             QMessageBox.critical(self.ui, "Errore Dati", f"Errore recupero dati sezione: {e}")
             return

        params = {
            'L': L, 'nx': nx, 'ny': ny, 'nz': nz, 'stirrup_step': stirrup_step,
            'load_value': load_val, 'load_dir': load_dir, 'load_locations': load_locs,
            'constraints': constraints,
            'steps': steps, 'iters': iters, 'tol': tol
        }

        if hasattr(self.ui, 'progressBar_beam'):
            self.ui.progressBar_beam.setValue(0)

        self.worker = FemWorkerDG(section, materials_db, params)
        self.worker.progress_update.connect(lambda s: print(f"[DG-FEM] {s}")) 
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished_computation.connect(self._on_success)
        
        if hasattr(self.ui, 'progressBar_beam'):
            self.worker.progress_percent.connect(self.ui.progressBar_beam.setValue)

        self.target_scale = scale_def
        self.worker.start()

    def _on_error(self, msg):
        QMessageBox.critical(self.ui, "Errore FEM", msg)
        if hasattr(self.ui, 'progressBar_beam'):
            self.ui.progressBar_beam.setValue(0)

    def _on_success(self, history, coords, solid_elems, bar_elems, max_disp, stress_history, max_stress):
        print(f"[DG-FEM] Analisi completata. Max disp: {max_disp:.4f}")
        
        gl_widget = self.mesh_generator._ensure_gl_widget_in_ui()
        if gl_widget:
            gl_widget.set_fem_results(history, coords, solid_elems, bar_elems, max_disp, stress_history, max_stress)
            gl_widget.deformation_scale = self.target_scale
            gl_widget.start_animation()
        else:
            QMessageBox.warning(self.ui, "Attenzione", "Widget 3D non trovato per visualizzare i risultati.")