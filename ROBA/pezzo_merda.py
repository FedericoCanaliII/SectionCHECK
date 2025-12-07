import pygame
import pymunk
import pymunk.pygame_util
import math
import numpy as np
from enum import Enum

class SimulationState(Enum):
    IDLE = 0
    RUNNING = 1
    FINISHED = 2

class RibaltamentoSimulator:
    def __init__(self):
        # Inizializzazione Pygame
        pygame.init()
        self.width, self.height = 1200, 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Ribaltamento (W = t) - diàtoni + backscatter + arresto a T")
        
        # Inizializzazione Pymunk
        self.space = pymunk.Space()
        self.space.gravity = (0, 981)  # 9.81 m/s² (invertito per coordinate pygame)
        
        # Parametri di default
        self.params = {
            'H': 6.0,           # Altezza facciata (m)
            't': 0.4,           # Spessore (m)
            'bx': 0.5,          # Larghezza blocchi (m)
            'by': 0.25,         # Altezza blocchi (m)
            'tilt': 2.0,        # Angolo iniziale (°)
            'diaOn': True,      # Abilita diàtoni
            'diaStep': 2.0,     # Passo verticale diàtoni (m)
            'diaH': 0.20,       # Altezza diatono (m)
            'slope': 0,         # Pendenza strada (‰)
            'mu': 0.70,         # Attrito blocchi
            'muG': 0.75,        # Attrito suolo
            'e': 0.02,          # Coefficiente di restituzione
            'rho': 1800,        # Densità (kg/m³)
            'breakOnImpact': True,  # Rottura a contatto suolo
            'noBack': True,     # No backscatter
            'ppm': 90,          # Scala (px/m)
            'dur': 15.0,        # Durata simulazione (s)
            'perc': 98          # Percentile lunghezza (%)
        }
        
        # Elementi fisici
        self.panel = None
        self.panel_shape = None
        self.hinge = None
        self.ground = None
        self.ground_shape = None
        self.foot = None
        self.backstop_vert = None
        self.backstop_wedge = None
        self.blocks = []
        self.diatones = []
        
        # Stato simulazione
        self.state = SimulationState.IDLE
        self.broken = False
        self.start_time = 0
        self.simulation_time = 0
        
        # Risultati
        self.Lp = 0
        self.Lmax = 0
        
        # Configurazione UI
        self.font = pygame.font.SysFont('Arial', 14)
        self.small_font = pygame.font.SysFont('Arial', 12)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # Area UI
        self.ui_width = 400
        self.sim_width = self.width - self.ui_width
        
        # Input attivo
        self.active_input = None
        self.input_text = ""
        
        # Inizializzazione controlli UI
        self.init_ui_controls()
        
        # Reset iniziale
        self.reset_simulation()
    
    def init_ui_controls(self):
        self.controls = {}
        y_pos = 50
        control_height = 30
        section_spacing = 40
        
        # Geometria facciata
        self.add_section_title("Geometria facciata (m)", y_pos)
        y_pos += section_spacing
        
        self.controls['H'] = self.create_number_input("Altezza H", 0.1, 20.0, self.params['H'], y_pos)
        y_pos += control_height
        self.controls['t'] = self.create_number_input("Spessore t (= W)", 0.05, 2.0, self.params['t'], y_pos)
        y_pos += control_height
        self.controls['bx'] = self.create_number_input("Blocchi (bx)", 0.05, 2.0, self.params['bx'], y_pos)
        y_pos += control_height
        self.controls['by'] = self.create_number_input("Blocchi (by)", 0.05, 1.0, self.params['by'], y_pos)
        y_pos += control_height
        self.controls['tilt'] = self.create_number_input("Angolo iniziale (°)", 0.5, 45.0, self.params['tilt'], y_pos)
        y_pos += control_height + 10
        
        # Diàtoni
        self.add_section_title("Diàtoni (fasce orizzontali)", y_pos)
        y_pos += section_spacing
        
        self.controls['diaOn'] = self.create_checkbox("Abilita diàtoni", self.params['diaOn'], y_pos)
        y_pos += control_height
        self.controls['diaStep'] = self.create_number_input("Passo verticale (m)", 0.1, 10.0, self.params['diaStep'], y_pos)
        y_pos += control_height
        self.controls['diaH'] = self.create_number_input("Altezza diatono hᵈ (m)", 0.01, 1.0, self.params['diaH'], y_pos)
        y_pos += control_height + 10
        
        # Suolo & rottura
        self.add_section_title("Suolo & rottura", y_pos)
        y_pos += section_spacing
        
        self.controls['slope'] = self.create_number_input("Pendenza strada (‰)", -50, 50, self.params['slope'], y_pos)
        y_pos += control_height
        self.controls['mu'] = self.create_number_input("Attrito blocchi μ", 0.05, 2.0, self.params['mu'], y_pos)
        y_pos += control_height
        self.controls['muG'] = self.create_number_input("Attrito suolo μ_g", 0.05, 2.0, self.params['muG'], y_pos)
        y_pos += control_height
        self.controls['e'] = self.create_number_input("Restituzione (pannello/suolo)", 0.01, 1.0, self.params['e'], y_pos)
        y_pos += control_height
        self.controls['rho'] = self.create_number_input("Densità ρ (kg/m³)", 100, 3000, self.params['rho'], y_pos)
        y_pos += control_height
        self.controls['breakOnImpact'] = self.create_checkbox("Rottura a contatto suolo?", self.params['breakOnImpact'], y_pos)
        y_pos += control_height
        self.controls['noBack'] = self.create_checkbox("No backscatter (blocco dietro piede)", self.params['noBack'], y_pos)
        y_pos += control_height + 10
        
        # Simulazione
        self.add_section_title("Simulazione", y_pos)
        y_pos += section_spacing
        
        self.controls['ppm'] = self.create_number_input("Scala (px/m)", 10, 200, self.params['ppm'], y_pos)
        y_pos += control_height
        self.controls['dur'] = self.create_number_input("Durata T (s)", 0.5, 60.0, self.params['dur'], y_pos)
        y_pos += control_height
        self.controls['perc'] = self.create_number_input("Percentile lunghezza (%)", 50, 100, self.params['perc'], y_pos)
        y_pos += control_height
        
        # Pulsanti
        self.controls['run_button'] = self.create_button("Simula", self.ui_width//2 - 100, y_pos + 10, 80, 30)
        self.controls['reset_button'] = self.create_button("Reset", self.ui_width//2 + 20, y_pos + 10, 80, 30)
    
    def add_section_title(self, title, y):
        title_surf = self.font.render(title, True, (0, 0, 0))
        self.screen.blit(title_surf, (20, y))
    
    def create_number_input(self, label, min_val, max_val, default_val, y):
        return {
            'type': 'number',
            'label': label,
            'value': default_val,
            'min': min_val,
            'max': max_val,
            'rect': pygame.Rect(200, y, 100, 25),
            'active': False
        }
    
    def create_checkbox(self, label, default_val, y):
        return {
            'type': 'checkbox',
            'label': label,
            'value': default_val,
            'rect': pygame.Rect(200, y, 20, 20)
        }
    
    def create_button(self, label, x, y, w, h):
        return {
            'type': 'button',
            'label': label,
            'rect': pygame.Rect(x, y, w, h)
        }
    
    def update_params_from_ui(self):
        for key, control in self.controls.items():
            if key not in ['run_button', 'reset_button']:
                self.params[key] = control['value']
    
    def reset_simulation(self):
        # Reset spazio fisico
        self.space = pymunk.Space()
        self.space.gravity = (0, 981)  # Invertito per coordinate pygame
        
        # Reset elementi
        self.panel = None
        self.panel_shape = None
        self.hinge = None
        self.ground = None
        self.ground_shape = None
        self.foot = None
        self.backstop_vert = None
        self.backstop_wedge = None
        self.blocks = []
        self.diatones = []
        
        # Reset stato
        self.state = SimulationState.IDLE
        self.broken = False
        self.start_time = 0
        self.simulation_time = 0
        self.Lp = 0
        self.Lmax = 0
        
        # Ricostruisci scena
        self.build_scene()
    
    def build_scene(self):
        # Parametri
        H = self.params['H']  # Altezza facciata
        t = self.params['t']  # Spessore
        bx = self.params['bx']  # Larghezza blocchi
        by = self.params['by']  # Altezza blocchi
        slope = self.params['slope'] / 1000  # Pendenza in radianti
        mu = self.params['mu']  # Attrito blocchi
        muG = self.params['muG']  # Attrito suolo
        e = self.params['e']  # Coefficiente di restituzione
        rho = self.params['rho']  # Densità
        ppm = self.params['ppm']  # Pixel per metro
        
        # Coordinate base
        base_x = 100 + self.ui_width
        base_y = self.height - 100
        
        # Creazione suolo
        ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground_shape = pymunk.Segment(ground_body, (0, base_y), (self.width, base_y), 5)
        ground_shape.friction = muG
        ground_shape.elasticity = e
        ground_body.angle = -math.atan(slope)
        self.space.add(ground_body, ground_shape)
        self.ground_body = ground_body
        self.ground_shape = ground_shape
        
        # Creazione piede (blu)
        foot_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        foot_shape = pymunk.Segment(foot_body, (base_x - 2, base_y - 12), (base_x - 2, base_y + 12), 4)
        foot_shape.color = (42, 127, 255, 255)  # Blu
        self.space.add(foot_body, foot_shape)
        self.foot = foot_shape
        
        # Creazione pannello
        mass = H * t * 1.0 * rho  # Massa per profondità unitaria
        moment = pymunk.moment_for_box(mass, (t * ppm, H * ppm))
        
        panel_body = pymunk.Body(mass, moment)
        panel_body.position = (base_x + t * ppm / 2, base_y - H * ppm / 2)
        panel_shape = pymunk.Poly.create_box(panel_body, (t * ppm, H * ppm))
        panel_shape.friction = mu
        panel_shape.elasticity = e
        panel_shape.color = (140, 106, 90, 255)  # Marrone
        
        self.space.add(panel_body, panel_shape)
        self.panel = panel_body
        self.panel_shape = panel_shape
        
        # Creazione cerniera
        static_body = self.space.static_body
        self.hinge = pymunk.PivotJoint(
            static_body,
            panel_body,
            (base_x, base_y),
            (-t * ppm / 2, H * ppm / 2)
        )
        self.space.add(self.hinge)
        
        # Applica inclinazione iniziale
        tilt_rad = math.radians(self.params['tilt'])
        panel_body.angle = -tilt_rad
        
        # Aggiungi collision handler per rilevare l'impatto con il suolo
        # Utilizziamo un approccio alternativo per versioni più vecchie di Pymunk
        self.collision_pairs = set()
    
    def check_collisions(self):
        """Controlla manualmente le collisioni tra pannello e suolo"""
        if self.broken or not self.params['breakOnImpact']:
            return
            
        # Verifica collisione approssimativa tra pannello e suolo
        if self.panel and self.ground_shape:
            # Ottieni i vertici del pannello
            panel_vertices = self.panel_shape.get_vertices()
            panel_world_vertices = [self.panel.local_to_world(v) for v in panel_vertices]
            
            # Controlla se qualche vertice è sotto il suolo
            ground_y = self.height - 100
            for v in panel_world_vertices:
                if v.y >= ground_y - 5:  # Tolleranza di 5 pixel
                    self.break_panel()
                    return
    
    def break_panel(self):
        if self.broken:
            return
            
        self.broken = True
        print("Pannello rotto! Creazione blocchi...")
        
        # Parametri
        H = self.params['H']
        t = self.params['t']
        bx = self.params['bx']
        by = self.params['by']
        mu = self.params['mu']
        e = self.params['e']
        rho = self.params['rho']
        ppm = self.params['ppm']
        diaOn = self.params['diaOn']
        diaStep = self.params['diaStep']
        diaH = self.params['diaH']
        
        # Coordinate base
        base_x = 100 + self.ui_width
        base_y = self.height - 100
        
        # Salva stato del pannello
        panel_vel = self.panel.velocity
        panel_ang_vel = self.panel.angular_velocity
        panel_angle = self.panel.angle
        
        # Rimuovi pannello e cerniera
        self.space.remove(self.panel, self.panel_shape, self.hinge)
        
        # Crea backstop se necessario
        if self.params['noBack']:
            # Parete verticale
            backstop_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            backstop_shape = pymunk.Segment(backstop_body, 
                                           (base_x - 2, 0), 
                                           (base_x - 2, self.height), 
                                           5)
            backstop_shape.elasticity = 0
            backstop_shape.friction = 1
            backstop_shape.color = (0, 0, 0, 0)  # Trasparente
            self.space.add(backstop_body, backstop_shape)
            self.backstop_vert = backstop_shape
            
            # Zeppa
            slope = self.params['slope'] / 1000
            wedge_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            wedge_shape = pymunk.Segment(wedge_body,
                                        (base_x - 36, base_y - 6),
                                        (base_x + 54, base_y - 6),
                                        12)
            wedge_body.angle = -math.atan(slope)
            wedge_shape.elasticity = 0
            wedge_shape.friction = 1
            wedge_shape.color = (0, 0, 0, 0)  # Trasparente
            self.space.add(wedge_body, wedge_shape)
            self.backstop_wedge = wedge_shape
        
        # Crea blocchi e diàtoni
        y = 0
        next_dia_y = diaStep if diaOn else float('inf')
        
        while y < H:
            if diaOn and abs(y - next_dia_y) < 1e-6:
                # Crea diatono
                h = min(diaH, H - y)
                mass = t * h * 1.0 * rho
                moment = pymunk.moment_for_box(mass, (t * ppm, h * ppm))
                
                body = pymunk.Body(mass, moment)
                body.position = (base_x + t * ppm / 2, base_y - (y + h/2) * ppm)
                shape = pymunk.Poly.create_box(body, (t * ppm, h * ppm))
                shape.friction = mu
                shape.elasticity = 0.005
                shape.color = (122, 90, 74, 255)  # Marrone scuro
                
                # Applica stato iniziale
                body.angle = panel_angle
                body.velocity = panel_vel
                body.angular_velocity = panel_ang_vel
                
                self.space.add(body, shape)
                self.blocks.append(body)
                
                y += h
                next_dia_y += diaStep
            else:
                # Crea blocchi normali
                h = min(by, H - y)
                if diaOn:
                    h = min(h, next_dia_y - y)
                
                n_full = int(t / bx)
                last_w = t - n_full * bx
                
                for i in range(n_full + (1 if last_w > 1e-6 else 0)):
                    w = bx if i < n_full else last_w
                    mass = w * h * 1.0 * rho
                    moment = pymunk.moment_for_box(mass, (w * ppm, h * ppm))
                    
                    body = pymunk.Body(mass, moment)
                    body.position = (base_x + (i * bx + w/2) * ppm, base_y - (y + h/2) * ppm)
                    shape = pymunk.Poly.create_box(body, (w * ppm, h * ppm))
                    shape.friction = mu
                    shape.elasticity = 0.01
                    shape.color = (155, 111, 94, 255)  # Marrone medio
                    
                    # Applica stato iniziale
                    body.angle = panel_angle
                    body.velocity = panel_vel
                    body.angular_velocity = panel_ang_vel
                    
                    self.space.add(body, shape)
                    self.blocks.append(body)
                
                y += h
        
        print(f"Creati {len(self.blocks)} blocchi")
    
    def apply_time_damping(self):
        # Applica smorzamento temporizzato
        elapsed = self.simulation_time
        duration = self.params['dur']
        
        if elapsed >= duration:
            return
        
        # Calcola fattore di smorzamento (0 -> 1)
        r = elapsed / duration
        gain = r * r * r  # Profilo morbido (cubo)
        
        # Smorzamento base e massimo
        base_air = 0.002
        max_air = 0.15
        
        # Applica a tutti i corpi
        bodies = self.blocks
        if not self.broken and self.panel:
            bodies = [self.panel]
            
        for body in bodies:
            # Applica smorzamento lineare (simulando friction_air)
            damping_factor = base_air + gain * (max_air - base_air)
            body.velocity = (body.velocity.x * (1 - damping_factor), 
                            body.velocity.y * (1 - damping_factor))
            body.angular_velocity *= (1 - damping_factor)
    
    def measure_results(self):
        base_x = 100 + self.ui_width
        bodies = self.blocks
        if not self.broken and self.panel:
            bodies = [self.panel]
        
        # Calcola posizioni X massime
        x_positions = []
        for body in bodies:
            # Per ogni corpo, trova il punto più a destra
            max_x = -float('inf')
            for shape in body.shapes:
                if isinstance(shape, pymunk.Poly):
                    # Trova il punto più a destra del poligono
                    verts = shape.get_vertices()
                    for v in verts:
                        world_v = body.local_to_world(v)
                        max_x = max(max_x, world_v.x)
            if max_x > -float('inf'):
                x_positions.append(max_x)
        
        if not x_positions:
            self.Lp = 0
            self.Lmax = 0
            return
            
        x_positions.sort()
        x_max = max(x_positions)
        
        # Calcola percentile
        p = self.params['perc']
        idx = int((p / 100) * (len(x_positions) - 1))
        x_p = x_positions[max(0, idx)]
        
        # Converti in metri
        ppm = self.params['ppm']
        self.Lmax = max(0, (x_max - base_x) / ppm)
        self.Lp = max(0, (x_p - base_x) / ppm)
    
    def draw_ui(self):
        # Sfondo UI
        ui_rect = pygame.Rect(0, 0, self.ui_width, self.height)
        pygame.draw.rect(self.screen, (245, 247, 251), ui_rect)
        
        # Titolo
        title_font = pygame.font.SysFont('Arial', 18, bold=True)
        title = title_font.render("Ribaltamento (innesco semplificato)", True, (0, 0, 0))
        self.screen.blit(title, (20, 10))
        
        subtitle = self.font.render("W = t (profondità 1 m) – diàtoni, backscatter rinforzato, arresto entro T", True, (0, 0, 0))
        self.screen.blit(subtitle, (20, 35))
        
        # Disegna controlli
        for key, control in self.controls.items():
            if control['type'] == 'number':
                self.draw_number_input(control, key)
            elif control['type'] == 'checkbox':
                self.draw_checkbox(control, key)
            elif control['type'] == 'button':
                self.draw_button(control)
        
        # Risultati
        results_y = self.height - 80
        results_text = f"Lₚ = {self.Lp:.2f} m  |  L_max = {self.Lmax:.2f} m"
        results_surf = self.font.render(results_text, True, (0, 0, 0))
        self.screen.blit(results_surf, (20, results_y))
        
        # Informazioni
        info_text = "Linea blu = piede. Righello 1 m. Arancione = Lₚ (percentile), rosso tratteggiato = L_max. Arresto a T attivo."
        info_surf = self.small_font.render(info_text, True, (100, 100, 100))
        self.screen.blit(info_surf, (20, results_y + 25))
    
    def draw_number_input(self, control, key):
        # Etichetta
        label_surf = self.font.render(control['label'], True, (0, 0, 0))
        self.screen.blit(label_surf, (control['rect'].x - 180, control['rect'].y + 5))
        
        # Casella di input
        border_color = (0, 100, 200) if self.active_input == key else (0, 0, 0)
        pygame.draw.rect(self.screen, (255, 255, 255), control['rect'])
        pygame.draw.rect(self.screen, border_color, control['rect'], 2)
        
        # Valore
        if self.active_input == key:
            value_text = self.input_text
        else:
            value_text = f"{control['value']:.2f}"
            
        value_surf = self.font.render(value_text, True, (0, 0, 0))
        self.screen.blit(value_surf, (control['rect'].x + 5, control['rect'].y + 5))
    
    def draw_checkbox(self, control, key):
        # Etichetta
        label_surf = self.font.render(control['label'], True, (0, 0, 0))
        self.screen.blit(label_surf, (control['rect'].x - 180, control['rect'].y))
        
        # Casella
        pygame.draw.rect(self.screen, (255, 255, 255), control['rect'])
        pygame.draw.rect(self.screen, (0, 0, 0), control['rect'], 1)
        
        # Segno di spunta
        if control['value']:
            pygame.draw.line(self.screen, (0, 0, 0), 
                            (control['rect'].x + 2, control['rect'].y + 10),
                            (control['rect'].x + 8, control['rect'].y + 16), 2)
            pygame.draw.line(self.screen, (0, 0, 0), 
                            (control['rect'].x + 8, control['rect'].y + 16),
                            (control['rect'].x + 16, control['rect'].y + 4), 2)
    
    def draw_button(self, control):
        # Pulsante primario (Simula) - nero
        if control['label'] == 'Simula':
            pygame.draw.rect(self.screen, (17, 17, 17), control['rect'], border_radius=10)
        # Pulsante secondario (Reset) - grigio
        else:
            pygame.draw.rect(self.screen, (102, 102, 102), control['rect'], border_radius=10)
            
        label_surf = self.font.render(control['label'], True, (255, 255, 255))
        label_rect = label_surf.get_rect(center=control['rect'].center)
        self.screen.blit(label_surf, label_rect)
    
    def draw_simulation(self):
        # Sfondo simulazione
        sim_rect = pygame.Rect(self.ui_width, 0, self.sim_width, self.height)
        pygame.draw.rect(self.screen, (245, 247, 251), sim_rect)
        
        # Disegna elementi fisici
        self.space.debug_draw(self.draw_options)
        
        # Disegna righello
        self.draw_ruler()
        
        # Disegna indicatori Lp e Lmax
        self.draw_indicators()
    
    def draw_ruler(self):
        base_x = 100 + self.ui_width
        base_y = self.height - 100
        ppm = self.params['ppm']
        
        # Disegna righello ogni metro
        max_x = self.width - 10
        x = base_x
        m = 0
        
        while x <= max_x:
            pygame.draw.line(self.screen, (154, 163, 173), 
                            (x, base_y), (x, base_y - 12), 1)
            
            meter_text = self.small_font.render(f"{m} m", True, (154, 163, 173))
            self.screen.blit(meter_text, (x + 2, base_y - 30))
            
            x += ppm
            m += 1
    
    def draw_indicators(self):
        base_x = 100 + self.ui_width
        base_y = self.height - 100
        ppm = self.params['ppm']
        
        y = base_y - 28
        
        # Linea Lp (arancione)
        if self.Lp > 0:
            pygame.draw.line(self.screen, (255, 180, 0), 
                            (base_x, y), 
                            (base_x + self.Lp * ppm, y), 3)
            
            lp_text = self.small_font.render(f"Lp ({self.params['perc']}%)", True, (17, 17, 17))
            self.screen.blit(lp_text, (base_x + self.Lp * ppm + 6, y - 4))
        
        # Linea Lmax (rosso tratteggiato)
        if self.Lmax > 0:
            # Simula linea tratteggiata
            dash_length = 8
            gap_length = 6
            x = base_x
            while x < base_x + self.Lmax * ppm:
                end_x = min(x + dash_length, base_x + self.Lmax * ppm)
                pygame.draw.line(self.screen, (225, 29, 72), 
                                (x, y + 8), 
                                (end_x, y + 8), 3)
                x += dash_length + gap_length
            
            lmax_text = self.small_font.render("Lmax", True, (17, 17, 17))
            self.screen.blit(lmax_text, (base_x + self.Lmax * ppm + 6, y + 4))
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Controlla click su UI
                if mouse_pos[0] < self.ui_width:
                    self.handle_ui_click(mouse_pos)
                else:
                    # Click su area simulazione - disattiva input
                    self.active_input = None
            
            elif event.type == pygame.KEYDOWN:
                # Gestione input numerico
                if self.active_input is not None:
                    if event.key == pygame.K_RETURN:
                        # Conferma input
                        try:
                            value = float(self.input_text)
                            control = self.controls[self.active_input]
                            # Applica limiti
                            value = max(control['min'], min(control['max'], value))
                            control['value'] = value
                            self.params[self.active_input] = value
                        except ValueError:
                            pass
                        self.active_input = None
                        self.input_text = ""
                    elif event.key == pygame.K_ESCAPE:
                        # Annulla input
                        self.active_input = None
                        self.input_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        # Cancella ultimo carattere
                        self.input_text = self.input_text[:-1]
                    else:
                        # Aggiungi carattere
                        char = event.unicode
                        if char in '0123456789.-':
                            # Controlla che ci sia solo un punto
                            if char == '.' and '.' in self.input_text:
                                pass
                            elif char == '-' and len(self.input_text) > 0:
                                pass
                            else:
                                self.input_text += char
        
        return True
    
    def handle_ui_click(self, mouse_pos):
        # Controlla click su controlli
        for key, control in self.controls.items():
            if control['rect'].collidepoint(mouse_pos):
                if control['type'] == 'button':
                    if key == 'run_button':
                        self.start_simulation()
                    elif key == 'reset_button':
                        self.reset_simulation()
                elif control['type'] == 'checkbox':
                    control['value'] = not control['value']
                    self.params[key] = control['value']
                elif control['type'] == 'number':
                    # Attiva input per questo controllo
                    self.active_input = key
                    self.input_text = f"{control['value']}"
                break
        
        # Se non è stato cliccato su un controllo, disattiva l'input
        else:
            self.active_input = None
    
    def start_simulation(self):
        self.state = SimulationState.RUNNING
        self.start_time = pygame.time.get_ticks()
        self.simulation_time = 0
        print("Simulazione avviata")
    
    def update_simulation(self):
        if self.state != SimulationState.RUNNING:
            return
        
        # Aggiorna tempo
        current_time = pygame.time.get_ticks()
        self.simulation_time = (current_time - self.start_time) / 1000.0
        
        # Controlla collisioni
        self.check_collisions()
        
        # Applica smorzamento temporizzato
        self.apply_time_damping()
        
        # Controlla fine simulazione
        if self.simulation_time >= self.params['dur']:
            self.state = SimulationState.FINISHED
            # Ferma tutti i corpi
            bodies = self.blocks
            if not self.broken and self.panel:
                bodies = [self.panel]
                
            for body in bodies:
                body.velocity = (0, 0)
                body.angular_velocity = 0
            print("Simulazione terminata")
        
        # Aggiorna fisica
        dt = 1.0 /60  # 60 FPS
        self.space.step(dt)
        
        # Calcola risultati
        self.measure_results()
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            running = self.handle_events()
            
            # Aggiorna simulazione
            self.update_simulation()
            
            # Disegna
            self.screen.fill((255, 255, 255))
            self.draw_ui()
            self.draw_simulation()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    simulator = RibaltamentoSimulator()
    simulator.run()