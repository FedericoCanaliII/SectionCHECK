"""
Beam Designer - Interfaccia grafica con visualizzazione 3D OpenGL

Descrizione:
Questo script fornisce una GUI (PyQt5) che permette di:
 - Disegnare la sezione trasversale della trave con un editor 2D (poligoni liberi + forme base)
 - Estrudere la sezione per ottenere il corpo 3D (lunghezza definibile)
 - Inserire barre longitudinali cliccando la posizione nella sezione 2D
 - Generare staffe (stirrups) rettangolari interne con passo definito
 - Visualizzare il tutto in 3D con QOpenGLWidget (rotazione/orbit, pan, zoom)
 - Esportare mesh e barre in file OBJ

Dipendenze:
 - PyQt5
 - PyOpenGL
 - numpy

Installazione (es. pip):
 pip install PyQt5 PyOpenGL numpy

Nota: il renderer OpenGL usa il "fixed pipeline" per semplicità e compatibilità.
Questo script è pensato come punto di partenza: puoi estenderlo (migliorare shading,
aggiungere ricampionamento mesh, staffe personalizzate, import/export avanzato, ecc.).

Istruzioni rapide:
 - Disegna la sezione cliccando "Edit polygon" e poi cliccando nel pannello sezione.
   Premi "Close polygon" per chiuderla.
 - Puoi inserire rettangolo/circle con "Add rectangle" / "Add circle".
 - Imposta "Beam length" e premi "Extrude" per generare il volume 3D.
 - Clicca "Add rebar" e poi clicca nella sezione per posizionare barre longitudinali.
 - Premi "Add stirrups" per generare staffe rettangolari interne (usa cover e spacing).
 - Ruota la vista 3D con il mouse sinistro, zoom con rotellina, pan con tasto destro.
 - Esporta in OBJ con "Export OBJ".

Autore: ChatGPT (script generato automaticamente)
"""

import sys
import math
from PyQt5 import QtWidgets, QtGui, QtCore, QtOpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


# ----------------------------- Utility geometriche -----------------------------

def polygon_area(pts):
    # shoelace
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def point_in_polygon(pt, poly):
    # ray casting
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
    return inside


def triangulate_polygon(poly):
    # simple ear clipping - poly as Nx2 numpy array
    poly = [tuple(p) for p in poly]
    if len(poly) < 3:
        return []
    verts = list(poly)
    indices = list(range(len(verts)))
    tris = []
    def is_convex(a,b,c):
        ax,ay = verts[a]
        bx,by = verts[b]
        cx,cy = verts[c]
        return ((bx-ax)*(cy-ay) - (by-ay)*(cx-ax)) < 0
    safety = 0
    while len(indices) > 3 and safety < 10000:
        found = False
        n = len(indices)
        for i in range(n):
            a = indices[i-1]
            b = indices[i]
            c = indices[(i+1)%n]
            if is_convex(a,b,c):
                tri = [verts[a], verts[b], verts[c]]
                # check no other vertex inside
                any_inside = False
                for j in indices:
                    if j in (a,b,c):
                        continue
                    if point_in_polygon((verts[j][0], verts[j][1]), tri):
                        any_inside = True
                        break
                if not any_inside:
                    tris.append((a,b,c))
                    indices.pop(i)
                    found = True
                    break
        if not found:
            # fallback: fan triangulation
            for i in range(1, len(indices)-1):
                tris.append((indices[0], indices[i], indices[i+1]))
            indices = [indices[0]]
        safety += 1
    if len(indices) == 3:
        tris.append((indices[0], indices[1], indices[2]))
    # convert to actual coordinates
    out_tris = []
    for tri in tris:
        out_tris.append((verts[tri[0]], verts[tri[1]], verts[tri[2]]))
    return out_tris


# ----------------------------- Cross-section editor widget -----------------------------

class SectionEditor(QtWidgets.QWidget):
    """Widget 2D per disegnare la sezione trasversale."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.polygon = []  # list of (x,y) in widget coords normalized to [-1,1]
        self.mode = 'idle'  # 'idle'|'editing'|'placing_rebar'
        self.rebars = []  # list of dicts {pos:(x,y),diam}
        self.cover = 20.0  # mm default
        self.scale = 100.0  # px per meter scaling for display convenience
        self.setMouseTracking(True)
        self.rubber_pt = None

    def to_local(self, widget_x, widget_y):
        # convert widget coords to local section coords in mm centering at widget center
        w = self.width(); h = self.height()
        cx = w/2.0; cy = h/2.0
        # origin at center, +y up
        x = (widget_x - cx) / self.scale * 1000.0  # in mm
        y = (cy - widget_y) / self.scale * 1000.0
        return (x, y)

    def to_widget(self, x_mm, y_mm):
        w = self.width(); h = self.height()
        cx = w/2.0; cy = h/2.0
        wx = cx + (x_mm/1000.0) * self.scale
        wy = cy - (y_mm/1000.0) * self.scale
        return (int(wx), int(wy))

    def clear(self):
        self.polygon = []
        self.rebars = []
        self.update()

    def add_rectangle(self, width_mm=300.0, height_mm=500.0):
        w = width_mm/2.0; h = height_mm/2.0
        self.polygon = [(-w,-h), (w,-h), (w,h), (-w,h)]
        self.update()

    def add_circle(self, radius_mm=150.0, nseg=48):
        self.polygon = []
        for i in range(nseg):
            a = 2*math.pi*i/nseg
            self.polygon.append((radius_mm*math.cos(a), radius_mm*math.sin(a)))
        self.update()

    def mousePressEvent(self, ev):
        pos = ev.pos()
        x,y = self.to_local(pos.x(), pos.y())
        if self.mode == 'editing':
            self.polygon.append((x,y))
            self.update()
        elif self.mode == 'placing_rebar':
            # add rebar if inside polygon
            if len(self.polygon) >= 3 and point_in_polygon((x,y), np.array(self.polygon)):
                self.rebars.append({'pos':(x,y), 'diam':16.0})
                self.update()
        else:
            # maybe pick and move a vertex? (not implemented)
            pass

    def mouseMoveEvent(self, ev):
        pos = ev.pos()
        x,y = self.to_local(pos.x(), pos.y())
        self.rubber_pt = (x,y)
        self.update()

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        # background
        qp.fillRect(self.rect(), QtGui.QColor(245,245,245))
        # axes
        pen = QtGui.QPen(QtGui.QColor(180,180,180))
        qp.setPen(pen)
        w = self.width(); h = self.height()
        cx = w/2; cy = h/2
        qp.drawLine(int(0), int(cy), int(w), int(cy))
        qp.drawLine(int(cx), int(0), int(cx), int(h))

        # draw polygon
        if len(self.polygon) > 0:
            pts = [QtCore.QPointF(*self.to_widget(x,y)) for (x,y) in self.polygon]
            brush = QtGui.QBrush(QtGui.QColor(200,220,255,80))
            pen = QtGui.QPen(QtGui.QColor(20,80,160))
            pen.setWidth(2)
            qp.setPen(pen)
            qp.setBrush(brush)
            qp.drawPolygon(QtGui.QPolygonF(pts))
            # vertices
            pen = QtGui.QPen(QtGui.QColor(20,80,160))
            qp.setPen(pen)
            for p in pts:
                qp.drawEllipse(p, 4, 4)
        # draw rebars
        for rb in self.rebars:
            wx, wy = self.to_widget(rb['pos'][0], rb['pos'][1])
            pen = QtGui.QPen(QtGui.QColor(180,30,30))
            qp.setPen(pen)
            qp.setBrush(QtGui.QBrush(QtGui.QColor(200,60,60)))
            r = max(4, int(rb['diam']/1000.0 * self.scale))
            qp.drawEllipse(QtCore.QPointF(wx,wy), r, r)
        # rubberband point
        if self.rubber_pt is not None and self.mode == 'editing':
            wx, wy = self.to_widget(self.rubber_pt[0], self.rubber_pt[1])
            pen = QtGui.QPen(QtGui.QColor(120,120,120))
            qp.setPen(pen)
            qp.drawEllipse(QtCore.QPointF(wx,wy), 3, 3)
        qp.end()


# ----------------------------- OpenGL 3D widget -----------------------------

class GLViewer(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        fmt = QtOpenGL.QGLFormat()
        fmt.setDoubleBuffer(True)
        super().__init__(fmt, parent)
        self.parent = parent
        # camera params
        self.rot_x = -20.0
        self.rot_y = -30.0
        self.distance = 2.0  # meters
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_pos = None
        # geometry
        self.surface_mesh = None  # dict with 'verts' Nx3 and 'tris' Mx3
        self.rebars_geo = []  # list of dicts with 'center' (x,y), 'diam', and 3D vertices for cylinder
        self.stirrups_geo = []
        self.beam_length = 1.0

    def initializeGL(self):
        glClearColor(0.95, 0.95, 0.95, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        light_pos = [5.0, 5.0, 10.0, 1.0]
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)

    def resizeGL(self, w, h):
        glViewport(0,0,w,h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(w)/float(max(1,h)), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # camera transform
        eye_x = self.pan_x
        eye_y = self.pan_y
        eye_z = self.distance
        glTranslatef(eye_x, eye_y, -eye_z)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)

        # draw axes
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        # x red
        glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(0.5,0,0)
        # y green
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,0.5,0)
        # z blue
        glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,0.5)
        glEnd()
        glEnable(GL_LIGHTING)

        # draw beam mesh
        if self.surface_mesh is not None:
            verts = self.surface_mesh['verts']
            tris = self.surface_mesh['tris']
            glColor3f(0.8,0.8,0.85)
            for tri in tris:
                glBegin(GL_TRIANGLES)
                for v in tri:
                    glNormal3fv(self.compute_normal(tri))
                    glVertex3fv(v)
                glEnd()
            # draw edges
            glDisable(GL_LIGHTING)
            glColor3f(0.2,0.2,0.2)
            glLineWidth(1.0)
            glBegin(GL_LINES)
            for tri in tris:
                glVertex3fv(tri[0]); glVertex3fv(tri[1])
                glVertex3fv(tri[1]); glVertex3fv(tri[2])
                glVertex3fv(tri[2]); glVertex3fv(tri[0])
            glEnd()
            glEnable(GL_LIGHTING)

        # draw rebars as cylinders (approx)
        for rb in self.rebars_geo:
            glPushMatrix()
            glTranslatef(rb['center'][0], rb['center'][1], rb['center'][2])
            # align along beam axis Z
            glColor3f(0.2,0.2,0.2)
            self.draw_cylinder(rb['diam']/2000.0, self.beam_length)  # convert mm->m radius
            glPopMatrix()

        # draw stirrups (wireframe)
        glDisable(GL_LIGHTING)
        glColor3f(0.1,0.1,0.6)
        for st in self.stirrups_geo:
            glBegin(GL_LINE_LOOP)
            for v in st:
                glVertex3fv(v)
            glEnd()
        glEnable(GL_LIGHTING)

    def compute_normal(self, tri):
        a = np.array(tri[0]); b = np.array(tri[1]); c = np.array(tri[2])
        n = np.cross(b-a, c-a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            return (0.0,0.0,1.0)
        return (n / norm).tolist()

    def draw_cylinder(self, radius, length, slices=18):
        # draw cylinder aligned with +z, centered at z=0 (we translate externally)
        half = length/2.0
        quad = gluNewQuadric()
        glPushMatrix()
        glTranslatef(0,0,-half)
        gluCylinder(quad, radius, radius, length, slices, 1)
        # draw end caps
        glTranslatef(0,0,length)
        gluDisk(quad, 0, radius, slices, 1)
        glPopMatrix()

    def mousePressEvent(self, ev):
        self.last_pos = ev.pos()

    def mouseMoveEvent(self, ev):
        if self.last_pos is None:
            self.last_pos = ev.pos()
            return
        dx = ev.x() - self.last_pos.x()
        dy = ev.y() - self.last_pos.y()
        buttons = ev.buttons()
        if buttons & QtCore.Qt.LeftButton:
            self.rot_x += dy*0.5
            self.rot_y += dx*0.5
        elif buttons & QtCore.Qt.RightButton:
            self.pan_x += dx/200.0
            self.pan_y -= dy/200.0
        self.last_pos = ev.pos()
        self.update()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y() / 120.0
        self.distance -= delta*0.1
        self.distance = max(0.1, self.distance)
        self.update()

    # ----------------- Geometry generation -----------------

    def build_from_section(self, polygon_mm, rebars, beam_length_m):
        # polygon_mm: list of (x_mm, y_mm)
        # rebars: list of {'pos':(x_mm,y_mm),'diam':diam_mm}
        # produce surface mesh (extrusion) in meters
        if len(polygon_mm) < 3:
            self.surface_mesh = None
            self.rebars_geo = []
            self.stirrups_geo = []
            self.update()
            return
        # convert to meters
        poly_m = [(x/1000.0, y/1000.0) for (x,y) in polygon_mm]
        # triangulate cross-section
        tris2d = triangulate_polygon(np.array(polygon_mm))
        verts3 = []
        faces = []
        # create top and bottom vertices
        z0 = -beam_length_m/2.0
        z1 = beam_length_m/2.0
        for (x_mm,y_mm) in poly_m:
            verts3.append((x_mm, y_mm, z0))
        for (x_mm,y_mm) in poly_m:
            verts3.append((x_mm, y_mm, z1))
        n = len(poly_m)
        # faces - caps
        tris3d = []
        for tri in tris2d:
            # map tri vertices to indices
            a = poly_m.index((tri[0][0], tri[0][1])) if (tri[0][0],tri[0][1]) in poly_m else 0
            # because triangulate_polygon returns float tuples it may not match exactly; instead pick nearest
            def index_of(p):
                p = (p[0]/1000.0,p[1]/1000.0) if abs(p[0])>1e-6 else (p[0]/1000.0,p[1]/1000.0)
                # use nearest
                dists = [ (i, (poly_m[i][0]-p[0])**2 + (poly_m[i][1]-p[1])**2) for i in range(len(poly_m)) ]
                return min(dists, key=lambda x:x[1])[0]
            ia = index_of(tri[0])
            ib = index_of(tri[1])
            ic = index_of(tri[2])
            # bottom face (z0) - keep orientation
            tris3d.append((verts3[ia], verts3[ib], verts3[ic]))
            # top face (z1) - reverse orientation
            tris3d.append((verts3[ic+n], verts3[ib+n], verts3[ia+n]))
        # side faces
        for i in range(n):
            a = i; b = (i+1)%n
            tris3d.append((verts3[a], verts3[b], verts3[b+n]))
            tris3d.append((verts3[b+n], verts3[a+n], verts3[a]))
        self.surface_mesh = {'verts': verts3, 'tris': tris3d}
        # build rebars geometry
        self.rebars_geo = []
        for rb in rebars:
            x = rb['pos'][0]/1000.0
            y = rb['pos'][1]/1000.0
            # center at z=0
            self.rebars_geo.append({'center':(x,y,0.0),'diam':rb['diam']})
        # build stirrups: compute offset polygon by cover
        self.stirrups_geo = []
        # attempt simple offset by shrinking bounding box inside cover
        xs = [p[0] for p in poly_m]; ys = [p[1] for p in poly_m]
        minx = min(xs); maxx = max(xs); miny = min(ys); maxy = max(ys)
        # convert cover mm to meters
        cover = self.parent.editor.cover/1000.0
        inner = (minx+cover, miny+cover, maxx-cover, maxy-cover)
        # simple rectangular stirrup following inner box
        # number of stirrups based on spacing
        spacing = self.parent.stir_spacing_spin.value()/1000.0
        if spacing > 0:
            nst = max(1, int(math.floor(self.beam_length / spacing)))
            z_positions = np.linspace(-self.beam_length/2.0, self.beam_length/2.0, nst)
            for zp in z_positions:
                pts = [ (inner[0], inner[1], zp), (inner[2], inner[1], zp), (inner[2], inner[3], zp), (inner[0], inner[3], zp) ]
                self.stirrups_geo.append(pts)
        self.update()

    def export_obj(self, filename):
        # export surface mesh and rebars as simple cylinders approximated by circles
        with open(filename, 'w') as f:
            f.write('# beam designer export\n')
            vtx_count = 0
            # write surface vertices
            for v in self.surface_mesh['verts']:
                f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            vtx_count += len(self.surface_mesh['verts'])
            # faces
            for tri in self.surface_mesh['tris']:
                # find vertex indices
                idxs = []
                for p in tri:
                    # search in verts
                    for i, vv in enumerate(self.surface_mesh['verts']):
                        if abs(vv[0]-p[0])<1e-9 and abs(vv[1]-p[1])<1e-9 and abs(vv[2]-p[2])<1e-9:
                            idxs.append(i+1)
                            break
                if len(idxs)==3:
                    f.write('f %d %d %d\n' % (idxs[0], idxs[1], idxs[2]))
            # rebars as line sets (not full cylinders for simplicity)
            for rb in self.rebars_geo:
                x,y,z = rb['center']
                r = rb['diam']/2000.0
                # sample circle at z=-L/2 and z=+L/2
                slices = 12
                base_idx = vtx_count
                for k in range(slices):
                    ang = 2*math.pi*k/slices
                    vx = x + r*math.cos(ang); vy = y + r*math.sin(ang); vz = -self.beam_length/2.0
                    f.write('v %f %f %f\n' % (vx,vy,vz))
                for k in range(slices):
                    ang = 2*math.pi*k/slices
                    vx = x + r*math.cos(ang); vy = y + r*math.sin(ang); vz = self.beam_length/2.0
                    f.write('v %f %f %f\n' % (vx,vy,vz))
                # side faces
                for k in range(slices):
                    i1 = base_idx + k + 1
                    i2 = base_idx + ((k+1)%slices) + 1
                    i3 = base_idx + slices + ((k+1)%slices) + 1
                    i4 = base_idx + slices + k + 1
                    f.write('f %d %d %d %d\n' % (i1, i2, i3, i4))
                vtx_count += slices*2
        return True


# ----------------------------- Main application window -----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Beam Designer - Travi in calcestruzzo armato (OpenGL)')
        self.resize(1200, 700)

        # central widget: splitter with left editor and right 3D
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        # section editor
        self.editor = SectionEditor()
        left_layout.addWidget(self.editor)

        # editor controls
        controls = QtWidgets.QHBoxLayout()
        self.edit_btn = QtWidgets.QPushButton('Edit polygon')
        self.close_poly_btn = QtWidgets.QPushButton('Close polygon')
        self.rect_btn = QtWidgets.QPushButton('Add rectangle')
        self.circle_btn = QtWidgets.QPushButton('Add circle')
        controls.addWidget(self.edit_btn)
        controls.addWidget(self.close_poly_btn)
        controls.addWidget(self.rect_btn)
        controls.addWidget(self.circle_btn)
        left_layout.addLayout(controls)

        # rebar & stirrup controls
        rebar_layout = QtWidgets.QFormLayout()
        self.add_rebar_btn = QtWidgets.QPushButton('Add rebar (click in section)')
        self.clear_rebars_btn = QtWidgets.QPushButton('Clear rebars')
        self.cover_spin = QtWidgets.QDoubleSpinBox(); self.cover_spin.setSuffix(' mm'); self.cover_spin.setValue(25.0)
        self.cover_spin.setRange(0,200)
        self.editor.cover = self.cover_spin.value()
        rebar_layout.addRow(self.add_rebar_btn, self.clear_rebars_btn)
        rebar_layout.addRow('Concrete cover', self.cover_spin)
        left_layout.addLayout(rebar_layout)

        # beam params
        beam_group = QtWidgets.QGroupBox('Beam parameters')
        bg_layout = QtWidgets.QFormLayout(beam_group)
        self.length_spin = QtWidgets.QDoubleSpinBox(); self.length_spin.setSuffix(' m'); self.length_spin.setRange(0.1, 100.0); self.length_spin.setValue(3.0)
        self.beam_length_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.beam_length_slider.setRange(10, 1000); self.beam_length_slider.setValue(300)
        self.beam_length_slider.valueChanged.connect(lambda v: self.length_spin.setValue(v/100.0))
        self.length_spin.valueChanged.connect(lambda v: self.beam_length_slider.setValue(int(v*100)))
        bg_layout.addRow('Beam length', self.length_spin)
        bg_layout.addRow('Length slider', self.beam_length_slider)
        self.extrude_btn = QtWidgets.QPushButton('Extrude / Update 3D')
        bg_layout.addRow(self.extrude_btn)
        left_layout.addWidget(beam_group)

        # stirrups
        st_group = QtWidgets.QGroupBox('Stirrups / Staffe')
        st_layout = QtWidgets.QFormLayout(st_group)
        self.stir_diam_spin = QtWidgets.QDoubleSpinBox(); self.stir_diam_spin.setSuffix(' mm'); self.stir_diam_spin.setRange(4,50); self.stir_diam_spin.setValue(8.0)
        self.stir_spacing_spin = QtWidgets.QDoubleSpinBox(); self.stir_spacing_spin.setSuffix(' mm'); self.stir_spacing_spin.setRange(10,2000); self.stir_spacing_spin.setValue(200.0)
        self.add_stir_btn = QtWidgets.QPushButton('Add stirrups')
        st_layout.addRow('Stirrup dia', self.stir_diam_spin)
        st_layout.addRow('Spacing', self.stir_spacing_spin)
        st_layout.addRow(self.add_stir_btn)
        left_layout.addWidget(st_group)

        # export
        export_layout = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton('Export OBJ')
        export_layout.addWidget(self.export_btn)
        left_layout.addLayout(export_layout)

        # spacer
        left_layout.addStretch()

        splitter.addWidget(left)

        # Right: OpenGL viewer
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.gl = GLViewer(self)
        right_layout.addWidget(self.gl)
        splitter.addWidget(right)

        self.setCentralWidget(splitter)

        # connections
        self.edit_btn.clicked.connect(self.toggle_edit_mode)
        self.close_poly_btn.clicked.connect(self.close_polygon)
        self.rect_btn.clicked.connect(lambda: self.editor.add_rectangle())
        self.circle_btn.clicked.connect(lambda: self.editor.add_circle())
        self.add_rebar_btn.clicked.connect(self.enable_place_rebar)
        self.clear_rebars_btn.clicked.connect(self.clear_rebars)
        self.extrude_btn.clicked.connect(self.update_3d)
        self.cover_spin.valueChanged.connect(self.update_cover)
        self.add_stir_btn.clicked.connect(self.add_stirrups)
        self.export_btn.clicked.connect(self.export_obj)

    def toggle_edit_mode(self):
        if self.editor.mode == 'editing':
            self.editor.mode = 'idle'
            self.edit_btn.setText('Edit polygon')
        else:
            self.editor.mode = 'editing'
            self.edit_btn.setText('Editing: click to add vertices')

    def close_polygon(self):
        # ensure polygon is closed by leaving list as-is; triangulation will use poly
        self.editor.mode = 'idle'
        self.edit_btn.setText('Edit polygon')
        self.editor.update()

    def enable_place_rebar(self):
        self.editor.mode = 'placing_rebar'
        self.add_rebar_btn.setText('Placing rebar: click in section')

    def clear_rebars(self):
        self.editor.rebars = []
        self.editor.update()

    def update_cover(self, v):
        self.editor.cover = v

    def update_3d(self):
        # gather data
        poly = self.editor.polygon
        rebars = self.editor.rebars
        L = self.length_spin.value()
        self.gl.beam_length = L
        self.gl.build_from_section(poly, rebars, L)

    def add_stirrups(self):
        # generates stirrups (handled in GLViewer.build_from_section automatically
        # but allow forcing update
        self.update_3d()

    def export_obj(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save OBJ', '', 'Wavefront OBJ (*.obj)')
        if fname:
            ok = self.gl.export_obj(fname)
            if ok:
                QtWidgets.QMessageBox.information(self, 'Export', 'OBJ saved to %s' % fname)


# ----------------------------- Run application -----------------------------

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
