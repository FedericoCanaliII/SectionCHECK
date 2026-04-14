"""
struttura_info.py – Finestra informativa sulla sintassi del testo strutturale.

Mostra all'utente come scrivere nodi, aste, shell, vincoli e carichi
con esempi pratici e riferimenti a materiali/sezioni del progetto.
Include una sezione per generare automaticamente il modello tramite AI.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QTextBrowser, QWidget,
    QPushButton, QHBoxLayout, QApplication
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon


_STYLE_DIALOG = """
QDialog {
    background-color: rgb(35, 35, 35);
}
QTabWidget::pane {
    border: 1px solid rgb(80, 80, 80);
    background-color: rgb(40, 40, 40);
    border-radius: 4px;
}
QTabBar::tab {
    background-color: rgb(50, 50, 50);
    color: rgb(200, 200, 200);
    border: 1px solid rgb(80, 80, 80);
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font: 10pt "Inter";
    min-width: 120px; /* Ridotto per ospitare più tab */
}
QTabBar::tab:selected {
    background-color: rgb(40, 40, 40);
    color: rgb(255, 255, 255);
    border-bottom-color: rgb(40, 40, 40);
}
QTabBar::tab:hover {
    background-color: rgb(55, 55, 55);
}
QTextBrowser {
    background-color: rgb(40, 40, 40);
    color: rgb(212, 212, 212);
    border: none;
    font: 10pt "Consolas";
    padding: 12px;
}
QPushButton {
    background-color: rgb(50, 50, 50);
    color: rgb(200, 200, 200);
    border: 1px solid rgb(100, 100, 100);
    border-radius: 4px;
    padding: 6px 20px;
    font: 10pt "Inter";
}
QPushButton:hover {
    background-color: rgb(60, 60, 60);
    border: 1px solid rgb(140, 140, 140);
}
QPushButton#copyButton {
    background-color: rgb(30, 100, 160);
    color: white;
    font-weight: bold;
}
QPushButton#copyButton:hover {
    background-color: rgb(40, 120, 190);
}

/* =========================================
   SCROLLBAR VERTICALE E ORIZZONTALE
   ========================================= */

QScrollBar:vertical {
    background-color: rgb(45, 45, 45);
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: rgb(120, 120, 120);
    min-height: 30px;
    border-radius: 4px;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::up-arrow:vertical,
QScrollBar::down-arrow:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: none;
    border: none;
    height: 0px;
}

QScrollBar:horizontal {
    background-color: rgb(45, 45, 45); 
    height: 8px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background-color: rgb(120, 120, 120);
    min-width: 30px;
    border-radius: 4px;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::left-arrow:horizontal,
QScrollBar::right-arrow:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: none;
    border: none;
    width: 0px;
}
"""

# ================================================================
#  CONTENUTI TABS
# ================================================================

_HTML_GENERALE = """
<h2 style="color:#569cd6;">Struttura del testo</h2>
<p>Il testo di definizione è organizzato in <b>sezioni</b>, ciascuna introdotta
da un commento intestazione. L'ordine consigliato è:</p>
<pre style="color:#ce9178;">
# ─── MATERIALI ───
# ─── SEZIONI ───
# ─── NODI ───
# ─── ASTE ───
# ─── SHELL ───
# ─── VINCOLI ───
# ─── CARICHI ───
</pre>
<p>Le intestazioni sono puramente visive e non obbligatorie. Il parser riconosce
i comandi indipendentemente dalla posizione.</p>

<h3 style="color:#569cd6;">Commenti</h3>
<p>Tutto ciò che segue <code style="color:#6a9955;">#</code> fino a fine riga
è un commento e viene ignorato dal parser.</p>
<pre style="color:#6a9955;">
# Questo è un commento intero
node 1  0.0  0.0  0.0   # commento inline
</pre>

<h3 style="color:#569cd6;">Unità di misura</h3>
<p>Il sistema usa unità coerenti. Si consiglia:</p>
<ul>
<li>Lunghezze: <b>metri (m)</b></li>
<li>Forze: <b>kN</b></li>
<li>Carichi distribuiti: <b>kN/m</b></li>
</ul>

<h3 style="color:#569cd6;">Integrazione con Materiali e Sezioni</h3>
<p>I materiali e le sezioni si definiscono <b>all'inizio</b> del testo strutturale,
prima dei nodi. Ogni elemento ha un <b>id intero univoco</b> e un <b>nome</b>
(racchiuso tra virgolette singole o doppie se contiene spazi).</p>

<p><b>1. Riferimento</b> – usa un materiale o sezione già definito nel programma
(il nome si illumina in <span style="color:#50c878;"><b>verde</b></span> se riconosciuto):</p>
<pre>
<span style="color:#569cd6;">material</span>  <span style="color:#b5cea8;">1</span>  <span style="color:#50c878;">'C25/30'</span>
<span style="color:#569cd6;">section</span>   <span style="color:#b5cea8;">1</span>  <span style="color:#50c878;">'R 400x600'</span>
</pre>

<p><b>2. Definizione inline</b> – definisci le proprietà direttamente nel testo:</p>
<pre>
<span style="color:#569cd6;">material</span>  <span style="color:#b5cea8;">2</span>  <span style="color:#ce9178;">'ClsCustom'</span>  <span style="color:#b5cea8;">2500</span>  <span style="color:#b5cea8;">31476</span>  <span style="color:#b5cea8;">13115</span>  <span style="color:#b5cea8;">0.2</span>
<span style="color:#569cd6;">section</span>   <span style="color:#b5cea8;">2</span>  <span style="color:#ce9178;">'Pilastro 1'</span>  <span style="color:#b5cea8;">0.09</span>  <span style="color:#b5cea8;">6.75e-4</span>  <span style="color:#b5cea8;">6.75e-4</span>  <span style="color:#9cdcfe;">material:</span> <span style="color:#50c878;">2</span>
</pre>

<p>Aste e shell fanno riferimento a sezioni/materiali definiti con la sintassi a
due punti <code style="color:#9cdcfe;">section:</code> /
<code style="color:#9cdcfe;">material:</code>. Il riferimento può essere espresso
indifferentemente come <b>id intero</b> o come <b>nome</b>: in entrambi i casi,
se valido, si illumina in <span style="color:#50c878;"><b>verde</b></span>.</p>
"""

_HTML_NODI = """
<h2 style="color:#569cd6;">Nodi</h2>
<p>Definiscono i punti nello spazio 3D della struttura.</p>

<h3>Sintassi</h3>
<pre>
<span style="color:#569cd6;">node</span>  <span style="color:#9cdcfe;">&lt;id&gt;</span>  <span style="color:#b5cea8;">&lt;x&gt;</span>  <span style="color:#b5cea8;">&lt;y&gt;</span>  <span style="color:#b5cea8;">&lt;z&gt;</span>
</pre>

<table style="color:#ddd; border-collapse:collapse;" cellpadding="6">
<tr><td style="color:#9cdcfe;"><b>id</b></td><td>Intero univoco identificativo del nodo</td></tr>
<tr><td style="color:#9cdcfe;"><b>x, y, z</b></td><td>Coordinate in metri (numeri decimali)</td></tr>
</table>

<h3>Esempio</h3>
<pre>
<span style="color:#6a9955;"># Nodi di un telaio 2 campate</span>
<span style="color:#569cd6;">node</span>  <span style="color:#b5cea8;">1</span>    <span style="color:#b5cea8;">0.0</span>    <span style="color:#b5cea8;">0.0</span>    <span style="color:#b5cea8;">0.0</span>
<span style="color:#569cd6;">node</span>  <span style="color:#b5cea8;">2</span>    <span style="color:#b5cea8;">5.0</span>    <span style="color:#b5cea8;">0.0</span>    <span style="color:#b5cea8;">0.0</span>
<span style="color:#569cd6;">node</span>  <span style="color:#b5cea8;">3</span>   <span style="color:#b5cea8;">10.0</span>    <span style="color:#b5cea8;">0.0</span>    <span style="color:#b5cea8;">0.0</span>
<span style="color:#569cd6;">node</span>  <span style="color:#b5cea8;">4</span>    <span style="color:#b5cea8;">0.0</span>    <span style="color:#b5cea8;">0.0</span>    <span style="color:#b5cea8;">3.5</span>
<span style="color:#569cd6;">node</span>  <span style="color:#b5cea8;">5</span>    <span style="color:#b5cea8;">5.0</span>    <span style="color:#b5cea8;">0.0</span>    <span style="color:#b5cea8;">3.5</span>
<span style="color:#569cd6;">node</span>  <span style="color:#b5cea8;">6</span>   <span style="color:#b5cea8;">10.0</span>    <span style="color:#b5cea8;">0.0</span>    <span style="color:#b5cea8;">3.5</span>
</pre>

<p style="color:#ce9178;">Nota: ogni nodo deve avere un ID univoco. IDs duplicati generano errore.</p>
"""

_HTML_ASTE = """
<h2 style="color:#569cd6;">Aste (Beam)</h2>
<p>Elementi monodimensionali (travi, pilastri, controventi) definiti da due
nodi estremi e da una sezione.</p>

<h3>Sintassi</h3>
<pre>
<span style="color:#569cd6;">beam</span>  <span style="color:#9cdcfe;">&lt;id&gt;</span>  <span style="color:#9cdcfe;">&lt;nodo_i&gt;</span>  <span style="color:#9cdcfe;">&lt;nodo_j&gt;</span>   <span style="color:#9cdcfe;">section:</span> &lt;id_o_nome&gt;
</pre>

<table style="color:#ddd; border-collapse:collapse;" cellpadding="6">
<tr><td style="color:#9cdcfe;"><b>id</b></td><td>Intero univoco dell'asta</td></tr>
<tr><td style="color:#9cdcfe;"><b>nodo_i, nodo_j</b></td><td>ID dei nodi estremi (devono esistere)</td></tr>
<tr><td style="color:#9cdcfe;"><b>section:</b></td><td>Riferimento alla sezione: id intero <b>oppure</b> nome (tra virgolette se contiene spazi). Se valido, diventa <span style="color:#50c878;">verde</span>.</td></tr>
</table>

<p style="color:#ce9178;">Nota: il materiale è già contenuto nella sezione,
quindi non va specificato sull'asta.</p>

<h3>Esempi</h3>
<pre>
<span style="color:#6a9955;"># Riferimento per ID</span>
<span style="color:#569cd6;">beam</span>  <span style="color:#b5cea8;">1</span>   <span style="color:#b5cea8;">1</span>  <span style="color:#b5cea8;">4</span>   <span style="color:#9cdcfe;">section:</span> <span style="color:#50c878;">1</span>

<span style="color:#6a9955;"># Riferimento per nome</span>
<span style="color:#569cd6;">beam</span>  <span style="color:#b5cea8;">2</span>   <span style="color:#b5cea8;">4</span>  <span style="color:#b5cea8;">5</span>   <span style="color:#9cdcfe;">section:</span> <span style="color:#50c878;">'Trave 30x50'</span>

<span style="color:#6a9955;"># Profilo standard del database</span>
<span style="color:#569cd6;">beam</span>  <span style="color:#b5cea8;">3</span>   <span style="color:#b5cea8;">1</span>  <span style="color:#b5cea8;">5</span>   <span style="color:#9cdcfe;">section:</span> <span style="color:#50c878;">'IPE300'</span>
</pre>
"""

_HTML_SHELL = """
<h2 style="color:#569cd6;">Shell (Solai / Pareti)</h2>
<p>Elementi bidimensionali a <b>3 o 4 nodi</b> (triangolari o quadrilateri) per solai, pareti, piastre.</p>

<h3>Sintassi</h3>
<pre>
<span style="color:#569cd6;">shell</span>  <span style="color:#9cdcfe;">&lt;id&gt;</span>  <span style="color:#9cdcfe;">&lt;n1&gt;</span> <span style="color:#9cdcfe;">&lt;n2&gt;</span> <span style="color:#9cdcfe;">&lt;n3&gt;</span> [<span style="color:#9cdcfe;">&lt;n4&gt;</span>]   <span style="color:#9cdcfe;">thickness:</span> <span style="color:#b5cea8;">&lt;t&gt;</span>   <span style="color:#9cdcfe;">material:</span> &lt;id_o_nome&gt;
</pre>

<table style="color:#ddd; border-collapse:collapse;" cellpadding="6">
<tr><td style="color:#9cdcfe;"><b>id</b></td><td>Intero univoco della shell</td></tr>
<tr><td style="color:#9cdcfe;"><b>n1..n3</b></td><td>ID dei 3 nodi (shell triangolare, ordine antiorario)</td></tr>
<tr><td style="color:#9cdcfe;"><b>n4</b></td><td>(opz.) Quarto nodo per shell quadrilatera</td></tr>
<tr><td style="color:#9cdcfe;"><b>thickness:</b></td><td>Spessore in m (default: 0.20)</td></tr>
<tr><td style="color:#9cdcfe;"><b>material:</b></td><td>Riferimento al materiale: id intero <b>oppure</b> nome (tra virgolette se contiene spazi). Se valido, diventa <span style="color:#50c878;">verde</span>.</td></tr>
</table>

<h3>Esempi</h3>
<pre>
<span style="color:#6a9955;"># Shell quadrilatera (4 nodi), materiale per nome</span>
<span style="color:#569cd6;">shell</span>  <span style="color:#b5cea8;">1</span>   <span style="color:#b5cea8;">5</span> <span style="color:#b5cea8;">6</span> <span style="color:#b5cea8;">8</span> <span style="color:#b5cea8;">7</span>   <span style="color:#9cdcfe;">thickness:</span> <span style="color:#b5cea8;">0.25</span>   <span style="color:#9cdcfe;">material:</span> <span style="color:#50c878;">'C25/30'</span>

<span style="color:#6a9955;"># Shell triangolare (3 nodi), materiale per id</span>
<span style="color:#569cd6;">shell</span>  <span style="color:#b5cea8;">2</span>   <span style="color:#b5cea8;">1</span> <span style="color:#b5cea8;">2</span> <span style="color:#b5cea8;">3</span>   <span style="color:#9cdcfe;">thickness:</span> <span style="color:#b5cea8;">0.20</span>   <span style="color:#9cdcfe;">material:</span> <span style="color:#50c878;">1</span>
</pre>
"""

_HTML_MATERIALI_SEZIONI = """
<h2 style="color:#569cd6;">Materiali e Sezioni</h2>
<p>I materiali e le sezioni vanno definiti <b>all'inizio</b> del testo strutturale,
prima dei nodi. Ogni elemento ha un <b>id intero univoco</b> e un <b>nome</b>
(racchiuso tra virgolette singole o doppie se contiene spazi). Sono ammessi due modi.</p>

<h3 style="color:#569cd6;">1. Riferimento a materiale/sezione esistente</h3>
<p>Si usa il nome di un materiale o sezione già presente nel database del programma
(o nel progetto). Se il nome è riconosciuto, sia l'<b>id</b> che il <b>nome</b>
si illuminano in <span style="color:#50c878;"><b>verde</b></span>.</p>
<pre>
<span style="color:#569cd6;">material</span>  <span style="color:#50c878;">1</span>  <span style="color:#50c878;">'C25/30'</span>
<span style="color:#569cd6;">material</span>  <span style="color:#50c878;">2</span>  <span style="color:#50c878;">'S355'</span>
<span style="color:#569cd6;">section</span>   <span style="color:#50c878;">1</span>  <span style="color:#50c878;">'R 400x600'</span>
<span style="color:#569cd6;">section</span>   <span style="color:#50c878;">2</span>  <span style="color:#50c878;">'IPE300'</span>
</pre>
<p style="color:#ce9178;">In fase di analisi, tutte le proprietà vengono recuperate
automaticamente dal database del programma. I nomi vanno racchiusi tra virgolette
singole o doppie (obbligatorio se contengono spazi).</p>

<h3 style="color:#569cd6;">2. Definizione inline</h3>
<p>Definisce le proprietà direttamente nel testo strutturale.</p>

<h4>Materiale inline</h4>
<pre>
<span style="color:#569cd6;">material</span>  <span style="color:#9cdcfe;">&lt;id&gt;</span>  <span style="color:#ce9178;">'&lt;nome&gt;'</span>  <span style="color:#b5cea8;">&lt;densità&gt;</span>  <span style="color:#b5cea8;">&lt;E&gt;</span>  <span style="color:#b5cea8;">&lt;G&gt;</span>  <span style="color:#b5cea8;">&lt;J&gt;</span>
</pre>
<table style="color:#ddd; border-collapse:collapse;" cellpadding="6">
<tr><td style="color:#9cdcfe;"><b>id</b></td><td>Intero univoco identificativo del materiale</td></tr>
<tr><td style="color:#9cdcfe;"><b>nome</b></td><td>Nome identificativo (tra virgolette se contiene spazi)</td></tr>
<tr><td style="color:#9cdcfe;"><b>densità</b></td><td>Densità del materiale (kg/m³)</td></tr>
<tr><td style="color:#9cdcfe;"><b>E</b></td><td>Modulo elastico (MPa)</td></tr>
<tr><td style="color:#9cdcfe;"><b>G</b></td><td>Modulo di taglio (MPa)</td></tr>
<tr><td style="color:#9cdcfe;"><b>J</b></td><td>Coefficiente di Poisson</td></tr>
</table>
<pre>
<span style="color:#569cd6;">material</span>  <span style="color:#b5cea8;">2</span>  <span style="color:#ce9178;">'ClsCustom'</span>      <span style="color:#b5cea8;">2500</span>  <span style="color:#b5cea8;">31476</span>   <span style="color:#b5cea8;">13115</span>   <span style="color:#b5cea8;">0.2</span>
<span style="color:#569cd6;">material</span>  <span style="color:#b5cea8;">3</span>  <span style="color:#ce9178;">'Acciaio_S355'</span>  <span style="color:#b5cea8;">7850</span>  <span style="color:#b5cea8;">210000</span>  <span style="color:#b5cea8;">80769</span>  <span style="color:#b5cea8;">0.3</span>
</pre>

<h4>Sezione inline</h4>
<pre>
<span style="color:#569cd6;">section</span>  <span style="color:#9cdcfe;">&lt;id&gt;</span>  <span style="color:#ce9178;">'&lt;nome&gt;'</span>  <span style="color:#b5cea8;">&lt;Area&gt;</span>  <span style="color:#b5cea8;">&lt;Iy&gt;</span>  <span style="color:#b5cea8;">&lt;Iz&gt;</span>  <span style="color:#9cdcfe;">material:</span> &lt;id_o_nome&gt;
</pre>
<table style="color:#ddd; border-collapse:collapse;" cellpadding="6">
<tr><td style="color:#9cdcfe;"><b>id</b></td><td>Intero univoco identificativo della sezione</td></tr>
<tr><td style="color:#9cdcfe;"><b>nome</b></td><td>Nome identificativo (tra virgolette se contiene spazi)</td></tr>
<tr><td style="color:#9cdcfe;"><b>Area</b></td><td>Area della sezione (m²)</td></tr>
<tr><td style="color:#9cdcfe;"><b>Iy</b></td><td>Momento d'inerzia asse Y (m⁴)</td></tr>
<tr><td style="color:#9cdcfe;"><b>Iz</b></td><td>Momento d'inerzia asse Z (m⁴)</td></tr>
<tr><td style="color:#9cdcfe;"><b>material:</b></td><td>Materiale associato per id intero <b>oppure</b> per nome. Se valido, diventa <span style="color:#50c878;">verde</span>.</td></tr>
</table>
<pre>
<span style="color:#569cd6;">section</span>  <span style="color:#b5cea8;">1</span>  <span style="color:#ce9178;">'Pilastro 30x30'</span>  <span style="color:#b5cea8;">0.09</span>  <span style="color:#b5cea8;">6.75e-4</span>   <span style="color:#b5cea8;">6.75e-4</span>   <span style="color:#9cdcfe;">material:</span> <span style="color:#50c878;">'C25/30'</span>
<span style="color:#569cd6;">section</span>  <span style="color:#b5cea8;">2</span>  <span style="color:#ce9178;">'Trave 30x50'</span>     <span style="color:#b5cea8;">0.15</span>  <span style="color:#b5cea8;">3.125e-3</span>  <span style="color:#b5cea8;">1.125e-3</span>  <span style="color:#9cdcfe;">material:</span> <span style="color:#50c878;">1</span>
</pre>

<p style="color:#ce9178;">Nota: ogni ID deve essere univoco; gli ID duplicati generano errore.
Le sezioni così definite vengono richiamate dalle aste tramite
<code style="color:#9cdcfe;">section:</code> indicando l'id o il nome.</p>
"""

_HTML_VINCOLI = """
<h2 style="color:#569cd6;">Vincoli</h2>
<p>Definiscono i gradi di libertà bloccati per ogni nodo.</p>

<h3>Sintassi</h3>
<pre>
<span style="color:#569cd6;">fix</span>  <span style="color:#9cdcfe;">&lt;nodo_id&gt;</span>  <span style="color:#b5cea8;">&lt;dx&gt;</span> <span style="color:#b5cea8;">&lt;dy&gt;</span> <span style="color:#b5cea8;">&lt;dz&gt;</span>  [<span style="color:#b5cea8;">&lt;rx&gt;</span> <span style="color:#b5cea8;">&lt;ry&gt;</span> <span style="color:#b5cea8;">&lt;rz&gt;</span>]
</pre>

<table style="color:#ddd; border-collapse:collapse;" cellpadding="6">
<tr><td style="color:#9cdcfe;"><b>nodo_id</b></td><td>ID del nodo vincolato</td></tr>
<tr><td style="color:#9cdcfe;"><b>dx, dy, dz</b></td><td>1 = bloccato, 0 = libero (traslazioni)</td></tr>
<tr><td style="color:#9cdcfe;"><b>rx, ry, rz</b></td><td>(opz.) 1 = bloccato, 0 = libero (rotazioni, default: 0)</td></tr>
</table>

<h3>Esempi</h3>
<pre>
<span style="color:#6a9955;"># Incastro (tutti i 6 gradi di libertà bloccati)</span>
<span style="color:#569cd6;">fix</span>  <span style="color:#b5cea8;">1</span>   <span style="color:#b5cea8;">1 1 1</span>  <span style="color:#b5cea8;">1 1 1</span>

<span style="color:#6a9955;"># Cerniera (traslazioni bloccate, rotazioni libere)</span>
<span style="color:#569cd6;">fix</span>  <span style="color:#b5cea8;">2</span>   <span style="color:#b5cea8;">1 1 1</span>  <span style="color:#b5cea8;">0 0 0</span>

<span style="color:#6a9955;"># Appoggio (solo traslazione verticale bloccata)</span>
<span style="color:#569cd6;">fix</span>  <span style="color:#b5cea8;">3</span>   <span style="color:#b5cea8;">0 0 1</span>  <span style="color:#b5cea8;">0 0 0</span>

<span style="color:#6a9955;"># Carrello (traslazione Y e Z bloccate)</span>
<span style="color:#569cd6;">fix</span>  <span style="color:#b5cea8;">4</span>   <span style="color:#b5cea8;">0 1 1</span>  <span style="color:#b5cea8;">0 0 0</span>

<span style="color:#6a9955;"># Versione corta (solo 3 valori → rotazioni = 0)</span>
<span style="color:#569cd6;">fix</span>  <span style="color:#b5cea8;">5</span>   <span style="color:#b5cea8;">1 1 1</span>
</pre>
"""

_HTML_CARICHI = """
<h2 style="color:#569cd6;">Carichi</h2>
<p>Forze concentrate e distribuite sulla struttura.</p>

<h3>Carico nodale (forza concentrata)</h3>
<pre>
<span style="color:#569cd6;">nodeLoad</span>  <span style="color:#9cdcfe;">&lt;nodo_id&gt;</span>  <span style="color:#b5cea8;">&lt;Fx&gt;</span>  <span style="color:#b5cea8;">&lt;Fy&gt;</span>  <span style="color:#b5cea8;">&lt;Fz&gt;</span>
</pre>
<table style="color:#ddd; border-collapse:collapse;" cellpadding="6">
<tr><td style="color:#9cdcfe;"><b>Fx, Fy, Fz</b></td><td>Componenti della forza (kN)</td></tr>
</table>

<h3>Carico distribuito su asta</h3>
<pre>
<span style="color:#569cd6;">beamLoad</span>  <span style="color:#9cdcfe;">&lt;asta_id&gt;</span>  <span style="color:#b5cea8;">&lt;wx&gt;</span>  <span style="color:#b5cea8;">&lt;wy&gt;</span>  <span style="color:#b5cea8;">&lt;wz&gt;</span>
</pre>
<table style="color:#ddd; border-collapse:collapse;" cellpadding="6">
<tr><td style="color:#9cdcfe;"><b>wx, wy, wz</b></td><td>Componenti del carico distribuito (kN/m)</td></tr>
</table>

<h3>Esempi</h3>
<pre>
<span style="color:#6a9955;"># Forza orizzontale di 10 kN sul nodo 5</span>
<span style="color:#569cd6;">nodeLoad</span>  <span style="color:#b5cea8;">5</span>   <span style="color:#b5cea8;">10.0</span>   <span style="color:#b5cea8;">0.0</span>   <span style="color:#b5cea8;">0.0</span>

<span style="color:#6a9955;"># Forza verticale verso il basso di 50 kN</span>
<span style="color:#569cd6;">nodeLoad</span>  <span style="color:#b5cea8;">3</span>    <span style="color:#b5cea8;">0.0</span>   <span style="color:#b5cea8;">0.0</span>  <span style="color:#b5cea8;">-50.0</span>

<span style="color:#6a9955;"># Carico distribuito verso il basso su asta 2</span>
<span style="color:#569cd6;">beamLoad</span>  <span style="color:#b5cea8;">2</span>    <span style="color:#b5cea8;">0.0</span>   <span style="color:#b5cea8;">0.0</span>  <span style="color:#b5cea8;">-12.0</span>

<span style="color:#6a9955;"># Carico distribuito laterale (vento) su asta 1</span>
<span style="color:#569cd6;">beamLoad</span>  <span style="color:#b5cea8;">1</span>    <span style="color:#b5cea8;">3.0</span>   <span style="color:#b5cea8;">0.0</span>   <span style="color:#b5cea8;">0.0</span>
</pre>

<p style="color:#ce9178;">Nota: le convenzioni di segno seguono il sistema di riferimento
globale (X, Y, Z). I carichi verso il basso hanno componente Z negativa.</p>
"""

# ================================================================
#  TESTO PROMPT AI
# ================================================================

_TESTO_PROMPT_PURO = """Agisci come un esperto ingegnere strutturista e programmatore.
Il tuo compito è generare il codice di definizione di un modello strutturale utilizzando un linguaggio di scripting proprietario, basato sulla seguente sintassi rigorosa:

# REGOLE DI SINTASSI
- Materiali (riferimento al database): material <id> '<nome>'
- Materiali (inline): material <id> '<nome>' <densita> <E> <G> <J>
- Sezioni (riferimento al database): section <id> '<nome>'
- Sezioni (inline): section <id> '<nome>' <Area> <Iy> <Iz> material: <id_o_nome>
- Nodi: node <id> <x> <y> <z>
- Aste: beam <id> <nodo_i> <nodo_j> section: <id_o_nome>
- Shell (3 o 4 nodi): shell <id> <n1> <n2> <n3> [<n4>] thickness: <t> material: <id_o_nome>
- Vincoli: fix <nodo_id> <dx> <dy> <dz> [<rx> <ry> <rz>] (1=bloccato, 0=libero)
- Carichi nodali: nodeLoad <nodo_id> <Fx> <Fy> <Fz>
- Carichi distribuiti su asta: beamLoad <asta_id> <wx> <wy> <wz>

# LINEE GUIDA IMPORTANTI
1. Utilizza i commenti (iniziando la riga con '#') per dividere chiaramente le sezioni: # --- MATERIALI ---, # --- SEZIONI ---, # --- NODI ---, # --- ASTE ---, # --- SHELL ---, # --- VINCOLI ---, # --- CARICHI ---.
2. Definisci materiali e sezioni PRIMA dei nodi. Ogni materiale e sezione ha un id intero univoco e un nome racchiuso tra virgolette singole o doppie (obbligatorio se contiene spazi).
3. Le sezioni inline hanno il riferimento al materiale tramite il parametro 'material:' (id intero o nome). Le aste hanno SOLO 'section:' (no 'material:'), perché il materiale è già contenuto nella sezione. Le shell hanno 'thickness:' e 'material:'.
4. I riferimenti tramite section: e material: possono usare indifferentemente l'id intero o il nome.
5. Unità di misura da considerare (non scriverle nel codice): [m] per coordinate e spessori, [kg/m³] per la densità, [MPa] per E e G, [kN] per le forze, [kN/m] per i carichi distribuiti. L'asse Z è verticale (verso l'alto).
6. L'ID di ogni elemento, materiale e sezione deve essere intero e progressivo, partendo da 1.
7. Le shell possono essere triangolari (3 nodi) o quadrilatere (4 nodi), in ordine antiorario.
8. I parametri con i due punti devono essere scritti come 'section:', 'material:', 'thickness:' (la 'keyword:' attaccata e il valore separato da spazio).

# ESEMPIO RAPIDO
material 1 'C25/30'
section  1 'Trave 30x50' 0.15 3.125e-3 1.125e-3 material: 1
node 1 0.0 0.0 0.0
node 2 5.0 0.0 0.0
beam 1 1 2 section: 'Trave 30x50'
fix 1 1 1 1 1 1 1
beamLoad 1 0.0 0.0 -10.0

Ora, in base a queste regole, genera il codice completo per la seguente struttura:
[SOSTITUISCI QUESTO TESTO CON LA DESCRIZIONE DELLA TUA STRUTTURA - ES: "Telaio 2D in c.a., 2 campate da 5m, 1 piano da 3.5m, incastri alla base"]
"""

_HTML_PROMPT = f"""
<h2 style="color:#569cd6;">Generazione Automatica con AI</h2>
<p>Vuoi creare una struttura in pochi secondi? Fai lavorare l'Intelligenza Artificiale al posto tuo!</p>
<p>Clicca il tasto <b>Copia Prompt</b> in alto a destra e incolla il testo in <i>ChatGPT</i>, <i>Claude</i> o <i>Gemini</i>. Ricordati di completare l'ultima riga del testo con la descrizione della struttura che desideri ottenere.</p>

<pre style="color:#ce9178; background-color:#2b2b2b; padding:10px; border-radius:5px;">
{_TESTO_PROMPT_PURO.replace('<', '&lt;').replace('>', '&gt;')}
</pre>
"""


# ================================================================
#  DIALOG INFO
# ================================================================

class StrutturaInfoDialog(QDialog):
    """Finestra informativa con tabs per ogni sezione della sintassi."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guida alla Sintassi – Struttura")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(_STYLE_DIALOG)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._make_tab(_HTML_GENERALE),            "Generale")
        tabs.addTab(self._make_tab(_HTML_MATERIALI_SEZIONI),   "Materiali e Sezioni")
        tabs.addTab(self._make_tab(_HTML_NODI),                "Nodi")
        tabs.addTab(self._make_tab(_HTML_ASTE),                "Aste")
        tabs.addTab(self._make_tab(_HTML_SHELL),               "Shell")
        tabs.addTab(self._make_tab(_HTML_VINCOLI),             "Vincoli")
        tabs.addTab(self._make_tab(_HTML_CARICHI),             "Carichi")
        tabs.addTab(self._make_prompt_tab(),                   "✨ Prompt AI")
        layout.addWidget(tabs)

        # Pulsante chiudi
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_chiudi = QPushButton("Chiudi")
        btn_chiudi.clicked.connect(self.close)
        btn_row.addWidget(btn_chiudi)
        layout.addLayout(btn_row)

    @staticmethod
    def _make_tab(html: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(html)
        lay.addWidget(browser)
        return w

    def _make_prompt_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 8, 0, 0)
        
        # Header con tasto copia
        header_lay = QHBoxLayout()
        header_lay.addStretch()
        
        self.btn_copia = QPushButton("📋 Copia Prompt")
        self.btn_copia.setObjectName("copyButton") # Per CSS personalizzato
        self.btn_copia.setCursor(Qt.PointingHandCursor)
        self.btn_copia.clicked.connect(self._copia_prompt)
        header_lay.addWidget(self.btn_copia)
        
        lay.addLayout(header_lay)

        # Contenuto Testuale
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(_HTML_PROMPT)
        lay.addWidget(browser)
        
        return w

    def _copia_prompt(self):
        """Copia il testo negli appunti e aggiorna momentaneamente il pulsante."""
        QApplication.clipboard().setText(_TESTO_PROMPT_PURO)
        
        self.btn_copia.setText("✅ Copiato!")
        self.btn_copia.setStyleSheet("background-color: rgb(40, 160, 60); color: white;")
        
        # Ripristina l'aspetto del bottone dopo 2 secondi
        QTimer.singleShot(2000, self._ripristina_bottone)

    def _ripristina_bottone(self):
        self.btn_copia.setText("📋 Copia Prompt")
        self.btn_copia.setStyleSheet("") # Ripristina il CSS dello stylesheet globale