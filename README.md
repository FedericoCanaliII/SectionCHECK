# SectionCHECK

<p align="center">
  <img src="interfaccia/icone/logo.png" width="120">
</p>

<p align="center">
  Software open-source per l'analisi strutturale di sezioni in calcestruzzo armato,<br>
  con rendering 2D/3D in tempo reale e analisi non lineare integrata.
</p>

<p align="center">
  <a href="#installazione">Installazione</a> &middot;
  <a href="#funzionalita">Funzionalita</a> &middot;
  <a href="#build-eseguibile">Build</a> &middot;
  <a href="#licenza">Licenza</a>
</p>

---

## Panoramica

SectionCHECK e un software desktop sviluppato in Python + PyQt5, pensato per la progettazione e la verifica strutturale di elementi in calcestruzzo armato. Integra un editor grafico per sezioni 2D, un modellatore 3D per elementi strutturali, un pacchetto completo di analisi non lineari e un agente AI integrato per assistere l'utente durante la modellazione e l'analisi.

---

## Funzionalita

### Materiali

Definizione dei materiali tramite legami costitutivi non lineari personalizzabili (calcestruzzo, acciaio da armatura, acciaio strutturale). Database precaricato con le principali normative.

### Sezioni

Editor 2D con strumenti di disegno dedicati (rettangolo, poligono, cerchio, fori, barre, staffe). Supporto per sezioni di qualsiasi forma con armatura posizionata liberamente.

### Elementi 3D

Modellatore tridimensionale per elementi strutturali (travi, pilastri, fondazioni, solai). Ogni elemento puo contenere piu oggetti geometrici (parallelepipedi, cilindri, forme personalizzate) con gestione di carichi e vincoli dedicata.

### Agente AI

Assistente intelligente integrato direttamente nel software, in grado di interagire con il modello strutturale. Supporta le principali API (Anthropic Claude, OpenAI GPT, Google Gemini, DeepSeek) e puo creare, modificare ed interrogare elementi, sezioni, materiali, carichi e vincoli tramite linguaggio naturale.

---

## Analisi

### Pressoflessione

Calcolo della capacita a pressoflessione retta e deviata con diagrammi interattivi.

<p align="center">
  <img src="interfaccia/immagini/img1.png" width="500">
</p>

### Dominio di Interazione N-M

Generazione della superficie di interazione tridimensionale Sforzo Normale – Momento flettente (N-Mx-My).

<p align="center">
  <img src="interfaccia/immagini/img2.png" width="500">
</p>

### Momento-Curvatura

Diagrammi momento-curvatura a 360 gradi con analisi incrementale non lineare.

<p align="center">
  <img src="interfaccia/immagini/img3.png" width="500">
</p>

### FEM Elemento

Analisi agli elementi finiti del singolo elemento strutturale con generazione automatica della mesh, materiali non lineari e post-processing dei risultati. Il solver si appoggia a [CalculiX](http://www.calculix.de/), un motore FEM open-source esterno.

---

## Installazione

### Requisiti

- Python 3.10+
- Windows 10/11 (testato), Linux/macOS (sperimentale)
- [CalculiX](http://www.calculix.de/) (necessario solo per le analisi FEM)

### Da sorgente

```bash
git clone https://github.com/<utente>/SectionCHECK.git
cd SectionCHECK

pip install -r requirements.txt

python main.py
```

### Dipendenze

| Pacchetto | Versione minima | Uso |
|-----------|----------------|-----|
| PyQt5 | 5.15 | Interfaccia grafica |
| numpy | 1.21 | Calcolo numerico |
| scipy | 1.7 | Algoritmi scientifici (Delaunay, KDTree) |
| shapely | 1.8 | Operazioni geometriche su poligoni |
| Pillow | 8.0 | Elaborazione immagini |
| PyOpenGL | 3.1 | Rendering 2D/3D |
| requests | 2.28 | Comunicazione HTTP (modulo AI) |

---

## Build eseguibile

Il progetto usa **cx_Freeze** per generare un eseguibile Windows standalone.

```bash
pip install cx_Freeze

python setup.py build
```

L'eseguibile `SectionCHECK.exe` viene creato nella cartella `build/`. Tutti i database, le icone e le risorse vengono inclusi automaticamente.

---

## Licenza

Distribuito sotto licenza **GNU Affero General Public License v3.0** – vedi il file [LICENSE](LICENSE) per i dettagli.
