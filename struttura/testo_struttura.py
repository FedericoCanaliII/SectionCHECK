"""
testo_struttura.py – Syntax highlighter e parser per il testo strutturale.

Fornisce:
  • StrutturaSyntaxHighlighter  – evidenziazione in tempo reale
  • parse_struttura()           – parser del testo → dati strutturali
  • TextoStrutturaManager       – gestisce plainTextEdit + text_control
"""

import re
from PyQt5.QtCore import Qt, QRect, QSize, QRegularExpression, QTimer
from PyQt5.QtGui import (
    QSyntaxHighlighter, QTextCharFormat, QColor, QFont,
    QPainter, QPen, QTextCursor,
)
from PyQt5.QtWidgets import (
    QPlainTextEdit, QWidget, QVBoxLayout, QLabel, QTextEdit,
)


# ================================================================
#  COLORI TEMA (stile IDE scuro, ispirato a VS Code Dark+)
# ================================================================

_COL_KEYWORD    = QColor(86,  156, 214)    # blu – keyword validi (node, beam, fix...)
_COL_SECTION    = QColor(206, 145, 120)    # arancione – intestazioni sezione (# ─── NODI)
_COL_COMMENT    = QColor(106, 153, 85)     # verde – commenti
_COL_NUMBER     = QColor(181, 206, 168)    # verde chiaro – numeri
_COL_STRING     = QColor(206, 145, 120)    # arancio – stringhe tra virgolette
_COL_PARAM      = QColor(156, 220, 254)    # celeste – nomi parametri (section:, material:)
_COL_ERROR_KW   = QColor(244, 71,  71)     # rosso – keyword errato
_COL_DEFAULT    = QColor(212, 212, 212)    # grigio chiaro – testo base
_COL_VALID_REF  = QColor(80, 200, 120)     # verde brillante – riferimento valido

# Keywords validi riconosciuti dal sistema
_KEYWORDS = {
    "material", "section",
    "node", "beam", "shell",
    "fix",
    "nodeLoad", "beamLoad", "shellLoad",
}

# Parametri con sintassi "keyword:" (section:, material:, thickness:)
_COLON_PARAMS = {"section:", "material:", "thickness:"}

# Pattern intestazioni sezione
_SECTION_HEADERS = re.compile(
    r"^#\s*[═─]+\s*(NODI|ASTE|SHELL|VINCOLI|CARICHI|MATERIALI|SEZIONI).*$",
    re.IGNORECASE,
)


def _norm_nome(s: str) -> str:
    """Normalizza un nome di materiale/sezione per il matching:
    - sostituisce il segno di moltiplicazione unicode '×' (U+00D7) con 'x'
    - toglie gli spazi iniziali/finali
    - lowercase
    Serve per riconoscere 'R 200x400' come 'R 200×400' del database."""
    if s is None:
        return ""
    return s.replace("×", "x").replace("X", "x").strip().lower()


# ================================================================
#  SYNTAX HIGHLIGHTER
# ================================================================


class StrutturaSyntaxHighlighter(QSyntaxHighlighter):
    """
    Evidenziazione sintattica per il testo di definizione strutturale.

    Regole di colorazione:
      • Intestazioni sezione (# ─── NODI ───)   → arancione bold
      • Commenti (# ...)                         → verde
      • Keywords validi (node, beam, fix...)     → blu
      • Parole non riconosciute a inizio riga    → rosso (errore)
      • Numeri                                   → verde chiaro
      • Stringhe '...' o "..."                   → arancio
      • Parametri (section:, material:)          → celeste
      • ID/nomi di materiali/sezioni riconosciuti → verde brillante
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Nomi globali (database programma + progetto aperto)
        self._mat_nomi_globali: set = set()
        self._sez_nomi_globali: set = set()
        self._mat_nomi_norm: set = set()
        self._sez_nomi_norm: set = set()
        # Definizioni locali nel testo strutturale corrente: {id_str: nome_str}
        self._mat_definiti: dict = {}
        self._sez_definiti: dict = {}
        self._mat_definiti_norm_values: set = set()
        self._sez_definiti_norm_values: set = set()
        self._build_formats()

    def _build_formats(self):
        self._fmt_keyword = QTextCharFormat()
        self._fmt_keyword.setForeground(_COL_KEYWORD)
        self._fmt_keyword.setFontWeight(QFont.Bold)

        self._fmt_section = QTextCharFormat()
        self._fmt_section.setForeground(_COL_SECTION)
        self._fmt_section.setFontWeight(QFont.Bold)

        self._fmt_comment = QTextCharFormat()
        self._fmt_comment.setForeground(_COL_COMMENT)
        self._fmt_comment.setFontItalic(True)

        self._fmt_number = QTextCharFormat()
        self._fmt_number.setForeground(_COL_NUMBER)

        self._fmt_string = QTextCharFormat()
        self._fmt_string.setForeground(_COL_STRING)

        self._fmt_param = QTextCharFormat()
        self._fmt_param.setForeground(_COL_PARAM)

        self._fmt_error = QTextCharFormat()
        self._fmt_error.setForeground(_COL_ERROR_KW)
        self._fmt_error.setFontUnderline(True)

        self._fmt_valid_ref = QTextCharFormat()
        self._fmt_valid_ref.setForeground(_COL_VALID_REF)
        self._fmt_valid_ref.setFontWeight(QFont.Bold)

    @staticmethod
    def _norm_set(nomi) -> set:
        return {_norm_nome(n) for n in nomi}

    def set_nomi_validi(self,
                        mat_nomi_globali: set, sez_nomi_globali: set,
                        mat_definiti: dict, sez_definiti: dict):
        """
        Aggiorna i riferimenti validi per l'illuminazione verde.

        mat_nomi_globali / sez_nomi_globali : insiemi di nomi presenti nel
            database del programma (per le righe 'material/section <id> <nome>'
            in modalità riferimento).
        mat_definiti / sez_definiti : dizionari {id_str: nome_str} delle
            definizioni presenti nel testo strutturale corrente, usati per
            validare i riferimenti in 'section:', 'material:' su sezioni
            inline, beam e shell.
        """
        self._mat_nomi_globali = mat_nomi_globali
        self._sez_nomi_globali = sez_nomi_globali
        # Set normalizzati per matching tollerante (x ↔ ×, case-insensitive)
        self._mat_nomi_norm = self._norm_set(mat_nomi_globali)
        self._sez_nomi_norm = self._norm_set(sez_nomi_globali)
        self._mat_definiti = mat_definiti
        self._sez_definiti = sez_definiti
        self._mat_definiti_norm_values = {_norm_nome(v) for v in mat_definiti.values()}
        self._sez_definiti_norm_values = {_norm_nome(v) for v in sez_definiti.values()}
        self.rehighlight()

    def _is_valid_material_ref(self, val: str) -> bool:
        """True se 'val' (id intero o nome) corrisponde a un materiale definito."""
        if val in self._mat_definiti:               # match per ID
            return True
        if _norm_nome(val) in self._mat_definiti_norm_values:  # match per nome
            return True
        return False

    def _is_valid_section_ref(self, val: str) -> bool:
        """True se 'val' (id intero o nome) corrisponde a una sezione definita."""
        if val in self._sez_definiti:
            return True
        if _norm_nome(val) in self._sez_definiti_norm_values:
            return True
        return False

    def highlightBlock(self, text: str):
        stripped = text.strip()
        if not stripped:
            return

        # 1) Intestazione sezione (# ─── NODI ─── etc.)
        if _SECTION_HEADERS.match(stripped):
            self.setFormat(0, len(text), self._fmt_section)
            return

        # 2) Commento puro
        if stripped.startswith("#"):
            self.setFormat(0, len(text), self._fmt_comment)
            return

        # 3) Rimuovi commento inline per l'analisi (gestendo entrambe le virgolette)
        code_part = text
        comment_start = -1
        quote_ch = None
        for i, ch in enumerate(text):
            if quote_ch:
                if ch == quote_ch:
                    quote_ch = None
            elif ch in ('"', "'"):
                quote_ch = ch
            elif ch == '#':
                comment_start = i
                break
        if comment_start >= 0:
            self.setFormat(comment_start, len(text) - comment_start, self._fmt_comment)
            code_part = text[:comment_start]

        # 4) Keyword a inizio riga
        first_word = ""
        m_kw = re.match(r"^(\s*)(\w+)", code_part)
        if m_kw:
            start = m_kw.start(2)
            word = m_kw.group(2)
            if word in _KEYWORDS:
                self.setFormat(start, len(word), self._fmt_keyword)
                first_word = word
            else:
                self.setFormat(start, len(word), self._fmt_error)

        # 5) Numeri
        for m in re.finditer(r"(?<![\"'\w])(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", code_part):
            self.setFormat(m.start(), m.end() - m.start(), self._fmt_number)

        # 6) Stringhe tra virgolette (singole o doppie)
        for m in re.finditer(r'"[^"]*"', code_part):
            self.setFormat(m.start(), m.end() - m.start(), self._fmt_string)
        for m in re.finditer(r"'[^']*'", code_part):
            self.setFormat(m.start(), m.end() - m.start(), self._fmt_string)

        # 7) Parametri con sintassi "keyword:" (section:, material:, thickness:)
        for m in re.finditer(r'\b(\w+):', code_part):
            tag = m.group(0)
            if tag in _COLON_PARAMS:
                self.setFormat(m.start(), len(tag), self._fmt_param)

        # 8) Illuminazione riferimenti (verde brillante)
        if first_word == "material":
            self._highlight_material_line(code_part)
        elif first_word == "section":
            self._highlight_section_line(code_part)
        elif first_word in ("beam", "shell"):
            self._highlight_colon_refs(code_part)

    def _highlight_material_line(self, code_part: str):
        """Riga 'material <id> <nome>' di riferimento → nome verde se in DB."""
        toks = _tokenize_with_pos(code_part)
        if len(toks) < 3:
            return
        if len(toks) == 3:
            nome_tok, nome_pos = toks[2]
            nome = _strip_quotes(nome_tok)
            if _norm_nome(nome) in self._mat_nomi_norm:
                self.setFormat(nome_pos, len(nome_tok), self._fmt_valid_ref)

    def _highlight_section_line(self, code_part: str):
        """
        Riga 'section <id> <nome>' di riferimento → nome verde se in DB.
        Riga inline 'section <id> <nome> <Area> <Iy> <Iz> material: <ref>'
        → 'ref' verde se identifica un materiale definito.
        """
        toks = _tokenize_with_pos(code_part)
        if len(toks) < 3:
            return
        has_material = any(t == "material:" for t, _ in toks)
        if not has_material:
            # Riferimento puro
            if len(toks) == 3:
                nome_tok, nome_pos = toks[2]
                nome = _strip_quotes(nome_tok)
                if _norm_nome(nome) in self._sez_nomi_norm:
                    self.setFormat(nome_pos, len(nome_tok), self._fmt_valid_ref)
        else:
            # Inline: il valore dopo 'material:' è il riferimento
            for i, (t, _) in enumerate(toks):
                if t == "material:" and i + 1 < len(toks):
                    val_tok, val_pos = toks[i + 1]
                    val = _strip_quotes(val_tok)
                    if self._is_valid_material_ref(val):
                        self.setFormat(val_pos, len(val_tok), self._fmt_valid_ref)
                    break

    def _highlight_colon_refs(self, code_part: str):
        """Per beam/shell evidenzia in verde il valore dopo section:/material:
        se identifica una sezione/materiale definito."""
        toks = _tokenize_with_pos(code_part)
        for i, (t, _) in enumerate(toks):
            if i + 1 >= len(toks):
                continue
            val_tok, val_pos = toks[i + 1]
            val = _strip_quotes(val_tok)
            if t == "section:":
                if self._is_valid_section_ref(val):
                    self.setFormat(val_pos, len(val_tok), self._fmt_valid_ref)
            elif t == "material:":
                if self._is_valid_material_ref(val):
                    self.setFormat(val_pos, len(val_tok), self._fmt_valid_ref)


# ================================================================
#  PARSER
# ================================================================

def parse_struttura(testo: str) -> tuple:
    """
    Parsa il testo strutturale e restituisce:
        (dati, errori)

    dati = {
        "nodi":               {id: (x, y, z), ...},
        "aste":               {id: {"nodo_i": int, "nodo_j": int,
                                    "sezione": str, "materiale": str}, ...},
        "shell":              {id: {"nodi": [int,...], "spessore": float,
                                    "materiale": str}, ...},
        "vincoli":            {nodo_id: [dx, dy, dz, rx, ry, rz], ...},
        "carichi_nodali":     [(nodo_id, fx, fy, fz), ...],
        "carichi_distribuiti": [(asta_id, wx, wy, wz), ...],
        "carichi_shell":       [(shell_id, qx, qy, qz), ...],
    }
    errori = [(riga, messaggio), ...]
    """
    dati = {
        "materiali": {},
        "sezioni": {},
        "nodi": {},
        "aste": {},
        "shell": {},
        "vincoli": {},
        "carichi_nodali": [],
        "carichi_distribuiti": [],
        "carichi_shell": [],
    }
    errori = []

    for lineno, raw_line in enumerate(testo.splitlines(), 1):
        line = raw_line.strip()

        # Salta righe vuote e commenti
        if not line or line.startswith("#"):
            continue

        # Rimuovi commento inline
        in_quote = False
        for i, ch in enumerate(line):
            if ch == '"':
                in_quote = not in_quote
            elif ch == '#' and not in_quote:
                line = line[:i].strip()
                break

        if not line:
            continue

        # Tokenizza preservando stringhe
        tokens = _tokenize(line)
        if not tokens:
            continue

        cmd = tokens[0]

        try:
            if cmd == "material":
                _parse_material(tokens, dati, errori, lineno)
            elif cmd == "section":
                _parse_section(tokens, dati, errori, lineno)
            elif cmd == "node":
                _parse_node(tokens, dati, errori, lineno)
            elif cmd == "beam":
                _parse_beam(tokens, dati, errori, lineno)
            elif cmd == "shell":
                _parse_shell(tokens, dati, errori, lineno)
            elif cmd == "fix":
                _parse_fix(tokens, dati, errori, lineno)
            elif cmd == "nodeLoad":
                _parse_node_load(tokens, dati, errori, lineno)
            elif cmd == "beamLoad":
                _parse_beam_load(tokens, dati, errori, lineno)
            elif cmd == "shellLoad":
                _parse_shell_load(tokens, dati, errori, lineno)
            else:
                errori.append((lineno, f"Comando sconosciuto: '{cmd}'"))
        except Exception as e:
            errori.append((lineno, f"Errore: {e}"))

    # Validazione incrociata
    _valida_riferimenti(dati, errori)

    return dati, errori


def _tokenize(line: str) -> list:
    """Tokenizza la riga preservando le stringhe tra virgolette singole o doppie."""
    return [t for t, _ in _tokenize_with_pos(line)]


def _tokenize_with_pos(line: str) -> list:
    """Come _tokenize ma restituisce coppie (token, posizione_iniziale)."""
    tokens = []
    i = 0
    n = len(line)
    while i < n:
        if line[i].isspace():
            i += 1
            continue
        if line[i] in ('"', "'"):
            quote = line[i]
            j = line.find(quote, i + 1)
            if j == -1:
                j = n - 1
            tokens.append((line[i:j+1], i))
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            tokens.append((line[i:j], i))
            i = j
    return tokens


def _strip_quotes(tok: str) -> str:
    """Rimuove le virgolette singole o doppie attorno al token."""
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ('"', "'"):
        return tok[1:-1]
    return tok


def _extract_colon_param(tokens: list, name: str) -> str:
    """
    Estrae il valore dopo un token 'name:'.
    Restituisce il primo token successivo (senza virgolette).
    I nomi con spazi vanno racchiusi tra virgolette singole o doppie:
        section: 'Pilastro 30x30'   →  "Pilastro 30x30"
        section: 1                  →  "1"
    """
    tag = name + ":"
    for i, t in enumerate(tokens):
        if t == tag and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt in _COLON_PARAMS:
                return ""
            return _strip_quotes(nxt)
    return ""


def _parse_node(tokens, dati, errori, lineno):
    # node <id> <x> <y> <z>
    if len(tokens) < 5:
        errori.append((lineno, f"node richiede: node <id> <x> <y> <z> (trovati {len(tokens)-1} argomenti)"))
        return
    nid = int(tokens[1])
    x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
    if nid in dati["nodi"]:
        errori.append((lineno, f"Nodo {nid} già definito"))
        return
    dati["nodi"][nid] = (x, y, z)


def _parse_beam(tokens, dati, errori, lineno):
    # beam <id> <nodo_i> <nodo_j>  section: <id_o_nome>
    if len(tokens) < 4:
        errori.append((lineno, "beam richiede: beam <id> <nodo_i> <nodo_j>  section: <id_o_nome>"))
        return
    bid = int(tokens[1])
    ni, nj = int(tokens[2]), int(tokens[3])
    sezione = _extract_colon_param(tokens, "section")
    if bid in dati["aste"]:
        errori.append((lineno, f"Asta {bid} già definita"))
        return
    dati["aste"][bid] = {
        "nodo_i": ni, "nodo_j": nj,
        "sezione": sezione,
    }


def _parse_shell(tokens, dati, errori, lineno):
    # shell <id> <n1> <n2> <n3> [<n4>]  [thickness: <t>] [material: nome]
    if len(tokens) < 5:
        errori.append((lineno, "shell richiede almeno: shell <id> <n1> <n2> <n3>"))
        return
    sid = int(tokens[1])
    # Determina quanti nodi ci sono (3 o 4) prima dei parametri opzionali
    n_nodi_end = 5  # default: 3 nodi (tokens[2..4])
    if len(tokens) > 5:
        # Se il token 5 è un intero (non un parametro con ':'), è il quarto nodo
        tok5 = tokens[5]
        if tok5 not in _COLON_PARAMS:
            try:
                int(tok5)
                n_nodi_end = 6
            except ValueError:
                pass
    nodi = [int(tokens[i]) for i in range(2, n_nodi_end)]
    spessore = _extract_colon_param(tokens, "thickness")
    spessore = float(spessore) if spessore else 0.20
    materiale = _extract_colon_param(tokens, "material")
    if sid in dati["shell"]:
        errori.append((lineno, f"Shell {sid} già definita"))
        return
    dati["shell"][sid] = {
        "nodi": nodi, "spessore": spessore, "materiale": materiale,
    }


def _parse_fix(tokens, dati, errori, lineno):
    # fix <nodo_id> <dx> <dy> <dz> <rx> <ry> <rz>
    if len(tokens) < 5:
        errori.append((lineno, "fix richiede almeno: fix <nodo_id> <dx> <dy> <dz>"))
        return
    nid = int(tokens[1])
    vals = [int(tokens[i]) for i in range(2, min(len(tokens), 8))]
    # Pad a 6 valori se ne sono stati dati solo 3
    while len(vals) < 6:
        vals.append(0)
    dati["vincoli"][nid] = vals[:6]


def _parse_node_load(tokens, dati, errori, lineno):
    # nodeLoad <nodo_id> <Fx> <Fy> <Fz>  [<Mx> <My> <Mz>]
    if len(tokens) < 5:
        errori.append((lineno, "nodeLoad richiede: nodeLoad <nodo_id> <Fx> <Fy> <Fz>"))
        return
    nid = int(tokens[1])
    fx, fy, fz = float(tokens[2]), float(tokens[3]), float(tokens[4])
    dati["carichi_nodali"].append((nid, fx, fy, fz))


def _parse_beam_load(tokens, dati, errori, lineno):
    # beamLoad <asta_id> <wx> <wy> <wz>
    if len(tokens) < 5:
        errori.append((lineno, "beamLoad richiede: beamLoad <asta_id> <wx> <wy> <wz>"))
        return
    bid = int(tokens[1])
    wx, wy, wz = float(tokens[2]), float(tokens[3]), float(tokens[4])
    dati["carichi_distribuiti"].append((bid, wx, wy, wz))


def _parse_shell_load(tokens, dati, errori, lineno):
    # shellLoad <shell_id> <qx> <qy> <qz>
    if len(tokens) < 5:
        errori.append((lineno, "shellLoad richiede: shellLoad <shell_id> <qx> <qy> <qz>"))
        return
    sid = int(tokens[1])
    qx, qy, qz = float(tokens[2]), float(tokens[3]), float(tokens[4])
    dati["carichi_shell"].append((sid, qx, qy, qz))


def _parse_material(tokens, dati, errori, lineno):
    # Riferimento:  material <id> <nome>
    # Inline:       material <id> <nome> <densita> <E> <G> <J>
    if len(tokens) < 3:
        errori.append((lineno, "material richiede almeno: material <id> <nome>"))
        return
    try:
        mid = int(tokens[1])
    except ValueError:
        errori.append((lineno, f"material: id deve essere intero, ricevuto '{tokens[1]}'"))
        return
    nome = _strip_quotes(tokens[2])
    if mid in dati["materiali"]:
        errori.append((lineno, f"Materiale {mid} già definito"))
        return
    if len(tokens) == 3:
        # Riferimento a materiale esistente nel programma
        dati["materiali"][mid] = {"nome": nome, "tipo": "riferimento"}
    elif len(tokens) == 7:
        # Definizione inline: material <id> <nome> <densita> <E> <G> <J>
        try:
            densita = float(tokens[3])
            E = float(tokens[4])
            G = float(tokens[5])
            J = float(tokens[6])
        except ValueError:
            errori.append((lineno, "material inline richiede: material <id> <nome> <densita> <E> <G> <J>"))
            return
        dati["materiali"][mid] = {
            "nome": nome, "tipo": "inline",
            "densita": densita, "E": E, "G": G, "J": J,
        }
    else:
        errori.append((lineno, "material: usa 'material <id> <nome>' oppure 'material <id> <nome> <densita> <E> <G> <J>'"))


def _parse_section(tokens, dati, errori, lineno):
    # Riferimento:  section <id> <nome>
    # Inline:       section <id> <nome> <Area> <Iy> <Iz>  material: <id_o_nome>
    if len(tokens) < 3:
        errori.append((lineno, "section richiede almeno: section <id> <nome>"))
        return
    try:
        sid = int(tokens[1])
    except ValueError:
        errori.append((lineno, f"section: id deve essere intero, ricevuto '{tokens[1]}'"))
        return
    nome = _strip_quotes(tokens[2])
    if sid in dati["sezioni"]:
        errori.append((lineno, f"Sezione {sid} già definita"))
        return

    has_material = "material:" in tokens
    if not has_material:
        # Riferimento puro
        if len(tokens) != 3:
            errori.append((lineno,
                "section riferimento: 'section <id> <nome>' "
                "(per la forma inline serve 'material: <id_o_nome>' alla fine)"))
            return
        dati["sezioni"][sid] = {"nome": nome, "tipo": "riferimento"}
        return

    # Inline: i token prima di 'material:' devono essere [section, id, nome, A, Iy, Iz]
    idx_mat = tokens.index("material:")
    if idx_mat != 6:
        errori.append((lineno,
            "section inline: 'section <id> <nome> <Area> <Iy> <Iz>  material: <id_o_nome>'"))
        return
    try:
        Area = float(tokens[3])
        Iy = float(tokens[4])
        Iz = float(tokens[5])
    except ValueError:
        errori.append((lineno, "section inline: Area, Iy, Iz devono essere numerici"))
        return
    materiale = _extract_colon_param(tokens, "material")
    dati["sezioni"][sid] = {
        "nome": nome, "tipo": "inline",
        "Area": Area, "Iy": Iy, "Iz": Iz,
        "materiale": materiale,
    }


def _valida_riferimenti(dati, errori):
    """Controlla riferimenti incrociati: nodi, materiali e sezioni."""
    nodi_ids = set(dati["nodi"].keys())
    aste_ids = set(dati["aste"].keys())

    # Set di riferimenti validi (id come stringa + nome) per materiali e sezioni
    mat_validi = set()
    for mid, m in dati["materiali"].items():
        mat_validi.add(str(mid))
        mat_validi.add(_norm_nome(m["nome"]))
    sez_validi = set()
    for sid, s in dati["sezioni"].items():
        sez_validi.add(str(sid))
        sez_validi.add(_norm_nome(s["nome"]))

    def _ref_match(val: str, pool: set) -> bool:
        return val in pool or _norm_nome(val) in pool

    for bid, asta in dati["aste"].items():
        if asta["nodo_i"] not in nodi_ids:
            errori.append((0, f"Asta {bid}: nodo_i={asta['nodo_i']} non definito"))
        if asta["nodo_j"] not in nodi_ids:
            errori.append((0, f"Asta {bid}: nodo_j={asta['nodo_j']} non definito"))
        sez = asta.get("sezione", "")
        if sez and not _ref_match(sez, sez_validi):
            errori.append((0, f"Asta {bid}: sezione '{sez}' non definita"))

    for sid, sh in dati["shell"].items():
        for nid in sh["nodi"]:
            if nid not in nodi_ids:
                errori.append((0, f"Shell {sid}: nodo {nid} non definito"))
        mat = sh.get("materiale", "")
        if mat and not _ref_match(mat, mat_validi):
            errori.append((0, f"Shell {sid}: materiale '{mat}' non definito"))

    for sid, s in dati["sezioni"].items():
        if s.get("tipo") == "inline":
            mat = s.get("materiale", "")
            if mat and not _ref_match(mat, mat_validi):
                errori.append((0, f"Sezione {sid}: materiale '{mat}' non definito"))

    for nid in dati["vincoli"]:
        if nid not in nodi_ids:
            errori.append((0, f"Vincolo: nodo {nid} non definito"))

    for (nid, *_) in dati["carichi_nodali"]:
        if nid not in nodi_ids:
            errori.append((0, f"nodeLoad: nodo {nid} non definito"))

    for (bid, *_) in dati["carichi_distribuiti"]:
        if bid not in aste_ids:
            errori.append((0, f"beamLoad: asta {bid} non definita"))

    shell_ids = set(dati["shell"].keys())
    for (sid, *_) in dati["carichi_shell"]:
        if sid not in shell_ids:
            errori.append((0, f"shellLoad: shell {sid} non definita"))


# ================================================================
#  LINE NUMBER AREA
# ================================================================

class _LineNumberArea(QWidget):
    """Gutter laterale che mostra i numeri di riga."""

    def __init__(self, editor: QPlainTextEdit):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):
        return QSize(self._editor._line_number_width(), 0)

    def paintEvent(self, event):
        self._editor._paint_line_numbers(event)


class StrutturaTextEdit(QPlainTextEdit):
    """QPlainTextEdit con numeri di riga integrati (stile IDE)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._line_area = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_area_width)
        self.updateRequest.connect(self._update_line_area)
        self._update_line_area_width()

    def _line_number_width(self) -> int:
        digits = max(1, len(str(self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_line_area_width(self, _=0):
        # Calcola la larghezza della barra dei numeri
        gutter_width = self._line_number_width()
        
        # Imposta il margine sinistro aggiungendo 10 pixel di spazio vuoto (padding)
        self.setViewportMargins(gutter_width, 0, 0, 0)

    def _update_line_area(self, rect, dy):
        if dy:
            self._line_area.scroll(0, dy)
        else:
            self._line_area.update(0, rect.y(),
                                   self._line_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_area_width()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        cr = self.contentsRect()
        self._line_area.setGeometry(
            QRect(cr.left(), cr.top(),
                  self._line_number_width(), cr.height()))

    def _paint_line_numbers(self, event):
        painter = QPainter(self._line_area)
        painter.fillRect(event.rect(), QColor(50,50,50))

        block = self.firstVisibleBlock()
        block_num = block.blockNumber()
        top = round(self.blockBoundingGeometry(block)
                    .translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())

        painter.setFont(self.font())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                num = str(block_num + 1)
                if block == self.textCursor().block():
                    painter.setPen(QColor(200, 200, 200))
                else:
                    painter.setPen(QColor(100, 100, 100))
                painter.drawText(0, top,
                                 self._line_area.width() - 6,
                                 self.fontMetrics().height(),
                                 Qt.AlignRight | Qt.AlignVCenter, num)
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            block_num += 1

        painter.end()


# ================================================================
#  MANAGER TESTO + CONTROLLO ERRORI
# ================================================================

class TextoStrutturaManager:
    """
    Gestisce il QPlainTextEdit e il widget text_control.
    Installa l'highlighter e mostra i feedback.
    """

    def __init__(self, ui, main_window):
        self._ui = ui
        self._main = main_window

        self._control: QWidget = ui.text_control

        # Sostituisci il QPlainTextEdit standard con StrutturaTextEdit
        # (che include i numeri di riga)
        old_editor = ui.struttura_plainTextEdit

        self._editor = StrutturaTextEdit(old_editor.parentWidget())
        self._editor.setObjectName("struttura_plainTextEdit")
        self._editor.setMinimumSize(old_editor.minimumSize())
        self._editor.setMaximumSize(old_editor.maximumSize())
        # 1. Stile solo per l'editor di testo
        stile_editor = """
        QPlainTextEdit {
            background-color: rgb(50, 50, 50);
            color: rgb(212, 212, 212);
            selection-background-color: rgb(38, 79, 120);
            selection-color: rgb(255, 255, 255);
            border: 1px solid rgb(120, 120, 120);
            border-left: 3px solid rgb(120, 120, 120);
            border-top: none;
            padding: 4px;
        }
        """
        self._editor.setStyleSheet(stile_editor)

        # 2. Stile per la scrollbar VERTICALE
        stile_scrollbar_vert = """
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
        """
        self._editor.verticalScrollBar().setStyleSheet(stile_scrollbar_vert)

        # 3. Stile per la scrollbar ORIZZONTALE
        stile_scrollbar_orizz = """
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
        self._editor.horizontalScrollBar().setStyleSheet(stile_scrollbar_orizz)

        # Il widget si trova in verticalLayout_3 (non nel layout del parent)
        vlay = ui.verticalLayout_3
        idx = vlay.indexOf(old_editor)
        vlay.removeWidget(old_editor)
        old_editor.setParent(None)
        old_editor.deleteLater()
        vlay.insertWidget(idx, self._editor)

        # Aggiorna il riferimento nell'UI
        ui.struttura_plainTextEdit = self._editor

        # Font monospazio per l'editor
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.Monospace)
        self._editor.setFont(font)

        self._editor.setTabStopDistance(32)
        self._editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._editor._update_line_area_width()

        # Highlighter
        self._highlighter = StrutturaSyntaxHighlighter(self._editor.document())

        # Cache nomi globali (database programma + progetto)
        self._mat_nomi_globali: set = set()
        self._sez_nomi_globali: set = set()
        self._refresh_nomi_globali()

        # Debounce per ricalcolo definizioni locali su modifica testo
        self._refresh_timer = QTimer(self._editor)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self._aggiorna_nomi_validi)
        self._editor.textChanged.connect(
            lambda: self._refresh_timer.start()
        )

        # Prima illuminazione
        self._aggiorna_nomi_validi()

        # Setup area feedback errori
        self._setup_text_control()

    def _setup_text_control(self):
        lay = self._control.layout()
        if lay is None:
            lay = QVBoxLayout(self._control)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setSpacing(4)

        self._label_feedback = QLabel("Pronto.")
        self._label_feedback.setWordWrap(True)
        
        # --- AGGIUNGI QUESTA RIGA ---
        self._label_feedback.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # ----------------------------

        self._label_feedback.setStyleSheet(
            "color: rgb(180, 180, 180);"
            "font: 9pt 'Consolas';"
            "border: none;"
        )
        self._label_feedback.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lay.addWidget(self._label_feedback)

    def _refresh_nomi_globali(self):
        """Aggiorna la cache dei nomi globali (database programma + progetto)."""
        nomi_mat = set()
        nomi_sez = set()
        try:
            from materiali.database_materiali import carica_database as db_mat
            db_m = db_mat()
            for cat in ("calcestruzzo", "barre", "acciaio", "personalizzati"):
                for nome in db_m.get(cat, {}):
                    nomi_mat.add(nome)
            from sezioni.database_sezioni import carica_database as db_sez
            db_s = db_sez()
            for cat in ("calcestruzzo_armato", "profili", "precompresso", "personalizzate"):
                for nome in db_s.get(cat, {}):
                    nomi_sez.add(nome)
            if self._main.ha_progetto():
                mat_prj = self._main.get_sezione("materiali") or {}
                for cat in ("calcestruzzo", "barre", "acciaio", "personalizzati"):
                    for nome in mat_prj.get(cat, {}):
                        nomi_mat.add(nome)
                sez_prj = self._main.get_sezione("sezioni") or {}
                for cat in ("calcestruzzo_armato", "profili", "precompresso", "personalizzate"):
                    for nome in sez_prj.get(cat, {}):
                        nomi_sez.add(nome)
        except Exception:
            pass
        self._mat_nomi_globali = nomi_mat
        self._sez_nomi_globali = nomi_sez

    def _aggiorna_nomi_validi(self):
        """
        Esegue una scansione veloce del testo per estrarre le definizioni
        locali di materiali e sezioni e le passa all'highlighter così che
        possa illuminare in verde i riferimenti riconosciuti.
        """
        testo = self._editor.toPlainText()
        mat_definiti: dict = {}   # {id_str: nome_str}
        sez_definiti: dict = {}
        for raw in testo.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Rimuovi commento inline
            quote_ch = None
            for i, ch in enumerate(line):
                if quote_ch:
                    if ch == quote_ch:
                        quote_ch = None
                elif ch in ('"', "'"):
                    quote_ch = ch
                elif ch == '#':
                    line = line[:i].strip()
                    break
            if not line:
                continue
            toks = _tokenize(line)
            if len(toks) < 3:
                continue
            cmd = toks[0]
            if cmd not in ("material", "section"):
                continue
            try:
                _id = int(toks[1])
            except ValueError:
                continue
            nome = _strip_quotes(toks[2])
            target = mat_definiti if cmd == "material" else sez_definiti
            target[str(_id)] = nome
        self._highlighter.set_nomi_validi(
            self._mat_nomi_globali, self._sez_nomi_globali,
            mat_definiti, sez_definiti,
        )

    def get_testo(self) -> str:
        return self._editor.toPlainText()

    def set_testo(self, testo: str):
        self._editor.blockSignals(True)
        self._editor.setPlainText(testo)
        self._editor.blockSignals(False)

        # AGGIUNTA: Forza il ricalcolo dei margini ora che il testo è caricato
        self._editor._update_line_area_width()

    # ------------------------------------------------------------------
    #  Selezione riga corrispondente ad un oggetto 3D
    # ------------------------------------------------------------------

    def seleziona_oggetto(self, kind: str, oid: int):
        """Evidenzia la riga di definizione dell'oggetto (kind, oid) nell'editor.

        kind ∈ {"nodo", "beam", "shell"}. L'id è quello numerico dichiarato
        nel testo. La riga viene colorata di un soft blu e scrollata in vista.
        """
        patterns = {
            "nodo":  rf"^\s*node\s+{oid}\b",
            "beam":  rf"^\s*beam\s+{oid}\b",
            "shell": rf"^\s*shell\s+{oid}\b",
        }
        pat = patterns.get(kind)
        if pat is None:
            self._clear_row_selection()
            return

        doc = self._editor.document()
        testo = self._editor.toPlainText()
        m = re.search(pat, testo, re.MULTILINE)
        if m is None:
            self._clear_row_selection()
            return

        # Converti indice carattere → riga
        prefix = testo[:m.start()]
        line_no = prefix.count("\n")

        block = doc.findBlockByNumber(line_no)
        if not block.isValid():
            self._clear_row_selection()
            return

        cursor = QTextCursor(block)

        sel = QTextEdit.ExtraSelection()
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(80, 125, 180, 90))   # soft blu
        fmt.setProperty(QTextCharFormat.FullWidthSelection, True)
        sel.format = fmt
        sel.cursor = cursor
        self._editor.setExtraSelections([sel])

        # Sposta il cursore e scrolla in vista (senza rubare il focus)
        self._editor.setTextCursor(cursor)
        self._editor.ensureCursorVisible()

    def _clear_row_selection(self):
        self._editor.setExtraSelections([])

    def parse_e_valida(self) -> tuple:
        """Esegue il parsing e aggiorna il widget text_control con i risultati."""
        # Aggiorna nomi globali (può essere cambiato il progetto) e definizioni locali
        self._refresh_nomi_globali()
        self._aggiorna_nomi_validi()

        testo = self.get_testo()
        dati, errori = parse_struttura(testo)

        # Genera feedback
        lines = []
        n_mat  = len(dati["materiali"])
        n_sez  = len(dati["sezioni"])
        n_nodi = len(dati["nodi"])
        n_aste = len(dati["aste"])
        n_shell = len(dati["shell"])
        n_vinc = len(dati["vincoli"])
        n_cn = len(dati["carichi_nodali"])
        n_cd = len(dati["carichi_distribuiti"])

        lines.append(f"<b style='color:#569cd6'>Riepilogo:</b>  "
                     f"Materiali: {n_mat}  |  Sezioni: {n_sez}  |  "
                     f"Nodi: {n_nodi}  |  Aste: {n_aste}  |  Shell: {n_shell}  |  "
                     f"Vincoli: {n_vinc}  |  Carichi nodali: {n_cn}  |  "
                     f"Carichi distribuiti: {n_cd}")

        if errori:
            lines.append("")
            lines.append(f"<b style='color:#f44747'>Errori ({len(errori)}):</b>")
            for riga, msg in errori:
                prefix = f"Riga {riga}: " if riga > 0 else ""
                lines.append(f"<span style='color:#f44747'>  • {prefix}{msg}</span>")
        else:
            lines.append("")
            lines.append("<span style='color:#6a9955'>✓ Nessun errore rilevato.</span>")

        # Verifica nodi liberi (senza vincoli e senza connessioni)
        nodi_connessi = set()
        for asta in dati["aste"].values():
            nodi_connessi.add(asta["nodo_i"])
            nodi_connessi.add(asta["nodo_j"])
        for sh in dati["shell"].values():
            nodi_connessi.update(sh["nodi"])
        nodi_isolati = set(dati["nodi"].keys()) - nodi_connessi
        if nodi_isolati:
            lines.append(f"<span style='color:#ce9178'>⚠ Nodi isolati (non connessi): "
                         f"{sorted(nodi_isolati)}</span>")

        self._label_feedback.setText("<br>".join(lines))
        return dati, errori
