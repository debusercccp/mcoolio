"""TUI per esplorare mappe di contatto Hi-C in formato .mcool.

La heatmap usa il carattere "▀": colore del testo = pixel superiore, sfondo = pixel inferiore.
Una cella del terminale è alta circa il doppio della larghezza, quindi i pixel risultano quadrati.
"""

import argparse
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

if not __package__:  # lanciato come script (python src/mcoolio/app.py): rende importabile il pacchetto
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "mcoolio"

import numpy as np
from rich.color import Color
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Footer, Input, Static

from .colormaps import COLORMAPS
from .fasta import Fasta, GCProfile, sniff
from .mcool import MCool

LEFT_W = 12
OUTSIDE_RGB = (24, 24, 28)
MASKED_RGB = (150, 150, 160)
BOUNDARY_RGB = (90, 110, 160)


@dataclass(frozen=True)
class View:
    x0: float  # coordinata globale (bp) del bordo sinistro
    y0: float  # coordinata globale (bp) del bordo superiore
    s: float  # bp per pixel (uguale sui due assi)
    w: int  # pixel orizzontali
    h: int  # pixel verticali (2 per riga di testo)
    binsize: int | None  # None = automatica
    norm: str
    cmap: str
    log: bool
    sat: float
    boundaries: bool


@dataclass
class Frame:
    view: View
    binsize: int
    strips: list[Strip]
    values: np.ndarray  # (h, w) valori campionati, NaN = mascherato / fuori
    gc_x: np.ndarray | None = None  # GC per colonna di pixel
    gc_y: np.ndarray | None = None  # GC per riga di testo (2 pixel)


# ---------------------------------------------------------------- utilità


def fmt_bp(v: float, step: float | None = None) -> str:
    """Formatta una posizione con le cifre decimali che servono a distinguere passi di ampiezza `step`."""
    if v == 0:
        return "0"
    mag = max(abs(v), step or 0)
    unit, suffix = (1e6, "M") if mag >= 1e6 else (1e3, "k") if mag >= 1e3 else (1, "")
    if not step:
        return f"{v / unit:g}{suffix}"
    decimals = max(0, int(np.ceil(-np.log10(step / unit) - 1e-9)))
    return f"{v / unit:.{decimals}f}{suffix}"


def nice_step(target: float) -> float:
    if target <= 0:
        return 1
    p = 10 ** np.floor(np.log10(target))
    for m in (1, 2, 5, 10):
        if m * p >= target:
            return m * p
    return 10 * p


def parse_bp(tok: str) -> float:
    m = re.fullmatch(r"([\d.]+)\s*([kmg]?)b?", tok.strip().lower().replace(",", "").replace("_", ""))
    if not m:
        raise ValueError(f"posizione non valida: {tok!r}")
    return float(m.group(1)) * {"": 1, "k": 1e3, "m": 1e6, "g": 1e9}[m.group(2)]


class Locator:
    def __init__(self, mc: MCool):
        self.mc = mc
        self.lookup = {}
        for i, n in enumerate(mc.chrom_names):
            self.lookup[n.lower()] = i
            self.lookup.setdefault(n.lower().removeprefix("chr"), i)

    def parse_one(self, s: str) -> tuple[float, float]:
        name, _, rng = s.partition(":")
        ci = self.lookup.get(name.lower(), self.lookup.get(name.lower().removeprefix("chr")))
        if ci is None:
            raise ValueError(f"cromosoma sconosciuto: {name!r}")
        cs, cl = int(self.mc.chrom_starts[ci]), int(self.mc.chrom_lengths[ci])
        if not rng:
            return cs, cs + cl
        a, _, b = rng.partition("-")
        lo = parse_bp(a)
        hi = parse_bp(b) if b else lo + 1
        lo, hi = max(0, min(lo, hi)), min(cl, max(lo, hi))
        return cs + lo, cs + max(hi, lo + 1)

    def parse(self, s: str):
        """'chr1', 'chr1:10M-20M', 'chr1:1-50M chr2' (righe, colonne)."""
        parts = [p for p in re.split(r"[\s;]+|\s+x\s+", s.strip()) if p and p.lower() != "x"]
        if not parts or len(parts) > 2:
            raise ValueError("usa: chr1 | chr1:10M-20M | chr1:10M-20M chr2:5M-15M")
        rows = self.parse_one(parts[0])
        cols = self.parse_one(parts[1]) if len(parts) == 2 else rows
        return rows, cols

    def describe(self, g0: float, g1: float) -> str:
        g0, g1 = max(0, g0), min(self.mc.genome_length, g1)
        if g1 <= g0:
            return "—"
        c0 = int(self.mc.chrom_index(np.array([g0]))[0])
        c1 = int(self.mc.chrom_index(np.array([g1 - 1]))[0])
        if c0 == c1:
            cs = self.mc.chrom_starts[c0]
            span = g1 - g0
            return f"{self.mc.chrom_names[c0]}:{fmt_bp(g0 - cs, span / 100)}-{fmt_bp(g1 - cs, span / 100)}"
        if c0 == 0 and c1 == len(self.mc.chrom_names) - 1 and g0 == 0:
            return "genoma"
        return f"{self.mc.chrom_names[c0]}…{self.mc.chrom_names[c1]}"

    def point(self, g: float) -> str:
        ci = int(self.mc.chrom_index(np.array([g]))[0])
        if ci < 0:
            return "—"
        return f"{self.mc.chrom_names[ci]}:{int(g - self.mc.chrom_starts[ci]):,}"


# ---------------------------------------------------------------- rendering


def render_frame(mc: MCool, v: View, gc: GCProfile | None = None) -> Frame:
    xs = v.x0 + (np.arange(v.w) + 0.5) * v.s
    ys = v.y0 + (np.arange(v.h) + 0.5) * v.s
    # In automatico: la risoluzione più fine con bin >= pixel (un bin per pixel, niente aliasing).
    want = v.binsize if v.binsize else v.s
    res = mc.pick_resolution(int(want), v.h * v.s, v.w * v.s)

    norm = v.norm if v.norm in res.norms else "none"
    header, mat = mc.matrix(res.binsize, v.y0, v.y0 + v.h * v.s, v.x0, v.x0 + v.w * v.s, norm)
    ri = mc.bin_ids(res, ys) - header["rows"]["b0"]
    ci = mc.bin_ids(res, xs) - header["cols"]["b0"]
    rok = (ri >= 0) & (ri < mat.shape[0])
    cok = (ci >= 0) & (ci < mat.shape[1])

    vals = np.full((v.h, v.w), np.nan, dtype=np.float32)
    if mat.size:
        sub = mat[np.clip(ri, 0, mat.shape[0] - 1)][:, np.clip(ci, 0, mat.shape[1] - 1)]
        vals = np.where(rok[:, None] & cok[None, :], sub, np.nan).astype(np.float32)
    inside = rok[:, None] & cok[None, :]

    rgb = colorize(vals, inside, v)

    if v.boundaries:
        min_px = mc.chrom_lengths / v.s >= 6  # niente linee tra cromosomi larghi pochi pixel
        for axis, coords in ((1, xs), (0, ys)):
            c = mc.chrom_index(coords)
            b = np.flatnonzero((np.diff(c) != 0) & (c[:-1] >= 0) & (c[1:] >= 0)) + 1
            b = b[min_px[c[b - 1]] & min_px[c[b]]]
            if axis == 1:
                rgb[:, b] = np.where(inside[:, b, None], BOUNDARY_RGB, rgb[:, b])
            else:
                rgb[b, :] = np.where(inside[b, :, None], BOUNDARY_RGB, rgb[b, :])

    frame = Frame(view=v, binsize=res.binsize, strips=to_strips(rgb), values=vals)
    if gc is not None:
        frame.gc_x = gc.fraction(v.x0 + np.arange(v.w + 1) * v.s)
        frame.gc_y = gc.fraction(v.y0 + np.arange((v.h + 1) // 2 + 1) * 2 * v.s)
    return frame


GC_LUT = COLORMAPS["viridis"]
VBARS = " ▁▂▃▄▅▆▇█"
HBARS = " ▏▎▍▌▋▊▉█"


def gc_bars(values: np.ndarray, lo: float, hi: float, chars: str, sep: str) -> Text:
    """Traccia GC come barre a blocchi colorate (altezza e colore crescono con il GC)."""
    t = Text()
    span = max(hi - lo, 1e-6)
    for k, g in enumerate(values):
        if k:
            t.append(sep)
        if np.isnan(g):
            t.append(" ")
            continue
        u = float(np.clip((g - lo) / span, 0, 1))
        r, gg, b = GC_LUT[int(u * 255)]
        t.append(chars[1 + int(round(u * (len(chars) - 2)))], Style(color=Color.from_rgb(int(r), int(gg), int(b))))
    return t


def colorize(vals: np.ndarray, inside: np.ndarray, v: View) -> np.ndarray:
    lut = COLORMAPS[v.cmap]
    finite = np.isfinite(vals)
    pos = vals[finite & (vals > 0)]
    t = np.zeros(vals.shape, dtype=np.float64)
    if pos.size:
        vmax = float(np.quantile(pos, v.sat))
        if v.log:
            vmin = float(np.quantile(pos, 0.02))
            vmin = min(vmin, vmax / 10) if vmax > 0 else vmin
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.log(np.where(vals > 0, vals, np.nan) / vmin) / np.log(vmax / vmin)
        else:
            t = vals / vmax if vmax > 0 else t
    t = np.nan_to_num(t, nan=0.0)
    idx = (np.clip(t, 0, 1) * 255).astype(np.uint8)
    rgb = lut[idx]
    rgb[inside & ~finite] = MASKED_RGB
    rgb[~inside] = OUTSIDE_RGB
    return rgb


_style_cache: dict[bytes, Style] = {}


def to_strips(rgb: np.ndarray) -> list[Strip]:
    h, w, _ = rgb.shape
    if h % 2:
        rgb = np.concatenate([rgb, np.full((1, w, 3), OUTSIDE_RGB, np.uint8)])
    if len(_style_cache) > 200_000:
        _style_cache.clear()
    strips = []
    for r in range(0, rgb.shape[0], 2):
        pair = np.concatenate([rgb[r], rgb[r + 1]], axis=1)  # (w, 6)
        segs = []
        prev_key, run = None, 0
        for c in range(w):
            key = pair[c].tobytes()
            if key == prev_key:
                run += 1
                continue
            if prev_key is not None:
                segs.append(Segment("▀" * run, _style_cache[prev_key]))
            if key not in _style_cache:
                a = pair[c]
                _style_cache[key] = Style(
                    color=Color.from_rgb(int(a[0]), int(a[1]), int(a[2])),
                    bgcolor=Color.from_rgb(int(a[3]), int(a[4]), int(a[5])),
                )
            prev_key, run = key, 1
        if prev_key is not None:
            segs.append(Segment("▀" * run, _style_cache[prev_key]))
        strips.append(Strip(segs, w))
    return strips


# ---------------------------------------------------------------- widget


class Heatmap(Widget, can_focus=True):
    DEFAULT_CSS = "Heatmap { width: 1fr; height: 1fr; }"

    def __init__(self, viewer: "Viewer", **kw):
        super().__init__(**kw)
        self.viewer = viewer
        self._drag = None

    def render_line(self, y: int) -> Strip:
        f = self.viewer.frame
        if f is None or y >= len(f.strips):
            return Strip.blank(self.size.width)
        return f.strips[y]

    def on_resize(self, event: events.Resize) -> None:
        self.viewer.on_heatmap_resize(event.size.width, event.size.height * 2)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self.viewer.zoom(1 / 1.25, event.x, event.y * 2 + 1)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self.viewer.zoom(1.25, event.x, event.y * 2 + 1)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._drag = (event.x, event.y, self.viewer.view)
        self.capture_mouse()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self._drag = None
        self.release_mouse()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._drag:
            x, y, v = self._drag
            self.viewer.set_view(replace(v, x0=v.x0 - (event.x - x) * v.s, y0=v.y0 - (event.y - y) * 2 * v.s))
        self.viewer.hover(event.x, event.y * 2)

    def on_leave(self, event: events.Leave) -> None:
        self.viewer.hover(None, None)


class Viewer(App):
    TITLE = "mcoolio"
    CSS = f"""
    #top {{ height: 2; }}
    #corner {{ width: {LEFT_W + 1}; color: $text-muted; }}
    #xruler {{ width: 1fr; color: $text; }}
    #yruler {{ width: {LEFT_W}; color: $text; }}
    #gcrow {{ height: 1; display: none; }}
    #gcrow.visible {{ display: block; }}
    #gclabel {{ width: {LEFT_W + 1}; color: $text-muted; }}
    #gcx {{ width: 1fr; }}
    #gcy {{ width: 1; }}
    #status {{ height: 1; background: $panel; color: $text; padding: 0 1; }}
    #hover {{ height: 1; background: $panel; color: $text-muted; padding: 0 1; }}
    #goto {{ display: none; }}
    #goto.visible {{ display: block; }}
    """
    BINDINGS = [
        Binding("q", "quit", "esci"),
        Binding("g,slash", "goto", "vai a"),
        Binding("0,home", "genome", "genoma"),
        Binding("plus,equals_sign", "zoom(0.7)", "zoom +"),
        Binding("minus", "zoom(1.4)", "zoom −"),
        Binding("left,h", "pan(-0.15,0)", "←", show=False),
        Binding("right,l", "pan(0.15,0)", "→", show=False),
        Binding("up,k", "pan(0,-0.15)", "↑", show=False),
        Binding("down,j", "pan(0,0.15)", "↓", show=False),
        Binding("shift+left,H", "pan(-0.5,0)", show=False),
        Binding("shift+right,L", "pan(0.5,0)", show=False),
        Binding("shift+up,K", "pan(0,-0.5)", show=False),
        Binding("shift+down,J", "pan(0,0.5)", show=False),
        Binding("d", "diagonal", "diagonale"),
        Binding("n", "norm", "norm."),
        Binding("c", "cmap", "colori"),
        Binding("s", "scale", "log/lin"),
        Binding("left_square_bracket", "sat(-1)", "sat −"),
        Binding("right_square_bracket", "sat(1)", "sat +"),
        Binding("r", "res(-1)", "ris. fine"),
        Binding("R", "res(1)", "ris. grossa"),
        Binding("a", "res_auto", "ris. auto"),
        Binding("b", "boundaries", "bordi", show=False),
        Binding("escape", "close_goto", show=False),
    ]

    SATS = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0]

    def __init__(self, mc: MCool, region: str | None, norm: str, cmap: str, fasta: Fasta | None = None):
        super().__init__()
        self.mc = mc
        self.fasta = fasta
        self.gc: GCProfile | None = None
        self._gc_note = ""
        self.loc = Locator(mc)
        self.binsizes = sorted(mc.resolutions)
        self.norms = ["none"] + next(iter(mc.resolutions.values())).norms
        self.initial_region = region
        self.view = View(0, 0, 1, 0, 0, None, norm if norm in self.norms else "none",
                         cmap, True, 0.98, True)
        self.frame: Frame | None = None
        self._pending: View | None = None
        self._busy = False
        self._force = False
        self._hover_xy = (None, None)

    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            yield Static(id="corner")
            yield Static(id="xruler")
        with Horizontal(id="gcrow"):
            yield Static(id="gclabel")
            yield Static(id="gcx")
        with Horizontal():
            yield Static(id="yruler")
            yield Static(id="gcy")
            yield Heatmap(self, id="heat")
        yield Static(id="hover")
        yield Static(id="status")
        yield Input(placeholder="chr1  |  chr1:10M-20M  |  chr1:10M-20M chr2:5M-15M   (Esc annulla)", id="goto")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = self.mc.path + (f"  +  {self.fasta.path}" if self.fasta else "")
        self.query_one(Heatmap).focus()
        if self.fasta:
            self.query_one("#gcrow").add_class("visible")
            self.check_assembly()
            self._gc_note = "GC: calcolo…"
            self._load_gc()

    def check_assembly(self) -> None:
        missing, differ = [], []
        for name, length in zip(self.mc.chrom_names, self.mc.chrom_lengths):
            fl = self.fasta.length(name)
            if fl is None:
                missing.append(name)
            elif fl != length:
                differ.append(name)
        if differ:
            self.notify(f"Lunghezze diverse tra .mcool e FASTA per {', '.join(differ[:6])}"
                        f"{'…' if len(differ) > 6 else ''}: forse un altro assembly?", severity="warning", timeout=10)
        if missing:
            self.notify(f"Cromosomi assenti dal FASTA: {', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}",
                        severity="warning", timeout=10)

    @work(thread=True, exit_on_error=False)
    def _load_gc(self) -> None:
        def progress(k, n, name):
            self.call_from_thread(self._set_gc_note, f"GC: calcolo {name} ({k + 1}/{n}, solo la prima volta)")
        try:
            prof = self.fasta.gc_profile(self.mc.chrom_names, self.mc.chrom_lengths, progress)
        except Exception as e:  # noqa: BLE001 - mostrato all'utente
            self.call_from_thread(self.notify, f"profilo GC non disponibile: {e}", severity="error")
            self.call_from_thread(self._set_gc_note, "")
            return
        self.call_from_thread(self._gc_ready, prof)

    def _set_gc_note(self, note: str) -> None:
        self._gc_note = note
        self.update_status()

    def _gc_ready(self, prof: GCProfile) -> None:
        self.gc = prof
        self._set_gc_note("")
        self._force = True  # nuovo frame anche se la vista non è cambiata
        self.request_render()

    # ---- gestione vista

    def on_heatmap_resize(self, w: int, h: int) -> None:
        first = self.view.w == 0
        v = self.view
        if not first:  # mantiene il centro
            cx, cy = v.x0 + v.w * v.s / 2, v.y0 + v.h * v.s / 2
            v = replace(v, w=w, h=h, x0=cx - w * v.s / 2, y0=cy - h * v.s / 2)
            self.set_view(v)
            return
        self.view = replace(v, w=w, h=h)
        if self.initial_region:
            try:
                self.fit(*self.loc.parse(self.initial_region))
                return
            except ValueError as e:
                self.notify(str(e), severity="error")
        self.action_genome()

    def clamp(self, v: View) -> View:
        if v.w == 0 or v.h == 0:
            return v
        smin = self.binsizes[0] / 16
        smax = self.mc.genome_length / min(v.w, v.h) * 1.5
        s = float(np.clip(v.s, smin, smax))
        if s != v.s:  # ricentra rispetto alla vecchia scala
            cx, cy = v.x0 + v.w * v.s / 2, v.y0 + v.h * v.s / 2
            v = replace(v, s=s, x0=cx - v.w * s / 2, y0=cy - v.h * s / 2)
        G = self.mc.genome_length
        x0 = float(np.clip(v.x0, -v.w * v.s * 0.8, G - v.w * v.s * 0.2))
        y0 = float(np.clip(v.y0, -v.h * v.s * 0.8, G - v.h * v.s * 0.2))
        return replace(v, x0=x0, y0=y0)

    def set_view(self, v: View) -> None:
        self.view = self.clamp(v)
        self.update_rulers()
        self.update_status()
        self.request_render()

    def fit(self, rows: tuple[float, float], cols: tuple[float, float]) -> None:
        v = self.view
        s = max((cols[1] - cols[0]) / v.w, (rows[1] - rows[0]) / v.h)
        cx, cy = (cols[0] + cols[1]) / 2, (rows[0] + rows[1]) / 2
        self.set_view(replace(v, s=s, x0=cx - v.w * s / 2, y0=cy - v.h * s / 2))

    def zoom(self, f: float, px: float, py: float) -> None:
        v = self.view
        gx, gy = v.x0 + px * v.s, v.y0 + py * v.s
        s = v.s * f
        self.set_view(replace(v, s=s, x0=gx - px * s, y0=gy - py * s))

    # ---- rendering asincrono: al massimo un worker alla volta, l'ultima vista vince

    def request_render(self) -> None:
        self._pending = self.view
        if not self._busy:
            self._busy = True
            self._force = False
            self._render_worker(self._pending)

    @work(thread=True, exit_on_error=False)
    def _render_worker(self, v: View) -> None:
        try:
            frame = render_frame(self.mc, v, self.gc)
        except Exception as e:  # noqa: BLE001 - mostrato all'utente
            self.call_from_thread(self._render_failed, v, e)
            return
        self.call_from_thread(self._render_done, frame)

    def _render_done(self, frame: Frame) -> None:
        self.frame = frame
        self.query_one(Heatmap).refresh()
        self.update_gc_tracks()
        self._next_render(frame.view)

    def update_gc_tracks(self) -> None:
        f = self.frame
        if f is None or f.gc_x is None:
            return
        both = np.concatenate([f.gc_x, f.gc_y])
        both = both[np.isfinite(both)]
        if both.size == 0:
            lo, hi = 0.3, 0.6
        else:  # contrasto adattato alla vista, ma almeno 10 punti percentuali di escursione
            lo, hi = np.quantile(both, [0.02, 0.98])
            mid = (lo + hi) / 2
            lo, hi = min(lo, mid - 0.05), max(hi, mid + 0.05)
        self.query_one("#gcx", Static).update(gc_bars(f.gc_x, lo, hi, VBARS, ""))
        self.query_one("#gcy", Static).update(gc_bars(f.gc_y, lo, hi, HBARS, "\n"))
        self.query_one("#gclabel", Static).update(Text(f"GC {lo * 100:.0f}–{hi * 100:.0f}%"))

    def _render_failed(self, v: View, e: Exception) -> None:
        self.notify(f"errore: {e}", severity="error")
        self._next_render(v)

    def _next_render(self, done: View) -> None:
        if self._pending is not None and (self._pending != done or self._force):
            self._force = False
            self._render_worker(self._pending)
        else:
            self._busy = False
        self.update_status()
        self.hover(*self._hover_xy)

    # ---- righelli e barre

    def _ticks(self, g0: float, g1: float, npx: int, every: int):
        """(pixel, etichetta) per tacche e inizi di cromosoma nell'intervallo [g0, g1)."""
        span = g1 - g0
        step = nice_step(span / max(1, npx / every))
        starts, ticks = [], []
        for name, cs, cl in zip(self.mc.chrom_names, self.mc.chrom_starts, self.mc.chrom_lengths):
            if cs + cl <= g0 or cs >= g1:
                continue
            if cs >= g0:
                starts.append((int((cs - g0) / span * npx), name))
            if cl / span * npx < every:  # cromosoma troppo stretto per le tacche
                continue
            first = max(step, np.ceil((g0 - cs) / step) * step)
            for p in np.arange(first, min(cl, g1 - cs), step):
                ticks.append((int((cs + p - g0) / span * npx), fmt_bp(p, step)))
        return starts, ticks

    def update_rulers(self) -> None:
        v = self.view
        if v.w == 0:
            return
        # asse x: riga 1 nomi dei cromosomi, riga 2 tacche
        names = [" "] * v.w
        marks = [" "] * v.w
        starts, ticks = self._ticks(v.x0, v.x0 + v.w * v.s, v.w, 12)
        if v.x0 > 0 and (not starts or starts[0][0] > 0):  # la vista inizia a metà cromosoma
            starts.insert(0, (0, self.loc.point(v.x0).split(":")[0]))
        for x, label in starts:
            _place(names, x, "▏")
        for x, label in starts:
            _place(names, x + 1, label, only_free=True)
        for x, label in ticks:
            if marks[x] == " ":
                _place(marks, x, "╵" + label, only_free=True)
        self.query_one("#xruler", Static).update(Text("".join(names) + "\n" + "".join(marks)))

        # asse y: una riga di testo = 2 pixel
        rows = v.h // 2
        lines = [""] * rows
        starts, ticks = self._ticks(v.y0, v.y0 + v.h * v.s, v.h, 6)
        for y, label in ticks:
            r = y // 2
            if r < rows and not lines[r]:
                lines[r] = f"{label} ╶".rjust(LEFT_W)
        for y, label in starts:
            r = y // 2
            if r < rows:
                lines[r] = f"{label[:LEFT_W - 3]} ━━".rjust(LEFT_W)
        t = Text("\n".join(lines))
        for i, line in enumerate(lines):
            if "━" in line:
                t.stylize("bold", sum(len(l) + 1 for l in lines[:i]), sum(len(l) + 1 for l in lines[:i + 1]))
        self.query_one("#yruler", Static).update(t)
        self.query_one("#corner", Static).update(Text("righe ↓\ncolonne →"))

    def update_status(self) -> None:
        v = self.view
        if v.w == 0:
            return
        rows = self.loc.describe(v.y0, v.y0 + v.h * v.s)
        cols = self.loc.describe(v.x0, v.x0 + v.w * v.s)
        region = rows if rows == cols else f"{rows} × {cols}"
        bs = self.frame.binsize if self.frame else None
        mode = "auto" if v.binsize is None else "fissa"
        busy = (" ⟳" if self._busy else "") + (f"   {self._gc_note}" if self._gc_note else "")
        st = Text()
        st.append(region, "bold")
        st.append(f"   ris {fmt_bp(bs) if bs else '?'} ({mode})   norm {v.norm}   {v.cmap}   "
                  f"{'log' if v.log else 'lin'}   sat {v.sat * 100:g}%{busy}")
        self.query_one("#status", Static).update(st)

    def hover(self, px: int | None, py: int | None) -> None:
        self._hover_xy = (px, py)
        out = self.query_one("#hover", Static)
        f = self.frame
        if px is None or f is None or not (0 <= py < f.values.shape[0] and 0 <= px < f.values.shape[1]):
            out.update("")
            return
        fv = f.view
        gx, gy = fv.x0 + (px + 0.5) * fv.s, fv.y0 + (py + 0.5) * fv.s
        val = f.values[py, px]
        sval = "mascherato" if np.isnan(val) else f"{val:.4g}"
        text = f"riga {self.loc.point(gy)}   col {self.loc.point(gx)}   valore {sval}"
        if f.gc_x is not None:
            gr, gcol = f.gc_y[min(py // 2, len(f.gc_y) - 1)], f.gc_x[px]
            fmt = lambda g: "—" if np.isnan(g) else f"{g * 100:.1f}%"  # noqa: E731
            text += f"   GC riga {fmt(gr)} · col {fmt(gcol)}"
        out.update(text)

    # ---- azioni

    def action_genome(self) -> None:
        G = self.mc.genome_length
        self.fit((0, G), (0, G))

    def action_zoom(self, f: float) -> None:
        self.zoom(f, self.view.w / 2, self.view.h / 2)

    def action_pan(self, fx: float, fy: float) -> None:
        v = self.view
        d = min(v.w, v.h) * v.s
        self.set_view(replace(v, x0=v.x0 + fx * d, y0=v.y0 + fy * d))

    def action_diagonal(self) -> None:
        v = self.view
        cx = v.x0 + v.w * v.s / 2
        self.set_view(replace(v, y0=cx - v.h * v.s / 2))

    def action_norm(self) -> None:
        i = self.norms.index(self.view.norm)
        self.set_view(replace(self.view, norm=self.norms[(i + 1) % len(self.norms)]))

    def action_cmap(self) -> None:
        names = list(COLORMAPS)
        i = names.index(self.view.cmap)
        self.set_view(replace(self.view, cmap=names[(i + 1) % len(names)]))

    def action_scale(self) -> None:
        self.set_view(replace(self.view, log=not self.view.log))

    def action_sat(self, d: int) -> None:
        i = min(range(len(self.SATS)), key=lambda k: abs(self.SATS[k] - self.view.sat))
        i = int(np.clip(i + d, 0, len(self.SATS) - 1))
        self.set_view(replace(self.view, sat=self.SATS[i]))

    def action_res(self, d: int) -> None:
        cur = self.view.binsize or (self.frame.binsize if self.frame else self.binsizes[-1])
        i = self.binsizes.index(cur) if cur in self.binsizes else 0
        i = int(np.clip(i + d, 0, len(self.binsizes) - 1))
        self.set_view(replace(self.view, binsize=self.binsizes[i]))

    def action_res_auto(self) -> None:
        self.set_view(replace(self.view, binsize=None))

    def action_boundaries(self) -> None:
        self.set_view(replace(self.view, boundaries=not self.view.boundaries))

    def action_goto(self) -> None:
        inp = self.query_one("#goto", Input)
        inp.add_class("visible")
        inp.value = ""
        inp.focus()

    def action_close_goto(self) -> None:
        self.query_one("#goto", Input).remove_class("visible")
        self.query_one(Heatmap).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            rows, cols = self.loc.parse(event.value)
        except ValueError as e:
            self.notify(str(e), severity="error")
            return
        self.action_close_goto()
        self.fit(rows, cols)


def _place(buf: list[str], x: int, label: str, only_free: bool = False) -> None:
    if x < 0 or x >= len(buf):
        return
    end = min(len(buf), x + len(label) + 1)
    if only_free and (x + len(label) > len(buf) or any(c != " " for c in buf[x:end])):
        return
    for k, ch in enumerate(label[: len(buf) - x]):
        buf[x + k] = ch


def main() -> None:
    p = argparse.ArgumentParser(prog="mcoolio", description="Visualizzatore TUI per file .mcool / .cool")
    p.add_argument("files", nargs="+", metavar="FILE",
                   help="un file .mcool/.cool e, facoltativo, un FASTA del genoma (in qualsiasi ordine)")
    p.add_argument("-f", "--fasta", help="FASTA del genoma di riferimento (traccia GC)")
    p.add_argument("-r", "--region", help="regione iniziale, es. chr1:10M-20M oppure 'chr1 chr2'")
    p.add_argument("-n", "--norm", default="weight", help="normalizzazione iniziale (none, weight, KR, …)")
    p.add_argument("-c", "--cmap", default="fall", choices=list(COLORMAPS), help="colormap iniziale")
    a = p.parse_args()

    cools, fastas = [], [a.fasta] if a.fasta else []
    for path in a.files:
        try:
            kind = sniff(path)
        except OSError as e:
            p.error(f"{path}: {e.strerror}")
        if kind == "hdf5":
            cools.append(path)
        elif kind in ("fasta", "gzip"):
            fastas.append(path)
        else:
            p.error(f"{path}: formato non riconosciuto (attesi .mcool/.cool o FASTA)")
    if len(cools) != 1:
        p.error("serve esattamente un file .mcool/.cool: il FASTA da solo non contiene i contatti Hi-C"
                if not cools else "passa un solo file .mcool/.cool alla volta")
    if len(fastas) > 1:
        p.error("passa un solo FASTA")

    try:
        fasta = Fasta(fastas[0]) if fastas else None
    except (OSError, ValueError) as e:
        p.error(str(e))
    Viewer(MCool(cools[0]), a.region, a.norm, a.cmap, fasta).run()


if __name__ == "__main__":
    main()
