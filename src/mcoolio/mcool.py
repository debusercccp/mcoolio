"""Lettura diretta di file .mcool / .cool (formato cooler, HDF5) senza dipendere da cooler."""

import json
import threading
from dataclasses import dataclass

import h5py
import numpy as np

# Vettori di normalizzazione in stile Juicer (importati da hic2cool): sono divisivi.
# "weight" (bilanciamento ICE di cooler) è invece moltiplicativo.
DIVISIVE_NORMS = {"KR", "VC", "VC_SQRT", "SCALE"}
BIN_COLUMNS = {"chrom", "start", "end"}

# Limite di sicurezza sul numero di celle per singola richiesta.
MAX_CELLS = 6_000_000


@dataclass
class Resolution:
    binsize: int
    group: h5py.Group
    nbins: int
    nnz: int
    chrom_offset: np.ndarray
    bin1_offset: h5py.Dataset
    norms: list[str]


class MCool:
    def __init__(self, path: str):
        self.path = path
        # Cache dei chunk generosa: i dataset pixel sono compressi a chunk piccoli.
        self.h5 = h5py.File(path, "r", rdcc_nbytes=512 * 1024**2, rdcc_nslots=1_000_003)
        self.lock = threading.Lock()

        if "resolutions" in self.h5:
            groups = {int(k): self.h5["resolutions"][k] for k in self.h5["resolutions"]}
        else:  # singolo .cool
            groups = {int(self.h5.attrs["bin-size"]): self.h5["/"]}

        self.resolutions: dict[int, Resolution] = {}
        for binsize, g in sorted(groups.items()):
            norms = [k for k in g["bins"] if k not in BIN_COLUMNS]
            self.resolutions[binsize] = Resolution(
                binsize=binsize,
                group=g,
                nbins=g["bins/start"].shape[0],
                nnz=g["pixels/count"].shape[0],
                chrom_offset=g["indexes/chrom_offset"][:].astype(np.int64),
                bin1_offset=g["indexes/bin1_offset"],
                norms=sorted(norms, key=lambda n: (n != "weight", n)),
            )

        g0 = next(iter(self.resolutions.values())).group
        self.chrom_names = [n.decode() if isinstance(n, bytes) else str(n) for n in g0["chroms/name"][:]]
        self.chrom_lengths = g0["chroms/length"][:].astype(np.int64)
        # Coordinate "globali": cromosomi concatenati nell'ordine del file.
        self.chrom_starts = np.concatenate([[0], np.cumsum(self.chrom_lengths)[:-1]])
        self.genome_length = int(self.chrom_lengths.sum())
        self.attrs = {k: _jsonable(v) for k, v in g0.attrs.items()}
        self._norm_cache: dict[tuple[int, str], np.ndarray] = {}

    def info(self) -> dict:
        return {
            "path": self.path,
            "attrs": self.attrs,
            "genomeLength": self.genome_length,
            "chroms": [
                {"name": n, "length": int(l), "start": int(s)}
                for n, l, s in zip(self.chrom_names, self.chrom_lengths, self.chrom_starts)
            ],
            "resolutions": [
                {"binsize": r.binsize, "nbins": r.nbins, "nnz": r.nnz, "norms": r.norms}
                for r in self.resolutions.values()
            ],
        }

    def _norm_vector(self, res: Resolution, norm: str) -> np.ndarray:
        """Vettore moltiplicativo per bin (NaN = bin mascherato)."""
        key = (res.binsize, norm)
        if key not in self._norm_cache:
            w = res.group["bins"][norm][:].astype(np.float64)
            if norm in DIVISIVE_NORMS:
                with np.errstate(divide="ignore"):
                    w = 1.0 / w
            w[~np.isfinite(w) | (w == 0)] = np.nan
            self._norm_cache[key] = w
        return self._norm_cache[key]

    def _bin_range(self, res: Resolution, g0: float, g1: float):
        """Converte un intervallo globale [g0, g1) in un range contiguo di bin + segmenti per cromosoma."""
        g0 = max(0, int(g0))
        g1 = min(self.genome_length, int(np.ceil(g1)))
        segments = []
        first = last = None
        for ci, (cs, cl) in enumerate(zip(self.chrom_starts, self.chrom_lengths)):
            lo, hi = max(g0 - cs, 0), min(g1 - cs, cl)
            if hi <= lo:
                continue
            lb, ub = lo // res.binsize, -(-hi // res.binsize)
            b0, b1 = int(res.chrom_offset[ci] + lb), int(res.chrom_offset[ci] + ub)
            if first is None:
                first = b0
            segments.append({
                "chrom": self.chrom_names[ci],
                "i0": b0 - first,
                "n": b1 - b0,
                "gstart": int(cs + lb * res.binsize),
                "gend": int(cs + min(ub * res.binsize, cl)),
            })
            last = b1
        if first is None:
            return 0, 0, []
        return first, last, segments

    def _read_rows(self, res: Resolution, r0: int, r1: int, c0: int, c1: int):
        """Pixel con bin1 in [r0, r1) e bin2 in [c0, c1) (triangolo superiore come salvato)."""
        if r1 <= r0 or c1 <= c0:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64)
        offs = res.bin1_offset[r0:r1 + 1].astype(np.int64)
        lo, hi = int(offs[0]), int(offs[-1])
        if hi <= lo:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64)
        pix = res.group["pixels"]
        bin2 = pix["bin2_id"][lo:hi]
        rows = np.repeat(np.arange(r0, r1, dtype=np.int64), np.diff(offs))
        m = (bin2 >= c0) & (bin2 < c1)
        counts = pix["count"][lo:hi]
        return rows[m], bin2[m].astype(np.int64), counts[m].astype(np.float64)

    def bin_ids(self, res: Resolution, g: np.ndarray) -> np.ndarray:
        """Id del bin per ogni coordinata globale (-1 fuori dal genoma)."""
        valid = (g >= 0) & (g < self.genome_length)
        gc = np.where(valid, g, 0)
        ci = np.searchsorted(self.chrom_starts, gc, side="right") - 1
        b = res.chrom_offset[ci] + ((gc - self.chrom_starts[ci]) // res.binsize).astype(np.int64)
        return np.where(valid, b, -1)

    def chrom_index(self, g: np.ndarray) -> np.ndarray:
        """Indice del cromosoma per ogni coordinata globale (-1 fuori dal genoma)."""
        ci = np.searchsorted(self.chrom_starts, g, side="right") - 1
        return np.where((g >= 0) & (g < self.genome_length), ci, -1)

    def pick_resolution(self, binsize: int, span_rows: float, span_cols: float) -> Resolution:
        """La risoluzione disponibile più vicina (>=) a quella chiesta, nel limite di MAX_CELLS."""
        candidates = list(self.resolutions.values())
        chosen = next((r for r in candidates if r.binsize >= binsize), candidates[-1])
        idx = candidates.index(chosen)
        while idx < len(candidates) - 1 and (span_rows / chosen.binsize) * (span_cols / chosen.binsize) > MAX_CELLS:
            idx += 1
            chosen = candidates[idx]
        return chosen

    def matrix(self, binsize: int, y0: float, y1: float, x0: float, x1: float, norm: str = "none"):
        """Matrice densa (righe = asse y, colonne = asse x) per un rettangolo in coordinate globali."""
        res = self.pick_resolution(binsize, y1 - y0, x1 - x0)
        r0, r1, rseg = self._bin_range(res, y0, y1)
        c0, c1, cseg = self._bin_range(res, x0, x1)
        mat = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)

        with self.lock:
            # Il file salva solo bin1 <= bin2: servono le due "metà" del rettangolo.
            i, j, v = self._read_rows(res, r0, r1, c0, c1)
            mat[i - r0, j - c0] = v
            if (r0, r1) == (c0, c1):
                mat[j - r0, i - c0] = v
            else:
                i, j, v = self._read_rows(res, c0, c1, r0, r1)
                mat[j - r0, i - c0] = v

            if norm != "none":
                if norm not in res.norms:
                    raise ValueError(f"normalizzazione sconosciuta: {norm}")
                w = self._norm_vector(res, norm)
                mat *= w[r0:r1, None].astype(np.float32)
                mat *= w[None, c0:c1].astype(np.float32)

        header = {
            "binsize": res.binsize,
            "norm": norm,
            "rows": {"b0": r0, "n": r1 - r0, "segments": rseg},
            "cols": {"b0": c0, "n": c1 - c0, "segments": cseg},
        }
        return header, mat


def _jsonable(v):
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, str) and v.startswith("{"):
        try:
            return json.loads(v)
        except ValueError:
            pass
    return v
