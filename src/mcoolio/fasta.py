"""FASTA indicizzato (.fai) e profilo di contenuto GC allineato ai cromosomi di un .mcool."""

import mmap
import os
from pathlib import Path

import numpy as np

# Il profilo GC si calcola una volta a questa risoluzione e si mette in cache su disco.
GC_BIN = 1000
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mcoolio"

_IS_GC = np.zeros(256, np.uint8)
_IS_GC[list(b"GCgcSs")] = 1
_IS_ACGT = np.zeros(256, np.uint8)
_IS_ACGT[list(b"ACGTacgtSsWw")] = 1


def sniff(path: str) -> str:
    """'hdf5', 'fasta', 'gzip' o 'unknown' in base ai primi byte."""
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "hdf5"
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if head.startswith(b">"):
        return "fasta"
    return "unknown"


class Fasta:
    def __init__(self, path: str):
        self.path = path
        kind = sniff(path)
        if kind == "gzip":
            raise ValueError(f"{path} è compresso: serve il FASTA non compresso (con o senza .fai)")
        if kind != "fasta":
            raise ValueError(f"{path} non sembra un file FASTA")
        fai = Path(path + ".fai")
        if not fai.exists():
            build_fai(path, fai)
        self.index: dict[str, tuple[int, int, int, int]] = {}
        for line in fai.read_text().splitlines():
            name, length, offset, linebases, linewidth = line.split("\t")[:5]
            self.index[name] = (int(length), int(offset), int(linebases), int(linewidth))
        f = open(path, "rb")
        self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        f.close()

    def length(self, chrom: str) -> int | None:
        e = self.index.get(chrom)
        return e[0] if e else None

    def _raw(self, chrom: str) -> np.ndarray:
        """Tutta la sequenza di un cromosoma come array di byte, senza a capo."""
        length, offset, lb, lw = self.index[chrom]
        nfull, rest = divmod(length, lb)
        buf = np.frombuffer(self._mm, np.uint8, count=nfull * lw + rest, offset=offset)
        seq = buf[: nfull * lw].reshape(nfull, lw)[:, :lb].ravel()
        return np.concatenate([seq, buf[nfull * lw:]]) if rest else seq

    def fetch(self, chrom: str, start: int, end: int) -> str:
        length, offset, lb, lw = self.index[chrom]
        start, end = max(0, start), min(length, end)
        if end <= start:
            return ""
        a = offset + (start // lb) * lw + start % lb
        b = offset + (end // lb) * lw + end % lb
        return self._mm[a:b].replace(b"\n", b"").replace(b"\r", b"").decode()

    def gc_profile(self, chrom_names: list[str], chrom_lengths: np.ndarray, progress=None):
        """Conteggi (GC, basi note) per bin da GC_BIN bp, nell'ordine dei cromosomi del .mcool.

        I cromosomi assenti dal FASTA restano a zero (GC non disponibile).
        """
        st = os.stat(self.path)
        key = f"{Path(self.path).name}-{st.st_size}-{int(st.st_mtime)}-{GC_BIN}"
        cache = CACHE_DIR / f"gc-{key}.npz"
        if cache.exists():
            with np.load(cache) as z:
                gc_by_chrom = {n: (z[f"gc/{n}"], z[f"n/{n}"]) for n in self.index if f"gc/{n}" in z}
        else:
            gc_by_chrom = {}
        todo = [n for n in chrom_names if n in self.index and n not in gc_by_chrom]
        for k, name in enumerate(todo):
            if progress:
                progress(k, len(todo), name)
            seq = self._raw(name)
            nb = -(-seq.size // GC_BIN)
            pad = np.zeros(nb * GC_BIN - seq.size, np.uint8)
            seq = np.concatenate([seq, pad]).reshape(nb, GC_BIN)
            gc_by_chrom[name] = (_IS_GC[seq].sum(1, dtype=np.uint16), _IS_ACGT[seq].sum(1, dtype=np.uint16))
        if todo:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp.npz")
            np.savez(tmp, **{f"gc/{n}": v[0] for n, v in gc_by_chrom.items()},
                     **{f"n/{n}": v[1] for n, v in gc_by_chrom.items()})
            tmp.replace(cache)

        gcs, ns = [], []
        for name, length in zip(chrom_names, chrom_lengths):
            nb = -(-int(length) // GC_BIN)
            gc, n = gc_by_chrom.get(name, (np.zeros(0, np.uint16), np.zeros(0, np.uint16)))
            # Se le lunghezze non coincidono (altro assembly) si tronca o si riempie con "ignoto".
            gcs.append(np.pad(gc[:nb], (0, max(0, nb - gc.size))))
            ns.append(np.pad(n[:nb], (0, max(0, nb - n.size))))
        return GCProfile(chrom_lengths, np.concatenate(gcs), np.concatenate(ns))


class GCProfile:
    """Frazione GC su intervalli qualsiasi in coordinate globali, tramite somme cumulative."""

    def __init__(self, chrom_lengths: np.ndarray, gc: np.ndarray, n: np.ndarray):
        self.chrom_starts = np.concatenate([[0], np.cumsum(chrom_lengths)[:-1]])
        self.genome_length = int(np.sum(chrom_lengths))
        nbins = -(-chrom_lengths // GC_BIN)
        self.bin_offset = np.concatenate([[0], np.cumsum(nbins)])
        self.cum_gc = np.concatenate([[0], np.cumsum(gc, dtype=np.int64)])
        self.cum_n = np.concatenate([[0], np.cumsum(n, dtype=np.int64)])

    def _bins(self, g: np.ndarray, ceil: bool) -> np.ndarray:
        g = np.clip(g, 0, self.genome_length)
        ci = np.clip(np.searchsorted(self.chrom_starts, g, side="right") - 1, 0, len(self.chrom_starts) - 1)
        local = g - self.chrom_starts[ci]
        b = self.bin_offset[ci] + (np.ceil(local / GC_BIN) if ceil else np.floor(local / GC_BIN)).astype(np.int64)
        return np.minimum(b, self.bin_offset[ci + 1])

    def fraction(self, edges: np.ndarray) -> np.ndarray:
        """GC per ogni intervallo [edges[i], edges[i+1]) (NaN se non ci sono basi note)."""
        last = len(self.cum_gc) - 1
        lo = np.minimum(self._bins(edges[:-1], ceil=False), last - 1)
        hi = np.minimum(np.maximum(self._bins(edges[1:], ceil=True), lo + 1), last)
        n = self.cum_n[hi] - self.cum_n[lo]
        gc = self.cum_gc[hi] - self.cum_gc[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = gc / n
        out[(n < GC_BIN // 10) | (edges[1:] <= 0) | (edges[:-1] >= self.genome_length)] = np.nan
        return out


def build_fai(path: str, fai: Path) -> None:
    """Crea l'indice .fai (come samtools faidx) se manca."""
    entries = []
    with open(path, "rb") as f:
        name = None
        length = offset = lb = lw = 0
        pos = 0
        for line in f:
            if line.startswith(b">"):
                if name is not None:
                    entries.append((name, length, offset, lb, lw))
                name = line[1:].split()[0].decode()
                length, lb, lw = 0, 0, 0
                pos += len(line)
                offset = pos
                continue
            if lb == 0:
                lb, lw = len(line.rstrip(b"\r\n")), len(line)
            length += len(line.rstrip(b"\r\n"))
            pos += len(line)
        if name is not None:
            entries.append((name, length, offset, lb, lw))
    fai.write_text("".join(f"{n}\t{l}\t{o}\t{b}\t{w}\n" for n, l, o, b, w in entries))
