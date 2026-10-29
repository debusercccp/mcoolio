"""Colormap come LUT 256x3 (uint8) costruite interpolando pochi punti di controllo."""

import numpy as np

_STOPS = {
    # Simile alla "fall" di HiGlass: bianco -> giallo -> rosso -> nero.
    "fall": ["#ffffff", "#ffffe0", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
             "#fc4e2a", "#e31a1c", "#bd0026", "#800026", "#000000"],
    "reds": ["#ffffff", "#ff0000"],
    "viridis": ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89",
                "#35b779", "#6ece58", "#b5de2b", "#fde725"],
    "hot": ["#000000", "#800000", "#ff0000", "#ff8000", "#ffff00", "#ffffff"],
    "gray": ["#ffffff", "#000000"],
}


def _hex(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _lut(stops: list[str]) -> np.ndarray:
    pts = np.array([_hex(c) for c in stops], dtype=np.float64)
    x = np.linspace(0, 1, len(pts))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, x, pts[:, k]) for k in range(3)], axis=1).round().astype(np.uint8)


COLORMAPS = {name: _lut(stops) for name, stops in _STOPS.items()}
