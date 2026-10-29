# mcoolio

Visualizzatore da terminale (TUI) per mappe di contatto Hi-C in formato `.mcool` / `.cool`.
Legge l'HDF5 direttamente con `h5py` (niente `cooler`) e disegna la heatmap con caratteri `▀` in truecolor:
ogni cella del terminale vale due pixel quadrati.

```sh
uv run mcoolio ../clew-hic/data/4DNFIXP4QG5B.mcool
uv run mcoolio file.mcool -r chr7:25M-29M -n KR -c reds
uv run mcoolio file.mcool -r "chr1 chr4"          # blocco trans: righe chr1, colonne chr4
```

### Genoma di riferimento (FASTA)

Puoi aggiungere il FASTA del genoma, come secondo file oppure con `-f`:

```sh
uv run mcoolio ../clew-hic/data/4DNFIXP4QG5B.mcool ../clew-hic/data/genomes/hg38/hg38.fa
uv run mcoolio file.mcool -f hg38.fa
```

Così compare una traccia di contenuto GC lungo i due assi (sopra le colonne, a sinistra delle righe),
il GC% di riga e colonna sotto il cursore, e un avviso se i cromosomi del FASTA non coincidono con quelli
del `.mcool` (assembly diverso). Il FASTA da solo non basta: i contatti stanno nel `.mcool`.

- Serve il FASTA **non compresso**; l'indice `.fai` si usa se c'è, altrimenti viene creato.
- La prima volta il profilo GC (bin da 1 kb) richiede ~15 s per hg38, poi resta in cache in `~/.cache/mcoolio/`.

Serve un terminale con supporto truecolor (kitty, WezTerm, Alacritty, GNOME Terminal, iTerm2, …).

## Comandi

| tasto | azione |
|---|---|
| frecce / `hjkl` (maiusc = passo lungo), trascinamento col mouse | sposta |
| `+` `-`, rotella | zoom (con la rotella, attorno al cursore) |
| `g` o `/` | vai a una regione: `chr1`, `chr1:10M-20M`, `chr1:10,000,000-20,000,000`, `chr1:1-50M chr2` |
| `0` | genoma intero |
| `d` | ricentra sulla diagonale |
| `n` | normalizzazione: none → weight (ICE) → KR → VC → VC_SQRT |
| `c` | colormap (fall, reds, viridis, hot, gray) |
| `s` | scala log / lineare |
| `[` `]` | saturazione (quantile usato come massimo del colore) |
| `r` / `R` / `a` | risoluzione più fine / più grossa / automatica |
| `b` | linee di confine tra cromosomi |
| `q` | esci |

Passando il mouse sulla mappa, la barra in basso mostra le coordinate di riga e colonna e il valore.

## Note

- In automatico viene usata la risoluzione più fine con bin ≥ 1 pixel, quindi ogni richiesta legge
  solo poche migliaia di righe della matrice anche sui file da decine di GB.
- `weight` (bilanciamento di cooler) è moltiplicativo; `KR`, `VC`, `VC_SQRT` (importati da Juicer)
  sono divisivi, e vengono invertiti automaticamente. I bin mascherati compaiono in grigio.
