# Multi-Source EU Trade Flow Map

**Group 14 — 2IMA20: Algorithms for Geovisualization, TU/e**

Visualizes intra-EU bilateral trade flows from one or more source countries simultaneously. Phase 1: minimal working pipeline with gross/net export mode and threshold filtering.

---

## Project Structure

```
Multi-Source-Flowmap_G14/
├── data/
│   ├── data-18886936.csv      # Eurostat bilateral trade data (2024)
│   └── label-18886936.csv     # ISO code ↔ verbose country name mapping
├── output/                    # Generated PNG maps (git-ignored)
├── src/
│   ├── data_loader.py         # CSV parsing → export & net-flow matrices
│   ├── map_utils.py           # Natural Earth boundaries + centroids
│   ├── flow_renderer.py       # Matplotlib flow line drawing
│   └── main.py                # CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+. Tested with geopandas 0.14.

---

## Usage

Run from the project root:

```bash
# Single source — gross exports from Germany (all flows)
python src/main.py --sources Germany

# Single source with threshold (only flows ≥ 1000 M EUR)
python src/main.py --sources Germany --threshold 1000 --output germany_gross.png

# Multi-source — gross exports from three countries
python src/main.py --sources Germany France Italy --output multi_gross.png

# Net exports mode (draws only flows where source is net exporter)
python src/main.py --sources Germany --mode net --threshold 500 --output germany_net.png
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--sources` | `Germany` | One or more source country names (space-separated) |
| `--mode` | `gross` | `gross` = total exports, `net` = exports minus imports |
| `--threshold` | `0` | Minimum flow to draw, in million EUR |
| `--output` | `flowmap.png` | Output filename (saved to `output/`) |

---

## Output

PNG files are saved to `output/`. Line width and opacity scale with flow magnitude. Each source country is color-coded.

---

## Data Source

Eurostat intra-EU bilateral trade statistics, 2024 annual totals (`VALUE_IN_EUR`, `EXPORT` direction). Downloaded from Eurostat Comext.

---

## Architecture (Phase 1)

| Module | Responsibility |
|---|---|
| `data_loader.py` | CSV parsing, name normalisation, matrix construction |
| `map_utils.py` | Shapefile loading, EU27 filtering, centroid computation |
| `flow_renderer.py` | Matplotlib rendering (arrows, scaling, legend) |
| `main.py` | CLI, orchestration |

Future phases will add: source selection algorithms, spiral-tree layout, crossing minimisation, quality metrics.
