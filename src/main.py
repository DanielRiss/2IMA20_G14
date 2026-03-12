"""
main.py — CLI entry point for the Multi-Source EU Trade Flow Map.

Usage examples:
  python src/main.py --sources Germany
  python src/main.py --sources Germany France --mode net --threshold 500
  python src/main.py --sources Germany France Italy --output multi_flow.png
"""

import argparse
import os
import sys

# Allow running as: python src/main.py from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_trade_data
from map_utils import load_eu_map, get_centroids
from flow_renderer import draw_flow_map

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


def main():
    parser = argparse.ArgumentParser(
        description='Render a multi-source EU trade flow map (2024 Eurostat data).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--sources', nargs='+', default=['Germany'],
        metavar='COUNTRY',
        help='Source country name(s), e.g. --sources Germany France Italy',
    )
    parser.add_argument(
        '--mode', choices=['gross', 'net'], default='gross',
        help='Flow mode: gross=total exports, net=exports minus imports',
    )
    parser.add_argument(
        '--threshold', type=float, default=0.0,
        metavar='MEUR',
        help='Minimum flow value to draw (million EUR)',
    )
    parser.add_argument(
        '--output', default='flowmap.png',
        metavar='FILE',
        help='Output filename (saved inside output/)',
    )
    args = parser.parse_args()

    data_path  = os.path.join(DATA_DIR, 'data-18886936.csv')
    label_path = os.path.join(DATA_DIR, 'label-18886936.csv')

    print("Loading trade data...")
    export_matrix, net_matrix, countries = load_trade_data(data_path, label_path)

    print("Loading EU map...")
    eu_gdf = load_eu_map()
    centroids = get_centroids(eu_gdf)

    # Validate requested source countries
    valid_sources = []
    for s in args.sources:
        if s in countries:
            valid_sources.append(s)
        else:
            print(f"  Warning: '{s}' not in EU27. Available: {', '.join(sorted(countries))}")

    if not valid_sources:
        print("No valid source countries specified. Exiting.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, args.output)

    title = f"EU Trade Flows from {', '.join(valid_sources)} (2024)"
    net_mode = (args.mode == 'net')

    print(f"Rendering: sources={valid_sources}, mode={args.mode}, threshold={args.threshold} M EUR")
    draw_flow_map(
        eu_gdf, centroids, export_matrix, valid_sources,
        threshold_meur=args.threshold,
        net_mode=net_mode,
        net_matrix=net_matrix,
        title=title,
        output_path=output_path,
    )


if __name__ == '__main__':
    main()
