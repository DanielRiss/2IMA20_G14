"""
map_utils.py — Load EU27 country boundaries and compute centroids.
Uses the Natural Earth 110m admin-0 countries shapefile.
Downloaded once from naciscdn.org and cached in data/ne_110m_admin_0_countries/.
"""

import os
import urllib.request
import zipfile
import tempfile
import geopandas as gpd

EU27_ISO = [
    'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
    'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT',
    'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'
]

ISO_SHORT = {
    'AT': 'Austria',     'BE': 'Belgium',    'BG': 'Bulgaria',
    'CY': 'Cyprus',      'CZ': 'Czechia',    'DE': 'Germany',
    'DK': 'Denmark',     'EE': 'Estonia',    'ES': 'Spain',
    'FI': 'Finland',     'FR': 'France',     'GR': 'Greece',
    'HR': 'Croatia',     'HU': 'Hungary',    'IE': 'Ireland',
    'IT': 'Italy',       'LT': 'Lithuania',  'LU': 'Luxembourg',
    'LV': 'Latvia',      'MT': 'Malta',      'NL': 'Netherlands',
    'PL': 'Poland',      'PT': 'Portugal',   'RO': 'Romania',
    'SE': 'Sweden',      'SI': 'Slovenia',   'SK': 'Slovakia',
}

# Manual centroid overrides for small/island/landlocked countries
# where automatic centroid may be inaccurate or the country absent from lowres
CENTROID_OVERRIDES = {
    'Cyprus':     (33.0,  35.1),   # island, may be absent from 110m
    'Malta':      (14.4,  35.9),   # too small for 110m shapefile
    'Luxembourg': (6.13, 49.81),   # very small country
    'France':     (2.35, 46.5),    # ISO_A2=-99 in NE 110m; use override as fallback
}


NE_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/"
    "ne_110m_admin_0_countries.zip"
)

# Cache directory relative to this file: ../data/ne_110m/
_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_HERE, '..', 'data', 'ne_110m')


def _get_shapefile_path():
    """Download Natural Earth 110m countries shapefile if not cached."""
    shp = os.path.join(_CACHE_DIR, 'ne_110m_admin_0_countries.shp')
    if os.path.exists(shp):
        return shp

    print("Downloading Natural Earth 110m countries shapefile (~500 KB)...")
    os.makedirs(_CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(_CACHE_DIR, 'ne_110m_admin_0_countries.zip')
    urllib.request.urlretrieve(NE_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(_CACHE_DIR)
    os.remove(zip_path)
    print(f"Cached to {_CACHE_DIR}")
    return shp


def load_eu_map():
    """
    Load EU27 country geometries from Natural Earth 110m.
    Shapefile is downloaded on first run and cached in data/ne_110m/.

    Returns a GeoDataFrame with columns: iso_a2, name_short, geometry.
    """
    shp_path = _get_shapefile_path()
    world = gpd.read_file(shp_path)

    # Natural Earth 110m uses 'ISO_A2' but France (and a few others) have '-99'
    # there; the correct code is in 'ISO_A2_EH'. Use EH as primary, fall back
    # to standard ISO_A2 where EH is also -99.
    if 'ISO_A2_EH' in world.columns:
        world['iso_a2'] = world['ISO_A2_EH'].where(
            world['ISO_A2_EH'] != '-99', world['ISO_A2']
        )
    else:
        world['iso_a2'] = world.get('ISO_A2', world.get('iso_a2', '-99'))

    eu = world[world['iso_a2'].isin(EU27_ISO)].copy()
    eu['name_short'] = eu['iso_a2'].map(ISO_SHORT)
    eu = eu.reset_index(drop=True)
    return eu


def get_centroids(eu_gdf):
    """
    Compute geographic centroids for each EU27 country.

    Returns dict: {country_name: (lon, lat)}
    Manual overrides applied for small/island countries.
    """
    centroids = {}
    for _, row in eu_gdf.iterrows():
        name = row.get('name_short')
        if name:
            c = row['geometry'].centroid
            centroids[name] = (c.x, c.y)

    # Apply overrides (and fill in missing countries)
    for name, xy in CENTROID_OVERRIDES.items():
        centroids[name] = xy

    return centroids
