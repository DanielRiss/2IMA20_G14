"""
map_utils.py — Load EU27 country boundaries and provide publication-quality basemap.
Uses the Natural Earth 110m admin-0 countries shapefile.
"""

import os
import urllib.request
import zipfile
import geopandas as gpd
import matplotlib
import matplotlib.axes

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

CENTROID_OVERRIDES = {
    'Cyprus':     (33.0,  35.1),
    'Malta':      (14.4,  35.9),
    'Luxembourg': (6.13, 49.81),
    'France':     (2.35, 46.5),
}

NE_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/"
    "ne_110m_admin_0_countries.zip"
)

_HERE      = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_HERE, '..', 'data', 'ne_110m')

# Cached world GDF (loaded once)
_world_gdf_cache = None


def _get_shapefile_path():
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


def _fix_iso(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Patch Natural Earth ISO_A2 so France is not '-99'."""
    if 'ISO_A2_EH' in gdf.columns:
        gdf = gdf.copy()
        gdf['iso_a2'] = gdf['ISO_A2_EH'].where(
            gdf['ISO_A2_EH'] != '-99', gdf.get('ISO_A2', '-99')
        )
    else:
        gdf = gdf.copy()
        gdf['iso_a2'] = gdf.get('ISO_A2', '-99')
    return gdf


def load_eu_map() -> gpd.GeoDataFrame:
    """
    Load EU27 country geometries from Natural Earth 110m.

    Returns GeoDataFrame with columns: iso_a2, name_short, geometry.
    """
    world = gpd.read_file(_get_shapefile_path())
    world = _fix_iso(world)
    eu = world[world['iso_a2'].isin(EU27_ISO)].copy()
    eu['name_short'] = eu['iso_a2'].map(ISO_SHORT)
    return eu.reset_index(drop=True)


def load_world_map() -> gpd.GeoDataFrame:
    """
    Load all-countries shapefile. Cached after first call.

    Returns GeoDataFrame with columns: iso_a2, geometry.
    """
    global _world_gdf_cache
    if _world_gdf_cache is None:
        world = gpd.read_file(_get_shapefile_path())
        _world_gdf_cache = _fix_iso(world).reset_index(drop=True)
    return _world_gdf_cache


def get_centroids(eu_gdf: gpd.GeoDataFrame) -> dict:
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
    for name, xy in CENTROID_OVERRIDES.items():
        centroids[name] = xy
    return centroids


def render_basemap(
    ax: matplotlib.axes.Axes,
    eu_gdf: gpd.GeoDataFrame,
    world_gdf: gpd.GeoDataFrame = None,
    highlighted_countries: list = None,
    xlim: tuple = (-25, 45),
    ylim: tuple = (34, 72),
) -> None:
    """
    Draw a publication-quality EU basemap onto ax.

    Parameters
    ----------
    eu_gdf               : EU27 GeoDataFrame (from load_eu_map)
    world_gdf            : full world GeoDataFrame (from load_world_map); if None
                           only EU countries are drawn
    highlighted_countries: country short names to render with a distinct fill
                           (e.g. currently selected source countries)
    xlim, ylim           : geographic bounding box [lon, lat]
    """
    ax.set_facecolor('#c8dff0')   # ocean blue

    # ── non-EU context countries ─────────────────────────────────────────
    if world_gdf is not None:
        non_eu = world_gdf[~world_gdf['iso_a2'].isin(EU27_ISO)]
        non_eu.plot(
            ax=ax, color='#e0ddd5', edgecolor='#b0b0a8',
            linewidth=0.3, zorder=1
        )

    # ── EU countries ──────────────────────────────────────────────────────
    hl = set(highlighted_countries or [])
    if hl:
        eu_normal = eu_gdf[~eu_gdf['name_short'].isin(hl)]
        eu_hl     = eu_gdf[eu_gdf['name_short'].isin(hl)]
        eu_normal.plot(ax=ax, color='#f0f4e8', edgecolor='#888888',
                       linewidth=0.5, zorder=2)
        eu_hl.plot(ax=ax, color='#fffde8', edgecolor='#999966',
                   linewidth=0.8, zorder=2)
    else:
        eu_gdf.plot(ax=ax, color='#f0f4e8', edgecolor='#888888',
                    linewidth=0.5, zorder=2)

    # ── 10° graticule ─────────────────────────────────────────────────────
    for lon in range(-30, 51, 10):
        ax.axvline(lon, color='#aabbcc', linewidth=0.25, linestyle='--',
                   alpha=0.55, zorder=1)
    for lat in range(30, 81, 10):
        ax.axhline(lat, color='#aabbcc', linewidth=0.25, linestyle='--',
                   alpha=0.55, zorder=1)

    # ── axis styling ─────────────────────────────────────────────────────
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.axis('off')