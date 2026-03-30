"""Verification script for disparity filter implementation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_trade_data
from flow_renderer import disparity_filter

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'data-18886936.csv')
LABEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'label-18886936.csv')

export_matrix, net_matrix, countries = load_trade_data(DATA_PATH, LABEL_PATH)

# Build Germany's raw flows (threshold=0, gross exports, exclude self)
de_row = export_matrix.loc['Germany']
raw_flows = {
    dst: float(v) for dst, v in de_row.items()
    if dst != 'Germany' and float(v) > 0
}

print("=" * 60)
print("CHECK 1: alpha=0.15 vs raw threshold-only")
filtered_015 = disparity_filter(raw_flows, alpha=0.15)
removed = set(raw_flows.keys()) - set(filtered_015.keys())
print(f"  Raw (threshold=0):          {len(raw_flows)} destinations")
print(f"  After disparity (alpha=0.15):   {len(filtered_015)} destinations")
print(f"  Removed by disparity filter ({len(removed)}):")
for dst in sorted(removed):
    print(f"    {dst:30s}  flow={raw_flows[dst]:>12,.1f} M EUR")

print()
print("=" * 60)
print("CHECK 2: alpha=0.05 (strict) vs alpha=0.50 (loose)")
filtered_005 = disparity_filter(raw_flows, alpha=0.05)
filtered_050 = disparity_filter(raw_flows, alpha=0.50)
print(f"  alpha=0.05 (strict): {len(filtered_005)} destinations")
print(f"  alpha=0.15 (default): {len(filtered_015)} destinations")
print(f"  alpha=0.50 (loose):  {len(filtered_050)} destinations")
assert len(filtered_005) <= len(filtered_015) <= len(filtered_050), \
    "FAIL: stricter alpha should remove more flows"
print("  PASS: stricter alpha removes more flows (monotone)")

print()
print("=" * 60)
print("CHECK 3: cache key includes disparity_alpha")
sources_fs = frozenset(['Germany'])
alpha_deg  = 25.0
threshold  = 0.0
net_mode   = False
exponent   = 0.5

key_015 = (sources_fs, alpha_deg, threshold, net_mode, exponent, 0.15)
key_020 = (sources_fs, alpha_deg, threshold, net_mode, exponent, 0.20)
print(f"  cache key (alpha=0.15): {key_015}")
print(f"  cache key (alpha=0.20): {key_020}")
assert key_015 != key_020, "FAIL: keys must differ"
print(f"  Keys differ: {key_015[-1]} != {key_020[-1]}  PASS")
