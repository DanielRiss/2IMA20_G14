"""
interactive.py — Tkinter GUI for exploring EU trade flow maps.

Launch with:
    python src/interactive.py
"""

import os
import sys

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_trade_data, ISO_SHORT
from map_utils import load_eu_map, load_world_map, get_centroids
from flow_renderer import render_to_axes, invalidate_spiral_cache
from spiral_tree import compute_tree_stats, count_crossings
from group_utils import DEFAULT_GROUPS, apply_groups

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'output')

DATA_FILE  = os.path.join(DATA_DIR, 'data-18886936.csv')
LABEL_FILE = os.path.join(DATA_DIR, 'label-18886936.csv')

EU27_COUNTRIES  = sorted(ISO_SHORT.values())
SHORT_TO_ISO    = {v: k for k, v in ISO_SHORT.items()}
AVAILABLE_YEARS = [2024]

# ── zoom region presets (xmin, xmax, ymin, ymax) ─────────────────────────────
ZOOM_PRESETS = {
    'Full EU':          (-25, 45, 34, 72),
    'Western Europe':   (-10, 15, 42, 58),
    'Northern Europe':  (  5, 35, 54, 72),
    'Central Europe':   (  9, 26, 45, 57),
    'Benelux + DE/FR':  (  1, 16, 47, 56),
    'Iberian Peninsula':(-10,  4, 35, 45),
    'Baltic States':    ( 20, 28, 53, 61),
}

# ── threshold slider ──────────────────────────────────────────────────────────
SLIDER_MAX = 200

def slider_to_threshold(pos: int) -> float:
    if pos == 0:
        return 0.0
    return round(10 ** (pos / 40.0))

def threshold_to_label(pos: int) -> str:
    t = slider_to_threshold(pos)
    if t == 0:     return "0 M €"
    if t >= 1000:  return f"{t / 1000:.0f} B €"
    return f"{t:.0f} M €"

def fmt_flow(val_meur: float) -> str:
    """Format a flow value (million EUR) as a compact string."""
    if val_meur >= 1_000_000:
        return f"€{val_meur/1e6:.1f}T"
    if val_meur >= 1_000:
        return f"€{val_meur/1e3:.0f}B"
    return f"€{val_meur:.0f}M"


# ── main app ───────────────────────────────────────────────────────────────────

class FlowMapApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("EU Trade Flow Map — Interactive Explorer")
        root.minsize(1200, 750)

        # ── cached data ────────────────────────────────────────────────────
        self.export_matrix = None
        self.net_matrix    = None
        self.eu_gdf        = None
        self.world_gdf     = None
        self.centroids     = None

        # ── control variables ──────────────────────────────────────────────
        self.data_mode_var  = tk.StringVar(value='gross')   # gross | net
        self.style_var      = tk.StringVar(value='straight')  # straight | spiral
        self.alpha_var      = tk.IntVar(value=25)
        self.threshold_var  = tk.IntVar(value=0)
        self.year_var       = tk.StringVar(value=str(AVAILABLE_YEARS[0]))
        self.zoom_region_var = tk.StringVar(value='Full EU')
        self.country_vars   = {c: tk.BooleanVar(value=False) for c in EU27_COUNTRIES}

        # ── groups ────────────────────────────────────────────────────────
        # Each entry: name → {'members': [...], 'var': BooleanVar}
        self._groups: dict = {}
        self._group_inner_frame: tk.Frame = None  # rebuilt on group changes
        for gname, members in DEFAULT_GROUPS.items():
            self._groups[gname] = {
                'members': list(members),
                'var': tk.BooleanVar(value=False),
            }

        # ── stats state ───────────────────────────────────────────────────
        self._current_trees = []

        self._redraw_after_id = None
        self._build_ui()
        self._load_data()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                               sashrelief=tk.RAISED, sashwidth=5)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── left control panel ─────────────────────────────────────────────
        ctrl = ttk.Frame(paned, padding=8)
        paned.add(ctrl, width=275)

        ttk.Label(ctrl, text="Controls",
                  font=('TkDefaultFont', 11, 'bold')).pack(anchor='w', pady=(0, 4))

        # Year
        yf = ttk.LabelFrame(ctrl, text="Year", padding=4)
        yf.pack(fill='x', pady=2)
        year_cb = ttk.Combobox(yf, textvariable=self.year_var,
                               values=[str(y) for y in AVAILABLE_YEARS],
                               state='readonly', width=10)
        year_cb.pack(anchor='w')
        year_cb.bind('<<ComboboxSelected>>', self._on_year_change)

        # Data mode
        df = ttk.LabelFrame(ctrl, text="Data Mode", padding=4)
        df.pack(fill='x', pady=2)
        ttk.Radiobutton(df, text="Gross Exports", variable=self.data_mode_var,
                        value='gross', command=self._schedule_redraw).pack(anchor='w')
        ttk.Radiobutton(df, text="Net Flows",     variable=self.data_mode_var,
                        value='net',   command=self._schedule_redraw).pack(anchor='w')

        # Rendering style
        sf = ttk.LabelFrame(ctrl, text="Rendering Style", padding=4)
        sf.pack(fill='x', pady=2)
        ttk.Radiobutton(sf, text="Straight Arrows", variable=self.style_var,
                        value='straight', command=self._on_style_change).pack(anchor='w')
        ttk.Radiobutton(sf, text="Spiral Trees",    variable=self.style_var,
                        value='spiral',   command=self._on_style_change).pack(anchor='w')

        # Alpha slider (hidden unless Spiral Trees is selected)
        self.alpha_frame = ttk.LabelFrame(ctrl, text="Restricting Angle α (°)", padding=4)
        self.alpha_label = ttk.Label(self.alpha_frame, text="25°")
        self.alpha_label.pack(anchor='w')
        alpha_sl = ttk.Scale(self.alpha_frame, from_=10, to=40,
                             variable=self.alpha_var, orient='horizontal',
                             command=self._on_alpha_move)
        alpha_sl.pack(fill='x')
        alpha_sl.bind('<ButtonRelease-1>', lambda _e: self._on_alpha_release())

        # Threshold
        self.thresh_frame = tf = ttk.LabelFrame(ctrl, text="Min Flow Threshold", padding=4)
        tf.pack(fill='x', pady=2)
        self.thresh_label = ttk.Label(tf, text="0 M €")
        self.thresh_label.pack(anchor='w')
        thresh_sl = ttk.Scale(tf, from_=0, to=SLIDER_MAX,
                              variable=self.threshold_var, orient='horizontal',
                              command=self._on_slider_move)
        thresh_sl.pack(fill='x')
        thresh_sl.bind('<ButtonRelease-1>', lambda _e: self._do_redraw())

        # Country groups
        self.groups_outer = ttk.LabelFrame(ctrl, text="Country Groups", padding=4)
        self.groups_outer.pack(fill='x', pady=2)
        self._rebuild_groups_panel()

        # Zoom region
        zf = ttk.LabelFrame(ctrl, text="Zoom Region", padding=4)
        zf.pack(fill='x', pady=2)
        zoom_cb = ttk.Combobox(zf, textvariable=self.zoom_region_var,
                               values=list(ZOOM_PRESETS.keys()),
                               state='readonly', width=20)
        zoom_cb.pack(anchor='w')
        zoom_cb.bind('<<ComboboxSelected>>', lambda _e: self._schedule_redraw())

        # Source countries (scrollable)
        cf = ttk.LabelFrame(ctrl, text="Source Countries", padding=4)
        cf.pack(fill='both', expand=True, pady=2)

        btn_row = ttk.Frame(cf)
        btn_row.pack(fill='x', pady=(0, 3))
        ttk.Button(btn_row, text="Select All", command=self._select_all).pack(
            side='left', padx=(0, 3))
        ttk.Button(btn_row, text="Clear All", command=self._clear_all).pack(side='left')

        sc = tk.Canvas(cf, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(cf, orient='vertical', command=sc.yview)
        sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        sc.pack(side='left', fill='both', expand=True)

        self._country_inner = ttk.Frame(sc)
        iid = sc.create_window((0, 0), window=self._country_inner, anchor='nw')
        self._country_inner.bind(
            '<Configure>',
            lambda e: sc.configure(scrollregion=sc.bbox('all')))
        sc.bind('<Configure>',
                lambda e: sc.itemconfig(iid, width=e.width))
        sc.bind_all('<MouseWheel>',
                    lambda e: sc.yview_scroll(int(-1 * e.delta / 120), 'units'))

        self._country_checkbuttons = {}
        for country in EU27_COUNTRIES:
            cb = ttk.Checkbutton(self._country_inner, text=country,
                                 variable=self.country_vars[country],
                                 command=self._schedule_redraw)
            cb.pack(anchor='w', pady=1)
            self._country_checkbuttons[country] = cb

        # Action buttons
        af = ttk.Frame(ctrl)
        af.pack(fill='x', pady=(4, 0))
        ttk.Button(af, text="Refresh",  command=self._do_redraw).pack(fill='x', pady=1)
        ttk.Button(af, text="Save PNG", command=lambda: self._save('png')).pack(
            fill='x', pady=1)
        ttk.Button(af, text="Save SVG", command=lambda: self._save('svg')).pack(
            fill='x', pady=1)

        # Status bar
        self.status_var = tk.StringVar(value="Loading…")
        ttk.Label(ctrl, textvariable=self.status_var, wraplength=250,
                  foreground='#555555', font=('TkDefaultFont', 8)).pack(
            anchor='w', pady=(4, 0))

        # ── right panel ────────────────────────────────────────────────────
        right = ttk.Frame(paned)
        paned.add(right)

        # Statistics panel (packed to bottom first so map expands above it)
        self._build_stats_panel(right)

        # Map figure
        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.04)
        self.ax      = self.fig.add_subplot(gs[0])
        self.zoom_ax = self.fig.add_subplot(gs[1])
        self.fig.patch.set_facecolor('white')

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_stats_panel(self, parent: ttk.Frame):
        """Build the statistics panel at the bottom of the right panel."""
        stats_outer = ttk.LabelFrame(parent, text="Statistics", padding=4)
        stats_outer.pack(side=tk.BOTTOM, fill=tk.X)

        # Per-tree table
        cols = ('source', 'flow', 'terminals', 'steiner', 'coverage', 'F_str')
        self.stats_tv = ttk.Treeview(stats_outer, columns=cols,
                                     show='headings', height=3)
        heads = ('Source', 'Total Flow', 'Terminals', 'Steiner', 'Coverage %', 'F_str')
        widths = (85, 80, 70, 60, 75, 60)
        for col, head, w in zip(cols, heads, widths):
            self.stats_tv.heading(col, text=head)
            self.stats_tv.column(col, width=w, anchor='e')
        self.stats_tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for treeview
        tv_vsb = ttk.Scrollbar(stats_outer, orient='vertical',
                               command=self.stats_tv.yview)
        self.stats_tv.configure(yscrollcommand=tv_vsb.set)
        tv_vsb.pack(side=tk.LEFT, fill=tk.Y)

        # Global stats (right side of stats panel)
        gf = ttk.Frame(stats_outer)
        gf.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 4))

        self._stat_vars = {}
        rows = [
            ('inter',     'Inter crossings:'),
            ('intra',     'Intra crossings:'),
            ('coverage',  'Total coverage:'),
            ('info_loss', 'Info loss:'),
        ]
        for key, label in rows:
            v = tk.StringVar(value='—')
            self._stat_vars[key] = v
            row_f = ttk.Frame(gf)
            row_f.pack(anchor='w', pady=1)
            ttk.Label(row_f, text=label, width=18, anchor='w').pack(side='left')
            self._stat_vars[f'{key}_lbl'] = ttk.Label(row_f, textvariable=v,
                                                       width=10, anchor='e')
            self._stat_vars[f'{key}_lbl'].pack(side='left')

    def _rebuild_groups_panel(self):
        """Rebuild the groups checkbox list (called after groups change)."""
        if self._group_inner_frame is not None:
            self._group_inner_frame.destroy()
        self._group_inner_frame = ttk.Frame(self.groups_outer)
        self._group_inner_frame.pack(fill='x')

        for gname, info in self._groups.items():
            row = ttk.Frame(self._group_inner_frame)
            row.pack(fill='x', pady=1)
            ttk.Checkbutton(row, text=gname, variable=info['var'],
                            command=self._schedule_redraw).pack(side='left')
            members_str = ', '.join(info['members'][:3])
            if len(info['members']) > 3:
                members_str += '…'
            ttk.Label(row, text=f"({members_str})", foreground='#777777',
                      font=('TkDefaultFont', 7)).pack(side='left', padx=(2, 0))

        ttk.Button(self._group_inner_frame, text="New Group…",
                   command=self._new_group_dialog).pack(anchor='w', pady=(3, 0))

    # ── data loading ────────────────────────────────────────────────────────

    def _load_data(self):
        self.status_var.set("Loading trade data…")
        self.root.update_idletasks()
        try:
            year = int(self.year_var.get())
            self.export_matrix, self.net_matrix, _ = load_trade_data(
                DATA_FILE, LABEL_FILE, year=year)
            invalidate_spiral_cache()

            if self.eu_gdf is None:
                self.status_var.set("Loading map shapefiles…")
                self.root.update_idletasks()
                self.eu_gdf    = load_eu_map()
                self.world_gdf = load_world_map()
                self.centroids = get_centroids(self.eu_gdf)

            self.status_var.set("Ready.")
            self._do_redraw()
        except Exception as exc:
            self.status_var.set(f"Error: {exc}")
            messagebox.showerror("Load Error", str(exc))

    def _on_year_change(self, _event=None):
        self._load_data()

    # ── mode / style callbacks ──────────────────────────────────────────────

    def _on_style_change(self):
        if self.style_var.get() == 'spiral':
            self.alpha_frame.pack(fill='x', pady=2, before=self.thresh_frame)
        else:
            self.alpha_frame.pack_forget()
        invalidate_spiral_cache()
        self._schedule_redraw()

    def _on_alpha_move(self, value):
        self.alpha_label.config(text=f"{int(float(value))}°")

    def _on_alpha_release(self):
        invalidate_spiral_cache()
        self._do_redraw()

    def _on_slider_move(self, value):
        self.thresh_label.config(text=threshold_to_label(int(float(value))))

    # ── groups dialog ────────────────────────────────────────────────────────

    def _new_group_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("New Country Group")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        ttk.Label(dlg, text="Group name:", padding=(8, 6, 8, 2)).pack(anchor='w')
        name_var = tk.StringVar()
        ttk.Entry(dlg, textvariable=name_var, width=24).pack(padx=8)

        ttk.Label(dlg, text="Member countries:", padding=(8, 6, 8, 2)).pack(anchor='w')

        frame = ttk.Frame(dlg, padding=(8, 0, 8, 0))
        frame.pack(fill='both')
        cvars = {}
        for i, c in enumerate(EU27_COUNTRIES):
            v = tk.BooleanVar(value=False)
            cvars[c] = v
            ttk.Checkbutton(frame, text=c, variable=v).grid(
                row=i // 3, column=i % 3, sticky='w', padx=3)

        def _ok():
            name = name_var.get().strip()
            members = [c for c, v in cvars.items() if v.get()]
            if not name:
                messagebox.showwarning("Missing name", "Enter a group name.", parent=dlg)
                return
            if len(members) < 2:
                messagebox.showwarning("Too few", "Select at least 2 countries.", parent=dlg)
                return
            self._groups[name] = {'members': members, 'var': tk.BooleanVar(value=True)}
            self._rebuild_groups_panel()
            self._schedule_redraw()
            dlg.destroy()

        btn_row = ttk.Frame(dlg, padding=8)
        btn_row.pack()
        ttk.Button(btn_row, text="Add Group", command=_ok).pack(side='left', padx=4)
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side='left')

    # ── effective data (with groups applied) ───────────────────────────────

    def _get_effective_data(self):
        """Return (exp_mx, net_mx, centroids) after applying active groups."""
        active = {gn: info['members'] for gn, info in self._groups.items()
                  if info['var'].get()}
        if not active:
            return self.export_matrix, self.net_matrix, self.centroids
        try:
            return apply_groups(self.export_matrix, self.net_matrix,
                                self.centroids, active)
        except ValueError as e:
            messagebox.showwarning("Group conflict", str(e))
            return self.export_matrix, self.net_matrix, self.centroids

    # ── redraw ──────────────────────────────────────────────────────────────

    def _schedule_redraw(self):
        if self._redraw_after_id is not None:
            self.root.after_cancel(self._redraw_after_id)
        self._redraw_after_id = self.root.after(60, self._do_redraw)

    def _do_redraw(self):
        self._redraw_after_id = None
        if self.export_matrix is None:
            return

        exp_mx, net_mx, centroids = self._get_effective_data()
        sources     = [c for c, v in self.country_vars.items() if v.get()
                       if c in centroids]
        net_mode    = (self.data_mode_var.get() == 'net')
        spiral_mode = (self.style_var.get() == 'spiral')
        threshold   = slider_to_threshold(self.threshold_var.get())
        alpha_deg   = float(self.alpha_var.get())
        year        = self.year_var.get()

        title = (f"EU Trade Flows from {', '.join(sources[:3])}{'…' if len(sources)>3 else ''} ({year})"
                 if sources else f"EU Trade Flows ({year})  —  select sources on the left")

        trees = render_to_axes(
            self.ax, self.eu_gdf, centroids, exp_mx, sources,
            threshold_meur=threshold,
            net_mode=net_mode,
            net_matrix=net_mx,
            title=title,
            spiral_mode=spiral_mode,
            alpha_deg=alpha_deg,
            world_gdf=self.world_gdf,
            xlim=(-25, 45), ylim=(34, 72),
        )

        # ── zoom panel ──────────────────────────────────────────────────
        zpreset = ZOOM_PRESETS.get(self.zoom_region_var.get(), (-25, 45, 34, 72))
        zxlim = (zpreset[0], zpreset[1])
        zylim = (zpreset[2], zpreset[3])
        render_to_axes(
            self.zoom_ax, self.eu_gdf, centroids, exp_mx, sources,
            threshold_meur=threshold,
            net_mode=net_mode,
            net_matrix=net_mx,
            title='',
            spiral_mode=spiral_mode,
            alpha_deg=alpha_deg,
            world_gdf=self.world_gdf,
            xlim=zxlim, ylim=zylim,
        )
        self.zoom_ax.set_title(self.zoom_region_var.get(), fontsize=8, pad=3)

        self.canvas.draw_idle()
        self._current_trees = trees

        # ── statistics ──────────────────────────────────────────────────
        self._update_stats(trees, net_mx, threshold)

        # Status bar
        mode_str = f"{'net' if net_mode else 'gross'} / {'spiral' if spiral_mode else 'straight'}"
        self.status_var.set(
            f"{len(sources)} source(s)  |  {mode_str}  |  "
            f"threshold {threshold_to_label(self.threshold_var.get())}"
        )

    # ── statistics update ────────────────────────────────────────────────────

    def _update_stats(self, trees, net_matrix, threshold_meur):
        # Clear treeview
        for row in self.stats_tv.get_children():
            self.stats_tv.delete(row)

        if not trees:
            for key in ('inter', 'intra', 'coverage', 'info_loss'):
                self._stat_vars[key].set('—')
            self._stat_vars['inter_lbl'].configure(foreground='black')
            return

        # Total EU flow (sum of all positive net flows in the matrix)
        total_eu = float(net_matrix[net_matrix > threshold_meur].sum().sum())

        # Per-tree rows
        for tree in trees:
            s = compute_tree_stats(tree, self._get_effective_data()[2], total_eu)
            self.stats_tv.insert('', 'end', values=(
                tree.source_name,
                fmt_flow(tree.total_flow),
                s['n_terminals'],
                s['n_steiner'],
                f"{s['coverage_pct']:.1f}%",
                f"{s['F_str']:.2f}",
            ))

        # Cross-tree stats
        inter, intra = count_crossings(trees)
        total_shown  = sum(t.total_flow for t in trees)
        coverage_pct = total_shown / total_eu * 100.0 if total_eu > 0 else 0.0

        # Information loss: sum min(exp(i→j), exp(j→i)) / total bilateral
        # Use raw net_matrix: loss occurs where flows partially cancel
        try:
            exp = self.export_matrix
            bilateral = float((exp + exp.T).values.sum()) / 2.0
            cancelled = float(exp.values.clip(0).min(exp.T.values.clip(0)).sum()) / 2.0 \
                if hasattr(exp.values, 'sum') else 0.0
            # Simpler: loss = sum_{i<j} min(exp_ij, exp_ji)
            import numpy as np
            exp_np = exp.values
            loss = float(np.minimum(exp_np, exp_np.T).sum()) / 2.0
            info_loss_pct = loss / bilateral * 100.0 if bilateral > 0 else 0.0
        except Exception:
            info_loss_pct = 0.0

        self._stat_vars['inter'].set(str(inter))
        self._stat_vars['intra'].set(str(intra))
        self._stat_vars['coverage'].set(f"{coverage_pct:.1f}%")
        self._stat_vars['info_loss'].set(f"{info_loss_pct:.1f}%")

        # Colour-code crossing count label
        lbl = self._stat_vars['inter_lbl']
        if inter == 0:
            lbl.configure(foreground='#228822')
        elif inter <= 5:
            lbl.configure(foreground='#cc8800')
        else:
            lbl.configure(foreground='#cc2222')

    # ── select/clear ─────────────────────────────────────────────────────────

    def _select_all(self):
        for v in self.country_vars.values():
            v.set(True)
        self._schedule_redraw()

    def _clear_all(self):
        for v in self.country_vars.values():
            v.set(False)
        self._schedule_redraw()

    # ── save ─────────────────────────────────────────────────────────────────

    def _save(self, fmt: str):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        sources  = [c for c, v in self.country_vars.items() if v.get()]
        iso_part = '_'.join(SHORT_TO_ISO.get(c, c[:2]) for c in sources) or 'none'
        data_str  = self.data_mode_var.get()
        style_str = self.style_var.get()
        thresh    = int(slider_to_threshold(self.threshold_var.get()))
        year      = self.year_var.get()

        fname = f"flowmap_{iso_part}_{data_str}_{style_str}_thresh{thresh}_{year}.{fmt}"
        path  = os.path.join(OUTPUT_DIR, fname)
        dpi   = 150 if fmt == 'png' else None
        self.fig.savefig(path, dpi=dpi, bbox_inches='tight',
                         format=fmt, facecolor='white')
        self.status_var.set(f"Saved: {fname}")
        messagebox.showinfo("Saved", f"Map saved to:\n{path}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    FlowMapApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()