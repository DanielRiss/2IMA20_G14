"""
interactive.py — Tkinter GUI for exploring EU trade flow maps.

Launch with:
    python src/interactive.py

Zoom / pan freely with the matplotlib toolbar (magnifier or hand icon).
The overview inset (bottom-left) always shows the full EU and highlights
the portion currently visible in the main view.  "Reset View" snaps back
to the full EU extent.
"""

import os
import sys
import threading

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_trade_data, ISO_SHORT
from map_utils import load_eu_map, load_world_map, get_centroids
from flow_renderer import render_to_axes, invalidate_spiral_cache
from spiral_tree import (compute_tree_stats, count_crossings,
                         optimize_multi_tree, compute_inter_tree_cost,
                         DEFAULT_OPT_WEIGHTS)
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

# ── geographic extent ─────────────────────────────────────────────────────────
_FULL_XLIM = (-25.0, 45.0)
_FULL_YLIM = (34.0,  72.0)


def _is_default_view(xlim, ylim) -> bool:
    """True if the axes are at the matplotlib-default (0,1) state (no data yet)."""
    return abs(xlim[1] - xlim[0] - 1.0) < 0.01 and abs(xlim[0]) < 0.01


def _has_custom_zoom(xlim, ylim) -> bool:
    """True when the user has zoomed away from the full EU extent."""
    if _is_default_view(xlim, ylim):
        return False
    return not (
        abs(xlim[0] - _FULL_XLIM[0]) < 1.5 and
        abs(xlim[1] - _FULL_XLIM[1]) < 1.5 and
        abs(ylim[0] - _FULL_YLIM[0]) < 1.5 and
        abs(ylim[1] - _FULL_YLIM[1]) < 1.5
    )


# ── threshold slider ──────────────────────────────────────────────────────────
SLIDER_MAX = 200


def slider_to_threshold(pos: int) -> float:
    if pos == 0:
        return 0.0
    return round(10 ** (pos / 40.0))


def threshold_to_label(pos: int) -> str:
    t = slider_to_threshold(pos)
    if t == 0:    return "0 M €"
    if t >= 1000: return f"{t / 1000:.0f} B €"
    return f"{t:.0f} M €"


def fmt_flow(val_meur: float) -> str:
    if val_meur >= 1_000_000: return f"€{val_meur/1e6:.1f}T"
    if val_meur >= 1_000:     return f"€{val_meur/1e3:.0f}B"
    return f"€{val_meur:.0f}M"


# ── main app ───────────────────────────────────────────────────────────────────

class FlowMapApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("EU Trade Flow Map — Interactive Explorer")
        root.minsize(275, 700)
        root.geometry("1200x800")

        # ── cached data ────────────────────────────────────────────────────
        self.export_matrix = None
        self.net_matrix    = None
        self.eu_gdf        = None
        self.world_gdf     = None
        self.centroids     = None

        # ── overview inset handle (recreated on every redraw) ──────────────
        self._overview_ax  = None
        self._in_redraw    = False   # guard against recursive callbacks

        # ── control variables ──────────────────────────────────────────────
        self.data_mode_var  = tk.StringVar(value='gross')
        self.style_var      = tk.StringVar(value='straight')
        self.alpha_var      = tk.IntVar(value=25)
        # Width scale slider: integer position 10–500, maps to 0.10×–5.00×
        # (position 100 = 1.00× = default)
        self.width_scale_var = tk.IntVar(value=100)
        self.exponent_var    = tk.IntVar(value=50)
        self.threshold_var  = tk.IntVar(value=0)
        self.year_var       = tk.StringVar(value=str(AVAILABLE_YEARS[0]))
        self.country_vars   = {c: tk.BooleanVar(value=False) for c in EU27_COUNTRIES}
        self.focus_var      = tk.BooleanVar(value=False)

        # ── groups ─────────────────────────────────────────────────────────
        self._groups: dict = {}
        self._group_inner_frame: tk.Frame = None
        for gname, members in DEFAULT_GROUPS.items():
            self._groups[gname] = {
                'members': list(members),
                'var': tk.BooleanVar(value=False),
            }

        # ── stats state ────────────────────────────────────────────────────
        self._current_trees = []
        self._redraw_after_id = None

        # ── multi-tree optimizer state ──────────────────────────────────────
        self._opt_thread:     threading.Thread | None = None
        self._opt_stop:       threading.Event         = threading.Event()
        self._opt_orig_trees: list | None             = None  # snapshot before opt
        self._opt_trees:      list | None             = None  # latest opt result

        self._build_ui()
        self._load_data()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                               sashrelief=tk.RAISED, sashwidth=5)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── left control panel (scrollable) ────────────────────────────────
        ctrl_outer = ttk.Frame(paned)
        paned.add(ctrl_outer, width=285)

        ctrl_canvas = tk.Canvas(ctrl_outer, borderwidth=0, highlightthickness=0)
        ctrl_vsb = ttk.Scrollbar(ctrl_outer, orient='vertical', command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_vsb.set)
        ctrl_vsb.pack(side='right', fill='y')
        ctrl_canvas.pack(side='left', fill='both', expand=True)

        ctrl = ttk.Frame(ctrl_canvas, padding=8)
        _ctrl_win = ctrl_canvas.create_window((0, 0), window=ctrl, anchor='nw')

        def _on_ctrl_configure(e):
            ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all'))
        ctrl.bind('<Configure>', _on_ctrl_configure)
        ctrl_canvas.bind('<Configure>',
                         lambda e: ctrl_canvas.itemconfig(_ctrl_win, width=e.width))
        ctrl_canvas.bind_all('<MouseWheel>',
                             lambda e: ctrl_canvas.yview_scroll(
                                 int(-1 * e.delta / 120), 'units'))

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
        mf = ttk.LabelFrame(ctrl, text="Data Mode", padding=4)
        mf.pack(fill='x', pady=2)
        ttk.Radiobutton(mf, text="Gross Exports", variable=self.data_mode_var,
                        value='gross', command=self._schedule_redraw).pack(anchor='w')
        ttk.Radiobutton(mf, text="Net Flows",     variable=self.data_mode_var,
                        value='net',   command=self._schedule_redraw).pack(anchor='w')

        # Rendering style
        sf = ttk.LabelFrame(ctrl, text="Rendering Style", padding=4)
        sf.pack(fill='x', pady=2)
        ttk.Radiobutton(sf, text="Straight Arrows", variable=self.style_var,
                        value='straight', command=self._on_style_change).pack(anchor='w')
        ttk.Radiobutton(sf, text="Spiral Trees",    variable=self.style_var,
                        value='spiral',   command=self._on_style_change).pack(anchor='w')

        # Alpha slider (only visible in spiral mode)
        self.alpha_frame = ttk.LabelFrame(ctrl, text="Restricting Angle α (°)", padding=4)
        self.alpha_label = ttk.Label(self.alpha_frame, text="25°")
        self.alpha_label.pack(anchor='w')
        alpha_sl = ttk.Scale(self.alpha_frame, from_=10, to=40,
                             variable=self.alpha_var, orient='horizontal',
                             command=self._on_alpha_move)
        alpha_sl.pack(fill='x')
        alpha_sl.bind('<ButtonRelease-1>', lambda _e: self._on_alpha_release())

        # Width scale slider (only visible in spiral mode, packed alongside alpha)
        self.width_frame = ttk.LabelFrame(ctrl, text="Edge Width Scale", padding=4)

        # ── collapsible header row ────────────────────────────────────────
        width_header = ttk.Frame(self.width_frame)
        width_header.pack(fill='x')
        self._width_toggle_btn = ttk.Button(width_header, text="\u25bc", width=2,
                                            command=self._toggle_width_panel)
        self._width_toggle_btn.pack(side='right')

        # ── collapsible body (starts expanded) ───────────────────────────
        self._width_body = ttk.Frame(self.width_frame)
        self._width_body.pack(fill='x', pady=(2, 0))

        self.width_label = ttk.Label(self._width_body, text="1.00×")
        self.width_label.pack(anchor='w')
        width_sl = ttk.Scale(self._width_body, from_=10, to=500,
                             variable=self.width_scale_var, orient='horizontal',
                             command=self._on_width_move)
        width_sl.pack(fill='x')
        width_sl.bind('<ButtonRelease-1>', lambda _e: self._do_redraw())
        self.power_label = ttk.Label(self._width_body,
                                     text='Width power: 0.50  (1=linear, 0.5=\u221a)')
        self.power_label.pack(anchor='w')
        power_sl = ttk.Scale(self._width_body, from_=10, to=100,
                             variable=self.exponent_var, orient='horizontal',
                             command=self._on_power_move)
        power_sl.pack(fill='x')
        power_sl.bind('<ButtonRelease-1>', lambda _e: self._on_power_release())

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

        # Source countries (scrollable checkboxes)
        cf = ttk.LabelFrame(ctrl, text="Source Countries", padding=4)
        cf.pack(fill='both', expand=True, pady=2)

        # Instruction label
        ttk.Label(cf, text="Select sources, then click Refresh",
                  font=('TkDefaultFont', 8), foreground='#666666').pack(
            anchor='w', pady=(0, 2))

        # Focus mode toggle
        focus_cb = ttk.Checkbutton(
            cf, text="Focus: only flows between selected",
            variable=self.focus_var, command=self._on_focus_change)
        focus_cb.pack(anchor='w', pady=(0, 4))

        btn_row = ttk.Frame(cf)
        btn_row.pack(fill='x', pady=(0, 3))
        ttk.Button(btn_row, text="Select All", command=self._select_all).pack(
            side='left', padx=(0, 3))
        ttk.Button(btn_row, text="Clear All",  command=self._clear_all).pack(side='left')

        # Selection summary label — packed before canvas so side='bottom' claims space first
        self._selection_label = ttk.Label(cf, text="0 sources selected",
                                          font=('TkDefaultFont', 8), foreground='#444444')
        self._selection_label.pack(side='bottom', anchor='w', pady=(2, 0))

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
        sc.bind('<Configure>', lambda e: sc.itemconfig(iid, width=e.width))
        sc.bind_all('<MouseWheel>',
                    lambda e: sc.yview_scroll(int(-1 * e.delta / 120), 'units'))

        self._country_checkbuttons = {}
        for country in EU27_COUNTRIES:
            cb = ttk.Checkbutton(self._country_inner, text=country,
                                 variable=self.country_vars[country],
                                 command=self._on_country_toggle)
            cb.pack(anchor='w', pady=1)
            self._country_checkbuttons[country] = cb

        # Multi-tree optimizer panel
        self._build_opt_panel(ctrl)

        # Action buttons
        af = ttk.Frame(ctrl)
        af.pack(fill='x', pady=(4, 0))
        ttk.Button(af, text="Refresh",    command=self._do_redraw).pack(fill='x', pady=1)
        ttk.Button(af, text="Reset View", command=self._reset_view).pack(fill='x', pady=1)

        # Save — collapsible, starts collapsed
        save_frame = ttk.LabelFrame(ctrl, text="Save", padding=4)
        save_frame.pack(fill='x', pady=2)

        save_header = ttk.Frame(save_frame)
        save_header.pack(fill='x')
        self._save_toggle_btn = ttk.Button(save_header, text="\u25b6", width=2,
                                           command=self._toggle_save_panel)
        self._save_toggle_btn.pack(side='right')

        self._save_body = ttk.Frame(save_frame)
        # body NOT packed — collapsed by default

        ttk.Button(self._save_body, text="Save PNG",
                   command=lambda: self._save('png')).pack(fill='x', pady=1)
        ttk.Button(self._save_body, text="Save SVG",
                   command=lambda: self._save('svg')).pack(fill='x', pady=1)

        # Status bar
        self.status_var = tk.StringVar(value="Loading…")
        ttk.Label(ctrl, textvariable=self.status_var, wraplength=250,
                  foreground='#555555', font=('TkDefaultFont', 8)).pack(
            anchor='w', pady=(4, 0))

        # ── right panel ────────────────────────────────────────────────────
        right = ttk.Frame(paned)
        paned.add(right)

        # Statistics (packed to bottom first so map fills the rest)
        self._build_stats_panel(right)

        # Map figure — single full-width axes
        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 9))
        self.fig.patch.set_facecolor('white')
        self.fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.01)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Update the overview inset after any toolbar zoom / pan
        self.canvas.mpl_connect('button_release_event', self._on_canvas_release)

    def _build_stats_panel(self, parent: ttk.Frame):
        stats_outer = ttk.LabelFrame(parent, text="Statistics", padding=4)
        stats_outer.pack(side=tk.BOTTOM, fill=tk.X)

        cols   = ('source', 'flow', 'terminals', 'steiner', 'coverage', 'F_str')
        heads  = ('Source', 'Total Flow', 'Terminals', 'Steiner', 'Coverage %', 'F_str')
        widths = (85, 80, 70, 60, 75, 60)
        self.stats_tv = ttk.Treeview(stats_outer, columns=cols,
                                     show='headings', height=3)
        for col, head, w in zip(cols, heads, widths):
            self.stats_tv.heading(col, text=head)
            self.stats_tv.column(col, width=w, anchor='e')
        self.stats_tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tv_vsb = ttk.Scrollbar(stats_outer, orient='vertical',
                               command=self.stats_tv.yview)
        self.stats_tv.configure(yscrollcommand=tv_vsb.set)
        tv_vsb.pack(side=tk.LEFT, fill=tk.Y)

        gf = ttk.Frame(stats_outer)
        gf.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 4))

        self._stat_vars = {}
        for key, label in [('inter',     'Inter crossings:'),
                            ('intra',     'Intra crossings:'),
                            ('coverage',  'Total coverage:'),
                            ('info_loss', 'Info loss:')]:
            v = tk.StringVar(value='—')
            self._stat_vars[key] = v
            row_f = ttk.Frame(gf)
            row_f.pack(anchor='w', pady=1)
            ttk.Label(row_f, text=label, width=18, anchor='w').pack(side='left')
            self._stat_vars[f'{key}_lbl'] = ttk.Label(
                row_f, textvariable=v, width=10, anchor='e')
            self._stat_vars[f'{key}_lbl'].pack(side='left')

    def _rebuild_groups_panel(self):
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

    # ── multi-tree optimizer UI ──────────────────────────────────────────────

    def _build_opt_panel(self, parent: ttk.Frame):
        of = ttk.LabelFrame(parent, text="Multi-Tree Optimization", padding=4)
        of.pack(fill='x', pady=2)

        # ── collapsible header row ────────────────────────────────────────
        header = ttk.Frame(of)
        header.pack(fill='x')
        self._opt_toggle_btn = ttk.Button(header, text="\u25b6", width=2,
                                          command=self._toggle_opt_panel)
        self._opt_toggle_btn.pack(side='right')

        # ── collapsible body (starts hidden) ─────────────────────────────
        self._opt_body = ttk.Frame(of)
        # body is NOT packed here — collapsed by default

        # Weight sliders (compact, two columns)
        wf = ttk.Frame(self._opt_body)
        wf.pack(fill='x', pady=(0, 3))
        self._opt_weight_vars: dict = {}
        for col, (key, label) in enumerate([('c_cross', 'Cross w:'),
                                             ('c_overlap', 'Overlap w:')]):
            v = tk.DoubleVar(value=DEFAULT_OPT_WEIGHTS[key])
            self._opt_weight_vars[key] = v
            ttk.Label(wf, text=label, width=9).grid(row=0, column=col*2, sticky='w')
            ttk.Entry(wf, textvariable=v, width=5).grid(row=0, column=col*2+1,
                                                         padx=(0, 6))

        # Iterations entry
        iter_f = ttk.Frame(self._opt_body)
        iter_f.pack(fill='x', pady=(0, 3))
        ttk.Label(iter_f, text="Max iter:").pack(side='left')
        self._opt_maxiter_var = tk.IntVar(value=2000)
        ttk.Entry(iter_f, textvariable=self._opt_maxiter_var, width=6).pack(
            side='left', padx=(2, 0))

        # Buttons row
        br = ttk.Frame(self._opt_body)
        br.pack(fill='x')
        self._opt_start_btn = ttk.Button(br, text="Optimize Layout",
                                         command=self._start_opt)
        self._opt_start_btn.pack(side='left', padx=(0, 3))
        self._opt_stop_btn  = ttk.Button(br, text="Stop",
                                         command=self._stop_opt, state='disabled')
        self._opt_stop_btn.pack(side='left', padx=(0, 3))
        self._opt_reset_btn = ttk.Button(br, text="Reset",
                                         command=self._reset_opt, state='disabled')
        self._opt_reset_btn.pack(side='left')

        # Status label
        self._opt_status_var = tk.StringVar(value="")
        ttk.Label(self._opt_body, textvariable=self._opt_status_var,
                  foreground='#555555', font=('TkDefaultFont', 8),
                  wraplength=240).pack(anchor='w', pady=(2, 0))

    # ── optimizer callbacks ──────────────────────────────────────────────────

    def _toggle_opt_panel(self):
        if self._opt_body.winfo_ismapped():
            self._opt_body.pack_forget()
            self._opt_toggle_btn.config(text="\u25b6")
        else:
            self._opt_body.pack(fill='x', pady=(2, 0))
            self._opt_toggle_btn.config(text="\u25bc")

    def _toggle_width_panel(self):
        if self._width_body.winfo_ismapped():
            self._width_body.pack_forget()
            self._width_toggle_btn.config(text="\u25b6")
        else:
            self._width_body.pack(fill='x', pady=(2, 0))
            self._width_toggle_btn.config(text="\u25bc")

    def _toggle_save_panel(self):
        if self._save_body.winfo_ismapped():
            self._save_body.pack_forget()
            self._save_toggle_btn.config(text="\u25b6")
        else:
            self._save_body.pack(fill='x', pady=(2, 0))
            self._save_toggle_btn.config(text="\u25bc")

    def _start_opt(self):
        if not self._current_trees:
            messagebox.showinfo("No trees",
                                "Render in Spiral Trees mode first, then optimize.")
            return
        if self.style_var.get() != 'spiral':
            messagebox.showinfo("Spiral only",
                                "Multi-tree optimization applies to Spiral Trees mode only.")
            return

        # Snapshot the trees before optimizing so Reset can restore them
        import copy
        self._opt_orig_trees = copy.deepcopy(self._current_trees)
        self._opt_trees      = None

        self._opt_start_btn.config(state='disabled')
        self._opt_stop_btn.config(state='normal')
        self._opt_reset_btn.config(state='disabled')
        self._opt_status_var.set("Starting…")

        self._opt_stop.clear()
        self._opt_thread = threading.Thread(
            target=self._run_opt_thread, daemon=True)
        self._opt_thread.start()

    def _run_opt_thread(self):
        """Background thread: runs SA optimizer and posts updates to main thread."""
        _, _, centroids = self._get_effective_data()
        weights = {k: v.get() for k, v in self._opt_weight_vars.items()}
        max_iter = max(1, self._opt_maxiter_var.get())

        def _on_update(it, cost, trees):
            # Schedule UI update on the main thread (thread-safe Tkinter API)
            try:
                self.root.after(
                    0,
                    lambda it=it, cost=cost, trees=trees:
                        self._apply_opt_update(it, cost, trees)
                )
            except Exception:
                pass   # root may have been destroyed

        optimize_multi_tree(
            trees=list(self._current_trees),
            centroids=centroids,
            stop_event=self._opt_stop,
            on_update=_on_update,
            weights=weights,
            max_iter=max_iter,
            update_every=max(1, max_iter // 30),   # ~30 visual updates
        )

    def _apply_opt_update(self, it: int, cost: float, trees: list):
        """Called on main thread; updates display and button states."""
        self._opt_trees     = trees
        self._current_trees = trees

        if it == -1:   # optimization finished / stopped
            self._opt_status_var.set(f"Done — cost: {cost:.3f}")
            self._opt_start_btn.config(state='normal')
            self._opt_stop_btn.config(state='disabled')
            self._opt_reset_btn.config(state='normal')
        else:
            self._opt_status_var.set(f"Iter {it}  |  cost: {cost:.3f}")

        self._redraw_with_trees(trees)
        _, net_mx, _ = self._get_effective_data()
        threshold    = slider_to_threshold(self.threshold_var.get())
        self._update_stats(trees, net_mx, threshold)

    def _stop_opt(self):
        self._opt_stop.set()
        self._opt_stop_btn.config(state='disabled')
        # _apply_opt_update with it=-1 will re-enable Start once the thread exits

    def _reset_opt(self):
        if self._opt_orig_trees is not None:
            self._current_trees = self._opt_orig_trees
            self._opt_trees     = None
            self._opt_reset_btn.config(state='disabled')
            self._opt_status_var.set("Reset to original layout.")
            self._redraw_with_trees(self._opt_orig_trees)

    def _redraw_with_trees(self, trees: list):
        """Re-render the map using the given (possibly optimised) trees."""
        if self.export_matrix is None:
            return

        exp_mx, net_mx, centroids = self._get_effective_data()
        sources = [c for c, v in self.country_vars.items() if v.get()
                   if c in centroids]

        # Focus-mode column filter (same as _do_redraw)
        if self.focus_var.get() and len(sources) >= 2:
            sources_set = set(sources)
            exp_mx = exp_mx.copy(); net_mx = net_mx.copy()
            for col in exp_mx.columns:
                if col not in sources_set:
                    exp_mx[col] = 0.0; net_mx[col] = 0.0

        net_mode    = (self.data_mode_var.get() == 'net')
        alpha_deg   = float(self.alpha_var.get())
        threshold   = slider_to_threshold(self.threshold_var.get())
        year        = self.year_var.get()

        old_xlim = self.ax.get_xlim()
        old_ylim = self.ax.get_ylim()
        zoomed   = _has_custom_zoom(old_xlim, old_ylim)

        if self._overview_ax is not None:
            try:
                self._overview_ax.remove()
            except Exception:
                pass
            self._overview_ax = None

        title = (
            f"EU Trade Flows from "
            f"{', '.join(sources[:3])}{'…' if len(sources) > 3 else ''} ({year})"
            if sources else
            f"EU Trade Flows ({year})"
        )

        self._in_redraw = True
        try:
            render_to_axes(
                self.ax, self.eu_gdf, centroids, exp_mx, sources,
                threshold_meur=threshold,
                net_mode=net_mode,
                net_matrix=net_mx,
                title=title,
                spiral_mode=True,
                alpha_deg=alpha_deg,
                world_gdf=self.world_gdf,
                xlim=_FULL_XLIM, ylim=_FULL_YLIM,
                precomputed_trees=trees,
                width_scale=self.width_scale_var.get() / 100.0,
                exponent=self.exponent_var.get() / 100.0,
            )
            if zoomed:
                self.ax.set_xlim(old_xlim)
                self.ax.set_ylim(old_ylim)
            self._draw_overview_inset()
            self.canvas.draw_idle()
        finally:
            self._in_redraw = False

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

    # ── style / mode callbacks ──────────────────────────────────────────────

    def _on_style_change(self):
        if self.style_var.get() == 'spiral':
            self.alpha_frame.pack(fill='x', pady=2, before=self.thresh_frame)
            self.width_frame.pack(fill='x', pady=2, before=self.thresh_frame)
        else:
            self.alpha_frame.pack_forget()
            self.width_frame.pack_forget()
        invalidate_spiral_cache()
        self._schedule_redraw()

    def _on_width_move(self, value):
        scale = int(float(value)) / 100.0
        self.width_label.config(text=f"{scale:.2f}×")

    def _on_power_move(self, value):
        exp = int(float(value)) / 100.0
        self.power_label.config(text=f'Width power: {exp:.2f}  (1=linear, 0.5=√)')

    def _on_power_release(self):
        invalidate_spiral_cache()
        self._do_redraw()

    def _on_alpha_move(self, value):
        self.alpha_label.config(text=f"{int(float(value))}°")

    def _on_alpha_release(self):
        invalidate_spiral_cache()
        self._do_redraw()

    def _on_focus_change(self):
        """Focus mode changed — cached trees used different flow values, must rebuild."""
        invalidate_spiral_cache()
        self._schedule_redraw()

    def _on_slider_move(self, value):
        self.thresh_label.config(text=threshold_to_label(int(float(value))))

    # ── canvas mouse release (toolbar zoom/pan end) ─────────────────────────

    def _on_canvas_release(self, _event):
        """Refresh the overview inset whenever the user finishes a zoom/pan."""
        if not self._in_redraw:
            self._draw_overview_inset()
            self.canvas.draw_idle()

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
            name    = name_var.get().strip()
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

    # ── effective data (groups applied) ────────────────────────────────────

    def _get_effective_data(self):
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
        # Any parameter change invalidates a running or completed optimisation
        if self._opt_thread is not None and self._opt_thread.is_alive():
            self._opt_stop.set()
        self._opt_trees      = None
        self._opt_orig_trees = None
        if hasattr(self, '_opt_reset_btn'):
            self._opt_reset_btn.config(state='disabled')
        if self._redraw_after_id is not None:
            self.root.after_cancel(self._redraw_after_id)
        self._redraw_after_id = self.root.after(60, self._do_redraw)

    def _do_redraw(self):
        self._redraw_after_id = None
        if self.export_matrix is None:
            return

        exp_mx, net_mx, centroids = self._get_effective_data()
        sources = [c for c, v in self.country_vars.items() if v.get()
                   if c in centroids]

        # Focus mode: restrict each source's destinations to only other
        # selected sources.  E.g. Germany + France → only Germany↔France flows.
        if self.focus_var.get() and len(sources) >= 2:
            sources_set = set(sources)
            exp_mx = exp_mx.copy()
            net_mx = net_mx.copy()
            for col in exp_mx.columns:
                if col not in sources_set:
                    exp_mx[col] = 0.0
                    net_mx[col] = 0.0
        net_mode    = (self.data_mode_var.get() == 'net')
        spiral_mode = (self.style_var.get() == 'spiral')
        threshold   = slider_to_threshold(self.threshold_var.get())
        alpha_deg   = float(self.alpha_var.get())
        year        = self.year_var.get()

        # ── preserve user zoom across redraws ─────────────────────────────
        old_xlim = self.ax.get_xlim()
        old_ylim = self.ax.get_ylim()
        zoomed   = _has_custom_zoom(old_xlim, old_ylim)

        # Remove stale overview before ax.clear() inside render_to_axes
        if self._overview_ax is not None:
            try:
                self._overview_ax.remove()
            except Exception:
                pass
            self._overview_ax = None

        title = (
            f"EU Trade Flows from "
            f"{', '.join(sources[:3])}{'…' if len(sources) > 3 else ''} ({year})"
            if sources else
            f"EU Trade Flows ({year})  —  select source countries on the left"
        )

        self._in_redraw = True
        try:
            trees = render_to_axes(
                self.ax, self.eu_gdf, centroids, exp_mx, sources,
                threshold_meur=threshold,
                net_mode=net_mode,
                net_matrix=net_mx,
                title=title,
                spiral_mode=spiral_mode,
                alpha_deg=alpha_deg,
                world_gdf=self.world_gdf,
                xlim=_FULL_XLIM, ylim=_FULL_YLIM,
                width_scale=self.width_scale_var.get() / 100.0,
                exponent=self.exponent_var.get() / 100.0,
            )

            # Restore zoom (render_to_axes resets to full EU)
            if zoomed:
                self.ax.set_xlim(old_xlim)
                self.ax.set_ylim(old_ylim)

            self._draw_overview_inset()
            self.canvas.draw_idle()
        finally:
            self._in_redraw = False

        self._current_trees = trees
        self._update_stats(trees, net_mx, threshold)

        mode_str  = f"{'net' if net_mode else 'gross'} / {'spiral' if spiral_mode else 'straight'}"
        focus_str = "  [focus]" if (self.focus_var.get() and len(sources) >= 2) else ""
        zoom_str  = "  [zoomed — use toolbar Home ⌂ to reset]" if zoomed else ""
        self.status_var.set(
            f"{len(sources)} source(s)  |  {mode_str}  |  "
            f"threshold {threshold_to_label(self.threshold_var.get())}{focus_str}{zoom_str}"
        )

    # ── overview inset ──────────────────────────────────────────────────────

    def _draw_overview_inset(self):
        """Mini overview map (bottom-left) with red rectangle = current view."""
        if self._overview_ax is not None:
            try:
                self._overview_ax.remove()
            except Exception:
                pass
            self._overview_ax = None

        if self.eu_gdf is None:
            return

        # Create inset: 17% wide × 22% tall, anchored to bottom-left of main ax
        ov = self.ax.inset_axes([0.005, 0.015, 0.17, 0.22])
        self._overview_ax = ov

        ov.set_facecolor('#c8dff0')
        self.eu_gdf.plot(ax=ov, color='#f0f4e8', edgecolor='#888888',
                         linewidth=0.25, zorder=2)
        ov.set_xlim(-25, 45)
        ov.set_ylim(34, 72)
        ov.set_aspect('equal')
        ov.axis('off')

        # Red rectangle showing the currently-visible portion of the main map
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        if _has_custom_zoom(xlim, ylim):
            rx = max(xlim[0], -25);  ry = max(ylim[0], 34)
            rw = min(xlim[1],  45) - rx
            rh = min(ylim[1],  72) - ry
            if rw > 0 and rh > 0:
                ov.add_patch(mpatches.Rectangle(
                    (rx, ry), rw, rh,
                    edgecolor='#cc2222', facecolor='#cc2222',
                    alpha=0.28, linewidth=1.5, zorder=3,
                ))

        ov.set_title('Overview', fontsize=5.5, pad=2, color='#666666')
        for spine in ov.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#999999')
            spine.set_linewidth(0.6)

    # ── view controls ───────────────────────────────────────────────────────

    def _reset_view(self):
        """Snap main map back to the full EU extent."""
        self.ax.set_xlim(_FULL_XLIM)
        self.ax.set_ylim(_FULL_YLIM)
        self._draw_overview_inset()
        self.canvas.draw_idle()
        self.status_var.set(self.status_var.get().split("  [zoomed")[0])

    # ── statistics update ────────────────────────────────────────────────────

    def _update_stats(self, trees, net_matrix, threshold_meur):
        for row in self.stats_tv.get_children():
            self.stats_tv.delete(row)

        if not trees:
            for key in ('inter', 'intra', 'coverage', 'info_loss'):
                self._stat_vars[key].set('—')
            self._stat_vars['inter_lbl'].configure(foreground='black')
            return

        total_eu = float(net_matrix[net_matrix > threshold_meur].sum().sum())

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

        inter, intra = count_crossings(trees)
        total_shown  = sum(t.total_flow for t in trees)
        coverage_pct = total_shown / total_eu * 100.0 if total_eu > 0 else 0.0

        try:
            import numpy as np
            exp_np    = self.export_matrix.values
            bilateral = float((exp_np + exp_np.T).sum()) / 2.0
            loss      = float(np.minimum(exp_np, exp_np.T).sum()) / 2.0
            info_loss_pct = loss / bilateral * 100.0 if bilateral > 0 else 0.0
        except Exception:
            info_loss_pct = 0.0

        self._stat_vars['inter'].set(str(inter))
        self._stat_vars['intra'].set(str(intra))
        self._stat_vars['coverage'].set(f"{coverage_pct:.1f}%")
        self._stat_vars['info_loss'].set(f"{info_loss_pct:.1f}%")

        lbl = self._stat_vars['inter_lbl']
        if inter == 0:
            lbl.configure(foreground='#228822')
        elif inter <= 5:
            lbl.configure(foreground='#cc8800')
        else:
            lbl.configure(foreground='#cc2222')

    # ── country toggle (no auto-rebuild) ─────────────────────────────────────

    def _on_country_toggle(self):
        """Checkbox callback: invalidates optimisation state but does NOT redraw.
        The user must click Refresh to trigger tree construction."""
        if self._opt_thread is not None and self._opt_thread.is_alive():
            self._opt_stop.set()
        self._opt_trees      = None
        self._opt_orig_trees = None
        if hasattr(self, '_opt_reset_btn'):
            self._opt_reset_btn.config(state='disabled')
        self._update_selection_label()

    def _update_selection_label(self):
        """Refresh the 'N sources selected — X B EUR' summary label."""
        selected = [c for c, v in self.country_vars.items() if v.get()]
        n = len(selected)
        if self.net_matrix is not None and n > 0:
            total_m = 0.0
            for c in selected:
                if c in self.net_matrix.index:
                    row = self.net_matrix.loc[c]
                    total_m += float(
                        row.drop(c, errors='ignore').clip(lower=0).sum()
                    )
            total_b = total_m / 1000.0
            text = (f"{n} source{'s' if n != 1 else ''} selected"
                    f" — {total_b:,.0f} B EUR total flow")
        else:
            text = f"{n} source{'s' if n != 1 else ''} selected"
        self._selection_label.config(text=text)

    # ── select / clear ───────────────────────────────────────────────────────

    def _select_all(self):
        for v in self.country_vars.values():
            v.set(True)
        self._on_country_toggle()

    def _clear_all(self):
        for v in self.country_vars.values():
            v.set(False)
        self._on_country_toggle()

    # ── save ─────────────────────────────────────────────────────────────────

    def _save(self, fmt: str):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        sources   = [c for c, v in self.country_vars.items() if v.get()]
        iso_part  = '_'.join(SHORT_TO_ISO.get(c, c[:2]) for c in sources) or 'none'
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
