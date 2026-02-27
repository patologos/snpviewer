"""
SNP Viewer - Touchstone/SNP File Viewer and Converter
A graphical application for loading, visualizing, and converting S-parameter files.

----------------------------------------------------------------------------
"THE BEER-WARE LICENSE" (Revision 42):
The author(s) of this file wrote it. As long as you retain this notice you
can do whatever you want with this stuff. If we meet some day, and you think
this stuff is worth it, you can buy me a beer in return.
----------------------------------------------------------------------------
"""

import sys
import os
import numpy as np

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QListWidget, QListWidgetItem, QPushButton, QGroupBox,
        QCheckBox, QGridLayout, QComboBox, QDoubleSpinBox, QLabel,
        QTabBar, QFileDialog, QStatusBar, QToolBar, QAction, QMessageBox,
        QInputDialog, QSizePolicy, QScrollArea, QFrame
    )
    from PyQt5.QtCore import Qt, pyqtSignal as Signal, QSize
    from PyQt5.QtGui import QIcon, QFont, QColor, QPalette
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg, NavigationToolbar2QT
    )
except ImportError:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QListWidget, QListWidgetItem, QPushButton, QGroupBox,
        QCheckBox, QGridLayout, QComboBox, QDoubleSpinBox, QLabel,
        QTabBar, QFileDialog, QStatusBar, QToolBar, QMessageBox,
        QInputDialog, QSizePolicy, QScrollArea, QFrame
    )
    from PySide6.QtCore import Qt, Signal, QSize
    from PySide6.QtGui import QIcon, QFont, QColor, QPalette, QAction
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg, NavigationToolbar2QT
    )

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from scipy.signal import find_peaks

import skrf as rf


# ---------------------------------------------------------------------------
# Frequency Scaling Helper
# ---------------------------------------------------------------------------

# Ordered from largest to smallest so the first match wins
_FREQ_PREFIXES = [
    (1e12, 'THz'),
    (1e9,  'GHz'),
    (1e6,  'MHz'),
    (1e3,  'kHz'),
    (1,    'Hz'),
]


def auto_freq_scale(freq_hz):
    """Pick the best engineering prefix for a frequency array in Hz.

    Returns (scaled_array, unit_string).  The chosen prefix is the largest
    one that keeps the maximum value >= 1.
    """
    f_max = np.max(np.abs(freq_hz)) if len(freq_hz) else 0
    for divisor, unit in _FREQ_PREFIXES:
        if f_max >= divisor:
            return freq_hz / divisor, unit
    return freq_hz, 'Hz'


# ---------------------------------------------------------------------------
# Settings Loader
# ---------------------------------------------------------------------------

class AppSettings:
    """Load and expose settings from a human-readable .conf file.

    The file is searched in this order:
      1. <directory of snpviewer.py>/snpviewer.conf
      2. ~/.snpviewer.conf

    Lines starting with '#' are comments.  Each setting is a simple
    ``key = value`` pair.  Unknown keys are silently ignored.
    """

    _VALID_PARAM_TYPES = {'S', 'Z', 'Y'}
    _VALID_FORMATS = {'DB', 'MA', 'RI'}

    def __init__(self):
        # Defaults (used when no conf file is found or a key is missing)
        self.param_type: str = 'S'
        self.s_default_params: list = [(0, 0), (1, 0)]  # S1,1 and S2,1
        self.z_default_params: list = [(0, 0)]
        self.y_default_params: list = [(0, 0)]
        self.plot_format: str = 'DB'
        self.z0: float = 50.0

        # Graph / font settings
        self.font_family: str = 'sans-serif'
        self.font_size: int = 10    # base body text
        self.label_size: int = 11   # axis labels
        self.title_size: int = 12   # plot title
        self.tick_size: int = 9     # tick labels
        self.legend_size: int = 9   # legend text

        self._path = self._find_conf_file()
        if self._path:
            self._load(self._path)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def default_params_for(self, param_type: str) -> list:
        """Return the default (m, n) list for the given param type."""
        pt = param_type.upper()
        if pt == 'Z':
            return list(self.z_default_params)
        if pt == 'Y':
            return list(self.y_default_params)
        return list(self.s_default_params)

    def conf_path(self):
        """Return the path to the loaded conf file, or None."""
        return self._path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_conf_file():
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'snpviewer.conf'),
            os.path.expanduser('~/.snpviewer.conf'),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _load(self, path):
        raw = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip().lower()
                value = value.strip()
                # Strip inline comments
                if '#' in value:
                    value = value[:value.index('#')].strip()
                raw[key] = value

        # param_type
        if 'param_type' in raw:
            v = raw['param_type'].upper()
            if v in self._VALID_PARAM_TYPES:
                self.param_type = v

        # default param selections
        self.s_default_params = self._parse_params(
            raw.get('s_default_params', ''), [(0, 0), (1, 0)])
        self.z_default_params = self._parse_params(
            raw.get('z_default_params', ''), [(0, 0)])
        self.y_default_params = self._parse_params(
            raw.get('y_default_params', ''), [(0, 0)])

        # plot_format
        if 'plot_format' in raw:
            v = raw['plot_format'].upper()
            if v in self._VALID_FORMATS:
                self.plot_format = v

        # z0
        if 'z0' in raw:
            try:
                v = float(raw['z0'])
                if 0.1 <= v <= 10000.0:
                    self.z0 = v
            except ValueError:
                pass

        # font / graph settings
        if 'font_family' in raw:
            self.font_family = raw['font_family']
        for _key, _attr, _lo, _hi in [
            ('font_size',   'font_size',   4, 32),
            ('label_size',  'label_size',  4, 32),
            ('title_size',  'title_size',  4, 32),
            ('tick_size',   'tick_size',   4, 24),
            ('legend_size', 'legend_size', 4, 24),
        ]:
            if _key in raw:
                try:
                    v = int(raw[_key])
                    if _lo <= v <= _hi:
                        setattr(self, _attr, v)
                except ValueError:
                    pass

    @staticmethod
    def _parse_params(value: str, default: list) -> list:
        """Parse a param list like '11, 21' into [(0,0),(1,0)].

        Special tokens:
          'all'  -> sentinel meaning "check everything"
          'none' -> empty list
        """
        v = value.strip().lower()
        if not v:
            return default
        if v == 'none':
            return []
        if v == 'all':
            return None  # None = "all available"

        result = []
        for token in v.split(','):
            token = token.strip()
            if len(token) == 2 and token.isdigit():
                m = int(token[0]) - 1
                n = int(token[1]) - 1
                if m >= 0 and n >= 0:
                    result.append((m, n))
        return result if result else default


# Module-level singleton — loaded once at import time
settings = AppSettings()


# ---------------------------------------------------------------------------
# Nature Journal Color Palette
# ---------------------------------------------------------------------------

class NatureColors:
    """Color palette inspired by Nature journal publications."""

    RED = '#E64B35'
    CYAN = '#4DBBD5'
    GREEN = '#00A087'
    BLUE = '#3C5488'
    SALMON = '#F39B7F'
    SLATE = '#8491B4'
    MINT = '#91D1C2'
    DARK_RED = '#DC0000'
    BROWN = '#7E6148'
    TAN = '#B09C85'

    CYCLE = [RED, BLUE, GREEN, CYAN, SALMON, SLATE, MINT, DARK_RED, BROWN, TAN]

    # Line-style cycle — advances once per full colour cycle so traces
    # are distinguished first by colour, then by dash pattern.
    LINESTYLES = ['-', '--', '-.', ':']

    # UI colors
    BG_LIGHT = '#FAFAFA'
    BG_SIDEBAR = '#F0F0F0'
    BORDER = '#D0D0D0'
    TEXT = '#333333'
    TEXT_SECONDARY = '#666666'

    @staticmethod
    def get_color(index: int) -> str:
        return NatureColors.CYCLE[index % len(NatureColors.CYCLE)]

    @staticmethod
    def get_linestyle(file_idx: int) -> str:
        """Return a line style that advances once per file.

        All parameters from the same file share the same dash pattern so
        files are immediately visually distinguishable even with just 2-3
        files loaded.  Colors still cycle across individual traces.
        """
        return NatureColors.LINESTYLES[file_idx % len(NatureColors.LINESTYLES)]

    @staticmethod
    def apply_matplotlib_defaults():
        """Set matplotlib rcParams for Nature-style plots.

        Font and size values are read from the AppSettings singleton so
        they can be tuned via snpviewer.conf without touching this file.
        """
        plt.rcParams.update({
            'font.family': settings.font_family,
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': settings.font_size,
            'axes.labelsize': settings.label_size,
            'axes.titlesize': settings.title_size,
            'axes.titleweight': 'bold',
            'axes.linewidth': 0.8,
            'axes.edgecolor': '#333333',
            'xtick.labelsize': settings.tick_size,
            'ytick.labelsize': settings.tick_size,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'legend.fontsize': settings.legend_size,
            'legend.framealpha': 0.9,
            'legend.edgecolor': '#cccccc',
            'figure.facecolor': 'white',
            'axes.facecolor': '#F5F5F5',
            'axes.grid': False,
            'grid.alpha': 0.0,
            'grid.linestyle': '-',
            'grid.color': '#cccccc',
            'lines.linewidth': 1.8,
            'axes.prop_cycle': plt.cycler('color', NatureColors.CYCLE),
        })


# ---------------------------------------------------------------------------
# Plot Canvas (matplotlib embedded in Qt)
# ---------------------------------------------------------------------------

class PlotCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for plotting S-parameter data."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = None
        self._hover_ann = None          # annotation shown on mouse-over
        self.mpl_connect('motion_notify_event', self._on_hover)
        self._show_placeholder()

    def _clear_fig(self):
        """Clear the figure and reset all transient per-plot state."""
        self._clear_fig()
        self._hover_ann = None          # old annotation is gone with the axes

    def _show_placeholder(self):
        """Show a placeholder message when no data is loaded."""
        self._clear_fig()
        self.ax = self.fig.add_subplot(111)
        self.ax.text(
            0.5, 0.5, 'Load an SNP file to begin',
            transform=self.ax.transAxes, ha='center', va='center',
            fontsize=14, color=NatureColors.TEXT_SECONDARY, style='italic'
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.draw()

    def _style_axes(self, title=''):
        """Apply Nature-style formatting to the current axes."""
        self.ax.set_title(title, pad=10)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.legend(loc='best', frameon=True)
        self.ax.set_facecolor('#F5F5F5')
        self.ax.grid(False)

    # ------------------------------------------------------------------
    # Math Memory support
    # ------------------------------------------------------------------

    def _interp_to_freq(self, ref_network, target_network, ref_data_key):
        """Interpolate ref_network's complex data onto target_network's
        frequency grid.  ref_data_key is 's', 'z', or 'y'.

        Returns a complex array shaped [len(target_freq), n_ports, n_ports]
        or None if interpolation is impossible.
        """
        ref_f = ref_network.frequency.f
        tgt_f = target_network.frequency.f
        ref_data = getattr(ref_network, ref_data_key)   # complex 3-D array

        n_ports_ref = ref_network.number_of_ports
        n_ports_tgt = target_network.number_of_ports
        n_ports = min(n_ports_ref, n_ports_tgt)

        out = np.zeros((len(tgt_f), n_ports, n_ports), dtype=complex)
        for m in range(n_ports):
            for n in range(n_ports):
                raw = ref_data[:, m, n]
                out[:, m, n] = (
                    np.interp(tgt_f, ref_f, raw.real) +
                    1j * np.interp(tgt_f, ref_f, raw.imag)
                )
        return out

    def plot_magnitude(self, networks, param_list,
                       mem_network=None, diff_only=False):
        """Plot S-parameters in dB for multiple networks.
        networks: list of (short_name, Network) tuples.
        mem_network: optional (name, Network) memory reference tuple.
        diff_only: if True, plot only differential traces (not raw traces).
        """
        self._clear_fig()
        self.ax = self.fig.add_subplot(111)

        if not param_list or not networks:
            self._show_no_params()
            return

        trace_idx = 0
        multi = len(networks) > 1
        # Determine best unit from the first network's raw Hz values
        _, freq_unit = auto_freq_scale(networks[0][1].frequency.f)

        # Pre-compute memory reference data if set
        mem_name = mem_network[0] if mem_network else None
        mem_net = mem_network[1] if mem_network else None

        # Plot memory reference trace (always shown when memory is set)
        if mem_net is not None:
            freq_m, _ = auto_freq_scale(mem_net.frequency.f)
            n_ports_m = mem_net.number_of_ports
            for m, n in param_list:
                if m >= n_ports_m or n >= n_ports_m:
                    continue
                s_mag = np.abs(mem_net.s[:, m, n])
                s_db = 20 * np.log10(np.where(s_mag == 0, 1e-30, s_mag))
                lbl = f'MEM {mem_name} S{m+1},{n+1}' if multi else f'MEM S{m+1},{n+1}'
                self.ax.plot(freq_m, s_db,
                             color='#AAAAAA', linewidth=1.4,
                             linestyle='--', label=lbl, alpha=0.75)

        for file_idx, (name, network) in enumerate(networks):
            freq, _ = auto_freq_scale(network.frequency.f)
            n_ports = network.number_of_ports
            is_mem_self = (network is mem_net)

            # Compute interpolated memory data for this network (if set),
            # but not when this network IS the memory (would be self - self = 0)
            mem_interp = None
            if mem_net is not None and not is_mem_self:
                mem_interp = self._interp_to_freq(mem_net, network, 's')

            for m, n in param_list:
                if m >= n_ports or n >= n_ports:
                    continue
                color = NatureColors.get_color(trace_idx)
                linestyle = NatureColors.get_linestyle(file_idx)

                # Raw trace
                s_raw = network.s[:, m, n]
                s_mag = np.abs(s_raw)
                s_db = 20 * np.log10(np.where(s_mag == 0, 1e-30, s_mag))

                if not diff_only:
                    label = f'{name} S{m+1},{n+1}' if multi else f'S{m+1},{n+1}'
                    self.ax.plot(freq, s_db, color=color,
                                 label=label, linewidth=1.8,
                                 linestyle=linestyle)

                # Differential trace (skip when this network is the memory itself)
                if mem_interp is not None and m < mem_interp.shape[1] and n < mem_interp.shape[2]:
                    diff_raw = s_raw - mem_interp[:, m, n]
                    diff_mag = np.abs(diff_raw)
                    diff_db = 20 * np.log10(np.where(diff_mag == 0, 1e-30, diff_mag))
                    diff_lbl = (f'\u0394{name} S{m+1},{n+1}' if multi
                                else f'\u0394S{m+1},{n+1}')
                    self.ax.plot(freq, diff_db, color=color,
                                 label=diff_lbl, linewidth=1.8,
                                 linestyle=':', alpha=0.9)

                trace_idx += 1

        title = 'Magnitude'
        if mem_net is not None:
            title += ' (Math Memory active)'
        self.ax.set_xlabel(f'Frequency ({freq_unit})')
        self.ax.set_ylabel('Magnitude (dB)')
        self._style_axes(title)
        self.draw()

    def plot_z_magnitude(self, networks, param_list,
                         mem_network=None, diff_only=False):
        """Plot Z-parameters magnitude (dB) for multiple networks."""
        self._clear_fig()
        self.ax = self.fig.add_subplot(111)

        if not param_list or not networks:
            self._show_no_params()
            return

        trace_idx = 0
        multi = len(networks) > 1
        _, freq_unit = auto_freq_scale(networks[0][1].frequency.f)

        mem_name = mem_network[0] if mem_network else None
        mem_net = mem_network[1] if mem_network else None

        if mem_net is not None:
            freq_m, _ = auto_freq_scale(mem_net.frequency.f)
            n_ports_m = mem_net.number_of_ports
            for m, n in param_list:
                if m >= n_ports_m or n >= n_ports_m:
                    continue
                z_mag = np.abs(mem_net.z[:, m, n])
                z_db = 20 * np.log10(np.where(z_mag == 0, 1e-30, z_mag))
                lbl = f'MEM {mem_name} Z{m+1},{n+1}' if multi else f'MEM Z{m+1},{n+1}'
                self.ax.plot(freq_m, z_db,
                             color='#AAAAAA', linewidth=1.4,
                             linestyle='--', label=lbl, alpha=0.75)

        for file_idx, (name, network) in enumerate(networks):
            freq, _ = auto_freq_scale(network.frequency.f)
            n_ports = network.number_of_ports
            mem_interp = None
            if mem_net is not None and network is not mem_net:
                mem_interp = self._interp_to_freq(mem_net, network, 'z')
            for m, n in param_list:
                if m >= n_ports or n >= n_ports:
                    continue
                color = NatureColors.get_color(trace_idx)
                linestyle = NatureColors.get_linestyle(file_idx)
                z_raw = network.z[:, m, n]
                z_mag = np.abs(z_raw)
                z_db = 20 * np.log10(np.where(z_mag == 0, 1e-30, z_mag))
                if not diff_only:
                    label = f'{name} Z{m+1},{n+1}' if multi else f'Z{m+1},{n+1}'
                    self.ax.plot(freq, z_db, color=color,
                                 label=label, linewidth=1.8,
                                 linestyle=linestyle)
                if mem_interp is not None and m < mem_interp.shape[1] and n < mem_interp.shape[2]:
                    diff_raw = z_raw - mem_interp[:, m, n]
                    diff_mag = np.abs(diff_raw)
                    diff_db = 20 * np.log10(np.where(diff_mag == 0, 1e-30, diff_mag))
                    diff_lbl = (f'\u0394{name} Z{m+1},{n+1}' if multi
                                else f'\u0394Z{m+1},{n+1}')
                    self.ax.plot(freq, diff_db, color=color,
                                 label=diff_lbl, linewidth=1.8,
                                 linestyle=':', alpha=0.9)
                trace_idx += 1

        title = 'Z-Parameters (Magnitude)'
        if mem_net is not None:
            title += ' (Math Memory active)'
        self.ax.set_xlabel(f'Frequency ({freq_unit})')
        self.ax.set_ylabel('|Z| (dB\u03A9)')
        self._style_axes(title)
        self.draw()

    def plot_y_magnitude(self, networks, param_list,
                         mem_network=None, diff_only=False):
        """Plot Y-parameters magnitude (dB) for multiple networks."""
        self._clear_fig()
        self.ax = self.fig.add_subplot(111)

        if not param_list or not networks:
            self._show_no_params()
            return

        trace_idx = 0
        multi = len(networks) > 1
        _, freq_unit = auto_freq_scale(networks[0][1].frequency.f)

        mem_name = mem_network[0] if mem_network else None
        mem_net = mem_network[1] if mem_network else None

        if mem_net is not None:
            freq_m, _ = auto_freq_scale(mem_net.frequency.f)
            n_ports_m = mem_net.number_of_ports
            for m, n in param_list:
                if m >= n_ports_m or n >= n_ports_m:
                    continue
                y_mag = np.abs(mem_net.y[:, m, n])
                y_db = 20 * np.log10(np.where(y_mag == 0, 1e-30, y_mag))
                lbl = f'MEM {mem_name} Y{m+1},{n+1}' if multi else f'MEM Y{m+1},{n+1}'
                self.ax.plot(freq_m, y_db,
                             color='#AAAAAA', linewidth=1.4,
                             linestyle='--', label=lbl, alpha=0.75)

        for file_idx, (name, network) in enumerate(networks):
            freq, _ = auto_freq_scale(network.frequency.f)
            n_ports = network.number_of_ports
            mem_interp = None
            if mem_net is not None and network is not mem_net:
                mem_interp = self._interp_to_freq(mem_net, network, 'y')
            for m, n in param_list:
                if m >= n_ports or n >= n_ports:
                    continue
                color = NatureColors.get_color(trace_idx)
                linestyle = NatureColors.get_linestyle(file_idx)
                y_raw = network.y[:, m, n]
                y_mag = np.abs(y_raw)
                y_db = 20 * np.log10(np.where(y_mag == 0, 1e-30, y_mag))
                if not diff_only:
                    label = f'{name} Y{m+1},{n+1}' if multi else f'Y{m+1},{n+1}'
                    self.ax.plot(freq, y_db, color=color,
                                 label=label, linewidth=1.8,
                                 linestyle=linestyle)
                if mem_interp is not None and m < mem_interp.shape[1] and n < mem_interp.shape[2]:
                    diff_raw = y_raw - mem_interp[:, m, n]
                    diff_mag = np.abs(diff_raw)
                    diff_db = 20 * np.log10(np.where(diff_mag == 0, 1e-30, diff_mag))
                    diff_lbl = (f'\u0394{name} Y{m+1},{n+1}' if multi
                                else f'\u0394Y{m+1},{n+1}')
                    self.ax.plot(freq, diff_db, color=color,
                                 label=diff_lbl, linewidth=1.8,
                                 linestyle=':', alpha=0.9)
                trace_idx += 1

        title = 'Y-Parameters (Magnitude)'
        if mem_net is not None:
            title += ' (Math Memory active)'
        self.ax.set_xlabel(f'Frequency ({freq_unit})')
        self.ax.set_ylabel('|Y| (dBS)')
        self._style_axes(title)
        self.draw()

    def plot_phase(self, networks, param_list,
                   mem_network=None, diff_only=False):
        """Plot S-parameters phase in degrees for multiple networks."""
        self._clear_fig()
        self.ax = self.fig.add_subplot(111)

        if not param_list or not networks:
            self._show_no_params()
            return

        trace_idx = 0
        multi = len(networks) > 1
        _, freq_unit = auto_freq_scale(networks[0][1].frequency.f)

        mem_name = mem_network[0] if mem_network else None
        mem_net = mem_network[1] if mem_network else None

        if mem_net is not None:
            freq_m, _ = auto_freq_scale(mem_net.frequency.f)
            n_ports_m = mem_net.number_of_ports
            for m, n in param_list:
                if m >= n_ports_m or n >= n_ports_m:
                    continue
                lbl = f'MEM {mem_name} S{m+1},{n+1}' if multi else f'MEM S{m+1},{n+1}'
                self.ax.plot(freq_m, mem_net.s_deg[:, m, n],
                             color='#AAAAAA', linewidth=1.4,
                             linestyle='--', label=lbl, alpha=0.75)

        for file_idx, (name, network) in enumerate(networks):
            freq, _ = auto_freq_scale(network.frequency.f)
            n_ports = network.number_of_ports
            mem_interp = None
            if mem_net is not None and network is not mem_net:
                mem_interp = self._interp_to_freq(mem_net, network, 's')
            for m, n in param_list:
                if m >= n_ports or n >= n_ports:
                    continue
                color = NatureColors.get_color(trace_idx)
                linestyle = NatureColors.get_linestyle(file_idx)
                s_deg = network.s_deg[:, m, n]
                if not diff_only:
                    label = f'{name} S{m+1},{n+1}' if multi else f'S{m+1},{n+1}'
                    self.ax.plot(freq, s_deg, color=color,
                                 label=label, linewidth=1.8,
                                 linestyle=linestyle)
                if mem_interp is not None and m < mem_interp.shape[1] and n < mem_interp.shape[2]:
                    mem_deg = np.degrees(np.angle(mem_interp[:, m, n]))
                    diff_deg = s_deg - mem_deg
                    diff_lbl = (f'\u0394{name} S{m+1},{n+1}' if multi
                                else f'\u0394S{m+1},{n+1}')
                    self.ax.plot(freq, diff_deg, color=color,
                                 label=diff_lbl, linewidth=1.8,
                                 linestyle=':', alpha=0.9)
                trace_idx += 1

        title = 'Phase'
        if mem_net is not None:
            title += ' (Math Memory active)'
        self.ax.set_xlabel(f'Frequency ({freq_unit})')
        self.ax.set_ylabel('Phase (degrees)')
        self._style_axes(title)
        self.draw()

    def plot_smith(self, networks, param_list):
        """Plot S-parameters on a Smith chart for multiple networks."""
        self._clear_fig()
        self.ax = self.fig.add_subplot(111)

        if not param_list or not networks:
            self._show_no_params()
            return

        trace_idx = 0
        multi = len(networks) > 1
        first_drawn = False

        for name, network in networks:
            n_ports = network.number_of_ports
            for m, n in param_list:
                if m >= n_ports or n >= n_ports:
                    continue
                color = NatureColors.get_color(trace_idx)
                label = f'{name} S{m+1},{n+1}' if multi else f'S{m+1},{n+1}'
                network.plot_s_smith(
                    m=m, n=n, ax=self.ax, color=color,
                    label=label, linewidth=1.8,
                    draw_labels=(not first_drawn), chart_type='z'
                )
                first_drawn = True
                trace_idx += 1

        self.ax.set_title('Smith Chart', pad=10, fontsize=12, fontweight='bold')
        self.ax.legend(loc='upper right', frameon=True)
        self.draw()

    def plot_vswr(self, networks, param_list):
        """Plot VSWR for multiple networks."""
        self._clear_fig()
        self.ax = self.fig.add_subplot(111)

        if not param_list or not networks:
            self._show_no_params()
            return

        trace_idx = 0
        multi = len(networks) > 1
        _, freq_unit = auto_freq_scale(networks[0][1].frequency.f)

        for file_idx, (name, network) in enumerate(networks):
            freq, _ = auto_freq_scale(network.frequency.f)
            n_ports = network.number_of_ports
            for m, n in param_list:
                if m >= n_ports or n >= n_ports:
                    continue
                if m != n:
                    continue  # VSWR only meaningful for Snn
                color = NatureColors.get_color(trace_idx)
                s_mag = np.abs(network.s[:, m, n])
                vswr = (1 + s_mag) / (1 - s_mag)
                vswr = np.clip(vswr, 1, 100)
                label = f'{name} VSWR(S{m+1},{n+1})' if multi else f'VSWR(S{m+1},{n+1})'
                self.ax.plot(freq, vswr, color=color,
                             label=label, linewidth=1.8,
                             linestyle=NatureColors.get_linestyle(file_idx))
                trace_idx += 1

        self.ax.set_xlabel(f'Frequency ({freq_unit})')
        self.ax.set_ylabel('VSWR')
        self.ax.set_ylim(bottom=1)
        self._style_axes('VSWR')
        self.draw()

    def plot_group_delay(self, networks, param_list,
                         mem_network=None, diff_only=False):
        """Plot group delay for multiple networks."""
        self._clear_fig()
        self.ax = self.fig.add_subplot(111)

        if not param_list or not networks:
            self._show_no_params()
            return

        trace_idx = 0
        multi = len(networks) > 1
        _, freq_unit = auto_freq_scale(networks[0][1].frequency.f)

        mem_name = mem_network[0] if mem_network else None
        mem_net = mem_network[1] if mem_network else None

        def _group_delay_ns(network, m, n):
            s_phase_rad = np.unwrap(np.angle(network.s[:, m, n]))
            omega = 2 * np.pi * network.frequency.f
            if len(omega) > 1:
                return -np.gradient(s_phase_rad, omega) * 1e9
            return None

        if mem_net is not None:
            freq_m, _ = auto_freq_scale(mem_net.frequency.f)
            n_ports_m = mem_net.number_of_ports
            for m, n in param_list:
                if m >= n_ports_m or n >= n_ports_m:
                    continue
                gd = _group_delay_ns(mem_net, m, n)
                if gd is not None:
                    lbl = f'MEM {mem_name} S{m+1},{n+1}' if multi else f'MEM S{m+1},{n+1}'
                    self.ax.plot(freq_m, gd,
                                 color='#AAAAAA', linewidth=1.4,
                                 linestyle='--', label=lbl, alpha=0.75)

        for file_idx, (name, network) in enumerate(networks):
            freq, _ = auto_freq_scale(network.frequency.f)
            n_ports = network.number_of_ports
            mem_interp = None
            if mem_net is not None and network is not mem_net:
                mem_interp = self._interp_to_freq(mem_net, network, 's')
            for m, n in param_list:
                if m >= n_ports or n >= n_ports:
                    continue
                color = NatureColors.get_color(trace_idx)
                linestyle = NatureColors.get_linestyle(file_idx)
                gd = _group_delay_ns(network, m, n)
                if gd is not None:
                    if not diff_only:
                        label = f'{name} S{m+1},{n+1}' if multi else f'S{m+1},{n+1}'
                        self.ax.plot(freq, gd, color=color,
                                     label=label, linewidth=1.8,
                                     linestyle=linestyle)
                    if (mem_interp is not None
                            and m < mem_interp.shape[1]
                            and n < mem_interp.shape[2]):
                        # Reconstruct a temporary network-like object for mem GD
                        mem_phase_rad = np.unwrap(np.angle(mem_interp[:, m, n]))
                        omega = 2 * np.pi * network.frequency.f
                        mem_gd = -np.gradient(mem_phase_rad, omega) * 1e9
                        diff_lbl = (f'\u0394{name} S{m+1},{n+1}' if multi
                                    else f'\u0394S{m+1},{n+1}')
                        self.ax.plot(freq, gd - mem_gd, color=color,
                                     label=diff_lbl, linewidth=1.8,
                                     linestyle=':', alpha=0.9)
                trace_idx += 1

        title = 'Group Delay'
        if mem_net is not None:
            title += ' (Math Memory active)'
        self.ax.set_xlabel(f'Frequency ({freq_unit})')
        self.ax.set_ylabel('Group Delay (ns)')
        self._style_axes(title)
        self.draw()

    def _show_no_params(self):
        """Show message when no parameters are selected."""
        self.ax.text(
            0.5, 0.5, 'Select parameters to plot',
            transform=self.ax.transAxes, ha='center', va='center',
            fontsize=13, color=NatureColors.TEXT_SECONDARY, style='italic'
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.draw()

    def show_placeholder(self):
        self._show_placeholder()

    # ------------------------------------------------------------------
    # Q-factor measurement via interactive span selection
    # ------------------------------------------------------------------

    def start_q_measurement(self, callback):
        """Enable a span-selector on the current axes for Q measurement.

        *callback* is called with a list of per-trace result dicts once
        the user finishes dragging a region.  The selector is automatically
        removed afterwards.
        """
        if self.ax is None:
            return

        self._q_callback = callback
        self._q_span = SpanSelector(
            self.ax,
            self._on_q_span_selected,
            direction='horizontal',
            useblit=True,
            props=dict(alpha=0.25, facecolor=NatureColors.CYAN),
            interactive=False,
        )
        self.setCursor(Qt.CrossCursor)

    def _on_q_span_selected(self, xmin, xmax):
        """Called by SpanSelector when the user releases the mouse."""
        # Tear down the selector immediately
        if hasattr(self, '_q_span') and self._q_span is not None:
            self._q_span.set_visible(False)
            self._q_span = None
        self.setCursor(Qt.ArrowCursor)

        results = self._compute_q_all_traces(xmin, xmax)
        if hasattr(self, '_q_callback') and self._q_callback:
            self._q_callback(results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_freq_unit(self):
        """Extract the frequency unit string from the x-axis label."""
        xlabel = self.ax.get_xlabel() if self.ax else ''
        for unit in ('THz', 'GHz', 'MHz', 'kHz', 'Hz'):
            if unit in xlabel:
                return unit
        return ''

    @staticmethod
    def _axis_unit_to_hz(unit):
        """Return the multiplier to convert a value in *unit* to Hz."""
        return {'THz': 1e12, 'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1}.get(unit, 1)

    @staticmethod
    def _format_bw(bw_axis, axis_unit):
        """Format a bandwidth value in the best engineering unit.

        *bw_axis* is in *axis_unit* (e.g. MHz).  Returns a string like
        '593.4 kHz' or '1.23 MHz', choosing whichever prefix avoids tiny
        or huge numbers.
        """
        hz = bw_axis * PlotCanvas._axis_unit_to_hz(axis_unit)
        for prefix, scale in (('THz', 1e12), ('GHz', 1e9), ('MHz', 1e6),
                               ('kHz', 1e3), ('Hz', 1)):
            if abs(hz) >= scale:
                val = hz / scale
                # Choose decimal places so we get 3–4 significant figures
                if val >= 100:
                    fmt = f"{val:.1f}"
                elif val >= 10:
                    fmt = f"{val:.2f}"
                else:
                    fmt = f"{val:.3f}"
                return f"{fmt} {prefix}"
        return f"{hz:.4g} Hz"

    def _compute_q_one_trace(self, xdata, ydata, xmin, xmax):
        """Compute Q for a single (xdata, ydata) trace within [xmin, xmax].

        Returns a dict on success, or None if the 3 dB crossings cannot be
        found.
        """
        mask = (xdata >= xmin) & (xdata <= xmax)
        if mask.sum() < 3:
            return None

        xr = xdata[mask]
        yr = ydata[mask]

        # Peak = maximum dB in range
        peak_idx = int(np.argmax(yr))
        f0 = xr[peak_idx]
        peak_db = yr[peak_idx]
        half_power_db = peak_db - 3.0

        # Left 3 dB crossing (search left of peak)
        left_f = None
        for i in range(peak_idx, 0, -1):
            if yr[i - 1] <= half_power_db:
                dx = xr[i] - xr[i - 1]
                dy = yr[i] - yr[i - 1]
                if dy != 0:
                    left_f = xr[i - 1] + (half_power_db - yr[i - 1]) * dx / dy
                break

        # Right 3 dB crossing (search right of peak)
        right_f = None
        for i in range(peak_idx, len(xr) - 1):
            if yr[i + 1] <= half_power_db:
                dx = xr[i + 1] - xr[i]
                dy = yr[i + 1] - yr[i]
                if dy != 0:
                    right_f = xr[i] + (half_power_db - yr[i]) * dx / dy
                break

        if left_f is None or right_f is None:
            return None

        bw = right_f - left_f
        if bw <= 0:
            return None

        return {
            'f0': f0,
            'bw': bw,
            'q': f0 / bw,
            'peak_db': peak_db,
            'half_power_db': half_power_db,
            'left_f': left_f,
            'right_f': right_f,
        }

    def _compute_q_all_traces(self, xmin, xmax):
        """Compute Q for every visible data line in [xmin, xmax].

        Returns a list of dicts (one per trace that succeeded), each
        containing the trace label, colour, and Q-factor values.
        Returns an empty list if no traces could be measured.
        """
        if self.ax is None:
            return []

        f_unit = self._get_freq_unit()
        results = []

        for ln in self.ax.lines:
            xd = ln.get_xdata()
            yd = ln.get_ydata()
            # Skip annotation lines (axhline/axvline have only 2 points)
            if xd is None or len(xd) <= 2:
                continue
            xdata = np.asarray(xd, dtype=float)
            ydata = np.asarray(yd, dtype=float)

            r = self._compute_q_one_trace(xdata, ydata, xmin, xmax)
            if r is not None:
                r['label'] = ln.get_label() or ''
                r['color'] = ln.get_color()
                r['f_unit'] = f_unit
                results.append(r)

        return results

    def annotate_q_results(self, results):
        """Draw Q-measurement annotations for all traces.

        *results* is the list returned by _compute_q_all_traces().
        Each trace gets its own colour-matched markers and a shared
        text box listing all results.
        """
        if not results or self.ax is None:
            return

        self._clear_q_annotations()
        self._q_annotations = []

        f_unit = results[0]['f_unit']

        # Build the combined result text (one line-pair per trace)
        text_lines = []
        for r in results:
            f0_str = f"{r['f0']:.6g}"
            bw_str = self._format_bw(r['bw'], f_unit)
            q_str  = f"{r['q']:.0f}"
            lbl = r['label']
            if lbl and not lbl.startswith('_'):
                header = lbl
            else:
                header = None

            if header:
                text_lines.append(header)
            text_lines.append(
                f"f0 = {f0_str} {f_unit}   BW = {bw_str}   Q = {q_str}"
            )

        txt = '\n'.join(text_lines)

        # One text box for all results, placed top-left in axes coords
        text_ann = self.ax.text(
            0.02, 0.97, txt,
            transform=self.ax.transAxes,
            fontsize=9, verticalalignment='top',
            family='monospace',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor='#555555',
                alpha=0.93,
            ),
            color='#222222',
        )
        self._q_annotations.append(text_ann)

        # Per-trace markers using each trace's own colour
        for r in results:
            c = r['color']
            f0     = r['f0']
            left_f = r['left_f']
            right_f = r['right_f']
            half_db = r['half_power_db']

            # Vertical line at f0 (solid)
            self._q_annotations.append(
                self.ax.axvline(f0, color=c, linestyle='-',
                                linewidth=1.4, alpha=0.75)
            )
            # Vertical lines at 3 dB crossings (dotted)
            self._q_annotations.append(
                self.ax.axvline(left_f, color=c, linestyle=':',
                                linewidth=1.1, alpha=0.75)
            )
            self._q_annotations.append(
                self.ax.axvline(right_f, color=c, linestyle=':',
                                linewidth=1.1, alpha=0.75)
            )
            # Horizontal dashed line at the -3 dB level
            self._q_annotations.append(
                self.ax.axhline(half_db, color=c, linestyle='--',
                                linewidth=1.1, alpha=0.75)
            )
            # Double-headed arrow spanning the BW
            self._q_annotations.append(
                self.ax.annotate(
                    '', xy=(right_f, half_db), xytext=(left_f, half_db),
                    arrowprops=dict(arrowstyle='<->', color=c, lw=1.4),
                )
            )

        self.draw()

    def _clear_q_annotations(self):
        """Remove any existing Q-factor annotation artists."""
        if hasattr(self, '_q_annotations'):
            for artist in self._q_annotations:
                try:
                    artist.remove()
                except Exception:
                    pass
        self._q_annotations = []

    # ------------------------------------------------------------------
    # Change detection on differential (math memory) traces
    # ------------------------------------------------------------------

    def find_and_annotate_changes(self, prominence_db=6.0):
        """Detect significant changes in all differential (Δ) traces currently
        plotted and annotate them.

        Algorithm per trace:
          1. Noise floor = median of the diff_db values (robust baseline).
          2. Threshold  = noise_floor + prominence_db  (above the median).
          3. find_peaks on the diff_db with prominence >= prominence_db
             to locate local maxima that stand clearly above the noise.
          4. Shade each contiguous above-threshold region as a translucent span.
          5. Mark each peak with a triangle + frequency label.

        Only lines whose label starts with 'Δ' are processed.
        Returns the number of peaks found across all traces.
        """
        if self.ax is None:
            return 0

        self._clear_change_annotations()
        self._change_annotations = []

        f_unit = self._get_freq_unit()
        total_peaks = 0

        for ln in self.ax.lines:
            lbl = ln.get_label() or ''
            if not lbl.startswith('\u0394'):   # only Δ traces
                continue

            xdata = np.asarray(ln.get_xdata(), dtype=float)
            ydata = np.asarray(ln.get_ydata(), dtype=float)

            if len(xdata) < 5:
                continue

            # --- noise floor & threshold ---
            noise_floor = np.median(ydata)
            prominence_req = prominence_db
            threshold = noise_floor + prominence_req

            # --- find peaks above threshold with sufficient prominence ---
            peaks, props = find_peaks(
                ydata,
                height=threshold,
                prominence=prominence_req,
            )

            if len(peaks) == 0:
                continue

            color = ln.get_color()

            # --- shade contiguous above-threshold regions ---
            above = ydata >= threshold
            # find run starts / ends
            padded = np.concatenate(([False], above, [False]))
            diff_mask = np.diff(padded.astype(int))
            starts = np.where(diff_mask == 1)[0]
            ends   = np.where(diff_mask == -1)[0]

            for s, e in zip(starts, ends):
                x0 = xdata[s]
                x1 = xdata[min(e, len(xdata) - 1)]
                span = self.ax.axvspan(
                    x0, x1,
                    alpha=0.18, color=color, linewidth=0,
                    zorder=1,
                )
                self._change_annotations.append(span)

            # --- mark each peak ---
            for pk in peaks:
                fx = xdata[pk]
                fy = ydata[pk]
                marker = self.ax.plot(
                    fx, fy, marker='v', markersize=8,
                    color=color, alpha=0.9,
                    linestyle='none', zorder=5,
                )
                self._change_annotations.extend(marker)

                txt = self.ax.annotate(
                    f'{fx:.5g} {f_unit}',
                    xy=(fx, fy),
                    xytext=(4, 10),          # 4 pt right, 10 pt above marker
                    textcoords='offset points',
                    fontsize=7.5, color=color,
                    va='bottom', ha='left',
                    rotation=90,
                    zorder=6,
                )
                self._change_annotations.append(txt)

            total_peaks += len(peaks)

        self.draw()
        return total_peaks

    def _clear_change_annotations(self):
        """Remove any existing change-detection annotation artists."""
        if hasattr(self, '_change_annotations'):
            for artist in self._change_annotations:
                try:
                    artist.remove()
                except Exception:
                    pass
        self._change_annotations = []

    # ------------------------------------------------------------------
    # Hover tooltip
    # ------------------------------------------------------------------

    def _on_hover(self, event):
        """Show a data-point tooltip when the cursor is near a plotted line."""
        if self.ax is None or event.inaxes != self.ax:
            self._hide_hover()
            return

        PIXEL_THRESH = 20           # snap radius in screen pixels
        best_dist = PIXEL_THRESH + 1
        best_x = best_y = best_label = best_color = None

        xmin, xmax = self.ax.get_xlim()

        for ln in self.ax.lines:
            xd = ln.get_xdata()
            yd = ln.get_ydata()
            if xd is None or len(xd) <= 2:
                continue                     # skip axhline / axvline markers
            lbl = ln.get_label() or ''
            if lbl.startswith('_'):
                continue                     # skip internal matplotlib lines

            xd = np.asarray(xd, dtype=float)
            yd = np.asarray(yd, dtype=float)

            # Restrict to the currently visible x-range for speed
            vis = (xd >= xmin) & (xd <= xmax)
            if not vis.any():
                continue
            xd_v, yd_v = xd[vis], yd[vis]

            # Convert data coords → display (pixel) coords, measure distance
            pts = self.ax.transData.transform(np.column_stack([xd_v, yd_v]))
            dists = np.hypot(pts[:, 0] - event.x, pts[:, 1] - event.y)
            idx = int(np.argmin(dists))
            if dists[idx] < best_dist:
                best_dist = dists[idx]
                best_x    = xd_v[idx]
                best_y    = yd_v[idx]
                best_label = lbl
                best_color = ln.get_color()

        if best_x is None:
            self._hide_hover()
            return

        # Build tooltip text
        f_unit = self._get_freq_unit()
        if f_unit:                           # frequency-based plot
            text = f'{best_x:.5g} {f_unit}\n{best_y:.4g}'
        else:                               # Smith chart — show Re / Im
            text = f'Re: {best_x:.4g}\nIm: {best_y:.4g}'
        if best_label and not best_label.startswith('_'):
            text = f'{best_label}\n' + text

        self._show_hover(best_x, best_y, text, best_color or '#555555')

    def _show_hover(self, x, y, text, color):
        """Create or update the hover annotation at data point (x, y)."""
        if self._hover_ann is None:
            self._hover_ann = self.ax.annotate(
                text,
                xy=(x, y),
                xytext=(12, 12),
                textcoords='offset points',
                fontsize=8,
                family='monospace',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor='lightyellow',
                    edgecolor=color,
                    alpha=0.93,
                    linewidth=1.2,
                ),
                zorder=20,
            )
        else:
            self._hover_ann.set_text(text)
            self._hover_ann.xy = (x, y)
            self._hover_ann.get_bbox_patch().set_edgecolor(color)
            self._hover_ann.set_visible(True)
        self.draw_idle()

    def _hide_hover(self):
        """Hide the hover annotation without destroying it."""
        if self._hover_ann is not None and self._hover_ann.get_visible():
            self._hover_ann.set_visible(False)
            self.draw_idle()


# ---------------------------------------------------------------------------
# File List Widget
# ---------------------------------------------------------------------------

class FileListWidget(QListWidget):
    """Widget displaying loaded SNP files with multi-select support."""

    selection_updated = Signal()  # emitted when selected files change

    def __init__(self, parent=None):
        super().__init__(parent)
        self._networks = {}  # item_text -> (short_name, Network)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.setAlternatingRowColors(True)
        self.setDragDropMode(QListWidget.NoDragDrop)
        self.setStyleSheet("""
            QListWidget {
                border: 1px solid #D0D0D0;
                border-radius: 3px;
                background-color: white;
                font-size: 10pt;
            }
            QListWidget::item {
                padding: 4px 6px;
            }
            QListWidget::item:selected {
                background-color: #3C5488;
                color: white;
            }
            QListWidget::item:alternate {
                background-color: #F8F8F8;
            }
        """)

    def add_network(self, filepath):
        """Load and add a Touchstone file. Returns (True, '') on success."""
        try:
            network = rf.Network(filepath)
        except Exception as e:
            return False, str(e)

        basename = os.path.basename(filepath)
        n_ports = network.number_of_ports
        n_points = len(network.frequency.f)
        freq_start = network.frequency.f_scaled[0]
        freq_stop = network.frequency.f_scaled[-1]
        freq_unit = network.frequency.unit

        display = f"{basename}  [{n_ports}-port, {n_points} pts]"

        # Avoid duplicates
        if display in self._networks:
            display = f"{display} ({filepath})"

        self._networks[display] = (basename, network)
        item = QListWidgetItem(display)
        item.setToolTip(
            f"File: {filepath}\n"
            f"Ports: {n_ports}\n"
            f"Points: {n_points}\n"
            f"Range: {freq_start:.4g} - {freq_stop:.4g} {freq_unit}"
        )
        self.addItem(item)
        # Select the new item (add to selection)
        item.setSelected(True)
        return True, ""

    def remove_selected(self):
        """Remove all currently selected files."""
        for item in self.selectedItems():
            text = item.text()
            self._networks.pop(text, None)
            self.takeItem(self.row(item))

    def get_selected_networks(self):
        """Return list of (short_name, Network) for all selected items."""
        result = []
        for item in self.selectedItems():
            entry = self._networks.get(item.text())
            if entry:
                result.append(entry)
        return result

    def get_current_network(self):
        """Return the Network for the current (focused) item, or None.
        Used for single-file operations like Save As."""
        item = self.currentItem()
        if item:
            entry = self._networks.get(item.text())
            if entry:
                return entry[1]
        return None

    def _on_selection_changed(self):
        self.selection_updated.emit()

    def get_network_count(self):
        return len(self._networks)


# ---------------------------------------------------------------------------
# Parameter Selector
# ---------------------------------------------------------------------------

class ParameterSelector(QGroupBox):
    """Dynamic checkbox grid for S/Z/Y-parameter selection."""

    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Parameters", parent)
        self._checkboxes = []
        self._layout = QGridLayout()
        self._layout.setSpacing(2)
        self._param_type = 'S'   # 'S', 'Z', or 'Y'
        self.setLayout(self._layout)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 10pt;
                border: 1px solid #D0D0D0;
                border-radius: 3px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QCheckBox {
                font-size: 9pt;
                spacing: 3px;
            }
        """)

    def set_param_type(self, param_type):
        """Switch the displayed parameter type ('S', 'Z', or 'Y') and
        relabel existing checkboxes without resetting their checked state."""
        self._param_type = param_type.upper()
        self.setTitle(f'{self._param_type}-Parameters')
        p = self._param_type
        for cb in self._checkboxes:
            m = cb.property('row')
            n = cb.property('col')
            cb.setText(f'{p}{m + 1},{n + 1}')

    def update_for_networks(self, networks):
        """Rebuild checkboxes for the union of ports across all networks.
        networks: list of (short_name, Network) tuples.
        """
        # Remember current checked state
        prev_checked = set()
        for cb in self._checkboxes:
            if cb.isChecked():
                prev_checked.add((cb.property('row'), cb.property('col')))

        # Clear existing
        for cb in self._checkboxes:
            cb.stateChanged.disconnect(self._on_changed)
            self._layout.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes = []

        if not networks:
            return

        # Use the max port count across all selected networks
        max_ports = max(net.number_of_ports for _, net in networks)
        p = self._param_type

        for m in range(max_ports):
            for n in range(max_ports):
                cb = QCheckBox(f'{p}{m + 1},{n + 1}')
                cb.setProperty('row', m)
                cb.setProperty('col', n)
                # Restore previous state, or use settings defaults on first build
                if prev_checked:
                    cb.setChecked((m, n) in prev_checked)
                else:
                    default = settings.default_params_for(self._param_type)
                    if default is None:  # 'all' sentinel
                        cb.setChecked(True)
                    else:
                        cb.setChecked((m, n) in default)
                cb.stateChanged.connect(self._on_changed)
                self._layout.addWidget(cb, m, n)
                self._checkboxes.append(cb)

        self.setTitle(f'{p}-Parameters')

    def get_selected_params(self):
        """Return list of (m, n) tuples for checked parameters."""
        params = []
        for cb in self._checkboxes:
            if cb.isChecked():
                m = cb.property('row')
                n = cb.property('col')
                params.append((m, n))
        return params

    def _on_changed(self, state):
        self.selection_changed.emit()


# ---------------------------------------------------------------------------
# Conversion Panel
# ---------------------------------------------------------------------------

class ConversionPanel(QGroupBox):
    """Controls for converting and saving Touchstone files."""

    def __init__(self, parent=None):
        super().__init__("Convert && Save", parent)

        layout = QVBoxLayout()
        layout.setSpacing(6)

        # Parameter type
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Parameter:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems(['S', 'Z', 'Y'])
        param_row.addWidget(self.param_combo)
        layout.addLayout(param_row)

        # Data format
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['DB', 'MA', 'RI'])
        format_row.addWidget(self.format_combo)
        layout.addLayout(format_row)

        # Reference impedance
        z0_row = QHBoxLayout()
        z0_row.addWidget(QLabel("Z0 (\u03A9):"))
        self.z0_spin = QDoubleSpinBox()
        self.z0_spin.setRange(0.1, 10000.0)
        self.z0_spin.setValue(50.0)
        self.z0_spin.setDecimals(1)
        self.z0_spin.setSuffix(' \u03A9')
        z0_row.addWidget(self.z0_spin)
        layout.addLayout(z0_row)

        # Save button
        self.save_btn = QPushButton("Save As...")
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {NatureColors.BLUE};
                color: white;
                border: none;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 10pt;
            }}
            QPushButton:hover {{
                background-color: #2A4070;
            }}
            QPushButton:pressed {{
                background-color: #1E3060;
            }}
            QPushButton:disabled {{
                background-color: #AAAAAA;
            }}
        """)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        self.setLayout(layout)
        self.setStyleSheet(self.styleSheet() + """
            QGroupBox {
                font-weight: bold;
                font-size: 10pt;
                border: 1px solid #D0D0D0;
                border-radius: 3px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QComboBox, QDoubleSpinBox {
                font-size: 9pt;
                padding: 2px 4px;
            }
            QLabel {
                font-size: 9pt;
                font-weight: normal;
            }
        """)

    def update_for_network(self, network):
        """Enable/disable options based on network port count."""
        if network is None:
            self.save_btn.setEnabled(False)
            return

        self.save_btn.setEnabled(True)
        # G and H parameters only for 2-port (not implementing those
        # as scikit-rf write_touchstone supports S, Z, Y primarily)

    def save_network(self, network):
        """Open save dialog and write the network in the chosen format."""
        if network is None:
            return

        n_ports = network.number_of_ports
        default_ext = f".s{n_ports}p"
        param = self.param_combo.currentText().lower()
        form = self.format_combo.currentText().lower()

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Touchstone File",
            f"converted{default_ext}",
            f"Touchstone Files (*{default_ext});;All Files (*)"
        )

        if not filepath:
            return

        try:
            z0 = self.z0_spin.value()
            net_copy = network.copy()
            if z0 != 50.0:
                net_copy.renormalize(z0)

            if param != 's':
                self._write_converted(net_copy, filepath, param, form, z0)
            else:
                net_copy.write_touchstone(filename=filepath, form=form)

            QMessageBox.information(
                self, "Saved",
                f"File saved successfully:\n{filepath}"
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Save Error",
                f"Failed to save file:\n{str(e)}"
            )

    def _write_converted(self, network, filepath, param, form, z0):
        """Write network with parameter conversion."""
        freq = network.frequency

        if param == 'z':
            data = network.z
        elif param == 'y':
            data = network.y
        else:
            data = network.s

        n_ports = network.number_of_ports

        with open(filepath, 'w') as f:
            freq_unit = freq.unit.upper()
            f.write(f"! Converted by SNP Viewer\n")
            f.write(f"# {freq_unit} {param.upper()} {form.upper()} R {z0}\n")

            # Build a 2-D matrix [N_freq × (1 + 2*n_ports²)] then write
            # in one C-level pass with np.savetxt — much faster than a
            # Python loop over frequency points.
            cols = [freq.f]
            for m in range(n_ports):
                for n in range(n_ports):
                    val = data[:, m, n]
                    if form == 'ri':
                        cols.append(val.real)
                        cols.append(val.imag)
                    elif form == 'ma':
                        cols.append(np.abs(val))
                        cols.append(np.degrees(np.angle(val)))
                    else:  # db
                        cols.append(20 * np.log10(np.abs(val) + 1e-30))
                        cols.append(np.degrees(np.angle(val)))

            np.savetxt(f, np.column_stack(cols), fmt='%.10g')


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------

class SNPViewerApp(QMainWindow):
    """Main application window for SNP Viewer."""

    PLOT_MAGNITUDE = 0
    PLOT_PHASE = 1
    PLOT_SMITH = 2
    PLOT_VSWR = 3
    PLOT_GROUP_DELAY = 4

    # Parameter types that show a magnitude (dB) plot and support Q measurement
    _MAGNITUDE_PLOTS = {PLOT_MAGNITUDE}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SNP Viewer")
        self.resize(1200, 750)
        self.setAcceptDrops(True)

        self._param_type = settings.param_type   # 'S', 'Z', or 'Y'

        # Math Memory state
        self._mem_network = None   # (short_name, Network) or None
        self._diff_only = False    # Show differential traces only

        NatureColors.apply_matplotlib_defaults()
        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._connect_signals()
        self._update_status("Ready. Open or drag-and-drop SNP files to begin.")

    def _build_ui(self):
        """Build the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)

        # --- Left Sidebar ---
        sidebar = QWidget()
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(350)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)
        sidebar_layout.setSpacing(6)

        # File list
        files_label = QLabel("Loaded Files")
        files_label.setStyleSheet(
            "font-weight: bold; font-size: 10pt; color: #333;"
        )
        sidebar_layout.addWidget(files_label)

        self.file_list = FileListWidget()
        sidebar_layout.addWidget(self.file_list, stretch=1)

        # File buttons
        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add Files...")
        self.btn_add.setStyleSheet(f"""
            QPushButton {{
                background-color: {NatureColors.GREEN};
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
                font-weight: bold;
                font-size: 9pt;
            }}
            QPushButton:hover {{ background-color: #008070; }}
        """)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                color: #333;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
                font-size: 9pt;
            }
            QPushButton:hover { background-color: #D0D0D0; }
        """)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        sidebar_layout.addLayout(btn_row)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet("color: #D0D0D0;")
        sidebar_layout.addWidget(sep1)

        # Parameter selector (in scroll area for many-port networks)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(180)
        self.param_selector = ParameterSelector()
        self.param_selector.set_param_type(self._param_type)
        scroll.setWidget(self.param_selector)
        sidebar_layout.addWidget(scroll)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #D0D0D0;")
        sidebar_layout.addWidget(sep2)

        # Conversion panel
        self.conversion_panel = ConversionPanel()
        sidebar_layout.addWidget(self.conversion_panel)

        # Apply settings to conversion panel defaults
        fmt_idx = self.conversion_panel.format_combo.findText(settings.plot_format)
        if fmt_idx >= 0:
            self.conversion_panel.format_combo.setCurrentIndex(fmt_idx)
        self.conversion_panel.z0_spin.setValue(settings.z0)
        param_idx = self.conversion_panel.param_combo.findText(self._param_type)
        if param_idx >= 0:
            self.conversion_panel.param_combo.setCurrentIndex(param_idx)

        sidebar_layout.addStretch()

        # --- Right Content Area ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(2)

        # Tab bar for plot types
        self.plot_tab_bar = QTabBar()
        self.plot_tab_bar.addTab("Magnitude (dB)")
        self.plot_tab_bar.addTab("Phase (deg)")
        self.plot_tab_bar.addTab("Smith Chart")
        self.plot_tab_bar.addTab("VSWR")
        self.plot_tab_bar.addTab("Group Delay")
        self.plot_tab_bar.setStyleSheet(f"""
            QTabBar::tab {{
                padding: 6px 16px;
                margin-right: 2px;
                border: 1px solid #D0D0D0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                background: #F0F0F0;
                font-size: 9pt;
            }}
            QTabBar::tab:selected {{
                background: white;
                border-bottom: 2px solid {NatureColors.BLUE};
                font-weight: bold;
                color: {NatureColors.BLUE};
            }}
            QTabBar::tab:hover {{
                background: #E8E8E8;
            }}
        """)
        right_layout.addWidget(self.plot_tab_bar)

        # Plot canvas
        self.canvas = PlotCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.canvas, stretch=1)

        # Navigation toolbar
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self)
        right_layout.addWidget(self.nav_toolbar)

        # Assemble splitter
        splitter.addWidget(sidebar)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 940])

        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().setStyleSheet(
            "font-size: 9pt; color: #666; padding: 2px 6px;"
        )

    def _build_menu(self):
        """Build the menu bar."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Files...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_files)
        file_menu.addAction(open_action)

        save_action = QAction("&Save As...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("&Help")
        settings_action = QAction("&Settings Info...", self)
        settings_action.triggered.connect(self._show_settings_info)
        help_menu.addAction(settings_action)
        help_menu.addSeparator()
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_toolbar(self):
        """Build the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setStyleSheet("""
            QToolBar {
                spacing: 4px;
                padding: 2px 4px;
                border-bottom: 1px solid #D0D0D0;
            }
            QToolButton {
                font-size: 9pt;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QToolButton:hover {
                background-color: #E0E0E0;
            }
        """)
        self.addToolBar(toolbar)

        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_files)
        toolbar.addAction(open_action)

        save_action = QAction("Save As", self)
        save_action.triggered.connect(self._on_save)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # S / Z / Y parameter-type toggle buttons
        _param_btn_style = """
            QToolButton {{
                font-size: 9pt;
                font-weight: bold;
                padding: 4px 10px;
                border-radius: 3px;
                border: 1px solid #B0B0B0;
                background: #F0F0F0;
                color: #333;
            }}
            QToolButton:checked {{
                background: {active_bg};
                color: white;
                border: 1px solid {active_bg};
            }}
            QToolButton:hover:!checked {{
                background: #E0E0E0;
            }}
        """
        self.s_action = QAction("S", self)
        self.s_action.setCheckable(True)
        self.s_action.setChecked(self._param_type == 'S')
        self.s_action.setToolTip("Display S-parameters")
        self.s_action.triggered.connect(lambda: self._on_param_type_changed('S'))
        toolbar.addAction(self.s_action)

        self.z_action = QAction("Z", self)
        self.z_action.setCheckable(True)
        self.z_action.setChecked(self._param_type == 'Z')
        self.z_action.setToolTip("Display Z-parameters (impedance)")
        self.z_action.triggered.connect(lambda: self._on_param_type_changed('Z'))
        toolbar.addAction(self.z_action)

        self.y_action = QAction("Y", self)
        self.y_action.setCheckable(True)
        self.y_action.setChecked(self._param_type == 'Y')
        self.y_action.setToolTip("Display Y-parameters (admittance)")
        self.y_action.triggered.connect(lambda: self._on_param_type_changed('Y'))
        toolbar.addAction(self.y_action)

        toolbar.addSeparator()

        self.q_action = QAction("Measure Q", self)
        self.q_action.setToolTip(
            "Drag a region around a peak to compute Q = f0 / (3 dB bandwidth)"
        )
        self.q_action.setCheckable(True)
        self.q_action.setEnabled(False)
        self.q_action.triggered.connect(self._on_q_action_toggled)
        toolbar.addAction(self.q_action)

        self.clear_q_action = QAction("Clear Q", self)
        self.clear_q_action.setToolTip("Remove Q-factor annotations from the plot")
        self.clear_q_action.setEnabled(False)
        self.clear_q_action.triggered.connect(self._on_clear_q)
        toolbar.addAction(self.clear_q_action)

        toolbar.addSeparator()

        self.mem_action = QAction("Set Mem", self)
        self.mem_action.setToolTip(
            "Capture the first selected trace as the math memory reference.\n"
            "Subsequent traces will show their difference from this reference."
        )
        self.mem_action.setEnabled(False)
        self.mem_action.triggered.connect(self._on_set_mem)
        toolbar.addAction(self.mem_action)

        self.clr_mem_action = QAction("Clr Mem", self)
        self.clr_mem_action.setToolTip("Clear the math memory reference")
        self.clr_mem_action.setEnabled(False)
        self.clr_mem_action.triggered.connect(self._on_clr_mem)
        toolbar.addAction(self.clr_mem_action)

        self.diff_only_action = QAction("Diff Only", self)
        self.diff_only_action.setToolTip(
            "Toggle: show only the differential (memory-subtracted) traces, "
            "hiding the raw measurement traces"
        )
        self.diff_only_action.setCheckable(True)
        self.diff_only_action.setChecked(False)
        self.diff_only_action.setEnabled(False)
        self.diff_only_action.triggered.connect(self._on_diff_only_toggled)
        toolbar.addAction(self.diff_only_action)

        self.find_changes_action = QAction("Find Changes", self)
        self.find_changes_action.setToolTip(
            "Auto-detect significant changes in the differential traces.\n"
            "Uses noise-floor estimation + peak finding to mark regions\n"
            "that deviate more than 6 dB above the median differential."
        )
        self.find_changes_action.setEnabled(False)
        self.find_changes_action.triggered.connect(self._on_find_changes)
        toolbar.addAction(self.find_changes_action)

        self.clear_changes_action = QAction("Clear Changes", self)
        self.clear_changes_action.setToolTip("Remove change-detection annotations")
        self.clear_changes_action.setEnabled(False)
        self.clear_changes_action.triggered.connect(self._on_clear_changes)
        toolbar.addAction(self.clear_changes_action)

    def _connect_signals(self):
        """Wire up all signals and slots."""
        self.file_list.selection_updated.connect(self._on_selection_changed)
        self.param_selector.selection_changed.connect(self._replot)
        self.plot_tab_bar.currentChanged.connect(self._on_tab_changed)
        self.btn_add.clicked.connect(self.open_files)
        self.btn_remove.clicked.connect(self._on_remove)
        self.conversion_panel.save_btn.clicked.connect(self._on_save)

    # --- Slots ---

    def open_files(self):
        """Open file dialog to load Touchstone files."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Touchstone Files",
            "",
            "Touchstone Files (*.s1p *.s2p *.s3p *.s4p *.s5p *.s6p "
            "*.s7p *.s8p *.s9p *.s10p *.s11p *.s12p *.snp);;"
            "All Files (*)"
        )
        if not filepaths:
            return

        errors = []
        for fp in filepaths:
            ok, err = self.file_list.add_network(fp)
            if not ok:
                errors.append(f"{os.path.basename(fp)}: {err}")

        if errors:
            QMessageBox.warning(
                self, "Load Errors",
                "Some files could not be loaded:\n\n" + "\n".join(errors)
            )

    def _on_selection_changed(self):
        """Handle file selection change (one or more files)."""
        networks = self.file_list.get_selected_networks()
        self.param_selector.update_for_networks(networks)
        # Enable save for the focused (current) item
        current_net = self.file_list.get_current_network()
        self.conversion_panel.update_for_network(current_net)
        self._replot()
        self._update_selection_status(networks)
        # Q button only active on magnitude tab when data is present
        on_magnitude = (self.plot_tab_bar.currentIndex() == self.PLOT_MAGNITUDE)
        self.q_action.setEnabled(on_magnitude and bool(networks))
        # Math memory: "Set Mem" available whenever at least one file is selected
        self.mem_action.setEnabled(bool(networks))

    def _on_remove(self):
        """Remove all selected files from the list."""
        self.file_list.remove_selected()
        networks = self.file_list.get_selected_networks()
        if not networks:
            self.canvas.show_placeholder()
            self.param_selector.update_for_networks([])
            self.conversion_panel.update_for_network(None)
            self.q_action.setEnabled(False)
            self.q_action.setChecked(False)
            self.clear_q_action.setEnabled(False)
            self.mem_action.setEnabled(False)
            self._update_status("No files loaded.")

    def _on_save(self):
        """Save the current (focused) network in the chosen format."""
        network = self.file_list.get_current_network()
        if network is None:
            QMessageBox.information(
                self, "No File",
                "Please select a file first."
            )
            return
        self.conversion_panel.save_network(network)

    def _replot(self):
        """Replot based on current state (all selected files overlaid)."""
        # Clear any pending Q span and annotations when the plot changes
        if hasattr(self.canvas, '_q_span') and self.canvas._q_span:
            self.canvas._q_span.set_visible(False)
            self.canvas._q_span = None
        self.canvas._clear_q_annotations()
        self.canvas._clear_change_annotations()
        self.q_action.setChecked(False)
        self.clear_q_action.setEnabled(False)
        self.clear_changes_action.setEnabled(False)

        networks = self.file_list.get_selected_networks()
        if not networks:
            self.canvas.show_placeholder()
            return

        params = self.param_selector.get_selected_params()
        plot_type = self.plot_tab_bar.currentIndex()

        # Math memory kwargs — only pass to plot types that support it
        mem_kw = dict(mem_network=self._mem_network,
                      diff_only=self._diff_only)

        if plot_type == self.PLOT_MAGNITUDE:
            if self._param_type == 'Z':
                self.canvas.plot_z_magnitude(networks, params, **mem_kw)
            elif self._param_type == 'Y':
                self.canvas.plot_y_magnitude(networks, params, **mem_kw)
            else:
                self.canvas.plot_magnitude(networks, params, **mem_kw)
        elif plot_type == self.PLOT_PHASE:
            self.canvas.plot_phase(networks, params, **mem_kw)
        elif plot_type == self.PLOT_SMITH:
            self.canvas.plot_smith(networks, params)
        elif plot_type == self.PLOT_VSWR:
            self.canvas.plot_vswr(networks, params)
        elif plot_type == self.PLOT_GROUP_DELAY:
            self.canvas.plot_group_delay(networks, params, **mem_kw)

    def _on_param_type_changed(self, param_type):
        """Switch between S, Z, and Y parameter display."""
        self._param_type = param_type

        # Keep the three toggle buttons mutually exclusive
        self.s_action.setChecked(param_type == 'S')
        self.z_action.setChecked(param_type == 'Z')
        self.y_action.setChecked(param_type == 'Y')

        # Relabel the parameter checkboxes
        self.param_selector.set_param_type(param_type)

        # Z/Y magnitude plots only make sense on the Magnitude tab;
        # switch to it if we are not already there.
        if self.plot_tab_bar.currentIndex() != self.PLOT_MAGNITUDE:
            # Suppress duplicate replot — tab-change will trigger _replot
            self.plot_tab_bar.setCurrentIndex(self.PLOT_MAGNITUDE)
        else:
            self._replot()

    def _on_tab_changed(self, index):
        """Handle plot-type tab change; replot and update Q button state."""
        self._replot()
        # Q measurement is meaningful on the Magnitude (dB) tab only
        on_magnitude = (index == self.PLOT_MAGNITUDE)
        networks = self.file_list.get_selected_networks()
        has_data = bool(networks)
        self.q_action.setEnabled(on_magnitude and has_data)
        if not on_magnitude:
            self.q_action.setChecked(False)
            self.canvas._clear_q_annotations()
            self.clear_q_action.setEnabled(False)

    def _on_q_action_toggled(self, checked):
        """Start or cancel Q-factor span selection."""
        if checked:
            self._update_status(
                "Q Measure: drag a region around a peak, then release."
            )
            self.canvas.start_q_measurement(self._on_q_result)
        else:
            # User un-toggled manually — cancel any pending selector
            if hasattr(self.canvas, '_q_span') and self.canvas._q_span:
                self.canvas._q_span.set_visible(False)
                self.canvas._q_span = None
            self.canvas.setCursor(Qt.ArrowCursor)
            self._update_status("Q measurement cancelled.")

    def _on_q_result(self, results):
        """Receive Q computation results (list, one per trace) and annotate."""
        # Button reverts to un-checked state
        self.q_action.setChecked(False)

        if not results:
            QMessageBox.warning(
                self, "Q Measurement Failed",
                "Could not find both 3 dB crossing points in the selected region "
                "for any trace.\n"
                "Try selecting a wider range around the peak, or check that the "
                "peak is clearly defined within the selection."
            )
            self._update_status("Q measurement failed — no valid 3 dB crossings found.")
            return

        self.canvas.annotate_q_results(results)
        self.clear_q_action.setEnabled(True)

        # Build a compact status-bar summary for all traces
        parts = []
        for r in results:
            lbl = r['label']
            unit = r['f_unit']
            f0_str = f"{r['f0']:.6g}"
            bw_str = self.canvas._format_bw(r['bw'], unit)
            q_str  = f"{r['q']:.0f}"
            entry = f"f0={f0_str} {unit}  BW={bw_str}  Q={q_str}"
            if lbl and not lbl.startswith('_'):
                entry = f"[{lbl}] " + entry
            parts.append(entry)
        self._update_status("Q: " + "    |    ".join(parts))

    def _on_clear_q(self):
        """Remove Q annotations from the plot."""
        self.canvas._clear_q_annotations()
        self.canvas.draw()
        self.clear_q_action.setEnabled(False)
        self._update_status("Q annotations cleared.")

    # --- Math Memory slots ---

    def _on_set_mem(self):
        """Capture the first selected network as the math memory reference."""
        networks = self.file_list.get_selected_networks()
        if not networks:
            return
        self._mem_network = networks[0]   # (short_name, Network)
        name = self._mem_network[0]
        self.clr_mem_action.setEnabled(True)
        self.diff_only_action.setEnabled(True)
        self.find_changes_action.setEnabled(True)
        self._update_status(f"Math memory set to: {name}")
        self._replot()

    def _on_clr_mem(self):
        """Clear the math memory reference."""
        self._mem_network = None
        self._diff_only = False
        self.clr_mem_action.setEnabled(False)
        self.diff_only_action.setEnabled(False)
        self.diff_only_action.setChecked(False)
        self.find_changes_action.setEnabled(False)
        self.clear_changes_action.setEnabled(False)
        self.canvas._clear_change_annotations()
        self._update_status("Math memory cleared.")
        self._replot()

    def _on_diff_only_toggled(self, checked):
        """Toggle differential-only display."""
        self._diff_only = checked
        self._replot()

    def _on_find_changes(self):
        """Run automatic change detection on the differential traces."""
        default_prominence = getattr(self, '_last_prominence_db', 6.0)
        prominence_db, ok = QInputDialog.getDouble(
            self,
            "Find Changes – Noise Floor",
            "Minimum peak prominence above median (dB):\n"
            "(Lower values detect smaller peaks; default is 6 dB)",
            default_prominence,   # current value
            0.1,                  # min
            100.0,                # max
            1,                    # decimals
        )
        if not ok:
            return
        self._last_prominence_db = prominence_db

        n = self.canvas.find_and_annotate_changes(prominence_db=prominence_db)
        if n == 0:
            QMessageBox.information(
                self, "Find Changes",
                "No significant changes detected.\n"
                f"No peaks exceed {prominence_db:.1f} dB above the median noise floor."
            )
            self._update_status("Find Changes: no significant peaks found.")
        else:
            self.clear_changes_action.setEnabled(True)
            self._update_status(
                f"Find Changes: {n} significant peak(s) marked "
                f"(threshold {prominence_db:.1f} dB above median)."
            )

    def _on_clear_changes(self):
        """Remove change-detection annotations."""
        self.canvas._clear_change_annotations()
        self.canvas.draw()
        self.clear_changes_action.setEnabled(False)
        self._update_status("Change annotations cleared.")

    def _update_selection_status(self, networks):
        """Update status bar with info about selected networks."""
        if not networks:
            self._update_status("No file selected.")
            return

        if len(networks) == 1:
            name, net = networks[0]
            n_ports = net.number_of_ports
            n_points = len(net.frequency.f)
            f_start = net.frequency.f_scaled[0]
            f_stop = net.frequency.f_scaled[-1]
            unit = net.frequency.unit
            self._update_status(
                f"{name}  |  {n_ports}-port  |  {n_points} pts  |  "
                f"{f_start:.4g} - {f_stop:.4g} {unit}"
            )
        else:
            names = ', '.join(n for n, _ in networks)
            self._update_status(f"{len(networks)} files selected: {names}")

    def _update_status(self, message):
        self.statusBar().showMessage(message)

    def _show_about(self):
        QMessageBox.about(
            self, "About SNP Viewer",
            "<h3>SNP Viewer</h3>"
            "<p>A Touchstone/SNP file viewer and converter.</p>"
            "<p>Supports .s1p through .snp files with visualization "
            "of magnitude, phase, Smith chart, VSWR, and group delay.</p>"
            "<p>Uses Nature Journal color scheme.</p>"
            "<p>Built with PyQt5, matplotlib, and scikit-rf.</p>"
            "<hr>"
            "<p><b>Author:</b> Daniel Hedlund<br>"
            "<b>Contact:</b> <a href='mailto:daniel.hedlund@gmail.com'>"
            "daniel.hedlund@gmail.com</a></p>"
            "<hr>"
            "<p style='font-size: 8pt; color: #666;'>"
            "<b>THE BEER-WARE LICENSE</b> (Revision 42):<br>"
            "Daniel Hedlund wrote this file. As long as you retain this notice "
            "you can do whatever you want with this stuff. If we meet some day, "
            "and you think this stuff is worth it, you can buy me a beer in return."
            "</p>"
        )

    def _show_settings_info(self):
        """Show which settings file is active and its current values."""
        conf_path = settings.conf_path()
        if conf_path:
            source = f"<p><b>Loaded from:</b><br><code>{conf_path}</code></p>"
        else:
            source = (
                "<p><b>No settings file found.</b> Using built-in defaults.<br>"
                "Create <code>snpviewer.conf</code> next to snpviewer.py "
                "or <code>~/.snpviewer.conf</code> to customise defaults.</p>"
            )

        def fmt_params(lst):
            if lst is None:
                return "all"
            if not lst:
                return "none"
            return ", ".join(f"{m+1}{n+1}" for m, n in lst)

        QMessageBox.information(
            self, "Settings",
            f"{source}"
            f"<table cellspacing='4'>"
            f"<tr><td><b>param_type</b></td><td>{settings.param_type}</td></tr>"
            f"<tr><td><b>s_default_params</b></td>"
            f"    <td>{fmt_params(settings.s_default_params)}</td></tr>"
            f"<tr><td><b>z_default_params</b></td>"
            f"    <td>{fmt_params(settings.z_default_params)}</td></tr>"
            f"<tr><td><b>y_default_params</b></td>"
            f"    <td>{fmt_params(settings.y_default_params)}</td></tr>"
            f"<tr><td><b>plot_format</b></td><td>{settings.plot_format}</td></tr>"
            f"<tr><td><b>z0</b></td><td>{settings.z0} Ω</td></tr>"
            f"</table>"
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            # Accept if any URL looks like a Touchstone file
            for url in event.mimeData().urls():
                fp = url.toLocalFile()
                if fp and (fp.lower().endswith('.snp') or
                           any(fp.lower().endswith(f'.s{i}p')
                               for i in range(1, 100))):
                    event.acceptProposedAction()
                    return
            # Also accept all files and let the loader sort it out
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        errors = []
        for url in event.mimeData().urls():
            fp = url.toLocalFile()
            if fp:
                ok, err = self.file_list.add_network(fp)
                if not ok:
                    errors.append(f"{os.path.basename(fp)}: {err}")
        if errors:
            QMessageBox.warning(
                self, "Load Errors",
                "Some files could not be loaded:\n\n" + "\n".join(errors)
            )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Apply a clean light palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor('#FAFAFA'))
    palette.setColor(QPalette.WindowText, QColor('#333333'))
    palette.setColor(QPalette.Base, QColor('#FFFFFF'))
    palette.setColor(QPalette.AlternateBase, QColor('#F5F5F5'))
    palette.setColor(QPalette.ToolTipBase, QColor('#FFFFF0'))
    palette.setColor(QPalette.ToolTipText, QColor('#333333'))
    palette.setColor(QPalette.Text, QColor('#333333'))
    palette.setColor(QPalette.Button, QColor('#F0F0F0'))
    palette.setColor(QPalette.ButtonText, QColor('#333333'))
    palette.setColor(QPalette.Highlight, QColor(NatureColors.BLUE))
    palette.setColor(QPalette.HighlightedText, QColor('#FFFFFF'))
    app.setPalette(palette)

    window = SNPViewerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
