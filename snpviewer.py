"""
SNP Viewer - Touchstone/SNP File Viewer and Converter
A graphical application for loading, visualizing, and converting S-parameter files.
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
        QSizePolicy, QScrollArea, QFrame
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
        QSizePolicy, QScrollArea, QFrame
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
    def apply_matplotlib_defaults():
        """Set matplotlib rcParams for Nature-style plots."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'axes.titleweight': 'bold',
            'axes.linewidth': 0.8,
            'axes.edgecolor': '#333333',
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'legend.fontsize': 9,
            'legend.framealpha': 0.9,
            'legend.edgecolor': '#cccccc',
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.2,
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
        self._show_placeholder()

    def _show_placeholder(self):
        """Show a placeholder message when no data is loaded."""
        self.fig.clear()
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
        self.ax.grid(True, alpha=0.2, linestyle='-', color='#cccccc')

    def plot_magnitude(self, networks, param_list):
        """Plot S-parameters in dB for multiple networks.
        networks: list of (short_name, Network) tuples.
        """
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if not param_list or not networks:
            self._show_no_params()
            return

        trace_idx = 0
        multi = len(networks) > 1
        # Determine best unit from the first network's raw Hz values
        _, freq_unit = auto_freq_scale(networks[0][1].frequency.f)

        for name, network in networks:
            freq, _ = auto_freq_scale(network.frequency.f)
            n_ports = network.number_of_ports
            for m, n in param_list:
                if m >= n_ports or n >= n_ports:
                    continue
                color = NatureColors.get_color(trace_idx)
                s_mag = np.abs(network.s[:, m, n])
                s_db = 20 * np.log10(np.where(s_mag == 0, 1e-30, s_mag))
                label = f'{name} S{m+1},{n+1}' if multi else f'S{m+1},{n+1}'
                self.ax.plot(freq, s_db, color=color,
                             label=label, linewidth=1.8)
                trace_idx += 1

        self.ax.set_xlabel(f'Frequency ({freq_unit})')
        self.ax.set_ylabel('Magnitude (dB)')
        self._style_axes('Magnitude')
        self.fig.tight_layout()
        self.draw()

    def plot_phase(self, networks, param_list):
        """Plot S-parameters phase in degrees for multiple networks."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if not param_list or not networks:
            self._show_no_params()
            return

        trace_idx = 0
        multi = len(networks) > 1
        _, freq_unit = auto_freq_scale(networks[0][1].frequency.f)

        for name, network in networks:
            freq, _ = auto_freq_scale(network.frequency.f)
            n_ports = network.number_of_ports
            for m, n in param_list:
                if m >= n_ports or n >= n_ports:
                    continue
                color = NatureColors.get_color(trace_idx)
                s_deg = network.s_deg[:, m, n]
                label = f'{name} S{m+1},{n+1}' if multi else f'S{m+1},{n+1}'
                self.ax.plot(freq, s_deg, color=color,
                             label=label, linewidth=1.8)
                trace_idx += 1

        self.ax.set_xlabel(f'Frequency ({freq_unit})')
        self.ax.set_ylabel('Phase (degrees)')
        self._style_axes('Phase')
        self.fig.tight_layout()
        self.draw()

    def plot_smith(self, networks, param_list):
        """Plot S-parameters on a Smith chart for multiple networks."""
        self.fig.clear()
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
        self.fig.tight_layout()
        self.draw()

    def plot_vswr(self, networks, param_list):
        """Plot VSWR for multiple networks."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if not param_list or not networks:
            self._show_no_params()
            return

        trace_idx = 0
        multi = len(networks) > 1
        _, freq_unit = auto_freq_scale(networks[0][1].frequency.f)

        for name, network in networks:
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
                             label=label, linewidth=1.8)
                trace_idx += 1

        self.ax.set_xlabel(f'Frequency ({freq_unit})')
        self.ax.set_ylabel('VSWR')
        self.ax.set_ylim(bottom=1)
        self._style_axes('VSWR')
        self.fig.tight_layout()
        self.draw()

    def plot_group_delay(self, networks, param_list):
        """Plot group delay for multiple networks."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if not param_list or not networks:
            self._show_no_params()
            return

        trace_idx = 0
        multi = len(networks) > 1
        _, freq_unit = auto_freq_scale(networks[0][1].frequency.f)

        for name, network in networks:
            freq, _ = auto_freq_scale(network.frequency.f)
            n_ports = network.number_of_ports
            for m, n in param_list:
                if m >= n_ports or n >= n_ports:
                    continue
                color = NatureColors.get_color(trace_idx)
                s_phase_rad = np.unwrap(np.angle(network.s[:, m, n]))
                omega = 2 * np.pi * network.frequency.f
                if len(omega) > 1:
                    group_delay = -np.gradient(s_phase_rad, omega)
                    label = f'{name} S{m+1},{n+1}' if multi else f'S{m+1},{n+1}'
                    self.ax.plot(freq, group_delay * 1e9, color=color,
                                 label=label, linewidth=1.8)
                trace_idx += 1

        self.ax.set_xlabel(f'Frequency ({freq_unit})')
        self.ax.set_ylabel('Group Delay (ns)')
        self._style_axes('Group Delay')
        self.fig.tight_layout()
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
    """Dynamic checkbox grid for S-parameter selection."""

    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__("S-Parameters", parent)
        self._checkboxes = []
        self._layout = QGridLayout()
        self._layout.setSpacing(2)
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

        for m in range(max_ports):
            for n in range(max_ports):
                cb = QCheckBox(f'S{m + 1},{n + 1}')
                cb.setProperty('row', m)
                cb.setProperty('col', n)
                # Restore previous state, or use defaults on first build
                if prev_checked:
                    cb.setChecked((m, n) in prev_checked)
                else:
                    if (m, n) == (0, 0):
                        cb.setChecked(True)
                    elif max_ports >= 2 and (m, n) == (1, 0):
                        cb.setChecked(True)
                cb.stateChanged.connect(self._on_changed)
                self._layout.addWidget(cb, m, n)
                self._checkboxes.append(cb)

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
        # scikit-rf supports writing with different parameter types
        # through the Touchstone file format
        freq = network.frequency

        if param == 'z':
            data = network.z
        elif param == 'y':
            data = network.y
        else:
            data = network.s

        n_ports = network.number_of_ports

        with open(filepath, 'w') as f:
            # Write header
            freq_unit = freq.unit.upper()
            if freq_unit == 'HZ':
                freq_unit = 'HZ'
            f.write(f"! Converted by SNP Viewer\n")
            f.write(f"# {freq_unit} {param.upper()} {form.upper()} R {z0}\n")

            for k in range(len(freq.f)):
                line = f"{freq.f[k]:.10g}"
                for m in range(n_ports):
                    for n in range(n_ports):
                        val = data[k, m, n]
                        if form == 'ri':
                            line += f"  {val.real:.10g}  {val.imag:.10g}"
                        elif form == 'ma':
                            mag = np.abs(val)
                            ang = np.degrees(np.angle(val))
                            line += f"  {mag:.10g}  {ang:.10g}"
                        elif form == 'db':
                            mag_db = 20 * np.log10(np.abs(val) + 1e-30)
                            ang = np.degrees(np.angle(val))
                            line += f"  {mag_db:.10g}  {ang:.10g}"
                        else:
                            line += f"  {val.real:.10g}  {val.imag:.10g}"
                f.write(line + "\n")


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

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SNP Viewer")
        self.resize(1200, 750)
        self.setAcceptDrops(True)

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

    def _connect_signals(self):
        """Wire up all signals and slots."""
        self.file_list.selection_updated.connect(self._on_selection_changed)
        self.param_selector.selection_changed.connect(self._replot)
        self.plot_tab_bar.currentChanged.connect(self._replot)
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

    def _on_remove(self):
        """Remove all selected files from the list."""
        self.file_list.remove_selected()
        networks = self.file_list.get_selected_networks()
        if not networks:
            self.canvas.show_placeholder()
            self.param_selector.update_for_networks([])
            self.conversion_panel.update_for_network(None)
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
        networks = self.file_list.get_selected_networks()
        if not networks:
            self.canvas.show_placeholder()
            return

        params = self.param_selector.get_selected_params()
        plot_type = self.plot_tab_bar.currentIndex()

        if plot_type == self.PLOT_MAGNITUDE:
            self.canvas.plot_magnitude(networks, params)
        elif plot_type == self.PLOT_PHASE:
            self.canvas.plot_phase(networks, params)
        elif plot_type == self.PLOT_SMITH:
            self.canvas.plot_smith(networks, params)
        elif plot_type == self.PLOT_VSWR:
            self.canvas.plot_vswr(networks, params)
        elif plot_type == self.PLOT_GROUP_DELAY:
            self.canvas.plot_group_delay(networks, params)

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
