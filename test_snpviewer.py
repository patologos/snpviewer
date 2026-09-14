"""Run with pytest; uses real Qt widgets and scikit-rf Networks offscreen."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import weakref

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
import skrf as rf

module_path = Path(os.environ.get('SNPVIEWER_PATH', Path(__file__).with_name('snpviewer.py')))
spec = importlib.util.spec_from_file_location('snpviewer_under_test', module_path)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


@pytest.fixture(scope='session')
def app():
    return viewer.QApplication.instance() or viewer.QApplication([])


@pytest.fixture
def canvas(app):
    widget = viewer.PlotCanvas()
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def network(n_ports=3, count=41, start=1e9, stop=2e9, z0=50):
    freq = rf.Frequency.from_f(np.linspace(start, stop, count), unit='hz')
    freq.unit = 'ghz'
    rng = np.random.default_rng(403)
    s = 0.03 * (rng.normal(size=(count, n_ports, n_ports)) +
                1j * rng.normal(size=(count, n_ports, n_ports)))
    return rf.Network(frequency=freq, s=s, z0=z0)


@pytest.mark.parametrize('data_key', ['s', 'z', 'y'])
def test_selected_interpolation_matches_numpy(canvas, data_key):
    ref, target = network(), network(2, 63, 0.7e9, 2.3e9)
    params = [(0, 0), (1, 0)]
    traces = canvas._memory_traces(ref, target, data_key, params + [(2, 2)])
    assert set(traces) == set(params)
    for param in params:
        values = getattr(ref, data_key)[:, param[0], param[1]]
        expected = np.interp(target.f, ref.f, values)
        np.testing.assert_allclose(traces[param], expected, rtol=1e-12, atol=1e-12)
    assert canvas._interp_cache_bytes == len(target.f) * len(params) * 16
    first = next(iter(canvas._interp_cache.values()))[2]
    canvas._memory_traces(ref, target, data_key, list(reversed(params)))
    assert next(iter(canvas._interp_cache.values()))[2] is first


@pytest.mark.parametrize('ref_count,target_count', [(1, 7), (7, 7), (7, 0), (0, 7)])
def test_interpolation_edge_grids(canvas, ref_count, target_count):
    ref, target = network(2, ref_count), network(2, target_count)
    actual = canvas._interp_to_freq(ref, target, 's')
    if not ref_count:
        assert actual is None
    else:
        assert actual.shape == (target_count, 2, 2)
        for m in range(2):
            for n in range(2):
                expected = np.interp(target.f, ref.f, ref.s[:, m, n])
                np.testing.assert_allclose(actual[:, m, n], expected)


def test_cache_budget_lru_and_collection(canvas):
    ref = network(2)
    targets = [network(2, 7, stop=(2 + i / 10) * 1e9) for i in range(4)]
    canvas._interp_cache_limit = 2 * 7 * 16
    for target in targets[:2]:
        canvas._memory_traces(ref, target, 's', [(0, 0)])
    # Touch the oldest entry; the other entry should be evicted next.
    canvas._memory_traces(ref, targets[0], 's', [(0, 0)])
    canvas._memory_traces(ref, targets[2], 's', [(0, 0)])
    assert {key[1] for key in canvas._interp_cache} == {id(targets[0]), id(targets[2])}
    assert canvas._interp_cache_bytes <= canvas._interp_cache_limit
    oversized = canvas._interp_to_freq(ref, targets[3], 's')
    assert oversized.nbytes > canvas._interp_cache_limit
    assert len(canvas._interp_cache) == 2
    ref_weak = weakref.ref(ref)
    del ref
    gc.collect()
    assert ref_weak() is None
    assert not canvas._interp_cache
    assert canvas._interp_cache_bytes == 0
    canvas.clear_data_cache()
    assert not canvas._network_cache


@pytest.mark.parametrize('method', ['plot_magnitude', 'plot_z_magnitude', 'plot_y_magnitude',
                                  'plot_phase', 'plot_group_delay'])
@pytest.mark.parametrize('diff_only', [False, True])
def test_memory_plot_values(canvas, method, diff_only):
    ref, target = network(2), network(3, 61, 0.8e9, 2.2e9)
    params = [(0, 0), (1, 0), (2, 2)]
    getattr(canvas, method)([('target', target)], params,
                            mem_network=('ref', ref), diff_only=diff_only)
    expected_count = 4 if diff_only else 7
    assert len(canvas.ax.lines) == expected_count
    data_key = {'plot_z_magnitude': 'z', 'plot_y_magnitude': 'y'}.get(method, 's')
    difference = [line for line in canvas.ax.lines if line.get_label().startswith('Δ')]
    for line, (m, n) in zip(difference, params):
        raw = getattr(target, data_key)[:, m, n]
        mem = np.interp(target.f, ref.f, getattr(ref, data_key)[:, m, n])
        if method == 'plot_phase':
            expected = np.degrees(np.angle(raw)) - np.degrees(np.angle(mem))
        elif method == 'plot_group_delay':
            expected = (-np.gradient(np.unwrap(np.angle(raw)), 2 * np.pi * target.f) +
                        np.gradient(np.unwrap(np.angle(mem)), 2 * np.pi * target.f)) * 1e9
        else:
            expected = 20 * np.log10(np.maximum(np.abs(raw - mem), 1e-30))
        np.testing.assert_allclose(line.get_ydata(), expected, rtol=1e-11, atol=1e-11)
    canvas.draw()


@pytest.mark.parametrize('method', ['plot_smith', 'plot_vswr', 'plot_mag_phase'])
def test_other_plots_render(canvas, method):
    getattr(canvas, method)([('sample', network(2))], [(0, 0), (1, 0)])
    canvas.draw()
    assert len(list(canvas._data_lines())) >= 1


def test_hover_avoids_identical_redraw(canvas, monkeypatch):
    draws = []
    monkeypatch.setattr(canvas, 'draw_idle', lambda: draws.append(True))
    canvas._show_hover(1, 2, 'value', '#555555')
    canvas._show_hover(1, 2, 'value', '#555555')
    assert len(draws) == 1
    canvas._hide_hover()
    canvas._show_hover(1, 2, 'value', '#555555')
    assert len(draws) == 3


def test_parameter_widgets_reused_and_empty_selection_retained(app):
    selector = viewer.ParameterSelector()
    selector.update_for_networks([('first', network(2))])
    widgets = tuple(selector._checkboxes)
    for checkbox in widgets:
        checkbox.setChecked(False)
    selector.update_for_networks([('second', network(2))])
    assert tuple(selector._checkboxes) == widgets
    selector.update_for_networks([('larger', network(3))])
    assert selector.get_selected_params() == []
    selector.close()


@pytest.mark.parametrize('data_key', ['s', 'z', 'y'])
def test_all_parameter_defaults(app, monkeypatch, data_key):
    monkeypatch.setattr(viewer.settings, f'{data_key}_default_params', None)
    selector = viewer.ParameterSelector()
    selector.set_param_type(data_key)
    selector.update_for_networks([('sample', network(3))])
    assert len(selector.get_selected_params()) == 9
    selector.close()


def test_batched_loading_selection_removal_and_duplicates(app, tmp_path):
    path = tmp_path / 'sample.s2p'
    network(2).write_touchstone(str(path))
    widget = viewer.FileListWidget()
    signals = []
    widget.selection_updated.connect(lambda: signals.append(True))
    errors = widget.add_networks([str(path)] * 3 + [str(tmp_path / 'missing.s2p')])
    assert len(signals) == 1
    assert len(errors) == 1
    assert widget.count() == widget.get_network_count() == 3
    assert len({widget.item(i).text() for i in range(3)}) == 3
    widget.select_by_indices([0, 2])
    assert len(signals) == 2
    assert len(widget.get_selected_networks()) == 2
    widget.remove_selected()
    assert len(signals) == 3
    assert widget.count() == widget.get_network_count() == 1
    widget.close()


@pytest.mark.parametrize('ports', [1, 2, 3, 5])
@pytest.mark.parametrize('param', ['s', 'z', 'y'])
@pytest.mark.parametrize('form', ['ri', 'ma', 'db'])
def test_converted_touchstone_roundtrip(app, tmp_path, ports, param, form):
    net = network(ports, 11, z0=75)
    path = tmp_path / f'converted.s{ports}p'
    viewer.ConversionPanel._write_converted(None, net, str(path), param, form, 75)
    restored = rf.Network(str(path))
    np.testing.assert_allclose(restored.f, net.f, rtol=1e-9)
    np.testing.assert_allclose(getattr(restored, param), getattr(net, param),
                               rtol=3e-8, atol=1e-10)
    np.testing.assert_allclose(restored.z0, 75)


def test_converted_writer_uses_multiple_bounded_chunks(app, monkeypatch, tmp_path):
    sizes = []
    savetxt = np.savetxt
    def record(file, values, **kwargs):
        sizes.append(values.nbytes)
        return savetxt(file, values, **kwargs)
    monkeypatch.setattr(viewer.np, 'savetxt', record)
    net = network(5, 3000)
    viewer.ConversionPanel._write_converted(None, net, str(tmp_path / 'large.s5p'), 's', 'ri', 50)
    assert len(sizes) > 1
    assert max(sizes) <= 1024 * 1024


@pytest.mark.parametrize('original_z0,requested_z0', [(75, 50), (50, 75), (50, 50)])
def test_save_uses_requested_impedance_without_mutating_input(app, monkeypatch, tmp_path,
                                                            original_z0, requested_z0):
    net = network(2, z0=original_z0)
    original_s = net.s.copy()
    path = tmp_path / 'saved.s2p'
    monkeypatch.setattr(viewer.QFileDialog, 'getSaveFileName', lambda *a, **k: (str(path), ''))
    monkeypatch.setattr(viewer.QMessageBox, 'information', lambda *a, **k: None)
    def fail(*args):
        pytest.fail(str(args))
    monkeypatch.setattr(viewer.QMessageBox, 'warning', fail)
    panel = viewer.ConversionPanel()
    panel.z0_spin.setValue(requested_z0)
    panel.save_network(net)
    restored = rf.Network(str(path))
    np.testing.assert_allclose(restored.z0, requested_z0)
    expected = net.copy()
    expected.renormalize(requested_z0)
    np.testing.assert_allclose(restored.s, expected.s, atol=1e-10)
    np.testing.assert_array_equal(net.s, original_s)
    np.testing.assert_array_equal(net.z0, original_z0)
    panel.close()


def test_full_application_load_switch_and_render(app, tmp_path):
    path = tmp_path / 'sample.s2p'
    network(2).write_touchstone(str(path))
    window = viewer.SNPViewerApp()
    assert window.file_list.add_networks([str(path)]) == []
    window._on_set_mem()
    for tab in range(6):
        window.plot_tab_bar.setCurrentIndex(tab)
        window._replot_timer.stop()
        window._perform_replot()
        window.canvas.draw()
    window._on_clr_mem()
    window._replot_timer.stop()
    window.close()
    window.deleteLater()
    app.processEvents()


def test_empty_file_reports_load_error(app, tmp_path):
    path = tmp_path / 'empty.s2p'
    path.write_text('# GHz S RI R 50\n', encoding='utf-8')
    widget = viewer.FileListWidget()
    errors = widget.add_networks([str(path)])
    assert len(errors) == 1
    assert widget.count() == 0
    widget.close()


def test_session_restores_empty_parameter_selection(app, monkeypatch, tmp_path):
    path = tmp_path / 'sample.s2p'
    network(2).write_touchstone(str(path))
    session = tmp_path / 'session.json'
    session.write_text(json.dumps({'files': [str(path)], 'selected_indices': [0],
                                   'checked_params': []}), encoding='utf-8')
    monkeypatch.setattr(viewer.QFileDialog, 'getOpenFileName',
                        lambda *a, **k: (str(session), ''))
    window = viewer.SNPViewerApp()
    monkeypatch.setattr(window, '_save_last_dir', lambda path: None)
    window._load_session()
    window._replot_timer.stop()
    assert window.file_list.get_selected_indices() == [0]
    assert len(window.param_selector._checkboxes) == 4
    assert window.param_selector.get_selected_params() == []
    window.close()
    window.deleteLater()
    app.processEvents()
