from pathlib import Path
import pytest
from kdm.visualization import plot_semantic_matrix, plot_method_tradeoff


def test_main_results_figures_use_explicit_complete_inputs(tmp_path):
    markers = ['UNKNOWN', 'UNCLEAR', 'UNSURE', 'I cannot identify it']
    rows = [
        {'marker': marker, 'reference_marker': reference, 'retention': 0.3}
        for marker in markers for reference in markers
    ]
    files = plot_semantic_matrix(rows, tmp_path / 'matrix', 'SOURCE: result receipt')
    files += plot_method_tradeoff(
        [{'method': 'Direct', 'accuracy': 0.5, 'retention': 0.4}],
        tmp_path / 'tradeoff', 'SOURCE: paired analysis receipt')
    assert len(files) == 4
    assert all(Path(path).stat().st_size > 1000 for path in files)
    with pytest.raises(ValueError, match='sixteen'):
        plot_semantic_matrix(rows[:-1], tmp_path / 'incomplete', 'SOURCE: result receipt')
