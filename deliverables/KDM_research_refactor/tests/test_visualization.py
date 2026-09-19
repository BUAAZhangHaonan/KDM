import json
from pathlib import Path
import pytest
from kdm.visualization import *

def test_all_figure_functions(tmp_path):
    root=Path(__file__).parents[1]
    files=plot_reported_fates(root/'data_example/reported_counts.json',tmp_path/'f1')
    files+=plot_lexical_boundary(tmp_path/'f2');files+=plot_decomposition(tmp_path/'f3');files+=plot_instruction_invariance(tmp_path/'f4')
    names=['UNKNOWN','UNCLEAR','UNSURE','I cannot identify it']
    rows=[{'marker':x,'reference_marker':y,'retention':.3} for x in names for y in names]
    files+=plot_semantic_matrix(rows,tmp_path/'f5','SIMULATED FIXTURE')
    files+=plot_method_tradeoff([{'method':'test','accuracy':.5,'retention':.5}],tmp_path/'f6','SIMULATED FIXTURE')
    assert len(files)==12 and all(Path(p).stat().st_size>1000 for p in files)
    with pytest.raises(ValueError):plot_semantic_matrix(rows[:-1],tmp_path/'bad','SIMULATED FIXTURE')
