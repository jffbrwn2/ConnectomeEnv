import numpy as np
import pytest
from proofreading.reachability import reachable_nodes


def test_outside_path_cannot_connect_disconnected_target_fragments():
    # A path through node99 is not inside the certified target graph.
    result=reachable_nodes([1,2,3,4,5],[[1,2],[2,99],[99,3],[3,4]],[1])
    assert np.array_equal(result,[1,2])
    assert np.array_equal(reachable_nodes([1,2,3,4,5],[[1,2],[3,4]],[1,5]),[1,2,5])
    assert np.array_equal(reachable_nodes([9],[],[9]),[9])
    with pytest.raises(ValueError):reachable_nodes([1,2],[],[3])
