"""Round-trip OBJ load/save."""
from __future__ import annotations

import numpy as np

from pag import load_obj, save_obj


def test_roundtrip_tetra(tmp_path):
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])

    p = tmp_path / "tetra.obj"
    save_obj(p, V, F)
    V2, F2 = load_obj(p)

    np.testing.assert_allclose(V, V2, atol=1e-7)
    np.testing.assert_array_equal(F, F2)


def test_load_obj_handles_polygon_fan(tmp_path):
    """A 4-gon should be triangulated as a fan from vertex 0."""
    p = tmp_path / "quad.obj"
    p.write_text(
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "f 1 2 3 4\n"
    )
    V, F = load_obj(p)
    assert V.shape == (4, 3)
    np.testing.assert_array_equal(F, [[0, 1, 2], [0, 2, 3]])


def test_load_obj_strips_uv_normal_refs(tmp_path):
    p = tmp_path / "uv.obj"
    p.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "f 1/10/100 2/20/200 3/30/300\n"
    )
    _, F = load_obj(p)
    np.testing.assert_array_equal(F, [[0, 1, 2]])
