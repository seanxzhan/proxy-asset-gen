"""Tests for scenario-viewer geometry export (no Polyscope window needed)."""
from __future__ import annotations

import numpy as np

from pag.eval_runner import Obstacle
from pag.eval_viz import export_eval_frame
from pag.io import load_obj


def test_export_eval_frame_writes_each_displayed_object(tmp_path):
    V_proxy = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    V_visual = 2.0 * V_proxy
    F = np.array([[0, 1, 2]], dtype=np.int64)
    offsets = [np.zeros(3), np.array([10.0, 0.0, 0.0])]
    sphere = Obstacle(
        name="test sphere",
        kind="sphere",
        center=np.array([1.0, 2.0, 3.0]),
        radius=0.5,
    )

    paths = export_eval_frame(
        tmp_path,
        7,
        V_proxy,
        F,
        V_visual,
        F,
        [sphere],
        offsets,
    )

    assert {path.name for path in paths} == {
        "M_proxy.obj",
        "M_visual_recon.obj",
        "obs0_test_sphere_proxy.obj",
        "obs0_test_sphere_recon.obj",
    }
    assert all(path.parent == tmp_path / "frame_0007" for path in paths)
    assert all(path.is_file() for path in paths)

    proxy_out, proxy_faces = load_obj(tmp_path / "frame_0007/M_proxy.obj")
    recon_out, recon_faces = load_obj(
        tmp_path / "frame_0007/M_visual_recon.obj"
    )
    np.testing.assert_allclose(proxy_out, V_proxy)
    np.testing.assert_allclose(recon_out, V_visual + offsets[1])
    np.testing.assert_array_equal(proxy_faces, F)
    np.testing.assert_array_equal(recon_faces, F)

    sphere_proxy, _ = load_obj(
        tmp_path / "frame_0007/obs0_test_sphere_proxy.obj"
    )
    sphere_recon, _ = load_obj(
        tmp_path / "frame_0007/obs0_test_sphere_recon.obj"
    )
    np.testing.assert_allclose(
        sphere_recon - sphere_proxy,
        np.broadcast_to(offsets[1], sphere_proxy.shape),
    )
    radii = np.linalg.norm(sphere_proxy - sphere.center, axis=1)
    np.testing.assert_allclose(radii, sphere.radius, atol=1e-7)
