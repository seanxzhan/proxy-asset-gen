#! /Users/szhan/miniforge3/envs/proxyasset/bin/python
"""
Render every .gltf/.glb in a directory to a single HTML gallery using pyrender.

For each file we load all geometries, drop any whose name contains "cage"
(case-insensitive), combine the rest into one mesh, and render it offscreen.
The PNG is base64-embedded so the HTML is self-contained.

Usage:
    python render_skirt_gallery.py /path/to/Skirt --output gallery.html
"""

import argparse
import base64
import io
import os
import platform
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# pyrender uses PYOPENGL_PLATFORM to pick its offscreen backend. On Linux egl is
# best (headless, GPU); on macOS there is no EGL and OpenGL is provided by the
# system, so leave the variable unset to use pyrender's default (a hidden pyglet
# window).
if platform.system() == "Linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import trimesh
import pyrender
from PIL import Image


CAGE_RE = re.compile(r"cage", re.IGNORECASE)


def _alphanumeric_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_non_cage_mesh(path: str) -> trimesh.Trimesh:
    """Load a gltf/glb and return all non-'cage' geometries concatenated."""
    loaded = trimesh.load(path, process=False)

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    if not isinstance(loaded, trimesh.Scene):
        raise ValueError(f"Unsupported load result type: {type(loaded)}")

    # Resolve names from scene graph so we know which geometry each node carries.
    # trimesh stores geometries in scene.geometry keyed by name; node->geometry
    # mapping lives in scene.graph.
    keep_names = []
    for name in loaded.geometry.keys():
        if not CAGE_RE.search(name):
            keep_names.append(name)

    # Also check node names — sometimes the node carries the "cage" label while
    # the underlying geometry has a generic name. Drop any geometry referenced
    # only by cage-named nodes.
    cage_geometries = set()
    node_geometries = {}
    for node_name in loaded.graph.nodes_geometry:
        try:
            _, geom_name = loaded.graph[node_name]
        except (KeyError, ValueError):
            continue
        if geom_name is None:
            continue
        node_geometries.setdefault(geom_name, []).append(node_name)

    for geom_name, node_names in node_geometries.items():
        if all(CAGE_RE.search(n) for n in node_names):
            cage_geometries.add(geom_name)

    keep_names = [n for n in keep_names if n not in cage_geometries]

    if not keep_names:
        raise ValueError(f"No non-cage geometry found in {path}")

    # Use scene.dump() with the filtered subset by building a sub-scene so node
    # transforms are baked in correctly.
    sub = trimesh.Scene()
    for node_name in loaded.graph.nodes_geometry:
        try:
            transform, geom_name = loaded.graph[node_name]
        except (KeyError, ValueError):
            continue
        if geom_name not in keep_names:
            continue
        geom = loaded.geometry[geom_name]
        sub.add_geometry(geom, node_name=node_name, transform=transform)

    if not sub.geometry:
        # Fallback: bake without node transforms
        meshes = [loaded.geometry[n] for n in keep_names]
        return trimesh.util.concatenate(meshes)

    combined = trimesh.util.concatenate(sub.dump())
    return combined


def render_mesh(mesh: trimesh.Trimesh, width: int = 512, height: int = 512) -> Image.Image:
    """Render a trimesh to a PIL image with a flat shaded look + wireframe."""
    mesh = mesh.copy()

    # Y-up convention + a viewing rotation, mirroring msimp's visualize_dataset.py.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(0, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-3 * np.pi / 4, [0, 1, 0]))

    bounds = mesh.bounds
    centroid = (bounds[0] + bounds[1]) / 2.0
    extent = float(np.max(bounds[1] - bounds[0]))

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

    mesh.vertex_normals = None
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=[200, 200, 200, 255])
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    for prim in pyr_mesh.primitives:
        prim.material.doubleSided = True
    scene.add(pyr_mesh)

    edges = mesh.edges_unique
    edge_points = mesh.vertices[edges.flatten()]
    edge_colors = np.full((len(edge_points), 4), [40, 40, 40, 255], dtype=np.uint8)
    line_prim = pyrender.Primitive(positions=edge_points, color_0=edge_colors, mode=1)
    scene.add(pyrender.Mesh(primitives=[line_prim]))

    fov = np.pi / 3.0
    dist = (extent / 2.0) / np.tan(fov / 2.0) * 2.0
    elevation = np.pi / 6
    azimuth = 0.0
    cam_pos = centroid + dist * np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.sin(elevation),
        np.cos(elevation) * np.sin(azimuth),
    ])

    forward = centroid - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos

    scene.add(pyrender.PerspectiveCamera(yfov=fov), pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=pose)

    r = pyrender.OffscreenRenderer(width, height)
    color, _ = r.render(scene)
    r.delete()
    return Image.fromarray(color)


def render_single(path: str, width: int, height: int):
    try:
        mesh = load_non_cage_mesh(path)
        nv, nf = len(mesh.vertices), len(mesh.faces)
        img = render_mesh(mesh, width, height)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return path, b64, nv, nf, None
    except Exception as e:
        return path, None, None, None, repr(e)


def generate_html(title: str, cards: list, html_path: str) -> None:
    items = ""
    for idx, (name, b64, nv, nf, err) in enumerate(cards):
        if err is not None:
            body = f'<div class="error">{err}</div>'
            stats = ""
        else:
            body = f'<img src="data:image/png;base64,{b64}" alt="{name}" />'
            stats = f"V: {nv:,} &nbsp; F: {nf:,}"
        items += f"""        <div class="image-card">
            <div class="image-name">[{idx}] {name}</div>
            <div class="image-container">{body}</div>
            <div class="stats">{stats}</div>
        </div>\n"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5; }}
    h1 {{ text-align: center; color: #333; margin-bottom: 30px; }}
    .grid-container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                       gap: 20px; max-width: 2000px; margin: 0 auto; }}
    .image-card {{ background: white; border-radius: 8px;
                   box-shadow: 0 2px 8px rgba(0,0,0,0.1); overflow: hidden; }}
    .image-name {{ padding: 10px 14px; background: #fafafa; border-bottom: 1px solid #eee;
                   font-weight: 500; color: #555; font-size: 13px; text-align: center;
                   word-break: break-all; }}
    .image-container {{ width: 100%; height: 280px; background: #fafafa;
                        display: flex; align-items: center; justify-content: center; overflow: hidden; }}
    .image-container img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
    .error {{ color: #c33; font-size: 12px; padding: 10px; text-align: center; }}
    .stats {{ padding: 8px 14px; background: #fafafa; border-top: 1px solid #eee;
              font-size: 12px; color: #888; text-align: center; }}
</style>
</head>
<body>
    <h1>{title}</h1>
    <div class="grid-container">
{items}    </div>
</body>
</html>"""

    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Render gltf/glb directory to HTML gallery (cage-filtered).")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--output", type=str, default=None,
                        help="HTML output path (default: <input_dir_name>.html in CWD)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel render workers (default: min(cpu_count, 8))")
    args = parser.parse_args()

    input_dir = args.input_dir
    name = Path(input_dir).name
    output = args.output or f"{name}.html"
    workers = args.workers or min(os.cpu_count() or 1, 8)

    files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir)
         if f.lower().endswith((".gltf", ".glb"))],
        key=lambda p: _alphanumeric_key(Path(p).stem),
    )
    if not files:
        print(f"No .gltf/.glb files in {input_dir}")
        return

    print(f"Rendering {len(files)} files with {workers} workers...")

    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(render_single, p, args.width, args.height): p for p in files}
        for i, fut in enumerate(as_completed(futures), 1):
            path, b64, nv, nf, err = fut.result()
            results[path] = (b64, nv, nf, err)
            tag = "ERR" if err else "ok"
            print(f"  [{i}/{len(files)}] {tag}: {Path(path).name}{' — ' + err if err else ''}")

    cards = [(Path(p).stem, *results[p]) for p in files]
    title = f"{name} ({len(files)} files)"
    generate_html(title, cards, output)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
