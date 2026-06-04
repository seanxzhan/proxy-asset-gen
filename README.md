# Implementation of Proxy Asset Generation for Cloth Simulation in Games

## Installation

```bash
conda env create -f env_mac.yml
pip install -e .
```

Also install pbd with
```bash
pip install -e /Users/szhan/projects/pbd 
```

## Run

First, get proxy mesh.
```bash
python scripts/get_proxy.py --input data/9423122485_cleaned.obj
```

Second, use the PBD project to generate training and testing data.
```bash
python scripts/data_gen/windblown_data_gen.py \
  --obj data/9423122485_cleaned_proxy.obj \
  --axis direction \
  --train-frames 600 --test-frames 200 \
  --pin-fraction 0.10 \
  --mag-train 0.3 0.3 --mag-test 0.3 0.3 \
  --turbulence-std 1.0 --coherence 0.95 --viz \
  --out data/9423122485_cleaned_proxy

# just visualize
python scripts/data_gen/vis_windblown_data.py \
    --obj data/9423122485_cleaned_proxy.obj \
    --pin-fraction 0.10 --magnitude 0.3 \
    --turbulence-std 1.0 --coherence 0.95
```

Third, optimize skinning weights. 
```bash
python scripts/get_skin_weights.py \
    --epochs 300 \
    --anim-dir data/9423122485_cleaned_proxy \
    --out results/9423122485_cleaned_proxy_skinning.npz
```

Playback the training and testing set.
```bash
python scripts/skinning_playback.py \
    --visual data/9423122485_cleaned.obj \
    --anim-dir data/9423122485_cleaned_proxy \
    --weights results/9423122485_cleaned_proxy_skinning.npz
```

## Test cases

Test proxy mesh generation:
```bash
python -m pytest tests/test_curvature.py tests/test_extract.py \
    tests/test_guide_graph.py tests/test_ilp.py tests/test_io.py \
    tests/test_mesh.py tests/test_pipeline.py tests/test_udf.py \
    --deselect tests/test_pipeline.py::test_pipeline_determinism
```

Test skinning optimization:
```bash
python -m pytest tests/test_anim.py tests/test_skinning_lbs.py \
    tests/test_skinning_losses.py tests/test_skinning_pipeline.py
```

These two need to be tested separately because of pytorch / vtk conflict