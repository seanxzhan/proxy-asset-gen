```bash
python examples/skinning_asset.py --epochs 200 --out /tmp/W.npz
python examples/skinning_playback.py \
    --visual data/9423122485_cleaned.obj \
    --anim-dir /Users/szhan/projects/pbd/data/9423122485_cleaned_proxy \
    --weights /tmp/W.npz
```

# Implementation of Proxy Asset Generation for Cloth Simulation in Games

First, get proxy mesh.
```bash
python scripts/get_proxy.py --input data/9423122485_cleaned.obj
```

Second, use the PBD project to generate training and testing data.
```bash
python scripts/windblown_data_gen.py \
  --obj ~/projects/proxy-asset-gen/data/9423122485_cleaned_proxy.obj \
  --axis direction \
  --train-frames 600 --test-frames 200 \
  --pin-fraction 0.10 \
  --mag-train 0.3 0.3 --mag-test 0.3 0.3 \
  --turbulence-std 0.0 --coherence 1.0 --viz \
  --out data/9423122485_cleaned_proxy
```

Third, optimize skinning weights. 
```bash
python scripts/get_skin_weights.py \
    --epochs 300 \
    --anim-dir /Users/szhan/projects/pbd/data/9423122485_cleaned_proxy \
    --out results/9423122485_cleaned_proxy_skinning.npz
```

Playback the training and testing set.
```bash
python scripts/skinning_playback.py \
    --visual data/9423122485_cleaned.obj \
    --anim-dir /Users/szhan/projects/pbd/data/9423122485_cleaned_proxy \
    --weights results/9423122485_cleaned_proxy_skinning.npz
```
