```bash
python examples/skinning_asset.py --epochs 200 --out /tmp/W.npz
python examples/skinning_playback.py \
    --visual data/9423122485_cleaned.obj \
    --anim-dir /Users/szhan/projects/pbd/data/9423122485_cleaned_proxy \
    --weights /tmp/W.npz
```