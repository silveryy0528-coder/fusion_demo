# Data format

Each dataset lives in its own folder (e.g. `data/synthetic/`) containing a
`manifest.json` plus one `.npz` file per modality.

## `manifest.json`

```json
{
  "sample_id": "synthetic",
  "pixel_size_nm": null,
  "shape": [256, 256],
  "modalities": {
    "haadf": {"file": "haadf.npz", "algorithms": ["SIRT"]},
    "Ag":    {"file": "Ag.npz",    "algorithms": ["SIRT"]},
    "Au":    {"file": "Au.npz",    "algorithms": ["SIRT"]}
  }
}
```

- `shape`: the shape every array in every modality must match (a 2-D slice
  or a small 3-D volume).
- `modalities`: one entry per modality (HAADF or an EDS element). `file` is
  the `.npz` file holding that modality's reconstructions; `algorithms` are
  the array keys expected inside it.

## `<modality>.npz`

Each `.npz` maps reconstruction-algorithm name to its array, e.g.:

```python
np.savez("Ti.npz", SIRT=sirt_array, FBP=fbp_array)
```

A modality can have one or several reconstructions. If it's chosen as the
**target** for fusion, multiple reconstructions are collapsed into one via
NMF (`fusion_demo.denoise.denoise`); if it's used as a **predictor**, every
reconstruction contributes its own feature columns.

See [`src/fusion_demo/io.py`](../src/fusion_demo/io.py) for the loading/validation code.
