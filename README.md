# Deep Learning Asset Pricing

## Tested Under: 
* **Tensorflow 1.12.0**
* **Python 3.6**

## Conda Environment
Conda virtual environment can be created with: conda create --name myenv --file requirements.txt python=3.6.10

## Usage
Check model_GAN.ipynb for training and model evaluation

## Datasets and Sample Checkpoints
Datasets and sample model trained checkpoints can be found at [Google Drive](https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l?usp=sharing)

## Glass-box extension (research code, PyTorch)

`glassbox/` adds an interpretable alternative to the GAN SDF and two research pipelines built on the same
data files. The original TensorFlow code above is unchanged.

* **Paper 1: glass-box SDF and auditor** ([design](docs/paper1_glassbox_sdf_auditor.md)). An additive,
  state-varying SDF whose shape functions *are* the model; a complexity ladder up to a CPZ-style neural SDF
  (the price of interpretability); and a rule-based auditor that finds where any SDF fails out of sample
  and classifies the failure as limits-to-arbitrage (mispricing-type) or state-dependent (risk-type).
* **Paper 2: the Rashomon set of SDFs** ([design](docs/paper2_rashomon_sdf.md)). For the glass-box SDF the
  set of near-optimal SDFs is an exact ellipsoid, which gives exact ranges for characteristic importance,
  slope signs and stock-level SDF weights ("SDF ambiguity"), plus tests of whether that ambiguity is priced.

```bash
pip install -r requirements_glassbox.txt
python scripts/make_synthetic.py                       # synthetic panel with planted ground truth
python scripts/run_paper1.py --config config_glassbox/synthetic.json --out results/paper1_synthetic
python scripts/run_paper2.py --config config_glassbox/synthetic.json --out results/paper2_synthetic
python -m pytest tests                                 # validation suite
```

First results on the CPZ data: [docs/cpz_results.md](docs/cpz_results.md). Extended CRSP + Open Source Asset Pricing panel through 2024: [docs/osap_results.md](docs/osap_results.md). Robustness to rule thresholds and knots: [docs/robustness.md](docs/robustness.md).

For the CPZ data, put the Google Drive files under `datasets/`, pick state variables with
`python scripts/run_paper1.py --config config_glassbox/cpz.json --list_macro`, and run with `config_glassbox/cpz.json`.
