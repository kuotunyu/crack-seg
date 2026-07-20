# GitHub Publishing Checklist

Use this checklist before pushing the project as a job-search portfolio item.

## Repository Hygiene

- [ ] Keep `dataset/`, `checkpoints/`, `outputs/`, and `runs/` out of Git.
- [ ] Do not commit `.h5` weights. If trained weights are available later, publish them via GitHub Releases or Hugging Face Hub.
- [ ] Run `git status --ignored` and confirm ignored large folders are not staged.
- [ ] Remove local caches such as `__pycache__/` and `.pytest_cache/` from the working tree if they are distracting.

## Reproducibility

- [ ] Record the dataset source and split policy.
- [ ] Train with a fixed seed, for example `--seed 42`.
- [ ] Keep `runs/config.json` and `runs/metrics.json` for each experiment.
- [ ] Report validation or test metrics in README with Dice, IoU, Precision, Recall, and F1.

## Verification

- [ ] Install CPU test dependencies with `pip install -r requirements-ci.txt`.
- [ ] Run Python syntax checks with `python -m py_compile app.py config.py data_preprocessing.py evaluate.py model.py predict.py segmentation_utils.py train.py video_predict.py`.
- [ ] Run `python -m pytest tests/ -v`.
- [ ] (Requires a trained checkpoint) Smoke-test Gradio: `python app.py --model_path checkpoints/best_model.h5`.

## Portfolio Presentation

- [ ] If weights are published later, add the release link to README.
- [ ] Add 2-3 representative sample images and overlays.
- [ ] Include a compact benchmark table comparing U-Net, DeepLabV3+, and SegFormer if all three are trained on the same split.
- [ ] Add the final GitHub Actions badge after the repository URL is known.
