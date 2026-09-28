# Historical nnU-Net reference environment

These requirement files are preserved from the original Full-16 benchmark.
They intentionally pin RankSEG 0.0.5 and Torch 2.8.0; changing them would not
turn historical results into a 0.0.7 evaluation. Use a separate environment:

```bash
python -m venv .venv-nnunet
.venv-nnunet/bin/python -m pip install --upgrade pip
.venv-nnunet/bin/python -m pip install -r environments/nnunet/requirements.txt
.venv-nnunet/bin/python -m pip install -e ".[nnunet]"
.venv-nnunet/bin/rankseg-bench nnunet verify-evidence evidence/nnunet
```

Select a compatible CPU/CUDA Torch wheel for your machine before installing
the remaining requirements. `requirements-lock.txt` records the full original
Linux/Python 3.10/CUDA environment, not a cross-platform installation promise.
