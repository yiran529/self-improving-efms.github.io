
# Reproducible Environment Setup (uv)

This project uses **uv** to manage Python dependencies reproducibly via:

- `pyproject.toml` (declared dependencies)
- `uv.lock` (fully resolved lockfile)

## Prerequisites

- Python 3.12
- uv installed

### Install uv

```bash
pip install -U uv
```

## Install (CPU)

1. Create a virtual environment (use your Python 3.12 interpreter)

```bash
uv venv --python /path/to/python3.12
```

2. Install dependencies exactly as pinned in `uv.lock`

```bash
uv sync --frozen
```

3. Verify

```bash
uv run python -c "import jax; print('devices:', jax.devices())"
```

## Optional: Install with GPU (CUDA 12.x)

If your machine has NVIDIA GPU available, install with the `gpu` extra:

```bash
uv sync --frozen --extra gpu
```

Verify:

```bash
uv run python -c "import jax; print(jax.devices())"
```

## Code Compatibility Update

To ensure compatibility with **matplotlib 3.10+**, the following two occurrences in the code:

```python
### in pointmass_notebook_old.ipynb
# Convert the rendered image to a numpy array
width, height = fig.get_size_inches() * fig.get_dpi()
image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
image = image.reshape(int(height), int(width), 3)
```

need to be replaced with:

```python
### in pointmass_notebook_modified.ipynb
canvas.draw()
width, height = canvas.get_width_height()
image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
image = image.reshape(height, width, 4)
image = image[:, :, :3]
```


---

# China Mainland Network (镜像源/代理)

The lockfile may reference the official PyPI registry (`https://pypi.org/simple`) and files hosted on `https://files.pythonhosted.org/...`.
In some China mainland network environments, these domains can be slow or inaccessible.
In that case, configure a mirror index for uv.

You can configure mirrors in **either** of the following ways:

## Option A: Configure in project config (recommended for team reproducibility)

Add index configuration to `pyproject.toml` under `[tool.uv]`:

```toml
[[tool.uv.index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```

You may replace the URL with any mirror you trust (e.g., TUNA / Aliyun / Tencent mirrors).

> uv supports persistent index configuration via `[[tool.uv.index]]` in `pyproject.toml` (project-level),
> or in `uv.toml` (user/system-level). See uv docs for details.

## Option B: Configure via environment variables (per-user, no repo changes)

Set a default index (mirror) via environment variable:

```bash
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
```

You can also add additional indexes via `UV_INDEX` (space-separated list of URLs) if needed.

After setting the mirror, run the install as usual:

```bash
uv sync --frozen
```
