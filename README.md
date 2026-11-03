# shapelens

See tensor shapes as boxes. Wrap a block of PyTorch code and get, per source line, the input
tensors, the op, and the output tensor drawn as rectangles, with your variable names and
dimension labels (`B`, `T`, `D`, ...).

![ViT patchify traced with shapelens: each source line, its input tensors, the op, and the output tensor drawn as boxes](https://raw.githubusercontent.com/hiimnhan/shapelens/main/docs/vit-patchify.png)

*A ViT patch embedding, traced: image `(B, C, H, W)` becomes tokens `(B, N+1, D)`.*

Built for Jupyter notebooks. In a cell:

```python
import shapelens as sl

with sl.Trace() as t:
    out = attention(x, Wq, Wk, Wv, H=4)
t   # last expression in the cell renders the boxes inline
```

- **Shape errors:** if an op fails inside the block, the trace is displayed immediately with the
  failing line highlighted in red, then the original error still propagates.
- `print(t)` gives a plain-text version of the same trace.

![A failing matmul highlighted in red: A has last dim 32 but B has first dim 64](https://raw.githubusercontent.com/hiimnhan/shapelens/main/docs/shape-error.png)

## Drawing rules

- Last two dims are the rectangle (rows x cols); the third-from-last is the cuboid depth;
  any leading dims are a label plus ghost copies behind.
- Sides are log-scaled, so relative sizes are not to scale.
- Dims of equal size share a color.

## Dimension labels

Any UPPERCASE `int` variable in your code (`B = 2`, `D = 32`) labels every dim of that size.
Add more by hand: `sl.Trace(S=197)`. Labels go by size, so equal sizes show every name (`H/W`).

## Tip

Use `device="meta"` tensors to see shapes without any compute or memory.

## Limits

PyTorch only. Smoke-tested on torch 2.0.1 to 2.8.0 and 2.14.1 (Python 3.10, macOS arm64); older
versions are untested. The tracer keeps references to op outputs while tracing, so prefer `meta`
tensors for big models. Only errors raised inside a torch op are highlighted.

## Development

```
uv sync              # installs the project plus jupyter/nbconvert (dev group)
uv run python test_shapelens.py   # smoke test (no pytest needed)
uv run jupyter lab   # open demo.ipynb: attention, ViT patchify, and a shape-bug cell
uv build             # wheel + sdist into dist/
```
