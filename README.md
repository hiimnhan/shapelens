# shapelens

Visualise Pytorch tensor dimensions at ease in Jupyter notebook.

![ViT patchify traced with shapelens: each source line, its input tensors, the op, and the output tensor drawn as boxes](https://raw.githubusercontent.com/hiimnhan/shapelens/main/docs/vit-patchify.png)

*A ViT patch embedding, traced: image `(B, C, H, W)` becomes tokens `(B, N+1, D)`.*

Built for Jupyter notebooks. In a cell:

```python
import shapelens as sl

with sl.trace() as t:
    out = attention(x, Wq, Wk, Wv, H=4)
t   # last expression in the cell renders the boxes inline
```

- **Shape errors:** if an op fails inside the block, the trace is displayed immediately with the
  failing line highlighted in red, then the original error still propagates.
- `print(t)` gives a plain-text version of the same trace.

![A failing matmul highlighted in red: A has last dim 32 but B has first dim 64](https://raw.githubusercontent.com/hiimnhan/shapelens/main/docs/shape-error.png)

## Example: self-attention

```python
import torch
import shapelens as sl

B, T, D = 2, 5, 8                         # batch, tokens, embedding dim
x  = torch.randn(B, T, D, device="meta")  # meta: shapes only, no compute
Wq = torch.randn(D, D, device="meta")
Wk = torch.randn(D, D, device="meta")
Wv = torch.randn(D, D, device="meta")

with sl.trace() as t:
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    scores = q @ k.transpose(-2, -1) / D ** 0.5   # every token against every token: (B, T, T)
    attn = scores.softmax(dim=-1)
    out = attn @ v                                # weighted sum of values: (B, T, D)
t   # in a notebook this renders boxes; print(t) gives the text below
```

![Single-head self-attention traced with shapelens: x is projected to q, k, v; scores are T by T; the output returns to T by D](https://raw.githubusercontent.com/hiimnhan/shapelens/main/docs/self-attention.png)

```
q = x @ Wq
    mm: x[B 2, T 5, D 8], Wq[D 8, D 8] -> q[B 2, T 5, D 8]
scores = q @ k.transpose(-2, -1) / D ** 0.5   # every token against every token: (B, T, T)
    transpose, bmm, div: k[B 2, T 5, D 8], q[B 2, T 5, D 8] -> scores[B 2, T 5, T 5]
attn = scores.softmax(dim=-1)
    _softmax: scores[B 2, T 5, T 5] -> attn[B 2, T 5, T 5]
out = attn @ v                                # weighted sum of values: (B, T, D)
    bmm: attn[B 2, T 5, T 5], v[B 2, T 5, D 8] -> out[B 2, T 5, D 8]
```

(The `k` and `v` projections are traced the same way as `q` and omitted here.) Note how `T 5` becomes
`T 5 × T 5` in `scores`, then collapses back to `D 8` in `out`.

## Dimension labels

Sizes are named from your code automatically:

- **UPPERCASE `int` variables** (`B = 2`, `D = 32`) label every dim of that size: `B 2`, `D 32`.
- **Integer arithmetic** on a traced line names its result by the expression itself. In
  `.view(B, T, H, D // H)` the head dim shows as `D // H = 8`.

To give an expression a short name, pass it with its **exact code**. No hard-coded number is needed:

```python
with sl.trace(Dh="D // H") as t:          # "Dh" names whatever `D // H` is in the code below
    out = attention(x, Wq, Wk, Wv, H=4)   # q: [B 2, H 4, T 5, Dh 8]
```

![The head dimension labelled automatically as "D // H = 8", then renamed "Dh" with sl.trace(Dh="D // H")](https://raw.githubusercontent.com/hiimnhan/shapelens/main/docs/dimension-labels.png)

A size with no variable or expression of its own can be named by value: `sl.trace(S=197)`.

**Priority**, highest first: names you pass, then the expression text, then variable names. The expression is
evaluated where that line actually runs, so it beats a leftover variable of the same size (say `D = 8` from an
earlier notebook cell), and a name you pass beats both. If the expression you name isn't found in the traced
code (a typo), you get a warning.

**Limits.** Labels go by size, so equal sizes show every name (`H/W`), and any other axis of the same size gets
the same label. Expressions must be integer arithmetic (`+ - * // % **`) on plain variables, on a single source
line. They are evaluated by a small whitelist-only evaluator, never with `eval`.

## Tip

Use `device="meta"` tensors to see shapes without any compute or memory.

