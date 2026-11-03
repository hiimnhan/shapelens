"""Helper for demo.ipynb: a plain attention block to trace."""


def attention(x, Wq, Wk, Wv, H):
    B, T, D = x.shape
    q = (x @ Wq).view(B, T, H, D // H).transpose(1, 2)
    k = (x @ Wk).view(B, T, H, D // H).transpose(1, 2)
    v = (x @ Wv).view(B, T, H, D // H).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / (D // H) ** 0.5
    w = scores.softmax(-1)
    return w @ v
