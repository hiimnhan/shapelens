"""shapelens: trace tensor shapes in a code block, render in Jupyter.

    import shapelens as sl
    with sl.trace() as t:
        out = attention(x)
    t            # last expression in a cell -> HTML table
"""
import html
import linecache
import math
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_SKIP = ("torch/", "contextlib.py", "/IPython/", "/ipykernel/")
_NOISE = {"view", "_unsafe_view", "expand", "clone", "reshape", "_reshape_alias", "detach", "alias"}
_PALETTE =["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f"]


def _side(d):
    return int(8 + 12 * math.log2(max(d, 1)))  # log scale so 224 vs 3 stays drawable


def _user_frame(f):
    while f and (f.f_code.co_filename == __file__ or any(s in f.f_code.co_filename for s in _SKIP)):
        f = f.f_back
    return f


class trace(TorchDispatchMode):
    def __init__(self, **dims):
        """dims: extra labels, e.g. trace(N=196). UPPERCASE int variables in your code are picked up automatically."""
        super().__init__()
        self.dims = {}  # size -> [labels]; ponytail: label by value, so equal sizes share names (H/W)
        for k, v in dims.items():
            self._label(v, k)
        self.rows = []  # (lineno, code, op, [(id, shape)], [(id, shape)])
        self.names = {}  # id(tensor) -> variable name, last seen
        self._keep = []  # keep outputs alive so ids stay unique

    def _label(self, size, name):
        labels = self.dims.setdefault(size, [])
        if name not in labels:
            labels.append(name)

    def _scan(self, frame):
        for k, v in frame.f_locals.items():
            if isinstance(v, torch.Tensor) and not k.startswith("_"):
                self.names[id(v)] = k
            elif type(v) is int and v > 1 and k.isupper():  # B, T, D, H2 ... but not loop vars like i
                self._label(v, k)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        try:
            out = func(*args, **(kwargs or {}))
        except Exception as e:  # record the failing op, then let the error propagate
            self._rec(func, args, [], (str(e).strip().splitlines() or [type(e).__name__])[0][:300])
            raise
        self._rec(func, args, out, None)
        return out

    def _rec(self, func, args, out, err):
        fr = _user_frame(sys._getframe(1))
        if fr is not None:
            self._scan(fr)
            code = linecache.getline(fr.f_code.co_filename, fr.f_lineno).strip()
            line = fr.f_lineno
        else:
            code, line = "", 0
        outs = [o for o in (out if isinstance(out, (tuple, list)) else [out]) if isinstance(o, torch.Tensor)]
        flat = [x for a in args for x in (a if isinstance(a, (list, tuple)) else [a])]  # cat/stack take lists
        ins = [(id(a), tuple(a.shape)) for a in flat if isinstance(a, torch.Tensor)]
        self._keep += outs
        self.rows.append((line, code, func._schema.name.split("::")[-1], ins,
                          [(id(o), tuple(o.shape)) for o in outs], err))

    def __exit__(self, *a):
        fr = _user_frame(sys._getframe(1))  # block's frame holds the final variable names
        if fr is not None:
            self._scan(fr)
        r = super().__exit__(*a)
        if a[0] is not None:  # block raised: the cell's trailing `t` never runs, so show the trace now
            try:
                from IPython.display import display
                display(self)
            except ImportError:  # never mask the user's real error
                pass
        return r

    def _lab(self, d):
        return f'{"/".join(self.dims.get(d, []))} {d}'.strip()

    def text(self):
        fmt = lambda i, s: f'{self.names.get(i, "?")}[{", ".join(self._lab(d) for d in s)}]'
        return "\n".join(
            f"{'✗ ' if err else ''}{code}\n    {op}: {', '.join(fmt(i, s) for i, s in ins)} -> "
            + (f"ERROR: {err}" if err else ", ".join(fmt(i, s) for i, s in outs))
            for code, op, ins, outs, err in self.lines()
        )

    __repr__ = text

    # --- rendering ---
    def _box(self, tid, shape, colors):
        """Draw a tensor: last 2 dims = rectangle, 3rd-from-last = cuboid depth, leading dims = ghost stack + label."""
        col = lambda d: colors.setdefault(d, _PALETTE[len(colors) % len(_PALETTE)])
        lab = lambda d: html.escape(self._lab(d))
        n = len(shape)
        W = _side(shape[-1]) if n else 12
        Hh = _side(shape[-2]) if n >= 2 else (16 if n else 12)
        dep = shape[-3] if n >= 3 else None
        D = min(40, _side(dep) // 2) if dep else 0
        lead = shape[:-3]
        g = min(math.prod(lead), 3) - 1 if lead else 0  # ghost copies behind
        rl, dl = (lab(shape[-2]) if n >= 2 else ""), (lab(dep) if dep else "")
        lt = " × ".join(lab(d) for d in lead)
        x0, y0 = 7 * len(rl) + 8, D + (16 if lead else 4)

        def cube(ox, oy, op):
            x, y = x0 + ox, y0 + oy
            s = (f'<g opacity="{op}"><rect x="{x}" y="{y}" width="{W}" height="{Hh}" '
                 f'fill="#4e79a7" fill-opacity=".18" stroke="#4e79a7"/>')
            if D:
                c = col(dep)
                s += (f'<polygon points="{x},{y} {x+D},{y-D} {x+W+D},{y-D} {x+W},{y}" fill="{c}" fill-opacity=".45" stroke="{c}"/>'
                      f'<polygon points="{x+W},{y} {x+W+D},{y-D} {x+W+D},{y+Hh-D} {x+W},{y+Hh}" fill="{c}" fill-opacity=".3" stroke="{c}"/>')
            return s + "</g>"

        def text(x, y, d, anchor, txt):
            return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="11" font-family="monospace" '
                    f'fill="{col(d)}">{txt}</text>')

        body = "".join(cube(8 * i, 8 * i, 0.5) for i in range(g, 0, -1)) + cube(0, 0, 1)
        if n:
            body += text(x0 + W / 2, y0 + Hh + 8 * g + 13, shape[-1], "middle", lab(shape[-1]))
        if n >= 2:
            body += text(x0 - 4, y0 + Hh / 2 + 4, shape[-2], "end", rl)
        if dep:
            body += text(x0 + W + D + 5, y0 - D / 2 + 4, dep, "start", dl)
        if lead:
            body += f'<text x="{x0}" y="11" font-size="11" font-family="monospace" fill="currentColor">{lt}</text>'
        wd = max(x0 + W + D + 8 * g + (7 * len(dl) + 8 if dl else 6), x0 + 7 * len(lt) if lead else 0)
        ht = y0 + Hh + 8 * g + 18
        name = html.escape(self.names.get(tid, ""))
        return (f'<div style="text-align:center;margin:0 6px"><svg width="{int(wd)}" height="{int(ht)}">{body}</svg>'
                f'<div style="font:bold 12px monospace">{name}</div></div>')

    def lines(self):
        """Collapse raw aten ops to one row per source line: named inputs -> named outputs."""
        groups = []  # [code, ops, ins, outs, last_out, all_ops, err, failing_op_ins]
        for line, code, op, ins, outs, err in self.rows:
            if not groups or groups[-1][0] != (line, code):
                groups.append([(line, code), [], {}, {}, None, [], None, []])
            g = groups[-1]
            if err:
                g[6], g[7] = err, ins
            if op not in g[5]:
                g[5].append(op)
            if op not in _NOISE and op not in g[1]:
                g[1].append(op)
            g[2].update({i: s for i, s in ins if i in self.names})
            g[3].update({i: s for i, s in outs if i in self.names})
            g[4] = outs
        for g in groups:
            if g[6]:  # failing op: your named inputs, else the raw shapes that clash
                g[2], g[3] = g[2] or dict(g[7]), {}
                continue
            if not g[3] and g[4]:  # statement result never bound to a name
                g[3] = dict(g[4])
            g[2] = {i: s for i, s in g[2].items() if i not in g[3]}  # drop in-place/aliased echoes
        return [(g[0][1], ", ".join(g[1] or g[5]), list(g[2].items()), list(g[3].items()), g[6]) for g in groups]

    def _repr_html_(self):
        colors = {}  # same dim size -> same color, so equal dims are visible at a glance

        def row(code, op, ins, outs, err):
            bad = "background:rgba(225,87,89,.12);border-left:4px solid #e15759;padding-left:8px;" if err else ""
            res = (f"<div style='color:#e15759;font:12px monospace;max-width:340px;white-space:pre-wrap'>"
                   f"<b>✗ {html.escape(op)} failed</b><br>{html.escape(err)}</div>"
                   if err else "".join(self._box(i, s, colors) for i, s in outs))
            return (
                f"<div style='display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #8884;{bad}'>"
                f"<div style='width:240px;flex:none;font:12px monospace;white-space:pre-wrap'>{html.escape(code)}"
                f"<div style='opacity:.6'>{html.escape(op)}</div></div>"
                + "".join(self._box(i, s, colors) for i, s in ins)
                + f"<div style='font-size:24px;padding:0 6px;{'color:#e15759' if err else ''}'>{'✗' if err else '→'}</div>"
                + res
                + "</div>"
            )

        return "".join(row(*r) for r in self.lines())
