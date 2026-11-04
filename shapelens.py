"""shapelens: trace tensor shapes in a code block, render in Jupyter.

    import shapelens as sl
    with sl.trace() as t:
        out = attention(x)
    t            # last expression in a cell -> HTML table
"""
import ast
import html
import linecache
import math
import operator
import sys
import warnings

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_SKIP = ("torch/", "contextlib.py", "/IPython/", "/ipykernel/")
_NOISE = {"view", "_unsafe_view", "expand", "clone", "reshape", "_reshape_alias", "detach", "alias"}
_PALETTE =["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f"]


_ARITH = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def _arith(node, env):
    """Evaluate pure integer arithmetic (int constants, int variables from env, + - * // % ** and unary minus).
    No eval(): anything else raises ValueError, so your code is never executed or given side effects."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name) and type(env.get(node.id)) is int:
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_arith(node.operand, env)
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH:
        a, b = _arith(node.left, env), _arith(node.right, env)
        if isinstance(node.op, ast.Pow) and not (0 <= b <= 16 and abs(a) <= 10**6):
            raise ValueError("pow too large")
        return _ARITH[type(node.op)](a, b)
    raise ValueError("not integer arithmetic")


def _value(node, frame, env):
    """Size of an expression YOU named, e.g. trace(Dh="self.d_model // self.n_heads"). Tries the safe integer
    evaluator first; otherwise evaluates like a debugger watch expression, in the scope where the line runs."""
    try:
        v = _arith(node, env)
    except (ValueError, ArithmeticError):
        v = eval(compile(ast.Expression(body=node), "<shapelens>", "eval"), frame.f_globals, frame.f_locals)
    if type(v) is float and v.is_integer():  # D / H -> 8.0
        v = int(v)
    if type(v) is bool:
        raise TypeError("evaluated to a bool, expected an int")
    try:
        return operator.index(v)
    except TypeError:
        raise TypeError(f"evaluated to a {type(v).__name__}, expected an int") from None


def _side(d):
    return int(8 + 12 * math.log2(max(d, 1)))  # log scale so 224 vs 3 stays drawable


def _user_frame(f):
    while f and (f.f_code.co_filename == __file__ or any(s in f.f_code.co_filename for s in _SKIP)):
        f = f.f_back
    return f


class trace(TorchDispatchMode):
    def __init__(self, **dims):
        """Dimension labels. UPPERCASE int variables (B, T, D) and integer arithmetic in your code (D // H) are
        named automatically. Names you pass win: trace(S=197) names size 197; trace(Dh="D // H") names whatever
        that exact expression evaluates to in your code. Any Python expression works there, e.g.
        trace(Dh="self.d_model // self.n_heads"); it is evaluated like a debugger watch expression."""
        super().__init__()
        self.dims = {}      # size -> [variable names]; ponytail: labels go by size, so equal sizes share names (H/W)
        self.explicit = {}  # size -> [names you passed in]
        self.exprs = {}     # size -> [source expressions like "D // H"]: fallback when nothing else names the size
        self._named = {}    # normalized expression text -> the name you gave it
        self._hit = set()   # named expressions that were found in the traced code
        self._failed = {}   # named expressions that were found but could not be evaluated -> reason
        self._trees = {}    # file -> (number of lines, parsed tree)
        self._ast = {}      # (file, line) -> [(expression node, source text)] of the whole statement
        for name, v in dims.items():
            if isinstance(v, str):
                self._named[ast.unparse(ast.parse(v, mode="eval").body)] = name
            else:
                self._add(self.explicit, v, name)
        self.rows = []  # (lineno, code, op, [(id, shape)], [(id, shape)])
        self.names = {}  # id(tensor) -> variable name, last seen
        self._keep = []  # keep outputs alive so ids stay unique

    @staticmethod
    def _add(table, size, name):
        names = table.setdefault(size, [])
        if name not in names:
            names.append(name)

    def _scan(self, frame):
        for k, v in frame.f_locals.items():
            if isinstance(v, torch.Tensor) and not k.startswith("_"):
                self.names[id(v)] = k
            elif type(v) is int and v > 1 and k.isupper():  # B, T, D ... not loop vars like i
                self._add(self.dims, v, k)

    def _nodes(self, fr):
        """(node, source text) for every expression in the WHOLE statement running at fr's line, so a call split
        over several lines still works. For compound statements (with/for/if) only the header is used."""
        fn, lineno = fr.f_code.co_filename, fr.f_lineno
        key = (fn, lineno)
        if key not in self._ast:
            nodes = []
            try:
                lines = linecache.getlines(fn)
                if fn not in self._trees or self._trees[fn][0] != len(lines):
                    self._trees[fn] = (len(lines), ast.parse("".join(lines)))
                stmts = [n for n in ast.walk(self._trees[fn][1])
                         if isinstance(n, ast.stmt) and n.lineno <= lineno <= n.end_lineno]
                if stmts:
                    stmt = min(stmts, key=lambda n: n.end_lineno - n.lineno)  # innermost statement
                    for field, value in ast.iter_fields(stmt):
                        if field in ("body", "orelse", "finalbody", "handlers"):
                            continue
                        for v in value if isinstance(value, list) else [value]:
                            if isinstance(v, ast.AST):
                                nodes += [m for m in ast.walk(v) if isinstance(m, ast.expr)]
            except (SyntaxError, ValueError):  # whole file doesn't parse (e.g. notebook %magics): use this line only
                line = linecache.getline(fn, lineno).strip()
                try:
                    nodes = [m for m in ast.walk(ast.parse(line + " pass" if line.endswith(":") else line))
                             if isinstance(m, ast.expr)]
                except SyntaxError:
                    nodes = []
            self._ast[key] = [(n, ast.unparse(n)) for n in nodes]
        return self._ast[key]

    def _scan_exprs(self, fr):
        nodes = self._nodes(fr)
        auto = [(n, t) for n, t in nodes if isinstance(n, ast.BinOp) and len(t) <= 16 and t not in self._named
                and any(isinstance(m, ast.Name) for m in ast.walk(n))]  # automatic: safe integer arithmetic only
        named = [(n, t) for n, t in nodes if t in self._named]
        if not auto and not named:
            return
        env = {**fr.f_globals, **fr.f_locals}  # evaluated in the scope where that line really runs
        for node, text in auto:
            try:
                v = _arith(node, env)
            except (ValueError, ArithmeticError):
                continue
            if v > 1:
                self._add(self.exprs, v, text)
        for node, text in named:
            try:
                v = _value(node, fr, env)
            except Exception as e:  # you named it, so tell you why it could not be used
                self._failed[text] = f"{type(e).__name__}: {e}"
                continue
            self._hit.add(text)
            self._add(self.explicit, v, self._named[text])

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
            self._scan_exprs(fr)
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
        for text, name in self._named.items():
            if text in self._hit:
                continue
            why = (f"found `{text}` in the traced code but could not evaluate it ({self._failed[text]})"
                   if text in self._failed else f"`{text}` was not found in the traced code")
            warnings.warn(f"shapelens: {why}, so the label '{name}' was not applied", stacklevel=2)
        if a[0] is not None:  # block raised: the cell's trailing `t` never runs, so show the trace now
            try:
                from IPython.display import display
                display(self)
            except ImportError:  # never mask the user's real error
                pass
        return r

    def _lab(self, d):
        names = self.explicit.get(d)  # priority: names you passed, then the expression in your code, then variables
        if names:
            return f'{"/".join(names)} {d}'
        if d in self.exprs:  # "D // H = 8": evaluated where that line ran, so it beats a stale global like D = 8
            return f'{self.exprs[d][0]} = {d}'
        if d in self.dims:
            return f'{"/".join(self.dims[d])} {d}'
        return str(d)

    def text(self):
        fmt = lambda i, s: f'{self.names.get(i, "?")}{tuple(s)}'  # plain shape, like torch: x(2, 5, 32)
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
        row_d = shape[-2] if n >= 2 else (1 if n == 1 else None)  # a 1-D tensor is drawn as one row: show its "1" side too
        rl, dl = (lab(row_d) if row_d is not None else ""), (lab(dep) if dep else "")
        lt = " × ".join(lab(d) for d in lead)
        cl = lab(shape[-1]) if n else ""  # the bottom label is centred: keep long ones ("D // H = 8") inside the SVG
        x0, y0 = max(7 * len(rl) + 8, int(3.5 * len(cl) - W / 2) + 4), D + (16 if lead else 4)

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
        if row_d is not None:
            body += text(x0 - 4, y0 + Hh / 2 + 4, row_d, "end", rl)
        if dep:
            body += text(x0 + W + D + 5, y0 - D / 2 + 4, dep, "start", dl)
        if lead:
            body += f'<text x="{x0}" y="11" font-size="11" font-family="monospace" fill="currentColor">{lt}</text>'
        wd = max(x0 + W + D + 8 * g + (7 * len(dl) + 8 if dl else 6), x0 + 7 * len(lt) if lead else 0,
                 x0 + W / 2 + 3.5 * len(cl) + 4)
        ht = y0 + Hh + 8 * g + 18
        name = html.escape(self.names.get(tid, ""))
        return (f'<div style="text-align:center;margin:0 6px"><svg width="{int(wd)}" height="{int(ht)}">{body}</svg>'
                f'<div style="font:bold 12px monospace">{name}</div>'
                f'<div style="font:11px monospace;opacity:.7">{html.escape(str(tuple(shape)))}</div></div>')

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
                + (f"<div style='font-size:24px;padding:0 6px;{'color:#e15759' if err else ''}'>{'✗' if err else '→'}</div>"
                   if ins or err else "")  # creation ops (ones, randn, ...) have no inputs: no empty gap, no arrow
                + res
                + "</div>"
            )

        return "".join(row(*r) for r in self.lines())
