"""Smoke test, no pytest needed:  python test_shapelens.py"""
import torch

import shapelens as sl


def test_trace_names_shapes_and_labels():
    B, T, D = 2, 5, 32
    x = torch.randn(B, T, D, device="meta")
    W = torch.randn(D, D, device="meta")
    with sl.trace(S=197) as t:
        q = (x @ W).view(B, T, 4, D // 4).transpose(1, 2)
        s = q @ q.transpose(-2, -1)
    rows = t.lines()
    assert len(rows) == 2, rows
    assert rows[0][3][0][1] == (2, 4, 5, 8)  # q
    assert rows[1][3][0][1] == (2, 4, 5, 5)  # s
    assert [t.names[i] for i, _ in rows[1][3]] == ["s"]
    assert t.dims[32] == ["D"] and t.dims[197] == ["S"]  # auto UPPERCASE ints + explicit label
    assert "<svg" in t._repr_html_() and "s[" in t.text()


def test_failing_op_is_recorded_and_error_propagates():
    x = torch.randn(2, 5, 32, device="meta")
    W_bad = torch.randn(64, 16, device="meta")
    t = sl.trace()
    try:
        with t:
            a = x + 1
            h = a @ W_bad
    except RuntimeError:
        pass
    else:
        raise AssertionError("shape error must still propagate")
    rows = t.lines()
    assert rows[0][4] is None and rows[-1][4], rows  # last row carries the error text
    assert "mm failed" in t._repr_html_() and t.text().startswith("a = x + 1")


if __name__ == "__main__":
    test_trace_names_shapes_and_labels()
    test_failing_op_is_recorded_and_error_propagates()
    print("ok torch", torch.__version__)
