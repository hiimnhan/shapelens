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
    assert t._lab(32) == "D 32" and t._lab(197) == "S 197"  # auto UPPERCASE ints + explicit label
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


def test_explicit_label_wins_over_a_stray_variable_of_the_same_size():
    D = 8  # leftover variable of the same size, like a `D = 8` from an earlier notebook cell
    x = torch.randn(2, 8, device="meta")
    with sl.trace(Dh=8) as t:
        y = x + 1
    assert t._lab(8) == "Dh 8", t._lab(8)  # was "Dh/D 8"


def test_expression_in_code_names_its_size():
    D, H = 32, 4
    x = torch.randn(2, 5, 32, device="meta")
    with sl.trace() as t:
        q = x.view(2, 5, H, D // H)
    assert t._lab(8) == "D // H = 8", t._lab(8)
    assert "q[2, 5, H 4, D // H = 8]" in t.text(), t.text()


def test_expression_beats_a_stale_variable_of_the_same_size():
    D, H, T = 32, 4, 8  # T == D // H == 8, like a leftover `D = 8` from an earlier notebook cell
    x = torch.randn(2, 5, 32, device="meta")
    with sl.trace() as t:
        q = x.view(2, 5, H, D // H)
    assert t._lab(8) == "D // H = 8", t._lab(8)


def test_name_an_expression_by_its_code():
    D, H = 32, 4
    x = torch.randn(2, 5, 32, device="meta")
    with sl.trace(Dh="D // H") as t:  # "Dh" names whatever `D // H` is in the code below, no hard-coded 8
        q = x.view(2, 5, H, D // H)
    assert t._lab(8) == "Dh 8", t._lab(8)


def test_unmatched_expression_warns():
    import warnings
    x = torch.randn(2, 3, device="meta")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with sl.trace(Dh="D // H"):  # no such expression in the traced code
            y = x + 1
    assert any("D // H" in str(m.message) for m in w), [str(m.message) for m in w]


def test_expression_evaluator_is_safe():
    import ast
    assert sl._arith(ast.parse("D // H + 1", mode="eval").body, {"D": 32, "H": 4}) == 9
    for bad in ("__import__('os').system('x')", "x.shape[0]", "D ** 99999", "D / H", "D // Z"):
        try:
            sl._arith(ast.parse(bad, mode="eval").body, {"D": 32, "H": 4})
        except (ValueError, ArithmeticError):
            continue
        raise AssertionError(f"should have been rejected: {bad}")


def test_no_arrow_without_inputs_and_1d_shows_both_sides():
    with sl.trace() as t:
        x = torch.ones(2, 3, device="meta")  # creation op: no inputs
    assert "→" not in t._repr_html_(), "no input tensors -> no empty gap and no arrow"
    with sl.trace() as t:
        x = torch.ones(2, 3, device="meta")
        y = x.sum(1)  # y is 1-D (2,)
    html = t._repr_html_()
    assert html.count("→") == 1  # only the sum row has inputs
    assert ">1</text>" in html and ">2</text>" in html, "a 1-D tensor must label its implicit '1' side as well"


if __name__ == "__main__":
    test_explicit_label_wins_over_a_stray_variable_of_the_same_size()
    test_expression_in_code_names_its_size()
    test_expression_beats_a_stale_variable_of_the_same_size()
    test_name_an_expression_by_its_code()
    test_unmatched_expression_warns()
    test_expression_evaluator_is_safe()
    test_no_arrow_without_inputs_and_1d_shows_both_sides()
    test_trace_names_shapes_and_labels()
    test_failing_op_is_recorded_and_error_propagates()
    print("ok torch", torch.__version__)
