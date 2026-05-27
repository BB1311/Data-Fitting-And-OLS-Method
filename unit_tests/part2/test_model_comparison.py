"""
test_model_comparison.py
================
Unit tests cho OLSFull.

"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part2.model_comparison import OLSFull


# ══════════════════════════════════════════════════════════════════════
# FIXTURES dùng chung
# ══════════════════════════════════════════════════════════════════════

def _perfect_data():
    """y = 2 + 3*x1 - x2 — kết quả đã biết: beta=[2,3,-1], R²=1.

    Lưu ý: x2 phải độc lập tuyến tính với cả cột intercept [1,1,1,1,1] và x1.
    x2 = [2,1,3,2,4]  → không phải tổ hợp tuyến tính của intercept và x1.
    """
    X = pd.DataFrame({"x1": [1.,2.,3.,4.,5.], "x2": [2.,1.,3.,2.,4.]})
    y = (2 + 3*X["x1"] - X["x2"]).values   # [3,4,5,6,7]
    return X, y

def _noisy_data():
    """y = 3 + 2*x1 + noise — R² < 1, metric > 0."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"x1": np.linspace(0, 10, 50)})
    y = 3. + 2.*X["x1"].values + rng.normal(0, .5, 50)
    return X, y


# ══════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════

def test_fit_correct_coefficients():
    """fit() — hệ số đúng trên dữ liệu tuyến tính hoàn hảo."""
    X, y = _perfect_data()
    beta = OLSFull(verbose=False).fit(X, y).coef_
    assert abs(beta[0] - 2.) < 1e-6, f"Intercept sai: {beta[0]}"
    assert abs(beta[1] - 3.) < 1e-6, f"Coef x1 sai: {beta[1]}"
    assert abs(beta[2] - (-1.)) < 1e-6, f"Coef x2 sai: {beta[2]}"


def test_fit_ndarray_feature_names():
    """fit() — nhận ndarray, tên biến fallback x0/x1."""
    X, y = _perfect_data()
    model = OLSFull(verbose=False).fit(X.values, y)
    assert model.n_features == 2
    assert model._feature_names == ["x0", "x1"]


def test_predict_perfect_fit():
    """predict() — khớp y thực trên dữ liệu hoàn hảo."""
    X, y = _perfect_data()
    model = OLSFull(verbose=False).fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-6)


def test_predict_dataframe_ndarray_equal():
    """predict() — DataFrame và ndarray cho cùng kết quả."""
    X, y = _noisy_data()
    model = OLSFull(verbose=False).fit(X, y)
    assert np.allclose(model.predict(X), model.predict(X.values), atol=1e-10)


def test_evaluate_perfect_metrics():
    """evaluate() — R²=1, MAE=0, RMSE=0 trên fit hoàn hảo."""
    X, y = _perfect_data()
    m = OLSFull(verbose=False).fit(X, y).evaluate(X, y, verbose=False)
    assert abs(m["r2"] - 1.) < 1e-6
    assert m["mae"]  < 1e-6
    assert m["rmse"] < 1e-6


def test_evaluate_noisy_metrics_valid():
    """evaluate() — metric hợp lệ trên dữ liệu có nhiễu."""
    X, y = _noisy_data()
    m = OLSFull(verbose=False).fit(X, y).evaluate(X, y, verbose=False)
    assert 0. <= m["r2"] <= 1.
    assert m["mae"]  > 0
    assert m["rmse"] > 0


def test_coef_table_shape_and_columns():
    """coef_table() — shape và tên cột đúng."""
    X, y = _noisy_data()
    tbl = OLSFull(verbose=False).fit(X, y).coef_table(X, y)
    assert set(tbl.columns) == {"feature","coef","std_error","t_stat","p_value","ci_lower","ci_upper"}
    assert len(tbl) == X.shape[1] + 1        # +1 cho Intercept
    assert tbl.iloc[0]["feature"] == "Intercept"


def test_coef_table_valid_statistics():
    """coef_table() — p_value ∈ [0,1], CI lower < upper."""
    X, y = _noisy_data()
    tbl = OLSFull(verbose=False).fit(X, y).coef_table(X, y)
    assert tbl["p_value"].between(0, 1).all()
    assert (tbl["ci_lower"] < tbl["ci_upper"]).all()


def test_to_result_keys_and_lam():
    """to_result() — keys đúng, lam=None (OLS không có λ)."""
    X, y = _noisy_data()
    result = OLSFull(verbose=False).fit(X, y).to_result(X, y)
    assert {"model","metrics","feature_names","lam"}.issubset(result.keys())
    assert result["lam"] is None
    assert result["feature_names"] == ["x1"]


def test_to_result_metrics_consistent():
    """to_result() — metrics khớp với evaluate()."""
    X, y = _noisy_data()
    model   = OLSFull(verbose=False).fit(X, y)
    res     = model.to_result(X, y)
    metrics = model.evaluate(X, y, verbose=False)
    assert abs(res["metrics"]["r2"]   - metrics["r2"])   < 1e-10
    assert abs(res["metrics"]["mae"]  - metrics["mae"])  < 1e-10
    assert abs(res["metrics"]["rmse"] - metrics["rmse"]) < 1e-10


# ══════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════

def _run_all():
    tests = [
        test_fit_correct_coefficients,
        test_fit_ndarray_feature_names,
        test_predict_perfect_fit,
        test_predict_dataframe_ndarray_equal,
        test_evaluate_perfect_metrics,
        test_evaluate_noisy_metrics_valid,
        test_coef_table_shape_and_columns,
        test_coef_table_valid_statistics,
        test_to_result_keys_and_lam,
        test_to_result_metrics_consistent,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n  {passed}/{passed+failed} tests PASS")


if __name__ == "__main__":
    _run_all()