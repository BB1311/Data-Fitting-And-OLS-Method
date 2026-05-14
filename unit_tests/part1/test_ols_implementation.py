import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path để import được part1
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from part1.ols_implementation import (
    ols_fit,
    OLSRegressor,
    compute_rss,
    compute_tss,
    compute_ess,
    compute_r2,
    compute_r2_adj,
    model_metrics,
    verify_with_sklearn,
)


def _assert_close(a, b, tol=1e-8, msg=""):
    """So sánh hai mảng (hoặc số) với sai số cho phép tol."""
    diff = np.max(np.abs(np.asarray(a) - np.asarray(b)))
    assert diff < tol, f"FAIL [{msg}]: max_diff={diff:.4e} (tol={tol:.0e})"


# ============================================================
# Core OLS & Class OLSRegressor
# ============================================================

def test_ols_fit_returns_correct_keys():
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([3, 5, 7], dtype=float)
    result = ols_fit(X, y)

    expected_keys = ("beta_hat", "sigma2_hat", "y_hat", "residuals", "X_design", "n", "k", "df")
    for key in expected_keys:
        assert key in result, f"FAIL: thiếu key '{key}'"

    assert result["n"] == 3, f"FAIL: n phải bằng 3, nhưng ra {result['n']}"
    assert result["k"] == 2, f"FAIL: k phải bằng 2, nhưng ra {result['k']}"
    assert result["df"] == 1, f"FAIL: df phải bằng 1, nhưng ra {result['df']}"


def test_ols_simple_known_data():
    """Test 1: Simple Linear Regression (y = 1 + 2x)."""
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([3, 5, 7], dtype=float)
    model = OLSRegressor(fit_intercept=True).fit(X, y)
    _assert_close(model.beta_hat[0], 1.0, msg="Intercept phải bằng 1")
    _assert_close(model.beta_hat[1], 2.0, msg="Slope phải bằng 2")
    _assert_close(model.residuals, [0, 0, 0], msg="Dữ liệu hoàn hảo, residuals = 0")


def test_ols_multiple_regression_known_data():
    """Test 2: Multiple Regression (y = 10 + 2x1 - 5x2) với n=4 để df > 0."""
    X = np.array([
        [1, 0],
        [2, 1],
        [3, 0],
        [4, 1]
    ], dtype=float)
    # Tính tay y = 10 + 2*x1 - 5*x2:
    # Dòng 1: 10 + 2(1) - 0 = 12
    # Dòng 2: 10 + 2(2) - 5 = 9
    # Dòng 3: 10 + 2(3) - 0 = 16
    # Dòng 4: 10 + 2(4) - 5 = 13
    y = np.array([12, 9, 16, 13], dtype=float)

    model = OLSRegressor(fit_intercept=True).fit(X, y)
    _assert_close(model.beta_hat, [10.0, 2.0, -5.0], msg="Beta multiple regression sai")


def test_regressor_properties():
    """residuals và fitted_values phải nhất quán với y và y_hat (tĩnh)."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.1, 3.9, 6.1, 7.9])
    model = OLSRegressor().fit(X, y)
    y_hat = model.predict(X)

    _assert_close(model.fitted_values, y_hat, msg="fitted_values == predict(X)")
    _assert_close(model.residuals, y - y_hat, msg="residuals == y - ŷ")
    assert model.residuals.shape == (4,), "FAIL: shape residuals"


def test_predict_before_fit_raises():
    """predict() phải raise RuntimeError nếu chưa gọi fit()."""
    model = OLSRegressor()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_fit_intercept_false():
    """fit_intercept=False: không thêm cột 1, beta_hat có p phần tử."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = 2.0 * X.ravel()
    model = OLSRegressor(fit_intercept=False).fit(X, y)
    assert model.beta_hat.shape == (1,), "FAIL: shape kỳ vọng (1,)"
    _assert_close(model.beta_hat[0], 2.0, msg="slope=2")


# ============================================================
# Các hàm tính Sum of Squares (RSS, TSS, ESS)
# ============================================================

def test_rss_known_values():
    y = np.array([1.0, 2.0, 3.0])
    _assert_close(compute_rss(y, y), 0.0, msg="RSS perfect fit")
    _assert_close(compute_rss(y, np.array([2.0, 2.0, 2.0])), 2.0, msg="RSS hand calc")


def test_tss_known_values():
    _assert_close(compute_tss(np.array([1.0, 2.0, 3.0])), 2.0, msg="TSS variance")
    _assert_close(compute_tss(np.array([5.0, 5.0, 5.0])), 0.0, msg="TSS constant")


def test_ess_known_values():
    y = np.array([1.0, 2.0, 3.0])
    _assert_close(compute_ess(y, y), 2.0, msg="ESS perfect fit")
    _assert_close(compute_ess(y, np.array([2.0, 2.0, 2.0])), 0.0, msg="ESS worst model")


# ============================================================
# Các hàm R2 và Metrics
# ============================================================

def test_r2_known_values():
    """Test R² = 1 - RSS/TSS với các kích thước mẫu khác nhau."""

    # Case 1: n = 2
    y_small = np.array([1.0, 3.0])         # mean = 2, TSS = (-1)² + 1² = 2.0
    y_hat_small = np.array([1.5, 2.5])     # RSS = 0.5² + (-0.5)² = 0.5
    _assert_close(compute_r2(y_small, y_hat_small), 0.75, msg="R2 n=2")
    _assert_close(compute_r2(y_small, y_small), 1.0, msg="R2 perfect n=2")

    # Case 2: n = 4
    y_large = np.array([2.0, 4.0, 6.0, 8.0])      # mean = 5.0
    y_hat_large = np.array([2.5, 3.5, 6.5, 7.5])

    # Tính tay:
    # TSS = (-3)² + (-1)² + 1² + 3² = 9 + 1 + 1 + 9 = 20.0
    # RSS = (-0.5)² + (0.5)² + (-0.5)² + (0.5)² = 0.25 * 4 = 1.0
    # R²  = 1 - (RSS / TSS) = 1 - (1.0 / 20.0) = 1 - 0.05 = 0.95

    _assert_close(compute_r2(y_large, y_hat_large), 0.95, msg="R2 n=4")
    _assert_close(compute_r2(y_large, y_large), 1.0, msg="R2 perfect n=4")


def test_r2_adj_and_metrics_hand_calc():
    """Test gộp R²_adj và model_metrics dựa trên tính toán thủ công"""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat = np.array([1.5, 1.5, 3.5, 3.5])

    _assert_close(compute_r2(y, y_hat), 0.8, msg="R2")
    _assert_close(compute_r2_adj(y, y_hat, p=1), 0.7, msg="R2_adj")

    m = model_metrics(y, y_hat, p=1, verbose=False)
    _assert_close(m["rss"], 1.0, msg="Metric RSS")
    _assert_close(m["tss"], 5.0, msg="Metric TSS")
    _assert_close(m["ess"], 4.0, msg="Metric ESS")
    _assert_close(m["f_stat"], 8.0, msg="Metric F-Stat")


def test_metrics_second_case():
    """Test Case 2: Kiểm tra model_metrics với định lý Pytago OLS."""
    y = np.array([2.0, 5.0, 5.0])
    y_hat = np.array([2.5, 4.0, 5.5])

    _assert_close(compute_r2(y, y_hat), 0.75, msg="R2 case 2")
    _assert_close(compute_r2_adj(y, y_hat, p=1), 0.5, msg="R2_adj case 2")

    m = model_metrics(y, y_hat, p=1, verbose=False)
    _assert_close(m["tss"], 6.0, msg="TSS case 2")
    _assert_close(m["ess"], 4.5, msg="ESS case 2")
    _assert_close(m["rss"], 1.5, msg="RSS case 2")
    _assert_close(m["ess"] + m["rss"], m["tss"], tol=1e-10, msg="TSS = ESS + RSS")


# ============================================================
# Edge Cases
# ============================================================

def test_ols_fit_raises_on_perfect_collinearity():
    """Đa cộng tuyến thực sự (rank < k) phải raise ValueError."""
    col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X_collinear = np.column_stack([col, col])  # hai cột giống hệt nhau
    y_dummy = col * 2
    with pytest.raises(ValueError, match="hạng|đa cộng tuyến"):
        ols_fit(X_collinear, y_dummy)


def test_compute_r2_raises_on_constant_y():
    """TSS = 0 (y hằng số) phải raise ValueError."""
    with pytest.raises(ValueError):
        compute_r2(np.array([4.0, 4.0]), np.array([1.0, 2.0]))


def test_ols_fit_raises_on_insufficient_degrees_of_freedom():
    """Thiếu bậc tự do (n <= k) phải raise ValueError."""
    with pytest.raises(ValueError, match="bậc tự do"):
        ols_fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0]))


def test_r2_adj_raises_insufficient_data():
    """compute_r2_adj() phải raise ValueError khi n <= p+1."""
    with pytest.raises(ValueError):
        compute_r2_adj(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 2.9]), p=2)


# ============================================================
# Kiểm chứng với sklearn
# ============================================================

def test_verify_sklearn_static():
    X = np.array([[1.5, 2.1], [3.0, 1.2], [4.5, 5.0], [2.2, 3.3]], dtype=float)
    y = np.array([5.1, 8.2, 12.5, 7.3], dtype=float)
    result = verify_with_sklearn(X, y)
    assert result["passed"], "FAIL: OLS tĩnh không khớp Sklearn"


# ============================================================
# Kiểm chứng Lý thuyết Thống kê (Theory Validation)
# ============================================================

def test_r2_adj_penalizes_extra_variables():
    """Kiểm chứng R²_adj giảm khi thêm biến nhiễu không có ý nghĩa."""
    # Dữ liệu gốc (n=5, p=1)
    X_base = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.1, 3.9, 6.1, 7.9, 10.1])

    model_base = OLSRegressor().fit(X_base, y)
    r2_adj_base = compute_r2_adj(y, model_base.predict(X_base), p=1)

    # Thêm cột nhiễu được thiết kế đặc biệt (vuông góc với phần dư của X_base)
    noise_col = np.array([[0.0], [1.0], [0.0], [-1.0], [0.0]])
    X_noise = np.hstack([X_base, noise_col])

    model_noise = OLSRegressor().fit(X_noise, y)
    r2_adj_noise = compute_r2_adj(y, model_noise.predict(X_noise), p=2)

    # Lúc này R² thông thường đứng im, nhưng do p tăng (từ 1 lên 2) nên R²_adj BẮT BUỘC giảm
    assert r2_adj_noise < r2_adj_base, f"FAIL: R²_adj không giảm! Base={r2_adj_base:.4f}, Noise={r2_adj_noise:.4f}"