import numpy as np
from scipy import stats
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
    hat_matrix,
    coef_inference,
    vif,
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


# ============================================================
# Kiểm thử hàm hat_matrix
# ============================================================

def test_hat_matrix_properties():
    """Test 1: Kiểm tra các tính chất toán học cốt lõi của Hat Matrix (kích thước, đối xứng, trace)."""
    X = np.array([
        [1.0, 2.1],
        [3.0, 4.5],
        [5.0, 6.2],
        [7.0, 8.9]
    ])
    
    H, is_idemp = hat_matrix(X, add_intercept=True)
    
    # 1. Kích thước của H phải là n x n (4 x 4)
    assert H.shape == (4, 4), f"FAIL: Kích thước kỳ vọng (4, 4), nhưng nhận được {H.shape}"
    
    # 2. Hàm phải xác nhận H là ma trận lũy đẳng (H^2 = H)
    assert is_idemp is True, "FAIL: Hàm báo cáo ma trận không lũy đẳng."
    
    # 3. Tính đối xứng: H = H^T
    _assert_close(H, H.T, msg="H phải là ma trận đối xứng")
    
    # 4. Tính chất Trace: Vết của ma trận chiếu H phải bằng hạng (rank) của X_design
    # X có 2 cột + 1 cột intercept = 3. Vậy Trace(H) = 3
    _assert_close(np.trace(H), 3.0, msg="Trace(H) phải bằng p + 1")


def test_hat_matrix_no_intercept():
    """Test 2: Kiểm tra khi tham số add_intercept = False."""
    # X có n=3, p=1
    X = np.array([[1.0], [2.0], [3.0]])
    
    H, is_idemp = hat_matrix(X, add_intercept=False)
    
    assert H.shape == (3, 3), "FAIL: Kích thước phải là (3, 3)"
    assert is_idemp is True, "FAIL: Ma trận vẫn phải lũy đẳng"
    
    # Vì không có intercept, số tham số mô hình k = p = 1. Trace(H) = 1
    _assert_close(np.trace(H), 1.0, msg="Trace(H) phải bằng 1 khi không có intercept")


def test_hat_matrix_raises_on_singular():
    """Test 3: Xử lý ngoại lệ khi ma trận (X^T X) không khả nghịch."""
    # Tạo ma trận có 2 cột giống hệt nhau (đa cộng tuyến hoàn hảo)
    col = np.array([1.0, 2.0, 3.0, 4.0])
    X_collinear = np.column_stack([col, col])
    
    # Phải bắt đúng ValueError với thông điệp liên quan đến ma trận suy biến/đa cộng tuyến
    with pytest.raises(ValueError, match="khả nghịch|suy biến|đa cộng tuyến"):
        hat_matrix(X_collinear, add_intercept=True)


# ============================================================
# Kiểm thử hàm coef_inference (Suy diễn thống kê)
# ============================================================

def test_coef_inference_correctness():
    """Test 1: Kiểm tra tính toán SE, t-stat, p-value với kết quả tính tay."""
    # Xây dựng bộ dữ liệu nhỏ có n=3, p=1
    # Mô hình: y = beta_0 + beta_1 * X + e
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 5.0])
    
    # Nghiệm được giải chính xác bằng toán học phân số:
    beta_hat_expected = np.array([2/3, 1.5])   # intercept = 0.666..., slope = 1.5
    sigma2_expected = 1/6                      # Phương sai nhiễu RSS/df = (1/6) / 1
    
    # Chạy hàm
    res = coef_inference(X, y, beta_hat_expected, sigma2_expected, verbose=False)
    
    # 1. Kiểm tra Standard Errors (SE)
    # Công thức: SE = sqrt(sigma2 * diag((X^T X)^-1))
    # (X^T X)^-1 của bài này là ma trận (1/6) * [[14, -6], [-6, 3]]
    se_expected = [np.sqrt((1/6) * (14/6)), np.sqrt((1/6) * (3/6))]
    _assert_close(res["se"], se_expected, msg="Sai số chuẩn (SE) tính sai")
    
    # 2. Kiểm tra t-statistics
    t_expected = [beta_hat_expected[0] / se_expected[0], beta_hat_expected[1] / se_expected[1]]
    _assert_close(res["t_stats"], t_expected, msg="t-statistics tính sai")
    
    # 3. Kiểm tra p-values và Khoảng tin cậy (với df = n - k = 3 - 2 = 1)
    p_expected = 2 * (1.0 - stats.t.cdf(np.abs(t_expected), df=1))
    _assert_close(res["p_values"], p_expected, msg="p-values tính sai")
    
    t_crit = stats.t.ppf(0.975, df=1)
    ci_lower_expected = beta_hat_expected - t_crit * np.array(se_expected)
    _assert_close(res["ci_lower"], ci_lower_expected, msg="Cận dưới CI 95% sai")


def test_coef_inference_exceptions():
    """Test 2: Kiểm tra exceptions của hàm."""
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    sigma2 = 1.0
    
    # Lỗi 1: Truyền vector beta_hat bị lệch kích thước so với số cột của X
    # X có 1 cột. Chèn intercept thành 2. Nhưng beta_hat lại có 3 phần tử.
    with pytest.raises(ValueError, match="kích thước|khớp"):
        coef_inference(X, y, np.array([1.0, 2.0, 3.0]), sigma2, verbose=False)
        
    # Lỗi 2: Không đủ bậc tự do (n=2, k=2 -> df=0)
    # X (1 cột) + intercept = 2 tham số. Mẫu có 2 điểm dữ liệu -> df = 0
    with pytest.raises(ValueError, match="bậc tự do"):
        coef_inference(X, y, np.array([1.0, 2.0]), sigma2, verbose=False)
        
    # Lỗi 3: Ma trận suy biến (Đa cộng tuyến hoàn hảo)
    # n=4, có 2 cột giống hệt nhau
    X_collinear = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    y_collinear = np.array([1, 2, 3, 4])
    beta_dummy = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="suy biến"):
        coef_inference(X_collinear, y_collinear, beta_dummy, sigma2, verbose=False)


def test_coef_inference_intercept_label():
    """
    Kiểm tra nhãn "Intercept" được nhận diện đúng trong cả hai trường hợp:
    a) Gọi với X gốc (hàm tự thêm cột 1)
    b) Gọi với X_design (đã có cột 1) — tương đương gọi qua summary()
 
    Cả hai phải tính SE như nhau (kết quả số học giống hệt).
    """
    X        = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y        = np.array([2.1, 3.9, 6.1, 7.9, 10.1])
    model    = OLSRegressor(fit_intercept=True).fit(X, y)
 
    # Gọi với X gốc
    res_raw  = coef_inference(X, y, model.beta_hat, model.sigma2_hat, verbose=False)
    # Gọi với X_design (giống summary())
    res_des  = coef_inference(model._X_design, y, model.beta_hat, model.sigma2_hat, verbose=False)
 
    # Kết quả số học phải giống hệt nhau
    _assert_close(res_raw["se"],       res_des["se"],       msg="SE nhất quán")
    _assert_close(res_raw["t_stats"],  res_des["t_stats"],  msg="t-stat nhất quán")
    _assert_close(res_raw["p_values"], res_des["p_values"], msg="p-value nhất quán")


# ============================================================
# Kiểm thử hàm vif (Variance Inflation Factor)
# ============================================================

def test_vif_orthogonal_data():
    """Test 1: Dữ liệu trực giao.
    Khi các cột hoàn toàn độc lập với nhau, R^2 của hồi quy chéo sẽ bằng 0.
    Dẫn đến VIF = 1 / (1 - 0) = 1.0 cho tất cả các biến.
    """
    # Xây dựng ma trận với 2 cột trực giao (tích vô hướng = 0)
    X_ortho = np.array([
        [ 1.0,  1.0],
        [ 1.0, -1.0],
        [-1.0,  1.0],
        [-1.0, -1.0]
    ])
    
    vif_scores = vif(X_ortho, verbose=False)
    
    # Kích thước phải bằng số lượng đặc trưng (p=2)
    assert len(vif_scores) == 2, "FAIL: Vector VIF phải có kích thước bằng số cột của X."
    
    # Giá trị VIF phải bằng 1.0
    _assert_close(vif_scores, [1.0, 1.0], msg="VIF của dữ liệu trực giao phải chính xác bằng 1.0")


def test_vif_perfect_collinearity():
    """Test 2: Đa cộng tuyến hoàn hảo.
    Nếu một biến là tổ hợp tuyến tính của các biến khác (R^2 = 1), 
    VIF phải trả về vô cực (np.inf) để tránh lỗi ZeroDivisionError.
    """
    x1 = np.array([1.0, 2.0, 3.0, 4.0])
    x2 = np.array([5.0, 1.0, 0.0, 2.0])
    
    # x3 phụ thuộc tuyến tính hoàn toàn vào x1 và x2
    x3 = 2.0 * x1 - 0.5 * x2 
    
    X_collinear = np.column_stack([x1, x2, x3])
    
    vif_scores = vif(X_collinear, verbose=False)
    
    # Tất cả các biến đều nằm trong mối quan hệ tuyến tính này nên VIF của cả 3 đều phải là vô cực
    assert np.isinf(vif_scores[0]), "FAIL: VIF của X1 phải là vô cực (inf)"
    assert np.isinf(vif_scores[1]), "FAIL: VIF của X2 phải là vô cực (inf)"
    assert np.isinf(vif_scores[2]), "FAIL: VIF của X3 phải là vô cực (inf)"


def test_vif_high_collinearity():
    """Test 3: Đa cộng tuyến cao.
    Kiểm tra xem hệ thống có nhận diện đúng các biến có VIF > 10 không.
    """
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2 = np.array([5.0, -1.0, 3.0, 8.0, 0.0])      # Độc lập hoàn toàn
    x3 = np.array([2.01, 4.02, 5.99, 8.05, 9.98])  # x3 gần như bằng 2*x1
    
    X = np.column_stack([x1, x2, x3])
    
    # Tính VIF
    vif_scores = vif(X, verbose=False)
    
    # x1 và x3 có quan hệ mật thiết -> VIF phải rất lớn (> 10)
    assert vif_scores[0] > 10.0, f"FAIL: VIF của x1 ({vif_scores[0]:.2f}) phải > 10 do đa cộng tuyến"
    assert vif_scores[2] > 10.0, f"FAIL: VIF của x3 ({vif_scores[2]:.2f}) phải > 10 do đa cộng tuyến"
    
    # x2 độc lập -> VIF phải rất nhỏ (xấp xỉ 1)
    assert vif_scores[1] < 2.0, f"FAIL: VIF của x2 ({vif_scores[1]:.2f}) phải nhỏ, vì nó độc lập"