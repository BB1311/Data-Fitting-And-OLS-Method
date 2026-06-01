"""
test_advanced_methods.py
========================
Unit tests cho advanced_methods.py — đã cập nhật để khớp với phiên bản mới nhất.

Nhóm test:
  1. KernelRidgeRegressor — Core & Math
  2. KernelRidgeRegressor — Kernel Functions
  3. KernelRidgeRegressor — CV Search & Comparison
  4. BayesianLinearRegressor — Core & Math
  5. BayesianLinearRegressor — Ridge Equivalence (lý thuyết quan trọng)
  6. BayesianLinearRegressor — Uncertainty Quantification
  7. Edge Cases & Exceptions
  8. So sánh với baseline (OLS/Ridge từ Part 1)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from part2.advanced_methods import (
    KernelRidgeRegressor,
    BayesianLinearRegressor,
    _rbf_kernel,
    _poly_kernel,
    compare_krr_vs_ols,
    compare_bayesian_vs_ols,
)
from part1.ridge_lasso import ridge_fit
from part1.ols_implementation import OLSRegressor, compute_r2


# ──────────────────────────────────────────────────────────────────────
# Tiện ích
# ──────────────────────────────────────────────────────────────────────
def _assert_close(a, b, tol=1e-6, msg=""):
    diff = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    assert diff < tol, f"FAIL [{msg}]: max_diff={diff:.4e} (tol={tol:.0e})"


# ======================================================================
# 1. KernelRidgeRegressor — Core & Math
# ======================================================================

class TestKRRCore:
    X_small = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_small = np.array([3.0, 7.0, 11.0])
    
    def test_fit_returns_self(self):
        """fit() phải trả về self (fluent interface)."""
        m = KernelRidgeRegressor(lam=1.0)
        ret = m.fit(self.X_small, self.y_small)
        assert ret is m

    def test_predict_shape(self):
        """predict() phải trả về (m,) với m là số dòng của X_new."""
        m = KernelRidgeRegressor(lam=0.1).fit(self.X_small, self.y_small)
        y_pred = m.predict(self.X_small)
        assert y_pred.shape == (3,)

    def test_perfect_fit_noiseless(self):
        """Với λ≈0 và dữ liệu không nhiễu, KRR phải fit gần hoàn hảo trên train."""
        m = KernelRidgeRegressor(kernel='rbf', lam=1e-8, length_scale=1.0).fit(self.X_small, self.y_small)
        y_pred = m.predict(self.X_small)
        rmse = float(np.sqrt(np.mean((self.y_small - y_pred) ** 2)))
        assert rmse < 1e-4, f"RMSE={rmse:.4e} quá lớn với dữ liệu hoàn hảo"

    def test_alpha_vector_shape(self):
        """Sau fit(), alpha_ phải có đúng n phần tử."""
        m = KernelRidgeRegressor(lam=1.0).fit(self.X_small, self.y_small)
        assert m.alpha_.shape == (3,)

    def test_large_lam_shrinks_predictions(self):
        """Lambda lớn → dự đoán bị co về 0 (shrinkage)."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=float)
        y = np.array([1.0, 2.0, 3.0], dtype=float)
        m_small = KernelRidgeRegressor(lam=1e-4).fit(X, y)
        m_large = KernelRidgeRegressor(lam=1e6).fit(X, y)
        assert np.std(m_large.predict(X)) < np.std(m_small.predict(X))

    def test_evaluate_returns_r2(self):
        """evaluate() phải trả về R² hợp lệ trong khoảng (-∞, 1]."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([3.0, 7.0, 11.0])
        m = KernelRidgeRegressor(lam=1.0).fit(X, y)
        result = m.evaluate(X, y)
        assert 'r2' in result
        r2 = result['r2']
        assert r2 <= 1.0

    def test_krr_beats_ridge_on_nonlinear(self):
        """
        Trên dữ liệu phi tuyến (sin), KRR (RBF) phải cho R² cao hơn Ridge tuyến tính.
        Đây là ưu điểm cốt lõi của Kernel Method.
        """
        X = np.linspace(-3, 3, 50).reshape(-1, 1)
        y = np.sin(X.ravel())
        
        m_krr = KernelRidgeRegressor(kernel='rbf', lam=0.01, length_scale=1.0).fit(X, y)
        r2_krr = m_krr.evaluate(X, y)['r2']

        res_ridge = ridge_fit(X, y, lam=0.01)
        r2_ridge = compute_r2(y, res_ridge["y_hat"])

        assert r2_krr > r2_ridge, (
            f"KRR (R²={r2_krr:.4f}) phải tốt hơn Ridge (R²={r2_ridge:.4f}) "
            f"trên dữ liệu phi tuyến"
        )


# ======================================================================
# 2. KernelRidgeRegressor — Kernel Functions
# ======================================================================

class TestKernelFunctions:
    X1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    X2 = np.array([[1.0, 1.0], [-1.0, 2.0]])
    
    def test_rbf_kernel_symmetry(self):
        """RBF kernel phải đối xứng: K(X1, X2) = K(X2, X1)^T."""
        K = _rbf_kernel(self.X1, self.X2, length_scale=1.0)
        K_T = _rbf_kernel(self.X2, self.X1, length_scale=1.0)
        _assert_close(K, K_T.T, msg="RBF không đối xứng")

    def test_rbf_kernel_diagonal_is_one(self):
        """k(x, x) = exp(0) = 1 cho tất cả điểm."""
        K = _rbf_kernel(self.X1, self.X1, length_scale=1.0)
        _assert_close(np.diag(K), np.ones(2), msg="Diagonal RBF phải = 1")

    def test_rbf_kernel_positive_semidefinite(self):
        """Gram matrix K = K(X, X) với RBF phải là PSD: eigenvalues >= 0."""
        X = np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 0.0]])
        K = _rbf_kernel(X, X, length_scale=1.0)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8), \
            f"Gram matrix RBF không phải PSD: min eigenvalue={eigenvalues.min():.4e}"

    def test_rbf_length_scale_effect(self):
        """Length_scale lớn → kernel decay chậm → điểm xa vẫn còn tương đồng."""
        x1 = np.array([[0.0, 0.0]])
        x2 = np.array([[5.0, 5.0]])
        k_small = _rbf_kernel(x1, x2, length_scale=0.1)[0, 0]
        k_large = _rbf_kernel(x1, x2, length_scale=100.0)[0, 0]
        assert k_large > k_small, \
            "Length_scale lớn → kernel giữa điểm xa phải lớn hơn"

    def test_poly_kernel_degree1_is_linear(self):
        """Poly kernel với degree=1, coef0=0 phải bằng xᵀx' (dot product)."""
        X1 = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
        X2 = np.array([[7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0]])
        K_poly = _poly_kernel(X1, X2, degree=1, coef0=0.0)
        K_dot = X1 @ X2.T
        _assert_close(K_poly, K_dot, msg="Poly(d=1, c=0) phải bằng dot product")

    def test_both_kernel_types_work(self):
        """Cả 'rbf' và 'poly' đều phải có thể fit và predict."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([1.0, 2.0, 3.0])
        for kernel in ['rbf', 'poly']:
            m = KernelRidgeRegressor(kernel=kernel, lam=1.0).fit(X, y)
            y_pred = m.predict(X)
            assert y_pred.shape == (3,), f"Kernel '{kernel}' predict sai shape"


# ======================================================================
# 3. KernelRidgeRegressor — CV Search
# ======================================================================

# ======================================================================
# 3. KernelRidgeRegressor — CV Search & Comparison
# ======================================================================

class TestKRRCV:
    X = np.array([
        [1.0, 2.0],
        [3.0, 5.0],
        [5.0, 3.0],
        [7.0, 8.0],
        [9.0, 1.0],
        [11.0, 4.0],
        [13.0, 9.0],
        [15.0, 6.0],
        [17.0, 7.0],
        [19.0, 10.0]
    ])
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 5.0   # y = 5 + 2*x1 + 3*x2
    
    def test_cv_search_returns_correct_keys(self):
        """cv_search() phải trả về các key cần thiết."""

        result = KernelRidgeRegressor.cv_search(
            self.X, self.y, lam_grid=[0.1, 1.0], ls_grid=[0.5, 1.0], k=5, verbose=False
        )
        for key in ('best_lam', 'best_length_scale', 'best_cv_mse', 'results_grid'):
            assert key in result, f"Thiếu key '{key}'"

    def test_cv_search_best_lam_in_grid(self):
        """best_lam phải nằm trong lam_grid."""
        lam_grid = [0.01, 0.1, 1.0, 10.0]
        result = KernelRidgeRegressor.cv_search(
            self.X, self.y, lam_grid=lam_grid, ls_grid=[1.0], k=5, verbose=False
        )
        assert result['best_lam'] in lam_grid

    def test_compare_krr_vs_ols_keys(self):
        """compare_krr_vs_ols() phải trả về đúng keys và winner hợp lệ."""
        result = compare_krr_vs_ols(self.X, self.y, lam=1.0, k=5, verbose=False)
        for key in ('ols_cv_mse', 'krr_cv_mse', 'winner'):
            assert key in result, f"Thiếu key '{key}'"
        assert result['winner'] in ('ols', 'krr'), \
            f"winner phải là 'ols' hoặc 'krr', nhận được '{result['winner']}'"

    def test_compare_krr_vs_ols_krr_wins_on_nonlinear(self):
        """
        Trên dữ liệu phi tuyến (sin), KRR phải thắng OLS rõ rệt.
        Đây là lý do tồn tại của Kernel Regression.
        """
        X = np.linspace(-3, 3, 30).reshape(-1, 1)
        y = np.sin(X.ravel())
        result = compare_krr_vs_ols(X, y, lam=0.01, length_scale=1.0, k=5, verbose=False)
        assert result['winner'] == 'krr', (
            f"KRR phải thắng OLS trên dữ liệu phi tuyến. "
            f"KRR MSE={result['krr_cv_mse']:.4f}, OLS MSE={result['ols_cv_mse']:.4f}"
        )

# ======================================================================
# 4. BayesianLinearRegressor — Core & Math
# ======================================================================

class TestBayesianCore:
    X = np.array([
        [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0], [6.0, 1.0],
        [7.0, 0.0], [8.0, 1.0], [9.0, 0.0], [10.0, 1.0], [11.0, 0.0], [12.0, 1.0]
    ])
    # y = 10 + 2*x1 - 5*x2, không có nhiễu
    y = np.array([12, 9, 16, 13, 20, 17, 24, 21, 28, 25, 32, 29], dtype=float)
    
    def test_fit_returns_self(self):
        """fit() phải trả về self."""
        m = BayesianLinearRegressor(alpha=1.0)
        assert m.fit(self.X, self.y) is m

    def test_posterior_mean_shape(self):
        """m_n_ phải có shape (k,) với k = p+1 (khi fit_intercept=True)."""
        m = BayesianLinearRegressor(alpha=1.0).fit(self.X, self.y)
        assert m.m_n_.shape == (3,), f"Shape m_n_ = {m.m_n_.shape}, kỳ vọng (3,)"

    def test_posterior_covariance_shape(self):
        """S_n_ phải là ma trận (k, k)."""
        m = BayesianLinearRegressor(alpha=1.0).fit(self.X, self.y)
        assert m.S_n_.shape == (3, 3), f"Shape S_n_ = {m.S_n_.shape}"

    def test_posterior_covariance_symmetric(self):
        """Covariance matrix phải đối xứng: Sₙ = Sₙᵀ."""
        m = BayesianLinearRegressor(alpha=1.0).fit(self.X, self.y)
        _assert_close(m.S_n_, m.S_n_.T, msg="Sₙ phải đối xứng")

    def test_posterior_covariance_positive_definite(self):
        """Covariance matrix phải positive definite (eigenvalues > 0)."""
        m = BayesianLinearRegressor(alpha=1.0).fit(self.X, self.y)
        eigvals = np.linalg.eigvalsh(m.S_n_)
        assert np.all(eigvals > -1e-8), f"Sₙ không PSD: min eigval={eigvals.min():.4e}"

    def test_predict_shape_with_std(self):
        """predict(return_std=True) phải trả về tuple (mean, std) có đúng shape."""
        m = BayesianLinearRegressor(alpha=1.0).fit(self.X, self.y)
        X_new = np.array([[2.0, 3.0], [4.0, 5.0]])
        y_mean, y_std = m.predict(X_new, return_std=True)
        assert y_mean.shape == (2,)
        assert y_std.shape == (2,)

    def test_predict_std_positive(self):
        """Predictive std phải >= 0 ở mọi điểm."""
        m = BayesianLinearRegressor(alpha=1.0).fit(self.X, self.y)
        _, y_std = m.predict(self.X, return_std=True)
        assert np.all(y_std >= 0.0), "std dự đoán phải >= 0"

    def test_evaluate_returns_r2(self):
        """evaluate() trả về R² hợp lệ."""
        m = BayesianLinearRegressor(alpha=1.0).fit(self.X, self.y)
        result = m.evaluate(self.X, self.y)
        assert result['r2'] <= 1.0 + 1e-8

    def test_large_alpha_shrinks_more(self):
        """Alpha lớn = prior mạnh → posterior mean nhỏ hơn (shrinkage mạnh)."""
        X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]])
        y = 10.0 + 2.0 * X[:, 0] - 5.0 * X[:, 1]
        m_small_alpha = BayesianLinearRegressor(alpha=0.01).fit(X, y)
        m_large_alpha = BayesianLinearRegressor(alpha=1000.0).fit(X, y)
        norm_small = float(np.linalg.norm(m_small_alpha.m_n_[1:]))
        norm_large = float(np.linalg.norm(m_large_alpha.m_n_[1:]))
        assert norm_large < norm_small, (
            f"Alpha lớn phải shrink hệ số. "
            f"norm(small_alpha)={norm_small:.4f}, norm(large_alpha)={norm_large:.4f}"
        )


# ======================================================================
# 5. BayesianLinearRegressor — Kết nối với Ridge (LÝ THUYẾT QUAN TRỌNG)
# ======================================================================

class TestBayesianRidgeEquivalence:
    """
    Kiểm chứng mệnh đề toán học:
        Bayesian posterior mean = Ridge estimate  khi prior = N(0, (σ²/α)I)
    
    Đây là mối liên hệ nền tảng giữa Bayesian inference và regularization.
    """
    X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]])
    y = 10.0 + 2.0 * X[:, 0] - 5.0 * X[:, 1]
    
    def test_ridge_equivalence_static(self):
        """Test tĩnh: posterior mean phải khớp β̂_Ridge với tol=1e-5."""
        alpha = 2.0

        # Bayesian với sigma2 đã biết
        sigma2_true = 0.5**2
        m = BayesianLinearRegressor(alpha=alpha, sigma2=sigma2_true).fit(self.X, self.y)

        # Đối chiếu qua ridge_equivalence_check
        result = m.ridge_equivalence_check(self.X, self.y, verbose=False)
        assert result["max_diff"] < 1e-6, (
            f"Bayesian và Ridge chênh nhau quá nhiều: {result['max_diff']:.4f}"
        )

    def test_ridge_equivalence_exact_known_sigma2(self):
        """
        Khi dữ liệu hoàn hảo (noise=0) và sigma2 biết chính xác,
        posterior mean ≈ Ridge với lam rất nhỏ (≈ OLS).
        """
        alpha = 1e-6
        sigma2 = 1.0

        m = BayesianLinearRegressor(alpha=alpha, sigma2=sigma2).fit(self.X, self.y)
        ols_model = OLSRegressor(fit_intercept=True).fit(self.X, self.y)

        max_diff = float(np.max(np.abs(m.m_n_ - ols_model.beta_hat)))
        assert max_diff < 0.1, (
            f"Bayesian với alpha≈0 phải xấp xỉ OLS. Diff={max_diff:.4f}"
        )

    def test_posterior_std_shrinks_with_more_data(self):
        """
        Bayesian property: posterior std phải giảm khi có nhiều dữ liệu hơn.
        Dữ liệu nhiều → uncertainty về β giảm.
        """
        X_small = self.X[:3]
        y_small = self.y[:3]

        m_small = BayesianLinearRegressor(alpha=1.0).fit(X_small, y_small)
        m_large = BayesianLinearRegressor(alpha=1.0).fit(self.X, self.y)

        std_small = float(np.mean(np.sqrt(np.diag(m_small.S_n_))))
        std_large = float(np.mean(np.sqrt(np.diag(m_large.S_n_))))

        assert std_large < std_small, (
            f"Posterior std phải giảm với n lớn hơn. "
            f"n=10: std={std_small:.4f}, n=200: std={std_large:.4f}"
        )


# ======================================================================
# 6. BayesianLinearRegressor — Uncertainty Quantification
# ======================================================================

class TestBayesianUncertainty:

    def test_credible_interval_coverage(self):
        """
        95% credible interval phải bao phủ giá trị thực (không nhiễu) với tỉ lệ ≥ 95%.
        """
        # Tập huấn luyện – 8 điểm cố định
        X_train = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0]
        ])
        # Hệ số thực của mô hình (không có nhiễu)
        intercept = 1.0
        slope1 = 2.0
        slope2 = -1.5
        y_train = intercept + slope1 * X_train[:, 0] + slope2 * X_train[:, 1]

        # Tập kiểm tra – 10 điểm cố định
        X_test = np.array([
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
            [8.0, 9.0],
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, 15.0],
            [16.0, 17.0],
            [18.0, 19.0],
            [20.0, 21.0]
        ])
        y_test_true = intercept + slope1 * X_test[:, 0] + slope2 * X_test[:, 1]

        # Bayesian với phương sai nhiễu rất nhỏ (dữ liệu gần như không nhiễu)
        sigma2_known = 1e-4
        alpha_prior = 0.1
        model = BayesianLinearRegressor(alpha=alpha_prior, sigma2=sigma2_known)
        model.fit(X_train, y_train)

        lower, upper = model.credible_interval(X_test, confidence=0.95)

        # Tỉ lệ điểm kiểm tra có giá trị thực nằm trong khoảng tin cậy
        coverage = np.mean((y_test_true >= lower) & (y_test_true <= upper))
        assert coverage >= 0.95, f"Coverage = {coverage:.2%}, kỳ vọng >= 95%"

    def test_extrapolation_has_higher_uncertainty(self):
        """
        Điểm ngoại suy (xa vùng train) phải có predictive std lớn hơn
        so với điểm trong vùng train.
        """
        X_train = np.array([[-1.0], [0.0], [1.0]])
        y_train = np.array([-1.0, 0.0, 1.0])

        m = BayesianLinearRegressor(alpha=1.0).fit(X_train, y_train)

        X_interpolate = np.array([[0.0]])     # nằm trong vùng train
        X_extrapolate = np.array([[100.0]])   # ngoại suy xa

        _, std_interp = m.predict(X_interpolate, return_std=True)
        _, std_extrap = m.predict(X_extrapolate, return_std=True)

        assert std_extrap[0] > std_interp[0], (
            f"Điểm ngoại suy phải có uncertainty cao hơn. "
            f"Interp std={std_interp[0]:.4f}, Extrap std={std_extrap[0]:.4f}"
        )

    def test_credible_interval_width(self):
        """Credible interval phải có độ rộng > 0 tại mọi điểm."""
        X = np.array([[1.0, 0.0], [2.0, 1.0]])
        y = np.array([3.0, 5.0])
        m = BayesianLinearRegressor(alpha=1.0).fit(X, y)
        lower, upper = m.credible_interval(X, confidence=0.95)
        assert np.all(upper > lower), "CI phải có upper > lower tại mọi điểm"

    def test_coef_summary_returns_correct_keys(self):
        """coef_summary() phải trả về dict đúng keys."""
        X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
        y = np.array([3.0, 7.0, 11.0])
        m = BayesianLinearRegressor(alpha=1.0).fit(X, y)
        result = m.coef_summary(verbose=False)
        for key in ("feature_names", "posterior_mean", "posterior_std"):
            assert key in result
        assert len(result["posterior_mean"]) == 3   # intercept + 2 features
        assert len(result["posterior_std"]) == 3


# ======================================================================
# 7. Edge Cases & Exceptions
# ======================================================================

class TestEdgeCases:

    def test_krr_predict_before_fit_raises(self):
        """predict() trước fit() phải raise RuntimeError."""
        m = KernelRidgeRegressor()
        with pytest.raises(RuntimeError):
            m.predict(np.array([[1.0, 2.0]]))

    def test_krr_negative_lambda_raises(self):
        """lambda < 0 phải raise ValueError."""
        with pytest.raises(ValueError, match="lam"):
            KernelRidgeRegressor(lam=-1.0)

    def test_krr_invalid_kernel_raises(self):
        """Kernel không hợp lệ phải raise ValueError."""
        with pytest.raises(ValueError, match="kernel"):
            KernelRidgeRegressor(kernel='sigmoid')

    def test_krr_negative_length_scale_raises(self):
        """length_scale <= 0 phải raise ValueError khi gọi kernel."""
        with pytest.raises(ValueError, match="length_scale"):
            _rbf_kernel(np.ones((2,1)), np.ones((2,1)), length_scale=-1.0)

    def test_bayes_predict_before_fit_raises(self):
        """predict() trước fit() phải raise RuntimeError."""
        m = BayesianLinearRegressor()
        with pytest.raises(RuntimeError):
            m.predict(np.array([[1.0, 2.0]]))

    def test_bayes_negative_alpha_raises(self):
        """alpha <= 0 phải raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            BayesianLinearRegressor(alpha=0.0)

    def test_krr_1d_input(self):
        """X 1D phải được reshape tự động."""
        X_1d = np.linspace(0, 5, 20)
        y = np.sin(X_1d)
        m = KernelRidgeRegressor(lam=0.1).fit(X_1d, y)
        assert m.X_train_.shape == (20, 1)

    def test_bayes_1d_input(self):
        """X 1D phải được reshape tự động."""
        X_1d = np.array([1.0, 2.0, 3.0])
        y = 5.0 + 3.0 * X_1d
        m = BayesianLinearRegressor(alpha=1.0).fit(X_1d, y)
        assert m.m_n_.shape == (2,)   # intercept + 1 feature


# ======================================================================
# 8. So sánh với baseline (OLS/Ridge từ Part 1)
# ======================================================================

class TestComparisonWithBaseline:
    X = np.array([
        [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0], [6.0, 1.0],
        [7.0, 0.0], [8.0, 1.0], [9.0, 0.0], [10.0, 1.0], [11.0, 0.0], [12.0, 1.0]
    ])
    # Dữ liệu tuyến tính hoàn hảo, không nhiễu
    y = 10.0 + 2.0 * X[:, 0] - 5.0 * X[:, 1]
    
    def test_compare_bayesian_vs_ols_keys(self):
        """compare_bayesian_vs_ols() trả về đúng keys."""
        result = compare_bayesian_vs_ols(self.X, self.y, k_cv=3, verbose=False)
        for key in ("ols_cv_mse", "bayes_cv_mse", "winner"):
            assert key in result
        assert result["winner"] in ("ols", "bayesian")

    def test_krr_linear_kernel_approx_ridge(self):
        """
        Khi dùng poly kernel bậc 1 không có bias (≈ linear), coef0=1 (tương đương có intercept),
        KRR phải cho kết quả gần với Ridge thông thường.
        """
        
        lam = 1.0
        m_krr = KernelRidgeRegressor(kernel='poly', lam=lam, degree=1, coef0=1.0).fit(self.X, self.y)
        res_ridge = ridge_fit(self.X, self.y, lam=lam)

        r2_krr = m_krr.evaluate(self.X, self.y)['r2']
        r2_ridge = compute_r2(self.y, res_ridge["y_hat"])

        # Cả hai phải có R² cao trên dữ liệu tuyến tính
        assert r2_krr > 0.8, f"KRR linear R² = {r2_krr:.4f} quá thấp"
        assert r2_ridge > 0.8, f"Ridge R² = {r2_ridge:.4f} quá thấp"

    def test_bayesian_r2_on_good_data(self):
        """Bayesian LR phải cho R² cao trên dữ liệu tuyến tính rõ ràng."""
        m = BayesianLinearRegressor(alpha=0.1).fit(self.X, self.y)
        r2 = m.evaluate(self.X, self.y)['r2']
        assert r2 > 0.9, f"Bayesian R² = {r2:.4f} quá thấp trên dữ liệu rõ ràng"

    def test_krr_rbf_r2_on_nonlinear_data(self):
        """KRR RBF phải có R² cao trên dữ liệu sin()."""
        X = np.linspace(-3, 3, 20).reshape(-1, 1)
        y = np.sin(X.ravel())
        m = KernelRidgeRegressor(kernel='rbf', lam=0.01, length_scale=1.0).fit(X, y)
        r2 = m.evaluate(X, y)['r2']
        assert r2 > 0.9, f"KRR RBF R² = {r2:.4f} quá thấp trên sin(x)"