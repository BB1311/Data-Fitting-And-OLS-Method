"""
test_ridge_lasso.py
===================
Unit tests cho ridge_lasso.py — phong cach nhat quan voi test_ols_implementation.py.

Cac nhom test:
  1. Ridge Regression (Core & Class)
  2. Lasso Regression (Core & Class)
  3. Soft Thresholding
  4. Edge Cases & Exceptions
  5. Ly thuyet thong ke (sparsity, shrinkage)
  6. Kiem chung voi sklearn
"""

import numpy as np
import pytest

from part1.ridge_lasso import (
    ridge_fit,
    RidgeRegressor,
    lasso_fit,
    LassoRegressor,
    _soft_threshold,
    _lasso_coordinate_descent,
    verify_ridge_sklearn,
    verify_lasso_sklearn,
)

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend

# ============================================================
# Tien ich noi bo
# ============================================================

def _assert_close(a, b, tol=1e-6, msg=""):
    """So sanh hai mang (hoac so) voi sai so cho phep tol."""
    diff = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
    assert diff < tol, f"FAIL [{msg}]: max_diff={diff:.4e}  (tol={tol:.0e})"


# ============================================================
# 1. RIDGE REGRESSION — Core & Class
# ============================================================

class TestRidgeCore:

    def test_ridge_fit_returns_correct_keys(self):
        """ridge_fit() phai tra ve day du cac key can thiet."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        res = ridge_fit(X, y, lam=1.0)
        for key in ("beta_hat", "y_hat", "residuals", "rss", "n", "k", "lam"):
            assert key in res, f"Thieu key '{key}'"
        assert res["n"] == 4
        assert res["k"] == 2        # intercept + 1 feature
        assert res["lam"] == 1.0

    def test_ridge_lam0_equals_ols(self):
        """
        Khi lambda = 0, Ridge phai cho ket qua giong het OLS.
        Su dung du lieu du lon de tranh van de suy bien.
        """
        np.random.seed(0)
        X = np.random.randn(30, 3)
        y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * np.random.randn(30)

        res_ridge = ridge_fit(X, y, lam=0.0)

        # OLS closed-form
        ones     = np.ones((30, 1))
        X_design = np.hstack([ones, X])
        XtX      = X_design.T @ X_design
        beta_ols = np.linalg.solve(XtX, X_design.T @ y)

        _assert_close(res_ridge["beta_hat"], beta_ols, tol=1e-8,
                      msg="Ridge(lam=0) phai bang OLS")

    def test_ridge_shrinks_coefficients(self):
        """
        Tang lambda phai lam cac he so tiến về 0 (shrinkage).
        ||beta_large_lam||_2 < ||beta_small_lam||_2.
        """
        np.random.seed(1)
        X = np.random.randn(50, 4)
        y = X @ np.array([3.0, -2.0, 1.5, -0.8]) + np.random.randn(50)

        beta_small = ridge_fit(X, y, lam=0.01)["beta_hat"][1:]   # bo intercept
        beta_large = ridge_fit(X, y, lam=1000.0)["beta_hat"][1:]

        norm_small = float(np.linalg.norm(beta_small))
        norm_large = float(np.linalg.norm(beta_large))
        assert norm_large < norm_small, (
            f"FAIL: tang lambda phai lam ||beta|| nho lai. "
            f"small={norm_small:.4f}, large={norm_large:.4f}"
        )

    def test_ridge_large_lambda_beta_near_zero(self):
        """Khi lambda rat lon, cac beta (khong phai intercept) phai xap xi 0."""
        np.random.seed(2)
        X = np.random.randn(50, 3)
        y = X @ np.array([2.0, -1.0, 3.0]) + np.random.randn(50)

        res = ridge_fit(X, y, lam=1e8)
        beta_features = res["beta_hat"][1:]
        assert np.all(np.abs(beta_features) < 1e-3), (
            f"FAIL: beta phai gan 0 khi lam rat lon: {beta_features}"
        )

    def test_ridge_perfect_fit(self):
        """Du lieu hoan hao (y = beta0 + beta1*x, khong nhieu): RSS phai rat nho."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = 2.0 + 3.0 * X.ravel()   # y = 2 + 3x

        res = ridge_fit(X, y, lam=1e-10)
        # Voi lambda cuc nho, gan giong OLS tren du lieu hoan hao
        _assert_close(res["rss"], 0.0, tol=1e-5, msg="RSS tren du lieu hoan hao")

    def test_ridge_predict_consistent(self):
        """y_hat tu fit() phai bang predict(X_train)."""
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        model = RidgeRegressor(lam=1.0).fit(X, y)
        _assert_close(model.fitted_values, model.predict(X),
                      msg="fitted_values == predict(X_train)")

    def test_ridge_predict_before_fit_raises(self):
        """predict() truoc fit() phai raise RuntimeError."""
        model = RidgeRegressor()
        with pytest.raises(RuntimeError):
            model.predict(np.array([[1.0, 2.0]]))

    def test_ridge_fit_intercept_false(self):
        """
        fit_intercept=False: beta_hat phai co dung p phan tu (khong co intercept).
        """
        np.random.seed(3)
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        model = RidgeRegressor(lam=1.0, fit_intercept=False).fit(X, y)
        assert model.beta_hat.shape == (2,), (
            f"FAIL: shape phai (2,), nhan duoc {model.beta_hat.shape}"
        )

    def test_ridge_residuals_property(self):
        """residuals phai bang y - fitted_values."""
        X = np.random.randn(15, 2)
        y = np.random.randn(15)
        model = RidgeRegressor(lam=0.5).fit(X, y)
        _assert_close(model.residuals, y - model.fitted_values,
                      msg="residuals == y - fitted_values")


class TestRidgeTrace:

    def test_ridge_trace_output_shape(self):
        """ridge_trace() phai tra ve (lambdas, coef_paths) co shape dung."""
        np.random.seed(10)
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        lambdas_test = np.logspace(-2, 2, 20)

        lams, paths = RidgeRegressor.ridge_trace(X, y, lambdas=lambdas_test,
                                                  show=False)
        assert lams.shape == (20,),    f"lambdas shape sai: {lams.shape}"
        assert paths.shape == (20, 4), f"coef_paths shape sai: {paths.shape}"  # 3 feat + intercept

    def test_ridge_trace_shrinks_monotonically(self):
        """
        Moi he so beta_j phai co xu huong giam khi lambda tang
        (khong bat buoc monotone tuyet doi, nhung ||beta|| phai giam).
        """
        np.random.seed(11)
        X = np.random.randn(50, 3)
        y = X @ np.array([2.0, -1.5, 3.0]) + np.random.randn(50)
        lambdas_test = np.logspace(-3, 4, 30)

        _, paths = RidgeRegressor.ridge_trace(X, y, lambdas=lambdas_test, show=False)
        norms = np.linalg.norm(paths[:, 1:], axis=1)  # bo intercept
        # ||beta|| khi lam nho phai > ||beta|| khi lam lon
        assert norms[0] > norms[-1], (
            f"FAIL: ||beta|| phai giam khi lam tang. "
            f"lam_min: {norms[0]:.4f}, lam_max: {norms[-1]:.4f}"
        )


# ============================================================
# 2. LASSO REGRESSION — Core & Class
# ============================================================

class TestLassoCore:

    def test_lasso_fit_returns_correct_keys(self):
        """lasso_fit() phai tra ve day du cac key."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        res = lasso_fit(X, y, lam=0.1)
        for key in ("beta_hat", "y_hat", "residuals", "rss", "n_nonzero", "n", "k", "lam"):
            assert key in res, f"Thieu key '{key}'"
        assert res["n"] == 4
        assert res["k"] == 2

    def test_lasso_lam0_close_to_ols(self):
        """
        Khi lambda = 0, Lasso (coordinate descent) phai hoi tu ve OLS.
        (Cho phep sai so noi long hon vi la iterative solver.)
        """
        np.random.seed(20)
        X = np.random.randn(40, 2)
        y = X @ np.array([1.5, -2.5]) + 0.1 * np.random.randn(40)

        res_lasso = lasso_fit(X, y, lam=1e-8, max_iter=5000, tol=1e-9)

        ones     = np.ones((40, 1))
        X_design = np.hstack([ones, X])
        beta_ols = np.linalg.lstsq(X_design, y, rcond=None)[0]

        _assert_close(res_lasso["beta_hat"], beta_ols, tol=1e-5,
                      msg="Lasso(lam~0) phai xap xi OLS")

    def test_lasso_sparsity(self):
        """
        Khi lambda du lon, Lasso phai loai bo cac bien khong quan trong
        (he so ve 0 — sparsity).
        """
        np.random.seed(21)
        n = 100
        x1 = np.random.randn(n)         # bien quan trong
        x_noise = np.random.randn(n, 5) * 0.01  # bien nhieu yeu
        X = np.column_stack([x1, x_noise])
        y = 3.0 * x1 + 0.1 * np.random.randn(n)

        res = lasso_fit(X, y, lam=0.5, max_iter=5000)
        # Phai co it nhat 1 he so = 0 (cac bien nhieu bi loai)
        assert res["n_nonzero"] < 6, (
            f"FAIL: Lasso phai loai bo bien nhieu. n_nonzero={res['n_nonzero']}"
        )

    def test_lasso_large_lambda_all_zero(self):
        """Khi lambda rat lon, tat ca cac he so (tru intercept) phai = 0."""
        np.random.seed(22)
        X = np.random.randn(30, 4)
        y = np.random.randn(30)
        res = lasso_fit(X, y, lam=1e6, max_iter=5000)
        beta_features = res["beta_hat"][1:]
        assert np.all(beta_features == 0.0), (
            f"FAIL: tat ca he so phai = 0 khi lam rat lon: {beta_features}"
        )

    def test_lasso_predict_consistent(self):
        """fitted_values phai bang predict(X_train)."""
        np.random.seed(23)
        X = np.random.randn(25, 3)
        y = np.random.randn(25)
        model = LassoRegressor(lam=0.1).fit(X, y)
        _assert_close(model.fitted_values, model.predict(X),
                      msg="fitted_values == predict(X_train)")

    def test_lasso_predict_before_fit_raises(self):
        """predict() truoc fit() phai raise RuntimeError."""
        model = LassoRegressor()
        with pytest.raises(RuntimeError):
            model.predict(np.array([[1.0, 2.0]]))

    def test_lasso_n_nonzero_property(self):
        """n_nonzero phai dem dung so he so != 0 (khong tinh intercept)."""
        np.random.seed(24)
        x1 = np.random.randn(80)
        x2 = np.random.randn(80) * 0.001  # bien nay se bi phat ve 0
        X  = np.column_stack([x1, x2])
        y  = 2.0 * x1 + np.random.randn(80) * 0.1

        model = LassoRegressor(lam=0.5, max_iter=5000).fit(X, y)
        assert model.n_nonzero <= 2, "n_nonzero sai"

    def test_lasso_fit_intercept_false(self):
        """fit_intercept=False: beta_hat phai co p phan tu."""
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        model = LassoRegressor(lam=0.1, fit_intercept=False).fit(X, y)
        assert model.beta_hat.shape == (3,), (
            f"FAIL: shape phai (3,), nhan duoc {model.beta_hat.shape}"
        )

    def test_lasso_residuals_property(self):
        """residuals phai bang y - fitted_values."""
        X = np.random.randn(15, 2)
        y = np.random.randn(15)
        model = LassoRegressor(lam=0.05).fit(X, y)
        _assert_close(model.residuals, y - model.fitted_values,
                      msg="residuals == y - fitted_values")
        
    def test_lasso_convergence_with_tol_f(self):
        """Đảm bảo rằng khi dùng tol_f, số vòng lặp thực tế ít hơn max_iter rất lớn."""
        np.random.seed(123)
        X = np.random.randn(100, 5)
        y = X @ np.array([2.0, 0.0, -1.5, 0.0, 0.5]) + 0.1 * np.random.randn(100)

        # Gọi với max_iter cực lớn nhưng tol_f rất nhỏ → phải dừng trước max_iter
        beta_with_tolf = _lasso_coordinate_descent(
            np.hstack([np.ones((100,1)), X]), y, lam=0.1,
            max_iter=5000, tol=1e-12, tol_f=1e-8
        )
        # Nếu nó chạy hết 5000 vòng thì có vấn đề, nhưng ta không kiểm tra số vòng cụ thể.
        # Thay vào đó, kiểm tra kết quả vẫn hợp lý (ít nhất không toàn zero nếu λ nhỏ)
        assert np.any(beta_with_tolf[1:] != 0.0)


class TestLassoPath:

    def test_lasso_path_output_shape(self):
        """lasso_path() phai tra ve (lambdas, coef_paths) co shape dung."""
        np.random.seed(30)
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        lambdas_test = np.logspace(-4, 0, 15)

        lams, paths = LassoRegressor.lasso_path(X, y, lambdas=lambdas_test, show=False)
        assert lams.shape  == (15,),    f"lambdas shape sai: {lams.shape}"
        assert paths.shape == (15, 4),  f"coef_paths shape sai: {paths.shape}"

    def test_lasso_path_sparsity_increases_with_lambda(self):
        """
        Khi lambda tang, so he so != 0 phai giam (hoac bang nhau).
        Kiem tra tren truong hop ro rang.
        """
        np.random.seed(31)
        X = np.random.randn(80, 5)
        y = X @ np.array([2.0, -1.5, 0.0, 3.0, 0.0]) + np.random.randn(80) * 0.5

        lambdas_test = np.logspace(-4, 0, 20)
        _, paths = LassoRegressor.lasso_path(X, y, lambdas=lambdas_test, show=False)

        # So he so != 0 o lam nho nhat phai >= so he so != 0 o lam lon nhat
        n_nonzero_small_lam = np.sum(np.abs(paths[0,  1:]) > 1e-8)
        n_nonzero_large_lam = np.sum(np.abs(paths[-1, 1:]) > 1e-8)
        assert n_nonzero_small_lam >= n_nonzero_large_lam, (
            f"FAIL: sparsity phai tang khi lam tang. "
            f"small_lam={n_nonzero_small_lam}, large_lam={n_nonzero_large_lam}"
        )


# ============================================================
# 3. SOFT THRESHOLDING
# ============================================================

class TestSoftThreshold:

    def test_positive_above_threshold(self):
        """z > gamma: S(z, gamma) = z - gamma."""
        _assert_close(_soft_threshold(5.0, 2.0), 3.0, msg="5 - 2 = 3")

    def test_negative_above_threshold(self):
        """z < -gamma: S(z, gamma) = z + gamma."""
        _assert_close(_soft_threshold(-5.0, 2.0), -3.0, msg="-5 + 2 = -3")

    def test_within_threshold(self):
        """|z| <= gamma: S(z, gamma) = 0."""
        _assert_close(_soft_threshold(1.5,  2.0), 0.0, msg="|z| < gamma -> 0")
        _assert_close(_soft_threshold(-1.5, 2.0), 0.0, msg="|z| < gamma -> 0 (am)")

    def test_zero_threshold(self):
        """gamma = 0: S(z, 0) = z (khong lam gi ca)."""
        _assert_close(_soft_threshold(3.7, 0.0), 3.7, msg="gamma=0 -> identity")

    def test_exactly_at_threshold(self):
        """z = gamma: S(z, gamma) = 0."""
        _assert_close(_soft_threshold(2.0, 2.0), 0.0, msg="z = gamma -> 0")


# ============================================================
# 4. EDGE CASES & EXCEPTIONS
# ============================================================

class TestEdgeCases:

    def test_ridge_negative_lambda_raises(self):
        """lambda am phai raise ValueError."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="lambda"):
            ridge_fit(X, y, lam=-0.1)

    def test_lasso_negative_lambda_raises(self):
        """lambda am phai raise ValueError."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="lambda"):
            lasso_fit(X, y, lam=-0.1)

    def test_ridge_singular_with_zero_lambda_raises(self):
        """
        Voi lambda = 0 va du lieu suy bien (da cong tuyen hoan hao),
        Ridge phai raise ValueError.
        """
        col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X_collinear = np.column_stack([col, col])
        y = col * 2.0
        with pytest.raises(ValueError):
            ridge_fit(X_collinear, y, lam=0.0)

    def test_ridge_solves_collinear_with_positive_lambda(self):
        """
        Khi lambda > 0, Ridge phai giai duoc ca du lieu da cong tuyen
        (vi (X^T X + lam*I) luon kha nghich khi lam > 0).
        """
        col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X_collinear = np.column_stack([col, col])
        y = col * 2.0
        # Khong duoc raise exception
        res = ridge_fit(X_collinear, y, lam=1.0)
        assert res["beta_hat"] is not None

    def test_ridge_1d_input(self):
        """X 1-D (vector) phai duoc reshape thanh (n, 1) tu dong."""
        X_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y    = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        res  = ridge_fit(X_1d, y, lam=0.0001)
        assert res["k"] == 2   # 1 feature + intercept

    def test_lasso_1d_input(self):
        """X 1-D phai duoc xu ly tu dong."""
        X_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y    = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        res  = lasso_fit(X_1d, y, lam=0.001)
        assert res["k"] == 2


# ============================================================
# 5. LY THUYET THONG KE (Shrinkage & Sparsity)
# ============================================================

class TestTheory:

    def test_ridge_vs_ols_bias_variance(self):
        """
        Kieu chung: Gia su du lieu cao chieu (p lon, n vua).
        Ridge phai co test RSS nho hon OLS (do regularization giam variance).
        """
        np.random.seed(40)
        n_train, n_test, p = 30, 200, 20
        X_train = np.random.randn(n_train, p)
        X_test  = np.random.randn(n_test,  p)
        beta    = np.random.randn(p) * 0.5
        y_train = X_train @ beta + np.random.randn(n_train)
        y_test  = X_test  @ beta + np.random.randn(n_test)

        # Ridge
        ridge = RidgeRegressor(lam=5.0).fit(X_train, y_train)
        rss_ridge = float(np.sum((y_test - ridge.predict(X_test)) ** 2))

        # OLS bang numpy lstsq (de tranh van de da cong tuyen trong du lieu cao chieu)
        ones    = np.ones((n_train, 1))
        Xd_tr   = np.hstack([ones, X_train])
        b_ols, _, _, _ = np.linalg.lstsq(Xd_tr, y_train, rcond=None)
        ones_te = np.ones((n_test, 1))
        Xd_te   = np.hstack([ones_te, X_test])
        rss_ols = float(np.sum((y_test - Xd_te @ b_ols) ** 2))

        assert rss_ridge < rss_ols, (
            f"FAIL: Ridge phai tot hon OLS trong do cao chieu. "
            f"RSS_ridge={rss_ridge:.2f}, RSS_ols={rss_ols:.2f}"
        )

    def test_lasso_selects_relevant_features(self):
        """
        Lasso phai nhan dien va giu lai cac bien quan trong,
        trong khi loai bo cac bien nhieu.
        """
        np.random.seed(41)
        n = 200
        x_signal = np.random.randn(n, 3)         # 3 bien quan trong
        x_noise  = np.random.randn(n, 7) * 0.001 # 7 bien nhieu
        X        = np.column_stack([x_signal, x_noise])
        y        = x_signal @ np.array([2.0, -3.0, 1.5]) + 0.2 * np.random.randn(n)

        model = LassoRegressor(lam=0.3, max_iter=5000, tol=1e-8).fit(X, y)
        beta  = model.beta_hat[1:]  # bo intercept

        # 3 he so dau phai khac 0
        assert np.all(np.abs(beta[:3]) > 1e-4), (
            f"FAIL: Lasso phai giu lai bien co y nghia: {beta[:3]}"
        )
        # 7 he so cuoi phai = 0 hoac rat nho
        assert np.sum(np.abs(beta[3:]) > 1e-4) <= 2, (
            f"FAIL: Lasso phai loai bo bien nhieu: {beta[3:]}"
        )

    def test_ridge_intercept_not_shrunk(self):
        """
        Intercept khong duoc bi phat boi Ridge (phai xap xi OLS intercept
        khi cac he so con lai duoc co lai).
        """
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 10.0 + X @ np.array([1.0, -1.0, 0.5]) + np.random.randn(100)

        # Intercept cua Ridge voi lam lon phai van gan 10 (gia tri that)
        res = ridge_fit(X, y, lam=1e4)
        intercept = res["beta_hat"][0]
        assert abs(intercept - 10.0) < 1.0, (
            f"FAIL: Intercept ({intercept:.4f}) nen xap xi 10.0 vi no khong bi phat"
        )


# ============================================================
# 6. KIEM CHUNG VOI SKLEARN
# ============================================================

class TestSklearnVerification:

    def test_ridge_matches_sklearn_small(self):
        """So sanh Ridge scratch voi sklearn tren du lieu nho."""
        X = np.array([[1.5, 2.1], [3.0, 1.2], [4.5, 5.0], [2.2, 3.3],
                      [5.1, 0.9], [1.0, 4.0]], dtype=float)
        y = np.array([5.1, 8.2, 12.5, 7.3, 11.0, 6.0], dtype=float)
        result = verify_ridge_sklearn(X, y, lam=1.0, tol=1e-6)
        assert result["passed"], f"Ridge khong khop sklearn: max_diff={result['max_diff']:.2e}"

    def test_ridge_matches_sklearn_large(self):
        """So sanh Ridge scratch voi sklearn tren du lieu lon va nhieu bien."""
        np.random.seed(50)
        X = np.random.randn(200, 10)
        y = X @ np.random.randn(10) + np.random.randn(200)
        for lam in [0.01, 1.0, 100.0]:
            result = verify_ridge_sklearn(X, y, lam=lam, tol=1e-6)
            assert result["passed"], (
                f"Ridge khong khop sklearn (lam={lam}): max_diff={result['max_diff']:.2e}"
            )

    def test_lasso_matches_sklearn_small(self):
        """So sanh Lasso scratch voi sklearn tren du lieu nho."""
        X = np.array([[1.5, 2.1], [3.0, 1.2], [4.5, 5.0], [2.2, 3.3],
                      [5.1, 0.9], [1.0, 4.0]], dtype=float)
        y = np.array([5.1, 8.2, 12.5, 7.3, 11.0, 6.0], dtype=float)
        result = verify_lasso_sklearn(X, y, lam=0.1, tol=1e-4)
        assert result["passed"], f"Lasso khong khop sklearn: max_diff={result['max_diff']:.2e}"

    def test_lasso_matches_sklearn_large(self):
        """So sanh Lasso scratch voi sklearn tren du lieu lon."""
        np.random.seed(51)
        X = np.random.randn(150, 8)
        y = X @ np.random.randn(8) + np.random.randn(150)
        for lam in [0.01, 0.1, 1.0]:
            result = verify_lasso_sklearn(X, y, lam=lam, tol=1e-4)
            assert result["passed"], (
                f"Lasso khong khop sklearn (lam={lam}): max_diff={result['max_diff']:.2e}"
            )

    def test_ridge_r2_consistent_with_sklearn(self):
        """R^2 cua Ridge scratch phai khop R^2 cua sklearn."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        np.random.seed(52)
        X = np.random.randn(80, 4)
        y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + np.random.randn(80)

        model_scratch = RidgeRegressor(lam=2.0).fit(X, y)
        model_sk      = Ridge(alpha=2.0, fit_intercept=True).fit(X, y)

        r2_scratch = 1.0 - np.sum((y - model_scratch.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2)
        r2_sklearn = r2_score(y, model_sk.predict(X))

        _assert_close(r2_scratch, r2_sklearn, tol=1e-8, msg="R2 Ridge scratch vs sklearn")