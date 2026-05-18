import numpy as np
import pytest
from part1.cross_validation import kfold_cv, compare_models_cv

# Tiện ích nội bộ
def _make_linear_data(n=200, p=3, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y   = 2.0 + X @ beta + noise * rng.standard_normal(n)
    return X, y, beta


# 1. kfold_cv — OLS
class TestKFoldOLS:

    def test_ols_perfect_data_mse_near_zero(self):
        """
        OLS trên dữ liệu tuyến tính hoàn hảo (không nhiễu):
        CV MSE phải gần 0, CV R² phải gần 1.
        """
        X, y, _ = _make_linear_data(noise=0.0, seed=1)
        res = kfold_cv(X, y, k=5, model="ols", random_state=0)
        assert res["cv_mse"] < 1e-10, f"CV MSE={res['cv_mse']:.2e} (kỳ vọng ≈ 0)"
        assert res["cv_r2"]  > 0.999,  f"CV R²={res['cv_r2']:.6f} (kỳ vọng ≈ 1)"

    def test_ols_returns_correct_keys(self):
        """kfold_cv() phải trả về đủ các key."""
        X, y, _ = _make_linear_data(noise=1.0, seed=2)
        res = kfold_cv(X, y, k=5, model="ols")
        for key in ("cv_mse", "cv_rmse", "cv_mae", "cv_r2",
                    "fold_mse", "fold_r2", "k", "model", "lam"):
            assert key in res, f"Thiếu key '{key}'"

    def test_ols_fold_count(self):
        """fold_mse phải có đúng k phần tử."""
        X, y, _ = _make_linear_data(noise=0.5, seed=3)
        for k in [3, 5, 10]:
            res = kfold_cv(X, y, k=k, model="ols")
            assert len(res["fold_mse"]) == k
            assert len(res["fold_r2"])  == k

    def test_ols_reproducible_with_same_seed(self):
        """Cùng random_state phải cho kết quả giống nhau."""
        X, y, _ = _make_linear_data(noise=1.0, seed=4)
        r1 = kfold_cv(X, y, k=5, model="ols", random_state=99)
        r2 = kfold_cv(X, y, k=5, model="ols", random_state=99)
        assert r1["cv_mse"] == r2["cv_mse"]

    def test_ols_different_seeds_differ(self):
        """Khác random_state thường cho CV MSE khác nhau."""
        X, y, _ = _make_linear_data(noise=1.0, seed=5)
        r1 = kfold_cv(X, y, k=5, model="ols", random_state=1)
        r2 = kfold_cv(X, y, k=5, model="ols", random_state=2)
        assert r1["cv_mse"] != r2["cv_mse"]

    def test_ols_cv_mse_equals_mean_fold_mse(self):
        """cv_mse phải bằng trung bình fold_mse theo công thức CV(k)."""
        X, y, _ = _make_linear_data(noise=0.5, seed=6)
        res = kfold_cv(X, y, k=5, model="ols")
        expected = float(np.mean(res["fold_mse"]))
        assert abs(res["cv_mse"] - expected) < 1e-12

# 2. kfold_cv — Ridge
class TestKFoldRidge:

    def test_ridge_lam0_close_to_ols(self):
        """
        Ridge(λ≈0) phải cho CV MSE gần bằng OLS.
        Khi λ → 0, Ridge → OLS.
        """
        X, y, _ = _make_linear_data(noise=1.0, seed=10)
        ols_res   = kfold_cv(X, y, k=5, model="ols",   random_state=42)
        ridge_res = kfold_cv(X, y, k=5, model="ridge", lam=1e-9, random_state=42)
        diff = abs(ols_res["cv_mse"] - ridge_res["cv_mse"])
        assert diff < 1e-6, f"|Δ MSE| = {diff:.2e} (kỳ vọng < 1e-6)"

    def test_ridge_large_lam_increases_mse(self):
        """
        Khi λ rất lớn, Ridge co hệ số về 0 → CV MSE tăng so với OLS.
        """
        X, y, _ = _make_linear_data(noise=0.5, seed=11)
        ols_res   = kfold_cv(X, y, k=5, model="ols",   random_state=42)
        ridge_res = kfold_cv(X, y, k=5, model="ridge", lam=1e8, random_state=42)
        assert ridge_res["cv_mse"] > ols_res["cv_mse"]

    def test_ridge_perfect_data(self):
        """Ridge(λ nhỏ) trên dữ liệu hoàn hảo: CV MSE gần 0."""
        X, y, _ = _make_linear_data(noise=0.0, seed=12)
        res = kfold_cv(X, y, k=5, model="ridge", lam=1e-9, random_state=0)
        assert res["cv_mse"] < 1e-8, f"CV MSE={res['cv_mse']:.2e}"

# 3. kfold_cv — Lasso
class TestKFoldLasso:

    def test_lasso_lam0_close_to_ols(self):
        """Lasso(λ≈0) phải cho CV MSE gần bằng OLS."""
        X, y, _ = _make_linear_data(noise=1.0, seed=20)
        ols_res   = kfold_cv(X, y, k=5, model="ols",   random_state=42)
        lasso_res = kfold_cv(X, y, k=5, model="lasso", lam=1e-8, random_state=42)
        diff = abs(ols_res["cv_mse"] - lasso_res["cv_mse"])
        assert diff < 1e-4, f"|Δ MSE| = {diff:.2e} (kỳ vọng < 1e-4)"

    def test_lasso_large_lam_increases_mse(self):
        """Khi λ rất lớn, Lasso co hệ số về 0 → CV MSE tăng."""
        X, y, _ = _make_linear_data(noise=0.5, seed=21)
        ols_res   = kfold_cv(X, y, k=5, model="ols",   random_state=42)
        lasso_res = kfold_cv(X, y, k=5, model="lasso", lam=1e4, random_state=42)
        assert lasso_res["cv_mse"] > ols_res["cv_mse"]

    def test_lasso_returns_correct_keys(self):
        """kfold_cv Lasso phải trả về đủ key."""
        X, y, _ = _make_linear_data(noise=1.0, seed=22)
        res = kfold_cv(X, y, k=5, model="lasso", lam=0.1)
        for key in ("cv_mse", "cv_rmse", "cv_mae", "cv_r2", "fold_mse"):
            assert key in res

# 4. compare_models_cv
class TestCompareModels:

    def test_returns_correct_keys(self):
        """compare_models_cv phải trả về đủ key."""
        X, y, _ = _make_linear_data(noise=1.0, seed=30)
        result = compare_models_cv(X, y, k=5,
                                   lam_grid=[0.1, 1.0, 10.0],
                                   random_state=42)
        for key in ("ols_result", "ridge_results", "lasso_results",
                    "best_lam_ridge", "best_lam_lasso",
                    "best_ridge", "best_lasso"):
            assert key in result

    def test_best_lam_is_in_grid(self):
        """best_lam_ridge và best_lam_lasso phải nằm trong lam_grid."""
        X, y, _ = _make_linear_data(noise=1.0, seed=31)
        lam_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
        result = compare_models_cv(X, y, k=5,
                                   lam_grid=lam_grid,
                                   random_state=42)
        assert result["best_lam_ridge"] in lam_grid
        assert result["best_lam_lasso"] in lam_grid

    def test_ridge_results_length(self):
        """ridge_results phải có đúng len(lam_grid) phần tử."""
        X, y, _ = _make_linear_data(noise=1.0, seed=32)
        lam_grid = [0.1, 1.0, 10.0]
        result = compare_models_cv(X, y, k=5, lam_grid=lam_grid)
        assert len(result["ridge_results"]) == len(lam_grid)
        assert len(result["lasso_results"]) == len(lam_grid)

# 5. Edge cases & Exceptions
class TestEdgeCases:

    def test_invalid_model_raises(self):
        """model không hợp lệ phải raise AssertionError."""
        X, y, _ = _make_linear_data(seed=40)
        with pytest.raises(AssertionError):
            kfold_cv(X, y, k=5, model="svm")

    def test_k_less_than_2_raises(self):
        """k < 2 phải raise AssertionError."""
        X, y, _ = _make_linear_data(seed=41)
        with pytest.raises(AssertionError):
            kfold_cv(X, y, k=1, model="ols")

    def test_n_less_than_k_raises(self):
        """n < k phải raise AssertionError."""
        X = np.random.randn(3, 2)
        y = np.random.randn(3)
        with pytest.raises(AssertionError):
            kfold_cv(X, y, k=5, model="ols")

    def test_handles_n_not_divisible_by_k(self):
        """n không chia hết cho k vẫn phải chạy đúng."""
        X, y, _ = _make_linear_data(n=103, noise=0.5, seed=42)
        res = kfold_cv(X, y, k=5, model="ols")
        assert len(res["fold_mse"]) == 5
        assert res["cv_mse"] > 0

    def test_k_equals_n_leave_one_out(self):
        """k = n là Leave-One-Out CV — phải chạy được."""
        X, y, _ = _make_linear_data(n=20, p=2, noise=0.5, seed=43)
        res = kfold_cv(X, y, k=20, model="ols")
        assert len(res["fold_mse"]) == 20