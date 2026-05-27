"""
test_model_comparison.py
========================
Unit tests cho class ModelComparison (model_comparison.py).

Mỗi method / hành vi có ít nhất 2 test case trên dữ liệu tổng hợp đã biết.
KHÔNG dùng AmesHousing.csv thật — tự sinh dữ liệu để test nhanh và reproducible.

Chạy:
    python -m pytest test_model_comparison.py -v
    # hoặc
    python test_model_comparison.py
"""

import sys, os
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend để test không mở cửa sổ

# ── path ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_ROOT, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from part2.model_comparison import ModelComparison

SEED = 42


# ══════════════════════════════════════════════════════════════════════
# FIXTURE — inject dữ liệu tổng hợp vào ModelComparison
# ══════════════════════════════════════════════════════════════════════
def _make_synthetic(n: int = 300, p: int = 8, seed: int = SEED):
    """
    Sinh dữ liệu hồi quy tuyến tính đã biết:
        y = 1·x0 + 2·x1 - 1.5·x2 + noise
    Dùng để kiểm tra metric và tính đúng đắn của mô hình.
    """
    rng  = np.random.default_rng(seed)
    X    = rng.standard_normal((n, p))
    beta = np.array([1.0, 2.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    y    = X @ beta + 0.5 * rng.standard_normal(n)
    return X, y


def _make_mc_with_data(n=300, p=8, lam_grid=None, seed=SEED) -> ModelComparison:
    """
    Tạo ModelComparison đã có sẵn X_train/X_test/y_train/y_test
    (bypass load_and_clean và DataPipeline để test nhanh).
    """
    X, y = _make_synthetic(n=n, p=p, seed=seed)
    split = int(n * 0.8)

    X_train = pd.DataFrame(X[:split], columns=[f"f{i}" for i in range(p)])
    X_test  = pd.DataFrame(X[split:], columns=[f"f{i}" for i in range(p)])
    y_train = y[:split]
    y_test  = y[split:]

    mc = ModelComparison(
        lam_grid     = lam_grid or list(np.logspace(-3, 3, 15)),
        random_state = seed,
    )
    mc.X_train = X_train
    mc.X_test  = X_test
    mc.y_train = y_train
    mc.y_test  = y_test
    return mc


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 1 — _mae / _rmse / _r2 (metric utilities)
# ══════════════════════════════════════════════════════════════════════
class TestMetricUtils(unittest.TestCase):
    """
    Các static method tính metric phải cho kết quả đúng trên dữ liệu đã biết.
    """

    # ── test 1: dự đoán hoàn hảo → MAE=RMSE=0, R²=1 ─────────────────
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(ModelComparison._mae(y, y),  0.0, places=10)
        self.assertAlmostEqual(ModelComparison._rmse(y, y), 0.0, places=10)
        self.assertAlmostEqual(ModelComparison._r2(y, y),   1.0, places=10)

    # ── test 2: dự đoán sai cố định → giá trị tính tay ───────────────
    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])   # lệch [+1, 0, -1]

        # MAE = (1+0+1)/3 = 2/3
        self.assertAlmostEqual(ModelComparison._mae(y_true, y_pred),
                               2/3, places=8)
        # RMSE = sqrt((1+0+1)/3) = sqrt(2/3)
        self.assertAlmostEqual(ModelComparison._rmse(y_true, y_pred),
                               np.sqrt(2/3), places=8)

    # ── test 3: MAE ≤ RMSE (luôn đúng với bất kỳ y nào) ─────────────
    def test_mae_leq_rmse(self):
        rng    = np.random.default_rng(SEED)
        y_true = rng.standard_normal(100)
        y_pred = rng.standard_normal(100)
        self.assertLessEqual(
            ModelComparison._mae(y_true, y_pred),
            ModelComparison._rmse(y_true, y_pred) + 1e-12,
        )

    # ── test 4: MAE/RMSE phải tính trên thang USD (expm1), không log ──
    def test_metrics_expm1(self):
        """
        Ames Housing dùng log1p(SalePrice) làm target.
        MAE/RMSE phải báo cáo bằng USD (sau expm1), không phải log-scale.

        Ví dụ: y_true = log1p(200_000), y_pred = log1p(225_000)
          → sai số thực tế = 25_000 USD
          → RMSE/MAE trên log-scale ≈ 0.118 (vô nghĩa kinh tế)
        """
        y_true = np.array([np.log1p(200_000.0)])
        y_pred = np.array([np.log1p(225_000.0)])

        actual_mae  = ModelComparison._mae(y_true, y_pred)
        actual_rmse = ModelComparison._rmse(y_true, y_pred)

        # Kỳ vọng: sai số ~25_000 USD (cho phép sai lệch nhỏ do float)
        self.assertAlmostEqual(actual_mae,  25_000.0, delta=1.0,
            msg=f"MAE = {actual_mae:.2f}, kỳ vọng ≈ 25000 USD")
        self.assertAlmostEqual(actual_rmse, 25_000.0, delta=1.0,
            msg=f"RMSE = {actual_rmse:.2f}, kỳ vọng ≈ 25000 USD")

        # Đảm bảo KHÔNG tính trên log-scale (≈ 0.118, không phải 25_000)
        self.assertGreater(actual_mae, 1_000,
            f"MAE = {actual_mae:.4f} có vẻ tính trên log-scale, "
            f"phải > 1000 USD (thang đo thực)")
        self.assertGreater(actual_rmse, 1_000,
            f"RMSE = {actual_rmse:.4f} có vẻ tính trên log-scale, "
            f"phải > 1000 USD (thang đo thực)")


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 2 — split (stratified train/test)
# ══════════════════════════════════════════════════════════════════════
class TestSplit(unittest.TestCase):
    """
    split() phải:
      - Kích thước train + test = tổng mẫu.
      - Tỉ lệ test ≈ test_size (±2%).
      - Reproducible với cùng random_state.
      - Không có hàng trùng lắp giữa train và test.
    """

    def _make_Xy(self, n=200):
        rng = np.random.default_rng(SEED)
        X   = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a","b","c"])
        y   = pd.Series(rng.uniform(100_000, 500_000, n), name="SalePrice")
        return X, y

    def _mc(self):
        return ModelComparison(test_size=0.2, random_state=SEED)

    # ── test 1: kích thước đúng ───────────────────────────────────────
    def test_split_sizes(self):
        X, y = self._make_Xy(n=200)
        mc   = self._mc()
        X_tr, X_te, y_tr, y_te = mc.split(X, y)

        self.assertEqual(len(X_tr) + len(X_te), 200)
        actual_ratio = len(X_te) / 200
        self.assertAlmostEqual(actual_ratio, 0.2, delta=0.02,
                               msg=f"Test ratio = {actual_ratio:.3f}, kỳ vọng ≈ 0.20")

    # ── test 2: reproducible ─────────────────────────────────────────
    def test_split_reproducible(self):
        X, y = self._make_Xy(n=200)
        mc1  = ModelComparison(test_size=0.2, random_state=SEED)
        mc2  = ModelComparison(test_size=0.2, random_state=SEED)
        _, X_te1, _, _ = mc1.split(X, y)
        _, X_te2, _, _ = mc2.split(X, y)
        pd.testing.assert_frame_equal(X_te1.reset_index(drop=True),
                                      X_te2.reset_index(drop=True))

    # ── test 3: không overlap giữa train và test ─────────────────────
    def test_no_overlap(self):
        X, y = self._make_Xy(n=200)
        mc   = self._mc()
        X_tr, X_te, _, _ = mc.split(X, y)

        # Dùng index gốc để kiểm tra
        train_set = set(X_tr.index.tolist())
        test_set  = set(X_te.index.tolist())
        self.assertEqual(len(train_set & test_set), 0,
                         "Có hàng xuất hiện ở cả train lẫn test")

    # ── test 4: StandardScaler chỉ fit trên X_train (không data leakage)
    def test_no_data_leakage(self):
        """
        StandardScaler phải được fit_transform trên X_train,
        và chỉ transform (không fit) trên X_test.

        Nếu bị leakage: scaler fit trên toàn bộ X → mean/std bị ảnh hưởng
        bởi X_test → thông tin tập test "rò rỉ" vào quá trình train.

        Kiểm tra: mean của scaler phải ≈ mean của X_train thô,
        KHÔNG phải mean của X_full (toàn bộ dữ liệu).
        """
        X, y = self._make_Xy(n=200)
        mc   = self._mc()
        X_tr, X_te, _, _ = mc.split(X, y)

        # ModelComparison phải lưu scaler sau split/fit
        self.assertTrue(
            hasattr(mc, "scaler_"),
            "mc phải có thuộc tính scaler_ sau khi split() để kiểm tra leakage"
        )

        scaler = mc.scaler_

        # Mean của scaler phải khớp với X_train (chưa scale), không phải X_full
        # Dùng X_tr gốc (trước khi transform) — mc phải lưu X_train_raw_
        self.assertTrue(
            hasattr(mc, "X_train_raw_"),
            "mc phải lưu X_train_raw_ (giá trị trước scale) để kiểm tra leakage"
        )
        train_mean = mc.X_train_raw_.mean(axis=0).values
        full_mean  = X.mean(axis=0).values

        np.testing.assert_allclose(
            scaler.mean_, train_mean, rtol=1e-5,
            err_msg=(
                "scaler.mean_ không khớp với mean của X_train → "
                "có thể scaler đã fit trên toàn bộ data (Data Leakage!)"
            )
        )

        # Đảm bảo mean của scaler KHÁC mean của full data
        # (nếu không có leakage, chúng phải khác nhau vì X_train ≠ X_full)
        if not np.allclose(train_mean, full_mean, rtol=1e-3):
            max_diff = np.max(np.abs(scaler.mean_ - full_mean))
            self.assertGreater(
                max_diff, 1e-6,
                "scaler.mean_ ≈ mean(X_full) → nghi ngờ scaler fit trên "
                "toàn bộ data thay vì chỉ X_train (Data Leakage)"
            )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 3 — fit() trên dữ liệu tổng hợp
# ══════════════════════════════════════════════════════════════════════
class TestFit(unittest.TestCase):
    """
    Sau khi fit():
      - self.results phải có key 'ridge' và 'lasso'.
      - Mỗi key chứa 'model', 'lam', 'metrics', 'y_pred'.
      - R² trên dữ liệu tổng hợp (n=300, tín hiệu rõ) phải > 0.7.
      - λ* phải nằm trong lam_grid.
    """

    def setUp(self):
        self.mc = _make_mc_with_data(n=300, p=8)
        self.mc.fit()

    # ── test 1: cấu trúc results đúng ────────────────────────────────
    def test_results_structure(self):
        for key in ["ridge", "lasso"]:
            self.assertIn(key, self.mc.results)
            r = self.mc.results[key]
            for field in ["model", "lam", "metrics", "y_pred",
                          "cv_result", "feature_names"]:
                self.assertIn(field, r,
                              f"results['{key}'] thiếu key '{field}'")

    # ── test 2: R² > 0.7 (dữ liệu có tín hiệu rõ) ───────────────────
    def test_r2_reasonable(self):
        for key in ["ridge", "lasso"]:
            r2 = self.mc.results[key]["metrics"]["r2"]
            self.assertGreater(r2, 0.70,
                               f"{key} R² = {r2:.4f} thấp bất thường")

    # ── test 3: λ* nằm trong lam_grid ────────────────────────────────
    def test_best_lam_in_grid(self):
        for key in ["ridge", "lasso"]:
            lam_best = self.mc.results[key]["lam"]
            self.assertIn(lam_best, self.mc.lam_grid,
                          f"{key} λ* = {lam_best} không thuộc lam_grid")

    # ── test 4: fit trước khi có data → RuntimeError ─────────────────
    def test_fit_without_data_raises(self):
        mc_empty = ModelComparison()   # chưa gán X_train, y_train
        with self.assertRaises((RuntimeError, AttributeError, TypeError)):
            mc_empty.fit()


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 4 — y_pred shape & range
# ══════════════════════════════════════════════════════════════════════
class TestPredictions(unittest.TestCase):
    """
    y_pred:
      - Đúng số hàng (= len(X_test)).
      - Không chứa NaN hay Inf.
      - Reproducible: fit 2 lần cùng seed → y_pred giống nhau.
    """

    def setUp(self):
        self.mc = _make_mc_with_data(n=300, p=8)
        self.mc.fit()

    # ── test 1: shape và finite ───────────────────────────────────────
    def test_pred_shape_and_finite(self):
        n_test = len(self.mc.X_test)
        for key in ["ridge", "lasso"]:
            pred = self.mc.results[key]["y_pred"]
            self.assertEqual(len(pred), n_test,
                             f"{key}: len(y_pred) = {len(pred)} ≠ {n_test}")
            self.assertTrue(np.all(np.isfinite(pred)),
                            f"{key}: y_pred chứa NaN hoặc Inf")

    # ── test 2: reproducible predictions ─────────────────────────────
    def test_pred_reproducible(self):
        mc2 = _make_mc_with_data(n=300, p=8)
        mc2.fit()
        for key in ["ridge", "lasso"]:
            np.testing.assert_allclose(
                self.mc.results[key]["y_pred"],
                mc2.results[key]["y_pred"],
                rtol=1e-8,
                err_msg=f"{key} predictions không reproducible"
            )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 5 — summary()
# ══════════════════════════════════════════════════════════════════════
class TestSummary(unittest.TestCase):
    """
    summary() phải:
      - Trả về DataFrame có đúng 2 hàng (ridge & lasso).
      - Có cột MAE, RMSE, R².
      - Gọi trước fit() → RuntimeError.
    """

    # ── test 1: cấu trúc DataFrame ───────────────────────────────────
    def test_summary_shape_and_cols(self):
        mc = _make_mc_with_data(n=200, p=6)
        mc.fit()
        df_cmp = mc.summary()

        self.assertEqual(df_cmp.shape[0], 2,
                         f"summary phải có 2 hàng, có {df_cmp.shape[0]}")
        for col in ["MAE", "RMSE", "R²"]:
            self.assertIn(col, df_cmp.columns,
                          f"summary thiếu cột '{col}'")

    # ── test 2: tất cả metric là số thực dương hợp lệ ─────────────────
    def test_metric_values_valid(self):
        mc = _make_mc_with_data(n=200, p=6)
        mc.fit()
        df_cmp = mc.summary()

        self.assertTrue((df_cmp["MAE"]  > 0).all(), "MAE phải > 0")
        self.assertTrue((df_cmp["RMSE"] > 0).all(), "RMSE phải > 0")
        self.assertTrue((df_cmp["R²"]   < 1.01).all(),
                        "R² phải ≤ 1 (có thể âm với mô hình rất kém)")

    # ── test 3: gọi summary() trước fit() → RuntimeError ─────────────
    def test_summary_before_fit_raises(self):
        mc = ModelComparison()
        with self.assertRaises(RuntimeError):
            mc.summary()


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 6 — plot methods (smoke tests)
# ══════════════════════════════════════════════════════════════════════
class TestPlots(unittest.TestCase):
    """
    Smoke tests: các hàm plot không raise exception và trả về Figure.
    Dùng matplotlib Agg backend (không mở cửa sổ).
    """

    @classmethod
    def setUpClass(cls):
        cls.mc = _make_mc_with_data(n=200, p=6)
        cls.mc.fit()

    # ── test 1: plot_cv_lambda trả về Figure ─────────────────────────
    def test_plot_cv_lambda_returns_figure(self):
        import matplotlib.pyplot as plt
        fig = self.mc.plot_cv_lambda(save=False)
        self.assertIsNotNone(fig)
        plt.close("all")

    # ── test 2: plot_feature_importance không raise ───────────────────
    def test_plot_feature_importance_no_error(self):
        import matplotlib.pyplot as plt
        for key in ["ridge", "lasso"]:
            fig = self.mc.plot_feature_importance(model_key=key, save=False)
            self.assertIsNotNone(fig)
        plt.close("all")

    # ── test 3: plot_actual_vs_predicted cho cả hai mô hình ──────────
    def test_plot_actual_vs_predicted(self):
        import matplotlib.pyplot as plt
        for key in ["ridge", "lasso"]:
            fig = self.mc.plot_actual_vs_predicted(model_key=key, save=False)
            self.assertIsNotNone(fig)
        plt.close("all")

    # ── test 4: plot_ridge_trace không raise ─────────────────────────
    def test_plot_ridge_trace_no_error(self):
        import matplotlib.pyplot as plt
        fig = self.mc.plot_ridge_trace(top_n=5, save=False)
        self.assertIsNotNone(fig)
        plt.close("all")

    # ── test 5: plot_lasso_coef_path không raise ─────────────────────
    def test_plot_lasso_coef_path_no_error(self):
        import matplotlib.pyplot as plt
        fig = self.mc.plot_lasso_coef_path(top_n=5, save=False)
        self.assertIsNotNone(fig)
        plt.close("all")


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 7 — __repr__ và _check_fitted
# ══════════════════════════════════════════════════════════════════════
class TestMiscBehavior(unittest.TestCase):
    """
    Kiểm tra trạng thái nội bộ và repr.
    """

    # ── test 1: repr chứa thông tin cơ bản ───────────────────────────
    def test_repr_contains_status(self):
        mc = ModelComparison()
        s  = repr(mc)
        self.assertIn("not fitted", s)

        mc2 = _make_mc_with_data()
        mc2.fit()
        self.assertIn("fitted", repr(mc2))

    # ── test 2: _fitted flag đúng trước/sau fit ───────────────────────
    def test_fitted_flag(self):
        mc = _make_mc_with_data()
        self.assertFalse(mc._fitted)
        mc.fit()
        self.assertTrue(mc._fitted)

    # ── test 3: cv_k và lam_grid được lưu đúng ───────────────────────
    def test_hyperparams_stored(self):
        lam_grid = [0.01, 0.1, 1.0, 10.0]
        mc = ModelComparison(cv_k=3, lam_grid=lam_grid, random_state=SEED)
        self.assertEqual(mc.cv_k, 3)
        self.assertEqual(mc.lam_grid, lam_grid)
        self.assertEqual(mc.random_state, SEED)


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 8 — Ridge shrinkage property
# ══════════════════════════════════════════════════════════════════════
class TestRidgeShrinkage(unittest.TestCase):
    """
    Tính chất Ridge: khi λ tăng → ||β||₂ giảm (shrinkage về 0).
    """

    # ── test 1: lambda lớn → hệ số nhỏ hơn lambda nhỏ ───────────────
    def test_larger_lambda_smaller_coef(self):
        mc_small = _make_mc_with_data(lam_grid=[0.001])
        mc_small.fit()
        mc_large = _make_mc_with_data(lam_grid=[1000.0])
        mc_large.fit()

        norm_small = np.linalg.norm(mc_small.results["ridge"]["model"].coef_)
        norm_large = np.linalg.norm(mc_large.results["ridge"]["model"].coef_)

        self.assertGreater(norm_small, norm_large,
                           "Hệ số Ridge phải nhỏ hơn khi λ lớn hơn")

    # ── test 2: metric của Ridge vs Lasso đều là số hữu hạn ──────────
    def test_both_models_finite_metrics(self):
        mc = _make_mc_with_data()
        mc.fit()
        for key in ["ridge", "lasso"]:
            for metric in ["mae", "rmse", "r2"]:
                val = mc.results[key]["metrics"][metric]
                self.assertTrue(np.isfinite(val),
                                f"{key}.{metric} = {val} không hữu hạn")