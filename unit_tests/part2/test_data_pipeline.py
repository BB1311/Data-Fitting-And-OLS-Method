"""
test_data_pipeline.py
=====================
Unit tests cho class DataPipeline (data_pipeline.py).

Chạy:
    python -m pytest test_data_pipeline.py -v
    # hoặc
    python test_data_pipeline.py
"""

import sys, os
import unittest
import numpy as np
import pandas as pd

# ── path ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_ROOT, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from part2.data_pipeline import DataPipeline, run_vif_check

def _make_simple_df() -> pd.DataFrame:
    """
    DataFrame tối giản với giá trị cố định.
    n=10 để tính quantile dễ kiểm tra tay.
    """
    return pd.DataFrame({
        "Gr Liv Area"   : [600, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2400, 2800],
        "Total Bsmt SF" : [  0, 300,  500,  700,  900, 1000, 1200, 1400, 1600, 1900],
        "Overall Qual"  : [  3,   4,    5,    5,    6,    6,    7,    8,    9,   10],
        "Age_At_Sale"   : [ 70,  55,   40,   35,   30,   20,   15,   10,    5,    1],
        "Garage Area"   : [  0, 100,  200,  300,  400,  500,  600,  700,  800,  880],
    }, dtype=float)


def _make_y() -> pd.Series:
    return pd.Series(
        [110_000, 145_000, 175_000, 195_000, 230_000,
         260_000, 290_000, 330_000, 380_000, 450_000],
        dtype=float, name="SalePrice"
    )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 1 — fit_transform / transform (no-leak contract)
# ══════════════════════════════════════════════════════════════════════
class TestFitTransform(unittest.TestCase):

    def setUp(self):
        df = _make_simple_df()        # 10 hàng
        self.X_train = df.iloc[:8].copy()
        self.X_test  = df.iloc[8:].copy()
        self.y_train = _make_y().iloc[:8]

    # ── test 1: output shape hợp lệ ─────────────────────────────────
    def test_output_shape(self):
        pipe = DataPipeline(
            outlier_method=None, scale=True, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )
        X_tr_out, y_tr_out = pipe.fit_transform(self.X_train, self.y_train)
        X_te_out            = pipe.transform(self.X_test)

        self.assertEqual(X_tr_out.shape[0], len(self.X_train))
        self.assertEqual(X_te_out.shape[0], len(self.X_test))
        self.assertEqual(X_tr_out.shape[1], X_te_out.shape[1])
        self.assertEqual(len(y_tr_out), len(self.X_train))

    # ── test 2: không còn NaN sau transform ─────────────────────────
    def test_no_nan_after_transform(self):
        pipe = DataPipeline(
            outlier_method=None, scale=True, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )
        X_tr_out, _ = pipe.fit_transform(self.X_train, self.y_train)
        X_te_out    = pipe.transform(self.X_test)

        self.assertFalse(X_tr_out.isnull().any().any(),
                         "Train vẫn còn NaN sau fit_transform")
        self.assertFalse(X_te_out.isnull().any().any(),
                         "Test vẫn còn NaN sau transform")

    # ── test 3: gọi transform trước fit → RuntimeError ──────────────
    def test_transform_before_fit_raises(self):
        pipe = DataPipeline()
        with self.assertRaises(RuntimeError):
            pipe.transform(self.X_test)


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 2 — Z-SCORE STANDARDIZATION
# ══════════════════════════════════════════════════════════════════════
class TestScaling(unittest.TestCase):

    def _make_pipe(self):
        return DataPipeline(
            outlier_method=None, scale=True, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )

    # ── test 1: mean ≈ 0 trên train ─────────────────────────────────
    def test_train_mean_near_zero(self):
        X_train = _make_simple_df()
        y       = _make_y()

        pipe = self._make_pipe()
        X_out, _ = pipe.fit_transform(X_train, y)

        for col in X_out.columns:
            self.assertAlmostEqual(
                X_out[col].mean(), 0.0, places=6,
                msg=f"mean của cột '{col}' ≠ 0 sau z-score"
            )

    # ── test 2: std ≈ 1 trên train ──────────────────────────────────
    def test_train_std_near_one(self):
        X_train = _make_simple_df()
        y       = _make_y()

        pipe = self._make_pipe()
        X_out, _ = pipe.fit_transform(X_train, y)

        for col in X_out.columns:
            self.assertAlmostEqual(
                X_out[col].std(ddof=1), 1.0, places=5,
                msg=f"std của cột '{col}' ≠ 1 sau z-score"
            )

    # ── test 3: transform test dùng tham số từ train (no leakage) ───
    def test_test_uses_train_params(self):
        X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        X_test  = pd.DataFrame({"x": [10.0, 20.0]})
        y_train = pd.Series([0.0] * 5, name="y")

        pipe = DataPipeline(
            outlier_method=None, scale=True, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )
        pipe.fit_transform(X_train, y_train)
        X_te_out = pipe.transform(X_test)

        mu_train  = 3.0
        std_train = float(np.std([1, 2, 3, 4, 5], ddof=1))

        expected_10 = (10.0 - mu_train) / std_train
        expected_20 = (20.0 - mu_train) / std_train

        self.assertAlmostEqual(X_te_out["x"].iloc[0], expected_10, places=5)
        self.assertAlmostEqual(X_te_out["x"].iloc[1], expected_20, places=5)


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 3 — WINSORIZE OUTLIER
# ══════════════════════════════════════════════════════════════════════
class TestWinsorize(unittest.TestCase):
    TRAIN_V = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0]

    def _pipe(self, threshold=0.10):
        return DataPipeline(
            outlier_method="winsorize",
            outlier_threshold=threshold,
            scale=False, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )

    def _fit(self, threshold=0.10):
        X_train = pd.DataFrame({"v": self.TRAIN_V})
        y_train = pd.Series([0.0] * len(self.TRAIN_V), name="y")
        pipe = self._pipe(threshold)
        pipe.fit_transform(X_train, y_train)
        return pipe

    @classmethod
    def _bounds(cls, threshold=0.10):
        s  = pd.Series(cls.TRAIN_V)
        lo = s.quantile(threshold)
        hi = s.quantile(1 - threshold)
        return lo, hi

    def test_upper_clipping_applied(self):
        _, hi = self._bounds()
        pipe  = self._fit()

        X_test  = pd.DataFrame({"v": [200.0]})
        X_out   = pipe.transform(X_test)

        self.assertAlmostEqual(
            X_out["v"].iloc[0], hi, places=5,
            msg=f"Giá trị 200 phải bị clip xuống hi={hi}"
        )

    def test_lower_clipping_applied(self):
        lo, _ = self._bounds()
        pipe  = self._fit()

        X_test  = pd.DataFrame({"v": [-999.0]})
        X_out   = pipe.transform(X_test)

        self.assertAlmostEqual(
            X_out["v"].iloc[0], lo, places=5,
            msg=f"Giá trị -999 phải bị clip lên lo={lo}"
        )

    def test_interior_values_unchanged(self):
        pipe    = self._fit()
        X_test  = pd.DataFrame({"v": [55.0]})
        X_out   = pipe.transform(X_test)

        self.assertAlmostEqual(
            X_out["v"].iloc[0], 55.0, places=5,
            msg="Giá trị nội bộ 55 bị thay đổi sai"
        )

    def test_fit_transform_clips_train(self):
        train_with_outlier = self.TRAIN_V + [999.0]
        X_train = pd.DataFrame({"v": train_with_outlier})
        y_train = pd.Series([0.0] * len(train_with_outlier), name="y")

        pipe    = self._pipe()
        X_out, _ = pipe.fit_transform(X_train, y_train)

        hi = pd.Series(train_with_outlier).quantile(0.90)
        self.assertLessEqual(
            X_out["v"].max(), hi + 1e-9,
            msg=f"Giá trị 999 trong train không bị clip xuống hi={hi}"
        )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 4 — ORDINAL ENCODING
# ══════════════════════════════════════════════════════════════════════
class TestOrdinalEncoding(unittest.TestCase):

    def _pipe_no_extras(self):
        return DataPipeline(
            outlier_method=None, scale=False, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False, encoding="auto",
        )

    def test_exter_qual_mapping(self):
        df = pd.DataFrame({
            "Exter Qual": ["Po", "TA", "Ex", "Gd", "Fa"],
            "num_col"   : [1.0,  2.0,  3.0,  4.0,  5.0],
        })
        y = pd.Series([0.0] * 5, name="y")

        pipe = self._pipe_no_extras()
        X_out, _ = pipe.fit_transform(df, y)

        expected = [1, 3, 5, 4, 2]
        actual   = X_out["Exter Qual"].tolist()
        self.assertEqual(actual, expected,
                         f"Ordinal mapping sai: {actual} ≠ {expected}")

    def test_ordinal_col_is_numeric(self):
        df = pd.DataFrame({
            "Exter Qual": ["TA", "Gd", "Ex"],
            "num_col"   : [10.0, 20.0, 30.0],
        })
        y = pd.Series([0.0] * 3, name="y")

        pipe = self._pipe_no_extras()
        X_out, _ = pipe.fit_transform(df, y)

        self.assertTrue(
            pd.api.types.is_numeric_dtype(X_out["Exter Qual"]),
            "Cột 'Exter Qual' phải là numeric sau ordinal encoding"
        )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 5 — ONE-HOT ENCODING (no leakage)
# ══════════════════════════════════════════════════════════════════════
class TestOneHotEncoding(unittest.TestCase):

    def _pipe(self):
        return DataPipeline(
            outlier_method=None, scale=False, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False, encoding="auto", drop_first=True,
        )

    def test_dummy_count_drop_first(self):
        df_train = pd.DataFrame({
            "Central Air": ["N", "Y", "N", "Y", "N", "Y"],
            "num_col"    : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        y_train = pd.Series([0.0] * 6, name="y")

        pipe = self._pipe()
        X_out, _ = pipe.fit_transform(df_train, y_train)

        dummy_cols = [c for c in X_out.columns if "Central Air" in c]
        self.assertEqual(len(dummy_cols), 1,
                         f"Phải có đúng 1 dummy col, có: {dummy_cols}")

    def test_unseen_category_filled_zero(self):
        df_train = pd.DataFrame({
            "Central Air": ["N", "Y", "N", "Y"],
            "num_col"    : [1.0, 2.0, 3.0, 4.0],
        })
        df_test = pd.DataFrame({
            "Central Air": ["M", "N"],
            "num_col"    : [5.0, 6.0],
        })
        y_train = pd.Series([0.0] * 4, name="y")

        pipe = self._pipe()
        X_tr_out, _ = pipe.fit_transform(df_train, y_train)
        X_te_out    = pipe.transform(df_test)

        self.assertEqual(
            sorted(X_tr_out.columns.tolist()),
            sorted(X_te_out.columns.tolist()),
            "Cột của train và test không khớp sau one-hot"
        )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 6 — drop_high_vif
# ══════════════════════════════════════════════════════════════════════
class TestDropHighVIF(unittest.TestCase):

    def _fitted_pipe(self, X: pd.DataFrame) -> DataPipeline:
        y = pd.Series([0.0] * len(X), name="y")
        pipe = DataPipeline(
            outlier_method=None, scale=False, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )
        pipe.fit_transform(X, y)
        return pipe

    def test_independent_cols_kept(self):
        X = pd.DataFrame({
            "a": [1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
            "b": [1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0],
            "c": [1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0],
            "d": [1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
        })
        pipe = self._fitted_pipe(X)
        X_reduced, dropped = pipe.drop_high_vif(X, threshold=10.0)

        self.assertEqual(len(dropped), 0,
                         f"Không nên loại cột nào, đã loại: {dropped}")
        self.assertEqual(X_reduced.shape[1], 4)

    def test_collinear_col_dropped(self):
        n  = 20
        x1 = [float(i) for i in range(n)]
        x2 = [float(2 * i + 1) for i in range(n)]
        x3 = [a + b for a, b in zip(x1, x2)]
        X  = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        pipe = self._fitted_pipe(X)
        X_reduced, dropped = pipe.drop_high_vif(X, threshold=10.0)

        self.assertGreater(len(dropped), 0,
                           "Phải loại ít nhất 1 cột do đa cộng tuyến")
        self.assertLess(X_reduced.shape[1], 3,
                        "Số cột sau drop phải nhỏ hơn 3")

    def test_vif_below_threshold_after_drop(self):
        x1 = [float(i) for i in range(1, 21)]
        x2 = [v + 0.01 * (i % 3 - 1) for i, v in enumerate(x1)]
        x3 = [float((i * 7 + 3) % 13) for i in range(20)]

        X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        pipe = self._fitted_pipe(X)

        threshold = 10.0
        X_reduced, _ = pipe.drop_high_vif(X, threshold=threshold)

        if X_reduced.shape[1] >= 2:
            vif_df  = run_vif_check(X_reduced)
            max_vif = vif_df["VIF"].replace([np.inf, -np.inf], np.nan).dropna().max()
            self.assertLessEqual(
                max_vif, threshold,
                f"VIF tối đa sau drop = {max_vif:.2f} > threshold {threshold}"
            )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 7 — log_target (SalePrice transform)
# ══════════════════════════════════════════════════════════════════════
class TestLogTarget(unittest.TestCase):

    def test_log_target_applied(self):
        y_in = pd.Series([100_000.0, 200_000.0, 300_000.0], name="SalePrice")
        X    = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

        pipe = DataPipeline(
            outlier_method=None, scale=False, log_target=True,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )
        _, y_out = pipe.fit_transform(X, y_in)

        expected = np.log1p(y_in.values)
        np.testing.assert_allclose(
            np.array(y_out), expected, rtol=1e-6,
            err_msg="log1p(y) không đúng khi log_target=True"
        )

    def test_no_log_target(self):
        y_in = pd.Series([1.0, 2.0, 3.0], name="y")
        X    = pd.DataFrame({"x": [10.0, 20.0, 30.0]})

        pipe = DataPipeline(
            outlier_method=None, scale=False, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )
        _, y_out = pipe.fit_transform(X, y_in)

        np.testing.assert_allclose(
            np.array(y_out), y_in.values, rtol=1e-9,
            err_msg="y bị biến đổi dù log_target=False"
        )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 8 — feature engineering
# ══════════════════════════════════════════════════════════════════════
class TestFeatureEngineering(unittest.TestCase):

    def _make_df(self):
        return pd.DataFrame({
            "Gr Liv Area"   : [1000.0, 1500.0, 2000.0],
            "Total Bsmt SF" : [ 500.0,  800.0, 1000.0],
            "Overall Qual"  : [   5.0,    7.0,    9.0],
        })

    def test_total_sqft_created(self):
        df = self._make_df()
        y  = pd.Series([0.0, 0.0, 0.0], name="y")

        pipe = DataPipeline(
            outlier_method=None, scale=False, log_target=False,
            engineer_features=True, log_skewed_features=False,
            add_interactions=False,
        )
        X_out, _ = pipe.fit_transform(df, y)
        self.assertIn("Total_SqFt", X_out.columns,
                      "Cột 'Total_SqFt' phải được tạo")

        expected = [1500.0, 2300.0, 3000.0]
        np.testing.assert_allclose(
            X_out["Total_SqFt"].values, expected, rtol=1e-6,
            err_msg="Total_SqFt ≠ Gr Liv Area + Total Bsmt SF"
        )

    def test_interaction_created(self):
        df = self._make_df()
        y  = pd.Series([0.0, 0.0, 0.0], name="y")

        pipe = DataPipeline(
            outlier_method=None, scale=False, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=True,
        )
        X_out, _ = pipe.fit_transform(df, y)
        self.assertIn("Qual_x_GrLivArea", X_out.columns,
                      "Cột 'Qual_x_GrLivArea' phải được tạo")

        expected = [5000.0, 10500.0, 18000.0]
        np.testing.assert_allclose(
            X_out["Qual_x_GrLivArea"].values, expected, rtol=1e-6,
            err_msg="Qual_x_GrLivArea ≠ Overall Qual × Gr Liv Area"
        )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 9 — LOG SKEWED FEATURES (MỚI)
# ══════════════════════════════════════════════════════════════════════
class TestLogSkewedFeatures(unittest.TestCase):

    def test_skewed_feature_logged(self):
        # Tạo data với:
        # - 1 cột lệch phải rất mạnh (skew > 0.75)
        # - 1 cột phân bố đều (skew thấp)
        # - 1 cột nhị phân (sẽ bị bỏ qua dù skew thế nào)
        df = pd.DataFrame({
            "skewed_col": [1.0, 1.5, 2.0, 2.5, 100.0],
            "normal_col": [10.0, 11.0, 10.5, 9.5, 10.0],
            "binary_col": [0.0, 1.0, 0.0, 1.0, 0.0]
        })
        y = pd.Series([0.0] * 5, name="y")

        pipe = DataPipeline(
            outlier_method=None, scale=False, log_target=False,
            engineer_features=False, log_skewed_features=True,
            add_interactions=False, encoding="auto"
        )

        X_out, _ = pipe.fit_transform(df, y)

        # skewed_col phải bị log1p
        expected_skewed = np.log1p([1.0, 1.5, 2.0, 2.5, 100.0])
        np.testing.assert_allclose(
            X_out["skewed_col"].values, expected_skewed, rtol=1e-6,
            err_msg="Cột lệch phải không được log1p đúng cách."
        )

        # normal_col và binary_col phải giữ nguyên
        np.testing.assert_allclose(
            X_out["normal_col"].values, [10.0, 11.0, 10.5, 9.5, 10.0], rtol=1e-6,
            err_msg="Cột phân bố bình thường bị biến đổi sai."
        )
        np.testing.assert_allclose(
            X_out["binary_col"].values, [0.0, 1.0, 0.0, 1.0, 0.0], rtol=1e-6,
            err_msg="Cột nhị phân bị biến đổi sai (đáng lẽ phải giữ nguyên)."
        )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 10 — INVERSE TRANSFORM Y (MỚI)
# ══════════════════════════════════════════════════════════════════════
class TestInverseTransformY(unittest.TestCase):

    def test_inverse_transform_correct(self):
        """Kiểm tra khôi phục lại giá trị y ban đầu qua np.expm1"""
        y_in = pd.Series([1000.0, 5000.0, 10000.0], name="SalePrice")
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

        pipe = DataPipeline(
            outlier_method=None, scale=False, log_target=True,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )
        _, y_log = pipe.fit_transform(X, y_in)

        # Dùng hàm inverse_transform_y để trả về USD
        y_inv = pipe.inverse_transform_y(y_log)

        np.testing.assert_allclose(
            y_inv, y_in.values, rtol=1e-6,
            err_msg="inverse_transform_y không khôi phục đúng giá gốc."
        )

    def test_inverse_transform_raises_when_not_fitted(self):
        """Gọi hàm khi pipeline chưa fit sẽ báo lỗi"""
        pipe = DataPipeline(log_target=True)
        with self.assertRaises(RuntimeError):
            pipe.inverse_transform_y(np.array([10.0]))

    def test_inverse_transform_raises_when_log_target_false(self):
        """Gọi hàm khi log_target=False (không áp dụng log) sẽ báo lỗi"""
        X = pd.DataFrame({"x": [1.0, 2.0]})
        y = pd.Series([100.0, 200.0])
        
        pipe = DataPipeline(log_target=False)
        pipe.fit_transform(X, y)
        
        with self.assertRaises(RuntimeError):
            pipe.inverse_transform_y(np.array([10.0]))

