"""
test_data_pipeline.py
=====================
Unit tests cho class DataPipeline (data_pipeline.py).

Mỗi method công khai có ít nhất 2 test case trên dữ liệu đã biết.
Toàn bộ dùng random_state cố định để reproducible.

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

SEED = 42
RNG  = np.random.default_rng(SEED)


# ══════════════════════════════════════════════════════════════════════
# FIXTURE — dữ liệu nhỏ dùng chung
# ══════════════════════════════════════════════════════════════════════
def _make_simple_df(n: int = 50, seed: int = SEED) -> pd.DataFrame:
    """
    DataFrame tối giản mô phỏng cấu trúc Ames Housing (sau clean_data),
    chỉ gồm các cột số — không có categorical — để test nhanh pipeline.
    """
    rng = np.random.default_rng(seed)
    df  = pd.DataFrame({
        "Gr Liv Area"   : rng.integers(600, 3000, n).astype(float),
        "Total Bsmt SF" : rng.integers(0, 2000, n).astype(float),
        "Overall Qual"  : rng.integers(1, 10, n).astype(float),
        "Age_At_Sale"   : rng.integers(0, 80, n).astype(float),
        "Garage Area"   : rng.integers(0, 900, n).astype(float),
    })
    return df


def _make_y(n: int = 50, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(100_000, 500_000, n).astype(float), name="SalePrice")


def _make_cat_df(n: int = 60, seed: int = SEED) -> pd.DataFrame:
    """DataFrame có cột categorical (ordinal + nominal) để test encoding."""
    rng = np.random.default_rng(seed)
    df  = _make_simple_df(n, seed)
    df["Exter Qual"]  = rng.choice(["Po", "Fa", "TA", "Gd", "Ex"], n)
    df["Central Air"] = rng.choice(["N", "Y"], n)          # nominal
    return df


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 1 — fit_transform / transform (no-leak contract)
# ══════════════════════════════════════════════════════════════════════
class TestFitTransform(unittest.TestCase):
    """
    fit_transform phải học tham số CHỈ từ train.
    transform trên test không được dùng thông tin ngoài train.
    """

    def setUp(self):
        df = _make_simple_df(n=80)
        self.X_train = df.iloc[:60].copy()
        self.X_test  = df.iloc[60:].copy()
        self.y_train = _make_y(n=80).iloc[:60]
        self.y_test  = _make_y(n=80).iloc[60:]

    # ── test 1: output shape hợp lệ ─────────────────────────────────
    def test_output_shape(self):
        pipe = DataPipeline(
            outlier_method=None, scale=True, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )
        X_tr_out, y_tr_out = pipe.fit_transform(self.X_train, self.y_train)
        X_te_out            = pipe.transform(self.X_test)

        # Số hàng giữ nguyên
        self.assertEqual(X_tr_out.shape[0], len(self.X_train))
        self.assertEqual(X_te_out.shape[0], len(self.X_test))

        # Số cột train == test (cùng feature space)
        self.assertEqual(X_tr_out.shape[1], X_te_out.shape[1])

        # y vẫn đủ dòng
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
    """
    Sau khi fit trên train:
      - mean ≈ 0, std ≈ 1 trên TẬP TRAIN (trong phạm vi sai số float).
      - Tham số scale học từ train (không dùng thông tin test).
    """

    def _make_pipe(self):
        return DataPipeline(
            outlier_method=None, scale=True, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )

    # ── test 1: mean ≈ 0 trên train ─────────────────────────────────
    def test_train_mean_near_zero(self):
        df = _make_simple_df(n=100)
        X_train, X_test = df.iloc[:80], df.iloc[80:]
        y = _make_y(n=100).iloc[:80]

        pipe = self._make_pipe()
        X_out, _ = pipe.fit_transform(X_train, y)

        for col in X_out.columns:
            self.assertAlmostEqual(
                X_out[col].mean(), 0.0, places=6,
                msg=f"mean của cột '{col}' ≠ 0 sau z-score"
            )

    # ── test 2: std ≈ 1 trên train ──────────────────────────────────
    def test_train_std_near_one(self):
        df = _make_simple_df(n=100)
        X_train = df.iloc[:80]
        y = _make_y(n=100).iloc[:80]

        pipe = self._make_pipe()
        X_out, _ = pipe.fit_transform(X_train, y)

        for col in X_out.columns:
            self.assertAlmostEqual(
                X_out[col].std(ddof=1), 1.0, places=5,
                msg=f"std của cột '{col}' ≠ 1 sau z-score"
            )

    # ── test 3: transform test dùng tham số từ train (no leakage) ───
    def test_test_uses_train_params(self):
        """
        Nếu test set có giá trị khác train, trung bình sau transform
        sẽ KHÔNG bằng 0 — đó là đúng (dùng μ, σ của train).
        Kiểm tra bằng cách so sánh với tính tay.
        """
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

        mu_train  = 3.0      # mean([1,2,3,4,5])
        std_train = float(np.std([1,2,3,4,5], ddof=1))   # ≈ 1.5811

        expected_10 = (10.0 - mu_train) / std_train
        expected_20 = (20.0 - mu_train) / std_train

        self.assertAlmostEqual(X_te_out["x"].iloc[0], expected_10, places=5)
        self.assertAlmostEqual(X_te_out["x"].iloc[1], expected_20, places=5)


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 3 — WINSORIZE OUTLIER
# ══════════════════════════════════════════════════════════════════════
class TestWinsorize(unittest.TestCase):
    """
    Sau winsorize(threshold=0.1):
      - Không có giá trị nào < quantile 10% hoặc > quantile 90% của train.
      - Các giá trị nội bộ không thay đổi.
    """

    def _simple_pipe(self, threshold=0.10):
        return DataPipeline(
            outlier_method="winsorize",
            outlier_threshold=threshold,
            scale=False, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )

    # ── test 1: cắt đuôi đúng ────────────────────────────────────────
    def test_clipping_applied(self):
        X_train = pd.DataFrame({"v": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
        y_train = pd.Series([0.0]*10, name="y")

        pipe = self._simple_pipe(threshold=0.10)
        X_out, _ = pipe.fit_transform(X_train, y_train)

        lo = X_train["v"].quantile(0.10)
        hi = X_train["v"].quantile(0.90)
        self.assertTrue((X_out["v"] >= lo).all(), "Có giá trị dưới ngưỡng lo")
        self.assertTrue((X_out["v"] <= hi).all(), "Có giá trị trên ngưỡng hi")

    # ── test 2: giá trị nội bộ không đổi ─────────────────────────────
    def test_interior_values_unchanged(self):
        X_train = pd.DataFrame({"v": [10.0, 20.0, 30.0, 40.0, 50.0,
                                      60.0, 70.0, 80.0, 90.0, 100.0]})
        y_train = pd.Series([0.0]*10, name="y")

        pipe = self._simple_pipe(threshold=0.10)
        X_out, _ = pipe.fit_transform(X_train, y_train)

        # Giá trị giữa (index 1..8) phải không đổi
        for i in range(1, 9):
            self.assertAlmostEqual(
                X_out["v"].iloc[i], X_train["v"].iloc[i], places=6,
                msg=f"Giá trị nội bộ tại index {i} bị thay đổi"
            )

    # ── test 3: test set cũng bị clip theo giới hạn của train ────────
    def test_test_clipped_by_train_limits(self):
        X_train = pd.DataFrame({"v": list(range(1, 11, 1), )})  # 1..10
        X_test  = pd.DataFrame({"v": [0.0, 5.0, 999.0]})
        y_train = pd.Series([0.0]*10, name="y")

        pipe = self._simple_pipe(threshold=0.10)
        pipe.fit_transform(X_train, y_train)
        X_te_out = pipe.transform(X_test)

        lo = X_train["v"].quantile(0.10)
        hi = X_train["v"].quantile(0.90)

        self.assertGreaterEqual(X_te_out["v"].iloc[0], lo,
                                "Giá trị 0.0 không được clip lên lo")
        self.assertAlmostEqual(X_te_out["v"].iloc[1], 5.0, places=5,
                               msg="Giá trị nội bộ 5.0 bị thay đổi sai")
        self.assertLessEqual(X_te_out["v"].iloc[2], hi,
                             "Giá trị 999 không được clip xuống hi")


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 4 — ORDINAL ENCODING
# ══════════════════════════════════════════════════════════════════════
class TestOrdinalEncoding(unittest.TestCase):
    """
    Kiểm tra ordinal encoding theo ORDINAL_MAPS:
      - Giá trị ánh xạ đúng thứ tự số học đã định nghĩa.
      - Cột ordinal biến mất khỏi output (đã được replace bằng số).
    """

    def _pipe_no_extras(self):
        return DataPipeline(
            outlier_method=None, scale=False, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False, encoding="auto",
        )

    # ── test 1: ánh xạ đúng theo ORDINAL_MAPS ────────────────────────
    def test_exter_qual_mapping(self):
        """'Exter Qual': Po=1, Fa=2, TA=3, Gd=4, Ex=5"""
        df = pd.DataFrame({
            "Exter Qual" : ["Po", "TA", "Ex", "Gd", "Fa"],
            "num_col"    : [1.0,  2.0,  3.0,  4.0,  5.0],
        })
        y = pd.Series([0.0]*5, name="y")

        pipe = self._pipe_no_extras()
        X_out, _ = pipe.fit_transform(df, y)

        expected = [1, 3, 5, 4, 2]
        actual   = X_out["Exter Qual"].tolist()
        self.assertEqual(actual, expected,
                         f"Ordinal mapping sai: {actual} ≠ {expected}")

    # ── test 2: cột vẫn là số (dtype numeric) sau encoding ───────────
    def test_ordinal_col_is_numeric(self):
        df = pd.DataFrame({
            "Exter Qual": ["TA", "Gd", "Ex"],
            "num_col"   : [10.0, 20.0, 30.0],
        })
        y = pd.Series([0.0]*3, name="y")

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
    """
    Kiểm tra one-hot encoding:
      - Số cột dummy đúng (n_categories - 1 nếu drop_first=True).
      - Test set có đúng cột như train (reindex, fill=0 với category mới).
    """

    def _make_central_air_df(self, vals, n_num=3):
        rng = np.random.default_rng(SEED)
        df  = pd.DataFrame({
            "Central Air": vals,
            "num_col"    : rng.standard_normal(len(vals)),
        })
        return df

    def _pipe(self):
        return DataPipeline(
            outlier_method=None, scale=False, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False, encoding="auto", drop_first=True,
        )

    # ── test 1: số cột dummy = n_cat - 1 (drop_first=True) ───────────
    def test_dummy_count_drop_first(self):
        """Central Air có 2 giá trị N/Y → drop_first → 1 dummy column"""
        df_train = self._make_central_air_df(["N","Y","N","Y","N","Y"])
        y_train  = pd.Series([0.0]*6, name="y")

        pipe = self._pipe()
        X_out, _ = pipe.fit_transform(df_train, y_train)

        # Số cột dummy của Central Air phải là 1
        dummy_cols = [c for c in X_out.columns if "Central Air" in c]
        self.assertEqual(len(dummy_cols), 1,
                         f"Phải có đúng 1 dummy col, có: {dummy_cols}")

    # ── test 2: test set gặp category chưa thấy → fill_value=0 ───────
    def test_unseen_category_filled_zero(self):
        """
        Train chỉ thấy ['N', 'Y'].
        Test xuất hiện category lạ 'M' (không có trong NOMINAL_COLS map).
        Phải reindex về cột của train, giá trị lạ → 0.
        """
        df_train = self._make_central_air_df(["N","Y","N","Y"])
        df_test  = self._make_central_air_df(["M","N"])   # 'M' là mới
        y_train  = pd.Series([0.0]*4, name="y")

        pipe = self._pipe()
        X_tr_out, _ = pipe.fit_transform(df_train, y_train)
        X_te_out     = pipe.transform(df_test)

        # Test set phải có đúng cột như train
        self.assertEqual(
            sorted(X_tr_out.columns.tolist()),
            sorted(X_te_out.columns.tolist()),
            "Cột của train và test không khớp sau one-hot"
        )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 6 — drop_high_vif
# ══════════════════════════════════════════════════════════════════════
class TestDropHighVIF(unittest.TestCase):
    """
    drop_high_vif phải:
      - Loại cột có VIF cao nhất trước (iterative).
      - Sau khi loại, VIF của tất cả cột còn lại ≤ threshold.
      - Cột hoàn toàn không tương quan giữ lại toàn bộ.
    """

    def _make_pipe(self):
        return DataPipeline(
            outlier_method=None, scale=False, log_target=False,
            engineer_features=False, log_skewed_features=False,
            add_interactions=False,
        )

    def _fitted_pipe(self, X: pd.DataFrame) -> DataPipeline:
        """Fit pipe đơn giản để dùng drop_high_vif."""
        y = pd.Series(np.zeros(len(X)), name="y")
        pipe = self._make_pipe()
        pipe.fit_transform(X, y)
        return pipe

    # ── test 1: cột độc lập hoàn toàn → không bị loại ───────────────
    def test_independent_cols_kept(self):
        """4 cột hoàn toàn độc lập → VIF thấp → không cột nào bị loại."""
        rng = np.random.default_rng(SEED)
        X   = pd.DataFrame(rng.standard_normal((200, 4)),
                            columns=["a", "b", "c", "d"])
        pipe = self._fitted_pipe(X)
        X_reduced, dropped = pipe.drop_high_vif(X, threshold=10.0)

        self.assertEqual(len(dropped), 0,
                         f"Không nên loại cột nào, đã loại: {dropped}")
        self.assertEqual(X_reduced.shape[1], 4)

    # ── test 2: cột hoàn toàn phụ thuộc → bị loại ───────────────────
    def test_collinear_col_dropped(self):
        """
        Tạo x3 = x1 + x2 (perfect collinearity) → x3 phải bị loại.
        """
        rng = np.random.default_rng(SEED)
        x1  = rng.standard_normal(200)
        x2  = rng.standard_normal(200)
        x3  = x1 + x2                    # hoàn toàn phụ thuộc
        X   = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        pipe = self._fitted_pipe(X)
        X_reduced, dropped = pipe.drop_high_vif(X, threshold=10.0)

        self.assertGreater(len(dropped), 0,
                           "Phải loại ít nhất 1 cột do đa cộng tuyến")
        self.assertIn("x3", dropped,
                      "Cột x3 (perfect collinear) phải bị loại")

    # ── test 3: sau drop, VIF của cột còn lại ≤ threshold ────────────
    def test_vif_below_threshold_after_drop(self):
        from part2.data_pipeline import run_vif_check
        rng = np.random.default_rng(SEED)
        x1  = rng.standard_normal(300)
        x2  = 0.99 * x1 + 0.01 * rng.standard_normal(300)  # gần tuyến tính
        x3  = rng.standard_normal(300)
        X   = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        pipe = self._fitted_pipe(X)

        threshold = 10.0
        X_reduced, _ = pipe.drop_high_vif(X, threshold=threshold)

        if X_reduced.shape[1] >= 2:
            vif_df = run_vif_check(X_reduced)
            max_vif = vif_df["VIF"].replace([np.inf, -np.inf], np.nan).dropna().max()
            self.assertLessEqual(
                max_vif, threshold,
                f"VIF tối đa sau drop = {max_vif:.2f} > threshold {threshold}"
            )


# ══════════════════════════════════════════════════════════════════════
# TEST SUITE 7 — log_target (SalePrice transform)
# ══════════════════════════════════════════════════════════════════════
class TestLogTarget(unittest.TestCase):
    """
    Khi log_target=True: y_out = log1p(y_in).
    Khi log_target=False: y_out = y_in (không đổi).
    """

    # ── test 1: log_target=True → y_out ≈ log1p(y_in) ───────────────
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

    # ── test 2: log_target=False → y_out không thay đổi ─────────────
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
    """
    Khi engineer_features=True:
      - 'Total_SqFt' = 'Gr Liv Area' + 'Total Bsmt SF' phải xuất hiện.
    Khi add_interactions=True:
      - 'Qual_x_GrLivArea' = 'Overall Qual' × 'Gr Liv Area' phải đúng giá trị.
    """

    def _make_df(self):
        return pd.DataFrame({
            "Gr Liv Area"   : [1000.0, 1500.0, 2000.0],
            "Total Bsmt SF" : [500.0,  800.0,  1000.0],
            "Overall Qual"  : [5.0,    7.0,    9.0],
        })

    # ── test 1: Total_SqFt được tạo đúng ────────────────────────────
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

        expected = (df["Gr Liv Area"] + df["Total Bsmt SF"]).values
        np.testing.assert_allclose(
            X_out["Total_SqFt"].values, expected, rtol=1e-6,
            err_msg="Total_SqFt ≠ Gr Liv Area + Total Bsmt SF"
        )

    # ── test 2: Qual_x_GrLivArea được tạo đúng ──────────────────────
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

        expected = (df["Overall Qual"] * df["Gr Liv Area"]).values
        np.testing.assert_allclose(
            X_out["Qual_x_GrLivArea"].values, expected, rtol=1e-6,
            err_msg="Qual_x_GrLivArea ≠ Overall Qual × Gr Liv Area"
        )

