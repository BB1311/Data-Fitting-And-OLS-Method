import numpy as np
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from part1.ols_implementation import vif

"""
data_pipeline.py
================
Pipeline tiền xử lý dữ liệu cho Đồ án 2 — Phần 2.

Quy trình (sau khi đã chạy clean_data.py):
  1. Tách target (SalePrice) khỏi features
  2. Phát hiện và xử lý outlier (Winsorize hoặc loại bỏ)
  3. Encoding biến phân loại:
       - Ordinal encoding (các cột có thứ bậc rõ ràng)
       - One-hot encoding (các cột danh nghĩa)
  4. Chuẩn hóa biến liên tục (z-score standardization)
  5. VIF check (phát hiện đa cộng tuyến)

Usage
-----
    from data_pipeline import DataPipeline, run_vif_check

    pipe = DataPipeline(
        outlier_method='winsorize',   # hoặc 'remove'
        outlier_threshold=0.05,       # winsorize: cắt 5% mỗi đầu; remove: IQR*1.5
        encoding='auto',              # 'auto' | 'onehot_only' | 'ordinal_only'
        scale=True,
    )

    X_train_clean, y_train_clean = pipe.fit_transform(X_train, y_train)
    X_test_clean  = pipe.transform(X_test)

    vif_df = run_vif_check(X_train_clean)
    X_train_no_mc, dropped = pipe.drop_high_vif(X_train_clean, threshold=10)
"""


# ======================================================================
# HÀM CẦU NỐI: VIF CHECK
# ======================================================================
def run_vif_check(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cầu nối: Chuyển DataFrame thành Numpy array để đưa vào hàm vif (Part 1), 
    sau đó trả về DataFrame chứa tên cột và điểm VIF tương ứng, sắp xếp giảm dần.
    """
    # 1. Chuyển DataFrame sang Numpy array
    X_array = X.values
    
    # 2. Gọi hàm vif từ Part 1
    vif_scores = vif(X_array, verbose=False)
    
    # 3. Tạo DataFrame kết quả kết hợp giữa Tên cột và Điểm VIF
    vif_df = pd.DataFrame({
        'feature': X.columns,
        'VIF': vif_scores
    })
    
    # 4. Sắp xếp giảm dần theo VIF để df.iloc[0] luôn là cột có VIF cao nhất
    vif_df = vif_df.sort_values(by='VIF', ascending=False).reset_index(drop=True)
    
    return vif_df
# ══════════════════════════════════════════════════════════════════════
# 0. MAPPING ORDINAL — tùy chỉnh theo Ames Housing Data Dictionary
# ══════════════════════════════════════════════════════════════════════
ORDINAL_MAPS = {
    # Chất lượng / tình trạng chung
    'Overall Qual':   {1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10},  # giữ nguyên
    'Overall Cond':   {1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10},

    # Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0
    'Exter Qual':     {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'Exter Cond':     {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'Bsmt Qual':      {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'Bsmt Cond':      {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'Heating QC':     {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'Kitchen Qual':   {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'Fireplace Qu':   {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'Garage Qual':    {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'Garage Cond':    {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'Pool QC':        {'None':0,'Fa':1,'TA':2,'Gd':3,'Ex':4},

    # Exposure
    'Bsmt Exposure':  {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4},

    # Finish types
    'BsmtFin Type 1': {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6},
    'BsmtFin Type 2': {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6},
    'Garage Finish':  {'None':0,'Unf':1,'RFn':2,'Fin':3},

    # Slope
    'Land Slope':     {'Gtl':0,'Mod':1,'Sev':2},

    # Lot Shape
    'Lot Shape':      {'IR3':0,'IR2':1,'IR1':2,'Reg':3},

    # Paved Drive
    'Paved Drive':    {'N':0,'P':1,'Y':2},

    # Fence (nếu chưa drop — ở đây đã drop, để sẵn)
    'Fence':          {'None':0,'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4},

    # Functional
    'Functional':     {'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7},
}

# Các cột danh nghĩa (không có thứ bậc) → One-hot encoding
NOMINAL_COLS = [
    'MS SubClass', 'MS Zoning', 'Lot Config', 'Neighborhood',
    'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl',
    'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
    'Foundation', 'Heating', 'Central Air',
    'Electrical', 'Garage Type',
    'Sale Type', 'Sale Condition',
    'Land Contour','Mo Sold'
]


# ══════════════════════════════════════════════════════════════════════
# CLASS DataPipeline
# ══════════════════════════════════════════════════════════════════════
class DataPipeline:
    """
    Pipeline tiền xử lý dữ liệu cho Phần 2.

    Các bước (theo thứ tự fit → transform):
      (A) Outlier detection & treatment
      (B) Ordinal encoding
      (C) One-hot encoding
      (D) Z-score standardization (chỉ các cột số)

    Tham số
    -------
    outlier_method : 'winsorize' | 'remove' | None
        - 'winsorize': Winsorize target và các cột số liên tục
          (giới hạn 2 đầu theo outlier_threshold).
        - 'remove'   : Xóa hàng nếu bất kỳ biến số liên tục nào nằm
          ngoài [Q1 - iqr_factor*IQR, Q3 + iqr_factor*IQR].
        - None       : Bỏ qua bước này.
    outlier_threshold : float (default 0.05)
        Chỉ dùng với 'winsorize': cắt outlier_threshold ở mỗi đuôi.
    iqr_factor : float (default 1.5)
        Chỉ dùng với 'remove': hệ số nhân IQR.
    outlier_cols : list | None
        Nếu None, tự động lấy tất cả cột float64 liên tục.
        Truyền list để chỉ định rõ cột nào cần xử lý outlier.
    encoding : 'auto' | 'onehot_only' | 'ordinal_only'
        'auto'          : ordinal với cột trong ORDINAL_MAPS, one-hot với NOMINAL_COLS.
        'onehot_only'   : one-hot toàn bộ categorical.
        'ordinal_only'  : ordinal toàn bộ (dùng mã số 0,1,2…).
    scale : bool (default True)
        Chuẩn hóa z-score các cột số.
    drop_first : bool (default True)
        Loại bỏ cột đầu tiên trong one-hot để tránh bẫy biến giả.
    """

    def __init__(
        self,
        outlier_method: str | None = 'winsorize',
        outlier_threshold: float = 0.05,
        iqr_factor: float = 1.5,
        outlier_cols: list | None = None,
        encoding: str = 'auto',
        scale: bool = True,
        drop_first: bool = True,
        # --- CÁC THAM SỐ MỚI THÊM VÀO ---
        log_target: bool = True,
        engineer_features: bool = True,
        log_skewed_features: bool = True,
        add_interactions: bool = True,
    ):
        self.outlier_method    = outlier_method
        self.outlier_threshold = outlier_threshold
        self.iqr_factor        = iqr_factor
        self.outlier_cols      = outlier_cols
        self.encoding          = encoding
        self.scale             = scale
        self.drop_first        = drop_first

        self.log_target          = log_target
        self.engineer_features   = engineer_features
        self.log_skewed_features = log_skewed_features
        self.add_interactions    = add_interactions

        # --- CÁC BIẾN LƯU TRỮ MỚI THÊM VÀO ---
        self._winsor_limits: dict  = {}
        self._iqr_bounds: dict     = {}
        self._scale_params: dict   = {}
        self._ordinal_medians: dict = {}
        self._skewed_cols_to_log: list = []
        self._onehot_cols_: list   = []
        self._ordinal_cols_: list  = []
        self._dummy_cols_: list    = []
        self._feature_names_: list = []
        self._ordinal_cat_order: dict = {}
        self._fitted = False


    # ------------------------------------------------------------------
    # PHẦN MỚI: FEATURE ENGINEERING & NON-LINEAR TRANSFORMS
    # ------------------------------------------------------------------
    def _create_new_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Tạo các biến tổng hợp (Những biến chưa tạo ở bước Clean Data)"""
        if not self.engineer_features: return X
        X = X.copy()
        if 'Gr Liv Area' in X.columns and 'Total Bsmt SF' in X.columns:
            X['Total_SqFt'] = X['Gr Liv Area'] + X['Total Bsmt SF']
        return X

    def _create_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Tạo các biến tương tác thủ công (Tránh bùng nổ chiều dữ liệu)"""
        if not self.add_interactions: return X
        X = X.copy()
        if 'Overall Qual' in X.columns and 'Gr Liv Area' in X.columns:
            X['Qual_x_GrLivArea'] = X['Overall Qual'] * X['Gr Liv Area']
        return X

    def _fit_log_features(self, X: pd.DataFrame):
        """Quét tìm các cột số bị lệch (skew > 0.75) trên tập Train"""
        if not self.log_skewed_features: return
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        valid_cols = [c for c in num_cols if X[c].min() >= 0]
        skewness = X[valid_cols].skew()

        self._skewed_cols_to_log = skewness[skewness > 0.75].index.tolist()

        binary_cols = [c for c in valid_cols if X[c].nunique() <= 2]
        self._skewed_cols_to_log = [
            c for c in self._skewed_cols_to_log
            if c not in self._ordinal_cols_
            and c not in binary_cols
        ]

    def _apply_log_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ép hàm log1p lên các cột đã phát hiện"""
        if not self.log_skewed_features: return X
        X = X.copy()
        for col in self._skewed_cols_to_log:
            if col in X.columns:
                X[col] = np.log1p(X[col])
        return X

    def compute_ols_pvalues(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Tính p-value từ OLS cho từng feature, trả về Series index = feature name.
        Dùng nội bộ cho drop_high_vif, không cần import model_comparison.
        """
        from scipy import stats

        X_arr = X.to_numpy()
        y_arr = y.to_numpy()
        n, k = X_arr.shape

        # Thêm intercept
        X_c = np.column_stack([np.ones(n), X_arr])

        # OLS: beta = (X'X)^-1 X'y
        XtX = X_c.T @ X_c
        Xty = X_c.T @ y_arr
        beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        # Residuals và sigma^2
        residuals = y_arr - X_c @ beta
        sigma2 = (residuals @ residuals) / (n - k - 1)

        # Standard errors
        cov = sigma2 * np.linalg.pinv(XtX)
        se = np.sqrt(np.diag(cov))

        # t-stats và p-values (two-tailed)
        t_stats = beta / np.where(se > 0, se, np.nan)
        p_values = 2 * stats.t.sf(np.abs(t_stats), df=n - k - 1)

        # Bỏ intercept (index 0), chỉ trả về features
        return pd.Series(p_values[1:], index=X.columns)
    

    # ------------------------------------------------------------------
    # PHẦN A: Outlier
    # ------------------------------------------------------------------
    def _detect_outlier_cols(self, X: pd.DataFrame) -> list:
        """Tự động chọn các cột số liên tục để xử lý outlier."""
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        continuous = [
            c for c in num_cols
            if X[c].nunique() > 10 and c not in ('Mo Sold',)
        ]
        return continuous

    def _fit_outlier(self, X: pd.DataFrame, y: pd.Series | None = None):
        cols = self.outlier_cols if self.outlier_cols else self._detect_outlier_cols(X)
        self._outlier_cols_fitted = cols

        if self.outlier_method == 'winsorize':
            lo = self.outlier_threshold
            hi = 1 - self.outlier_threshold
            for col in cols:
                vals = X[col].dropna()
                self._winsor_limits[col] = (vals.quantile(lo), vals.quantile(hi))
            if y is not None:
                vals_y = y.dropna()
                self._winsor_limits['__target__'] = (
                    vals_y.quantile(lo), vals_y.quantile(hi)
                )

        elif self.outlier_method == 'remove':
            for col in cols:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self._iqr_bounds[col] = (
                    q1 - self.iqr_factor * iqr,
                    q3 + self.iqr_factor * iqr,
                )

    def _transform_outlier_X(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.outlier_method == 'winsorize':
            for col, (lo, hi) in self._winsor_limits.items():
                if col == '__target__' or col not in X.columns:
                    continue
                X[col] = X[col].clip(lower=lo, upper=hi)

        elif self.outlier_method == 'remove':
            for col, (lb, ub) in self._iqr_bounds.items():
                if col in X.columns:
                    X[col] = X[col].clip(lower=lb, upper=ub)
        return X

    def _remove_outlier_rows(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Xóa hàng outlier (chỉ dùng trên train set với method='remove')."""
        if self.outlier_method != 'remove':
            return X, y
        mask = pd.Series(True, index=X.index)
        for col, (lb, ub) in self._iqr_bounds.items():
            if col in X.columns:
                mask &= (X[col] >= lb) & (X[col] <= ub)
        n_removed = (~mask).sum()
        if n_removed > 0:
            print(f"  [Outlier-remove] Đã xóa {n_removed} hàng outlier trên train set.")
        return X[mask].copy(), y[mask].copy()

    # ------------------------------------------------------------------
    # PHẦN B: Encoding
    # ------------------------------------------------------------------
    def _fit_encoding(self, X: pd.DataFrame):
        cat_cols = X.select_dtypes(include=['object', 'str', 'category']).columns.tolist()

        if self.encoding == 'auto':
            self._ordinal_cols_ = [c for c in ORDINAL_MAPS if c in cat_cols]
            remaining = [c for c in cat_cols if c not in self._ordinal_cols_]
            self._onehot_cols_ = [c for c in NOMINAL_COLS if c in remaining]
            uncategorized = [c for c in remaining if c not in self._onehot_cols_]
            if uncategorized:
                print(f"  [Encoding] {len(uncategorized)} cột tự động one-hot: {uncategorized}")
                self._onehot_cols_ += uncategorized

        elif self.encoding == 'onehot_only':
            self._ordinal_cols_ = []
            self._onehot_cols_ = cat_cols
        elif self.encoding == 'ordinal_only':
            self._ordinal_cols_ = cat_cols
            self._onehot_cols_ = []
        else:
            raise ValueError(f"encoding phải là 'auto', 'onehot_only', hoặc 'ordinal_only'.")

        # --- FIX LEAKAGE: Lưu median của tập Train ---
        for col in self._ordinal_cols_:
            if col in X.columns and col in ORDINAL_MAPS:
                mapped = X[col].map(ORDINAL_MAPS[col])
                self._ordinal_medians[col] = mapped.median()

        for col in self._ordinal_cols_:
            if col in X.columns and col not in ORDINAL_MAPS:
                self._ordinal_cat_order[col] = list(pd.Categorical(X[col]).categories)

    def _apply_ordinal(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self._ordinal_cols_:
            if col not in X.columns:
                continue
            if col in ORDINAL_MAPS:
                X[col] = X[col].map(ORDINAL_MAPS[col])
                if X[col].isna().any():
                    med_val = self._ordinal_medians.get(col, 0)
                    X[col] = X[col].fillna(med_val)
            else:
                if col in self._ordinal_cat_order:
                    cat = pd.Categorical(X[col], categories=self._ordinal_cat_order[col])
                    X[col] = cat.codes
                else:
                    X[col] = pd.Categorical(X[col]).codes
        return X

    def _fit_onehot(self, X: pd.DataFrame):
        """Ghi nhớ tên cột dummy từ train set."""
        if not self._onehot_cols_:
            self._dummy_cols_ = []
            return
        X_sub = X[[c for c in self._onehot_cols_ if c in X.columns]]
        dummies = pd.get_dummies(X_sub, drop_first=self.drop_first, dtype=int)
        self._dummy_cols_ = dummies.columns.tolist()

    def _apply_onehot(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._onehot_cols_:
            return X
        cols_present = [c for c in self._onehot_cols_ if c in X.columns]
        X_sub = X[cols_present]
        dummies = pd.get_dummies(X_sub, drop_first=self.drop_first, dtype=int)

        # Đảm bảo test set có đúng cột như train set
        dummies = dummies.reindex(columns=self._dummy_cols_, fill_value=0)

        X = X.drop(columns=cols_present)
        X = pd.concat([X, dummies], axis=1)
        return X

    # ------------------------------------------------------------------
    # PHẦN C: Standardization
    # ------------------------------------------------------------------
    def _fit_scale(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            mu  = X[col].mean()
            std = X[col].std(ddof=1)
            self._scale_params[col] = (mu, std if std > 1e-8 else 1.0)

    def _apply_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (mu, std) in self._scale_params.items():
            if col in X.columns:
                X[col] = (X[col] - mu) / std
        return X

    # ------------------------------------------------------------------
    # PHẦN D: _fit_core (dùng chung bởi fit và fit_transform)
    # ------------------------------------------------------------------
    def _fit_core(self, X_temp: pd.DataFrame):
        """Học encoding + log + scale trên X đã qua outlier transform."""
        print("  [Encoding] Phân loại cột categorical...")
        self._fit_encoding(X_temp)
        X_temp = self._apply_ordinal(X_temp)
        X_temp = self._create_interactions(X_temp)
        self._fit_log_features(X_temp)
        X_temp = self._apply_log_features(X_temp)
        self._fit_onehot(X_temp)
        if self.scale:
            X_enc = self._apply_onehot(X_temp)
            print("  [Scale] Học tham số z-score...")
            self._fit_scale(X_enc)

    # ------------------------------------------------------------------
    # PHẦN E: fit / transform / fit_transform
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> 'DataPipeline':
        """
        Học tham số pipeline trên tập train.
        Dùng cùng flow với fit_transform để đảm bảo tương đương:
          - Apply outlier TRƯỚC khi học encoding/scale.
        """
        print("=== DataPipeline.fit() ===")
        _y = np.log1p(y) if (self.log_target and y is not None) else y

        X_temp = self._create_new_features(X)

        # Bước 1: học outlier params
        if self.outlier_method:
            print(f"  [Outlier] Học tham số outlier (method='{self.outlier_method}')...")
            self._fit_outlier(X_temp, _y)

        # Bước 2: apply outlier TRƯỚC khi học encoding/scale (đồng nhất với fit_transform)
        if self.outlier_method == 'remove' and _y is not None:
            X_temp, _y = self._remove_outlier_rows(X_temp, _y)
        elif self.outlier_method == 'winsorize':
            X_temp = self._transform_outlier_X(X_temp)

        # Bước 3: học encoding/scale trên X đã xử lý outlier
        self._fit_core(X_temp)
        self._fitted = True
        print("  fit() hoàn tất.\n")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted: raise RuntimeError("Gọi fit() trước khi transform().")

        X = self._create_new_features(X)
        X = self._transform_outlier_X(X)
        X = self._apply_ordinal(X)
        X = self._create_interactions(X)
        X = self._apply_log_features(X)
        X = self._apply_onehot(X)
        if self.scale:
            X = self._apply_scale(X)

        self._feature_names_ = X.columns.tolist()
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Học và transform trên tập train.
        Trả về tuple (X_out, y) — không phải chỉ X_out.
        """
        if self.log_target and y is not None:
            print("  [Target] Áp dụng np.log1p(y)")
            y = np.log1p(y)

        X_temp = self._create_new_features(X)

        # Bước 1: học outlier params
        if self.outlier_method:
            print(f"  [Outlier] Học tham số outlier (method='{self.outlier_method}')...")
            self._fit_outlier(X_temp, y)

        # Bước 2: apply outlier TRƯỚC khi học encoding/scale
        if self.outlier_method == 'remove' and y is not None:
            X_temp, y = self._remove_outlier_rows(X_temp, y)
        elif self.outlier_method == 'winsorize':
            X_temp = self._transform_outlier_X(X_temp)
            if y is not None and '__target__' in self._winsor_limits:
                lo, hi = self._winsor_limits['__target__']
                y = y.clip(lower=lo, upper=hi)

        # Bước 3: học encoding/scale trên X đã winsorize
        self._fit_core(X_temp)
        self._fitted = True

        # Bước 4: transform
        X_out = self._apply_ordinal(X_temp)
        X_out = self._create_interactions(X_out)
        X_out = self._apply_log_features(X_out)
        X_out = self._apply_onehot(X_out)
        if self.scale:
            X_out = self._apply_scale(X_out)

        self._feature_names_ = X_out.columns.tolist()
        print(f"  Shape sau pipeline: {X_out.shape}")
        return X_out, y

    # ------------------------------------------------------------------
    # PHẦN F: VIF — loại cột đa cộng tuyến
    # ------------------------------------------------------------------

    def drop_high_vif(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 10.0,
        max_iter: int = None,  # mặc định = số cột, không hardcode 20
    ) -> tuple[pd.DataFrame, list]:
        """
        Lặp loại bỏ biến kém ý nghĩa thống kê nhất trong tập vi phạm VIF.
        
        Thứ tự ưu tiên loại:
        1. p-value lớn nhất (kém ý nghĩa nhất)
        2. |corr với y| nhỏ nhất (tiebreaker)
        
        Chỉ xét trong tập biến đang vi phạm (VIF > threshold hoặc VIF = inf).
        """
        if max_iter is None:
            max_iter = X.shape[1]  # không bao giờ loại nhiều hơn số cột ban đầu

        dropped = []
        y_aligned = y.loc[X.index]

        for iteration in range(max_iter):
            # ── 1. Tính VIF ────────────────────────────────────────────────
            # Đảm bảo run_vif_check đã được định nghĩa trong class hoặc import sẵn
            vif_df = run_vif_check(X) 

            # ── 2. Tìm max VIF đúng cách (không dùng iloc[0] khi chưa sort) ──
            max_vif = vif_df['VIF'].replace([np.inf, -np.inf], np.nan).max()
            if pd.isna(max_vif) or max_vif <= threshold:
                break  # tất cả VIF đã an toàn

            # ── 3. Chỉ lấy tập vi phạm để xét loại ───────────────────────
            candidates = vif_df[
                (vif_df['VIF'] > threshold) | np.isinf(vif_df['VIF'])
            ].copy()

            # ── 4. p-value từ OLS nội bộ (chỉ tính trên candidates) ──────
            candidates['p_value'] = 1.0
            try:
                # Gọi hàm tính p-values thay vì ols_coefficient_table
                p_map = self.compute_ols_pvalues(X, y_aligned)
                candidates['p_value'] = candidates['feature'].map(p_map).fillna(1.0)
            except Exception:
                pass  # ma trận suy biến → giữ p_value=1.0, OLS không chạy được

            # ── 5. |corr với y| làm tiebreaker ────────────────────────────
            correlations = X[candidates['feature']].corrwith(y_aligned).abs()
            candidates['abs_corr'] = candidates['feature'].map(correlations).fillna(0.0)

            # ── 6. Sort 2 tầng trong tập vi phạm ──────────────────────────
            # Tầng 1: p_value giảm dần → kém ý nghĩa nhất lên đầu
            # Tầng 2: abs_corr tăng dần → ít liên quan y nhất lên đầu (tiebreaker)
            candidates = candidates.sort_values(
                by=['p_value', 'abs_corr'],
                ascending=[False, True]
            )

            worst = candidates.iloc[0]

            print(
                f"  [VIF iter {iteration+1}] Loại '{worst['feature']}' "
                f"(VIF={worst['VIF']:.2f}, p={worst['p_value']:.4f}, |corr|={worst['abs_corr']:.4f})"
            )

            X = X.drop(columns=[worst['feature']])
            dropped.append(worst['feature'])

        print(f"\n[VIF] Đã loại {len(dropped)} biến kém ý nghĩa: {dropped}")
        return X, dropped

    # ------------------------------------------------------------------
    # PHẦN G: Inverse transform target (log-space --> giá gốc)
    # ------------------------------------------------------------------
    def inverse_transform_y(self, y_log: np.ndarray | pd.Series) -> np.ndarray:
        """
        Chuyển prediction từ log-space về không gian giá gốc (USD).

        Dùng sau khi mô hình predict trên y đã log1p để tính
        MAE/RMSE/R² có ý nghĩa thực tế với stakeholder.

        Parameters
        ----------
        y_log : array-like
            Giá trị dự đoán (hoặc y_test) đang ở log-scale
            (kết quả của np.log1p áp dụng lên SalePrice gốc).

        Returns
        -------
        y_original : np.ndarray
            Giá trị đã được inverse (np.expm1), đơn vị USD.

        Raises
        ------
        RuntimeError
            Nếu pipeline chưa fit, hoặc log_target=False
            (target không bị log → không cần inverse).

        Examples
        --------
        y_pred_log  = model.predict(X_test_clean)          # log-scale
        y_pred_usd  = pipe.inverse_transform_y(y_pred_log) # USD
        y_test_usd  = pipe.inverse_transform_y(y_test)     # USD
        metrics     = evaluate_model(y_test_usd, y_pred_usd, inverse_transform_y=False)
        """
        if not self._fitted:
            raise RuntimeError("Pipeline chưa được fit. Gọi fit() hoặc fit_transform() trước.")
        if not self.log_target:
            raise RuntimeError(
                "log_target=False — target không bị log1p nên không cần inverse_transform_y."
            )
        return np.expm1(np.asarray(y_log, dtype=float))

    # ------------------------------------------------------------------
    # Thông tin pipeline
    # ------------------------------------------------------------------
    def summary(self):
        """In tóm tắt pipeline sau khi fit."""
        if not self._fitted:
            print("Pipeline chưa được fit.")
            return
        print("=" * 55)
        print("DataPipeline Summary")
        print("=" * 55)
        print(f"  Outlier method      : {self.outlier_method}")
        if self.outlier_method == 'winsorize':
            print(f"  Winsorize threshold : {self.outlier_threshold} (mỗi đuôi)")
        elif self.outlier_method == 'remove':
            print(f"  IQR factor          : {self.iqr_factor}")
        print(f"  Encoding            : {self.encoding}")
        print(f"  Ordinal cols        : {len(self._ordinal_cols_)}")
        print(f"  One-hot cols        : {len(self._onehot_cols_)}")
        print(f"  One-hot dummies     : {len(self._dummy_cols_)}")
        print(f"  Scale (z-score)     : {self.scale}")
        print(f"  Scale params learned: {len(self._scale_params)} cột")
        print(f"  Feature names out   : {len(self._feature_names_)} cột")
        print("=" * 55)