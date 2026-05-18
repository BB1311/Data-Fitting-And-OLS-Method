"""
Cấu trúc:
  Step 0 – Load
  Step 1 – Drop biến định danh và biến >80% missing
  Step 2 – Sửa lỗi dữ liệu rõ ràng
  Step 3 – Feature Engineering (biến mới có ý nghĩa vật lý)
  Step 4 – Missing Values: xử lý theo nhóm nhân-quả
      Group A – "None" semantics còn lại (MNAR cấu trúc)
      Group B – Garage cluster (nhân-quả)
      Group C – Basement cluster (nhân-quả)
      Group D – Fireplace cluster
      Group E – MasVnr cluster
      Group F – Lot Frontage (MAR)
      Group G – Còn lại (MCAR / missing rất nhỏ)
  Step 5 – Feature Engineering sau impute (các biến tổng hợp cần dữ liệu sạch)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# CONSTANTS
ID_COLS = ["Order", "PID"]
HIGH_MISSING_THRESHOLD = 0.80

# STEP 0 – Load
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv('data/AmesHousing.csv')

# STEP 1 – Drop cột không cần thiết
def drop_uninformative_cols(df: pd.DataFrame, verbose: bool = True):
    """
    Loại bỏ:
      1. Biến định danh (Order, PID): không có giá trị dự báo, chỉ là index
      2. Biến có tỷ lệ missing > 80%:
         - Pool QC (99.6%): chỉ 13 nhà có hồ bơi, thông tin cực kỳ thưa
         - Misc Feature (96.4%): tương tự
         - Alley (93.2%): hơn 93% không có ngõ hẻm
         Các biến này gây nhiễu và không có giá trị thống kê đáng kể.
    """
    df = df.copy()
    df.drop(columns=[c for c in ID_COLS if c in df.columns], inplace=True)

    missing_rate = df.isnull().mean()
    high_missing = missing_rate[missing_rate > HIGH_MISSING_THRESHOLD].index.tolist()
    df.drop(columns=high_missing, inplace=True)

    if verbose:
        print(f"[Step 1] Dropped ID cols: {ID_COLS}")
        print(f"[Step 1] Dropped high-missing (>{HIGH_MISSING_THRESHOLD:.0%}): {high_missing}")
        print(f"[Step 1] Shape after drop: {df.shape}")

    return df, high_missing

# STEP 2 – Sửa lỗi dữ liệu rõ ràng
def fix_data_errors(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Các lỗi đã xác nhận qua EDA:

    1. Garage Yr Blt = 2207 (1 quan sát, row 2260):
       - Year Built = 2006, Yr Sold = 2007
       - Garage không thể xây năm 2207 → lỗi nhập liệu (2007 gõ nhầm thành 2207)
       - Xử lý: gán bằng Year Built của ngôi nhà đó

    2. MS SubClass được lưu dạng số (20, 30, 60...) nhưng thực chất là mã
       phân loại kiểu nhà (20 = 1-STORY, 60 = 2-STORY...).
       Để nguyên dạng số khiến mô hình hiểu nhầm 60 > 20 theo nghĩa số học.
       Xử lý: chuyển thành string (categorical nominal).
    """
    df = df.copy()

    mask_err = df["Garage Yr Blt"] > 2100
    if mask_err.any():
        df.loc[mask_err, "Garage Yr Blt"] = df.loc[mask_err, "Year Built"]
        if verbose:
            print(f"[Step 2] Fixed {mask_err.sum()} row(s): Garage Yr Blt > 2100 → Year Built")

    df["MS SubClass"] = df["MS SubClass"].astype(str)
    if verbose:
        print("[Step 2] MS SubClass cast to string (nominal category, not numeric)")

    return df

# STEP 3 – Feature Engineering (trước impute)
def feature_engineering_pre(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Tạo biến mới có ý nghĩa vật lý / kinh tế hơn biến gốc, nhưng chỉ sử dụng
    các biến không bị missing hoặc không cần impute trước.

    FE-1  Age_at_Sale        = Yr Sold - Year Built
          Tuổi nhà lúc bán → phản ánh khấu hao trực tiếp hơn năm xây tuyệt đối.

    FE-2  Remod_Age          = Yr Sold - Year Remod/Add
          Tuổi tính từ lần cải tạo gần nhất → quan trọng với người mua.

    FE-3  Has_Remodel (0/1)
          Year Remod/Add == Year Built → chưa bao giờ sửa chữa = 0.
          53.5% quan sát thuộc nhóm này; cờ nhị phân này mang thông tin
          không thể tách ra từ Remod_Age đơn thuần.

    FE-4  Has_Garage (0/1)   — dùng làm cờ nhân-quả ở Step 4 Group B.

    FE-5  Has_Basement (0/1) — dùng làm cờ nhân-quả ở Step 4 Group C.
    """
    df = df.copy()

    df["Age_at_Sale"]    = df["Yr Sold"] - df["Year Built"]
    df["Remod_Age"]      = df["Yr Sold"] - df["Year Remod/Add"]
    df["Has_Remodel"]    = (df["Year Remod/Add"] != df["Year Built"]).astype(int)
    df["Has_Garage"]     = df["Garage Type"].notna().astype(int)
    df["Has_Basement"]   = df["Bsmt Qual"].notna().astype(int)

    if verbose:
        pre_cols = ["Age_at_Sale", "Remod_Age", "Has_Remodel", "Has_Garage", "Has_Basement"]
        print(f"[Step 3] Created pre-impute features: {pre_cols}")

    return df

# Hàm k-NN impute cho Lot Frontage
def knn_impute_lot_frontage(df, k=5):
    """
    Cài đặt k-NN imputation cho Lot Frontage.
    Sử dụng các biến: Lot Area, Lot Shape (ordinal), Lot Config (ordinal),
    Land Contour (ordinal), Neighborhood (ordinal).
    Trả về Series đã được impute.
    """
    # Chọn các biến để tính khoảng cách
    feat_cols = ['Lot Area', 'Lot Shape', 'Lot Config', 'Land Contour', 'Neighborhood']

    # Tạo mapping ordinal cho các biến phân loại
    ordinal_maps = {}
    for col in ['Lot Shape', 'Lot Config', 'Land Contour', 'Neighborhood']:
        unique_vals = df[col].dropna().unique()
        # Sắp xếp để có thứ tự nhất quán
        unique_vals = sorted(unique_vals)
        ordinal_maps[col] = {v: i for i, v in enumerate(unique_vals)}

    # Tạo bản sao dữ liệu số cho khoảng cách
    numeric_df = df[feat_cols].copy()
    for col in ['Lot Shape', 'Lot Config', 'Land Contour', 'Neighborhood']:
        numeric_df[col] = numeric_df[col].map(ordinal_maps[col])

    # Chuẩn hóa các biến số (Lot Area) về cùng scale (0-1) để tránh trội
    lot_area_min = numeric_df['Lot Area'].min()
    lot_area_max = numeric_df['Lot Area'].max()
    if lot_area_max - lot_area_min > 0:
        numeric_df['Lot Area'] = (numeric_df['Lot Area'] - lot_area_min) / (lot_area_max - lot_area_min)

    # Xác định các hàng có missing Lot Frontage
    missing_mask = df['Lot Frontage'].isna()
    if not missing_mask.any():
        return df['Lot Frontage']

    # Chia thành train (không missing) và test (missing)
    train_idx = ~missing_mask
    test_idx = missing_mask

    train_X = numeric_df.loc[train_idx].values
    train_y = df.loc[train_idx, 'Lot Frontage'].values
    test_X = numeric_df.loc[test_idx].values

    # k-NN: với mỗi điểm test, tìm k điểm train gần nhất
    imputed_values = []
    for test_point in test_X:
        # Tính khoảng cách Euclidean
        dists = np.sqrt(((train_X - test_point) ** 2).sum(axis=1))
        # Lấy k index nhỏ nhất
        k_indices = np.argsort(dists)[:k]
        # Giá trị impute = trung bình của k láng giềng
        imputed = np.mean(train_y[k_indices])
        imputed_values.append(imputed)

    result = df['Lot Frontage'].copy()
    result.loc[test_idx] = imputed_values
    return result

# STEP 4 – Missing Value Imputation (domain-aware)
def impute_missing(df: pd.DataFrame,
                   method: str = "mv2",
                   knn_k: int = 5,
                   verbose: bool = True):
    """
    Xử lý missing theo 7 nhóm nhân-quả.
    method: 'mv2' (median/mode) | 'mv4' (k-NN cho Group F – Lot Frontage)
    """
    df = df.copy()
    log = []

    # ── Group A: "None" semantics – Fence ─────────────────────────
    if "Fence" in df.columns and df["Fence"].isnull().any():
        n = df["Fence"].isnull().sum()
        df["Fence"] = df["Fence"].fillna("None")
        log.append(f"[Group A] Fence: {n} NaN → 'None' (no fence)")

    # ── Group B: Garage cluster ────────────────────────────────────
    # Logic: HAS_GARAGE = 0 → TẤT CẢ garage cols PHẢI là "None"/0.
    #        HAS_GARAGE = 1 nhưng vẫn missing → lỗi nhập liệu, impute có điều kiện.
    #
    # Garage Type:     NaN ↔ không có garage (đã xác nhận 157/157 rows)
    # Garage Yr Blt:   NaN no-garage → 0 | NaN has-garage (2 rows) → Year Built
    # Garage Finish, Qual, Cond: NaN no-garage → 'None' | has-garage → mode trong nhóm
    # Garage Cars, Area: NaN no-garage → 0 | has-garage → median trong nhóm

    no_garage  = df["Has_Garage"] == 0
    has_garage = df["Has_Garage"] == 1

    if "Garage Type" in df.columns:
        n = df.loc[no_garage, "Garage Type"].isnull().sum()
        df.loc[no_garage, "Garage Type"] = df.loc[no_garage, "Garage Type"].fillna("None")
        log.append(f"[Group B] Garage Type: {n} NaN (no garage) → 'None'")

    # Garage Yr Blt
    if "Garage Yr Blt" in df.columns:
        # No garage → 0 (không có garage → không có năm xây)
        n1 = df.loc[no_garage, "Garage Yr Blt"].isnull().sum()
        df.loc[no_garage, "Garage Yr Blt"] = df.loc[no_garage, "Garage Yr Blt"].fillna(0)
        # Has garage but still missing (2 rows) → Year Built của ngôi nhà đó
        miss_has = has_garage & df["Garage Yr Blt"].isnull()
        n2 = miss_has.sum()
        if n2:
            df.loc[miss_has, "Garage Yr Blt"] = df.loc[miss_has, "Year Built"]
        log.append(f"[Group B] Garage Yr Blt: {n1} NaN (no garage)→0 | {n2} NaN (has garage)→Year Built")

    for col in ["Garage Finish", "Garage Qual", "Garage Cond"]:
        if col in df.columns:
            n_no  = df.loc[no_garage, col].isnull().sum()
            df.loc[no_garage, col] = df.loc[no_garage, col].fillna("None")
            miss_has = has_garage & df[col].isnull()
            n_has = miss_has.sum()
            if n_has:
                mode_val = df.loc[has_garage, col].mode()[0]
                df.loc[miss_has, col] = mode_val
            log.append(f"[Group B] {col}: {n_no} (no garage)→'None' | {n_has} (has garage)→mode")

    for col in ["Garage Cars", "Garage Area"]:
        if col in df.columns:
            n_no = df.loc[no_garage, col].isnull().sum()
            df.loc[no_garage, col] = df.loc[no_garage, col].fillna(0)
            miss_has = has_garage & df[col].isnull()
            n_has = miss_has.sum()
            if n_has:
                med = df.loc[has_garage, col].median()
                df.loc[miss_has, col] = med
            log.append(f"[Group B] {col}: {n_no} (no garage)→0 | {n_has} (has garage)→median")

    # ── Group C: Basement cluster ──────────────────────────────────
    # Has_Basement = 0 → tất cả bsmt cols = "None" / 0
    # Has_Basement = 1 nhưng missing (Bsmt Exposure: 3 rows) → mode trong nhóm

    no_bsmt  = df["Has_Basement"] == 0
    has_bsmt = df["Has_Basement"] == 1

    for col in ["Bsmt Qual", "Bsmt Cond", "BsmtFin Type 1", "BsmtFin Type 2"]:
        if col in df.columns:
            n = df.loc[no_bsmt, col].isnull().sum()
            df.loc[no_bsmt, col] = df.loc[no_bsmt, col].fillna("None")
            if n: log.append(f"[Group C] {col}: {n} NaN (no basement) → 'None'")

    for col in ["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
                "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath"]:
        if col in df.columns:
            n_no = df.loc[no_bsmt, col].isnull().sum()
            df.loc[no_bsmt, col] = df.loc[no_bsmt, col].fillna(0)
            miss_has = has_bsmt & df[col].isnull()
            n_has = miss_has.sum()
            if n_has:
                med = df.loc[has_bsmt, col].median()
                df.loc[miss_has, col] = med
            if n_no or n_has:
                log.append(f"[Group C] {col}: {n_no} (no bsmt)→0 | {n_has} (has bsmt)→median")

    # Bsmt Exposure: 3 rows có basement nhưng missing → mode trong nhóm có basement
    if "Bsmt Exposure" in df.columns:
        df.loc[no_bsmt, "Bsmt Exposure"] = df.loc[no_bsmt, "Bsmt Exposure"].fillna("None")
        miss_has = has_bsmt & df["Bsmt Exposure"].isnull()
        n_has = miss_has.sum()
        if n_has:
            mode_val = df.loc[has_bsmt, "Bsmt Exposure"].mode()[0]
            df.loc[miss_has, "Bsmt Exposure"] = mode_val
            log.append(f"[Group C] Bsmt Exposure: {n_has} NaN (has basement) → mode='{mode_val}'")

    # ── Group D: Fireplace cluster ─────────────────────────────────
    # Fireplace Qu NaN ↔ Fireplaces == 0: 100% khớp (đã xác minh).
    # Không dùng mode toàn cục vì sẽ gán chất lượng fireplace cho nhà không có lò sưởi.
    if "Fireplace Qu" in df.columns:
        no_fp  = df["Fireplaces"] == 0
        has_fp = df["Fireplaces"] > 0
        df.loc[no_fp, "Fireplace Qu"] = df.loc[no_fp, "Fireplace Qu"].fillna("None")
        miss_has = has_fp & df["Fireplace Qu"].isnull()
        if miss_has.any():
            mode_val = df.loc[has_fp, "Fireplace Qu"].mode()[0]
            df.loc[miss_has, "Fireplace Qu"] = mode_val
        log.append(f"[Group D] Fireplace Qu: no-fireplace rows → 'None' (0 fireplace = 0 quality)")

    # ── Group E: Masonry Veneer cluster ───────────────────────────
    # 3 tình huống khác nhau:
    #   (a) Type NaN và Area = 0 (1745 rows) → không có tường ốp → Type='None', Area=0
    #   (b) Type NaN nhưng Area > 0 (7 rows) → lỗi nhập liệu, có tường ốp
    #       → gán Type = mode trong nhóm có Area > 0
    #   (c) Area NaN khi Type đã có (23 rows):
    #       - Type = 'None' → Area = 0
    #       - Type khác → median theo cùng type

    if "Mas Vnr Type" in df.columns and "Mas Vnr Area" in df.columns:
        # (a)
        mask_a = df["Mas Vnr Type"].isnull() & (df["Mas Vnr Area"].fillna(0) == 0)
        df.loc[mask_a, "Mas Vnr Type"] = "None"
        df.loc[mask_a, "Mas Vnr Area"] = df.loc[mask_a, "Mas Vnr Area"].fillna(0)
        log.append(f"[Group E] Mas Vnr: {mask_a.sum()} (Type NaN, Area=0) → Type='None', Area=0")

        # (b)
        mask_b = df["Mas Vnr Type"].isnull() & (df["Mas Vnr Area"].fillna(0) > 0)
        if mask_b.any():
            mode_type = df.loc[df["Mas Vnr Type"].notna() & (df["Mas Vnr Area"] > 0),
                                "Mas Vnr Type"].mode()[0]
            df.loc[mask_b, "Mas Vnr Type"] = mode_type
            log.append(f"[Group E] Mas Vnr Type: {mask_b.sum()} (Type NaN, Area>0) → mode='{mode_type}'")

        # (c)
        mask_c = df["Mas Vnr Area"].isnull()
        if mask_c.any():
            mask_c_none = mask_c & (df["Mas Vnr Type"] == "None")
            df.loc[mask_c_none, "Mas Vnr Area"] = 0
            mask_c_has  = mask_c & (df["Mas Vnr Type"] != "None") & df["Mas Vnr Type"].notna()
            if mask_c_has.any():
                for vtype in df.loc[mask_c_has, "Mas Vnr Type"].unique():
                    med = df.loc[df["Mas Vnr Type"] == vtype, "Mas Vnr Area"].median()
                    df.loc[mask_c_has & (df["Mas Vnr Type"] == vtype), "Mas Vnr Area"] = med
            log.append(f"[Group E] Mas Vnr Area: {mask_c.sum()} NaN → 0 (None type) or median by type")

    # ── Group F: Lot Frontage (MAR) ────────────────────────────────
    # Bằng chứng MAR: tỷ lệ missing thay đổi rõ theo Neighborhood
    # (GrnHill 100%, NWAmes 35%, Inside lots 13%)
    # → Grouped imputation theo Neighborhood phù hợp hơn global median.
    #
    # MV2: grouped median (Neighborhood) – nhanh, interpret được
    # MV4: k-NN trên {Lot Area, Lot Shape, Lot Config, Land Contour, Neighborhood}
    #      – tận dụng quan hệ đa biến, bảo toàn phân phối tốt hơn

    if "Lot Frontage" in df.columns and df["Lot Frontage"].isnull().any():
        n = df["Lot Frontage"].isnull().sum()
        if method == "mv2":
            df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(
                lambda x: x.fillna(x.median())
            )
            df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())
            log.append(f"[Group F] Lot Frontage: {n} NaN → grouped median by Neighborhood (MV2)")
        elif method == "mv4":
            df["Lot Frontage"] = knn_impute_lot_frontage(df, k=knn_k)
            log.append(f"[Group F] Lot Frontage: {n} NaN → k-NN (k={knn_k}) (MV4)")

    # ── Group G: Còn lại – MCAR / missing cực nhỏ ─────────────────
    if "Electrical" in df.columns and df["Electrical"].isnull().any():
        mode_val = df["Electrical"].mode()[0]
        n = df["Electrical"].isnull().sum()
        df["Electrical"] = df["Electrical"].fillna(mode_val)
        log.append(f"[Group G] Electrical: {n} NaN → mode='{mode_val}' (MCAR, 1 obs)")

    # Fallback cho bất kỳ cột nào vẫn còn missing
    for col in [c for c in df.columns if df[c].isnull().any()]:
        n = df[col].isnull().sum()
        if pd.api.types.is_numeric_dtype(df[col]):
            val = df[col].median()
            df[col] = df[col].fillna(val)
            log.append(f"[Group G] {col}: {n} NaN → median={val:.2f} (fallback)")
        else:
            val = df[col].mode()[0]
            df[col] = df[col].fillna(val)
            log.append(f"[Group G] {col}: {n} NaN → mode='{val}' (fallback)")

    if verbose:
        print(f"\n[Step 4] Imputation log ({method.upper()}):")
        for entry in log:
            print(f"  {entry}")
        print(f"\n[Step 4] Missing remaining: {df.isnull().sum().sum()}")

    return df, log

# STEP 5 – Feature Engineering (sau impute)
def feature_engineering_post(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Tạo biến tổng hợp cần dữ liệu đã được impute đầy đủ.

    FE-4  Total_Area         = Total Bsmt SF + 1st Flr SF + 2nd Flr SF
          Tổng diện tích sử dụng. Giữ 3 biến thành phần riêng lẻ gây
          đa cộng tuyến với Gr Liv Area (vì Gr Liv Area = 1st + 2nd Flr SF).

    FE-5  Total_Porch_SF     = Wood Deck + Open Porch + Enclosed Porch
                               + 3Ssn Porch + Screen Porch
          5 biến diện tích sân/hiên, mỗi biến có phân phối lệch nặng
          (median = 0). Gom lại giảm chiều và giảm thưa.

    FE-6  Total_Bath         = Full Bath + 0.5*Half Bath
                               + Bsmt Full Bath + 0.5*Bsmt Half Bath
          Quy đổi half bath = 0.5 theo thông lệ bất động sản.

    FE-7  Garage_Age         = Yr Sold - Garage Yr Blt (chỉ khi Has_Garage=1)
          Tuổi garage lúc bán; đã có Has_Garage nên an toàn.
    """
    df = df.copy()

    df["Total_Area"]     = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"]
    df["Total_Porch_SF"] = (df["Wood Deck SF"] + df["Open Porch SF"]
                             + df["Enclosed Porch"] + df["3Ssn Porch"]
                             + df["Screen Porch"])
    df["Total_Bath"]     = (df["Full Bath"] + 0.5 * df["Half Bath"]
                             + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"])
    df["Garage_Age"]     = np.where(
        df["Has_Garage"] == 1,
        df["Yr Sold"] - df["Garage Yr Blt"],
        0
    )

    if verbose:
        post_cols = ["Total_Area", "Total_Porch_SF", "Total_Bath", "Garage_Age"]
        print(f"[Step 5] Created post-impute features: {post_cols}")

    return df