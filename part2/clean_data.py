import numpy as np
import pandas as pd
import builtins
from pathlib import Path

"""
Quy trình:
  0. Đọc dữ liệu
  1. Xóa cột vô dụng / near-zero variance / định danh
  2. Kiểm tra và sửa tính nhất quán giữa các biến
  3. Xử lý missing có cấu trúc (structural missing) theo quan hệ nhân-quả
  4. Imputation Lot Frontage và Electrical
  5. Tạo biến mới, xóa biến gốc
  6. Kiểm tra lần cuối — báo lỗi nếu còn missing
"""


def clean_data(input_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Đọc, làm sạch và trả về DataFrame đã xử lý.

    Parameters
    ----------
    input_path : str
        Đường dẫn tới file CSV gốc (ví dụ: 'data/AmesHousing.csv').

    Returns
    -------
    pd.DataFrame
        DataFrame đã qua toàn bộ pipeline làm sạch, không còn missing values.
    """
    def _log(*args, **kwargs):
        if verbose:
            builtins.print(*args, **kwargs)

    # Giữ nguyên các lệnh print hiện có, nhưng chỉ cho phép in khi `verbose` được bật.
    print = _log

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 0: ĐỌC DỮ LIỆU
    # ══════════════════════════════════════════════════════════════════════

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    print(f"BƯỚC 0: DỮ LIỆU GỐC — {df.shape[0]} dòng, {df.shape[1]} cột")

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 1: XÓA CỘT ĐỊNH DANH VÀ NEAR-ZERO VARIANCE
    # ══════════════════════════════════════════════════════════════════════
    print("\nBƯỚC 1: XÓA CỘT ĐỊNH DANH VÀ NEAR-ZERO VARIANCE")

    # Biến định danh: unique mỗi hàng, không mang thông tin dự báo
    id_cols = ['Order', 'PID']

    # Biến >80% missing: quá ít thông tin, giữ lại thông tin "có/không" bằng cờ
    # Pool QC (99.6%), Misc Feature (96.4%), Alley (93.2%), Fence (80.5%)
    # Tạo cờ trước khi drop
    df['Has_Pool']         = (df['Pool Area'] > 0).astype(int)
    df['Has_Misc_Feature'] = df['Misc Feature'].notna().astype(int)
    df['Has_Alley']        = df['Alley'].notna().astype(int)
    df['Has_Fence']        = df['Fence'].notna().astype(int)
    high_missing_cols = ['Pool QC', 'Misc Feature', 'Alley', 'Fence']

    # Flag garage (dùng fillna(0) để xác định có garage ngay cả khi có missing)
    # Anchor: đối chiếu chéo 3 cột để tránh quy kết ẩu
    # Nhà có garage nếu: Garage Type notna HOẶC Garage Area > 0 HOẶC Garage Cars > 0
    # (vì có thể bỏ sót Garage Type nhưng diện tích/số xe vẫn được ghi)
    has_garage = (
        (df['Garage Area'].fillna(0) > 0) |
        (df['Garage Cars'].fillna(0) > 0) |
        (df['Garage Type'].notna() & (df['Garage Type'] != 'None'))
    )
    df['Has_Garage'] = has_garage.astype(int)

    # Near-zero variance: gần như hằng số, không phân biệt được nhà
    # Utilities: 99.9% = AllPub; Street: 99.6% = Pave
    nzv_cols = ['Utilities', 'Street']

    # MS SubClass: dtype int64 nhưng là MÃ LOẠI NHÀ, không có thứ tự số học -> convert sang string (categorical)
    df['MS SubClass'] = df['MS SubClass'].astype(str).astype(object)

    # Mo Sold: dtype int64 nhưng mang tính mùa vụ (không có tính tuyến tính) -> convert sang string để One-hot
    df['Mo Sold'] = df['Mo Sold'].astype(str).astype(object)

    # Các cột chỉ có ý nghĩa khi kết hợp với biến đã xóa, hoặc quá thưa thớt
    extra_drop = ['Pool Area', 'Misc Val']

    cols_to_drop = id_cols + high_missing_cols + nzv_cols + extra_drop
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"  Đã xóa {len(cols_to_drop)} cột: {cols_to_drop}")
    print(f"  Còn lại: {df.shape[1]} cột")

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 2: KIỂM TRA VÀ SỬA TÍNH NHẤT QUÁN
    # ══════════════════════════════════════════════════════════════════════
    print("\nBƯỚC 2: KIỂM TRA TÍNH NHẤT QUÁN")

    # 2.1. Gr Liv Area = 1st Flr SF + 2nd Flr SF + Low Qual Fin SF
    # Nếu lệch >10% thì dùng tổng các tầng làm chuẩn
    sum_floors = df['1st Flr SF'] + df['2nd Flr SF'] + df['Low Qual Fin SF']
    denom = df['Gr Liv Area'].replace(0, np.nan)  # tránh ZeroDivisionError
    mask = (sum_floors > 0) & (np.abs(sum_floors - df['Gr Liv Area']) / denom > 0.1)
    if mask.any():
        print(f"  2.1. {mask.sum()} dòng Gr Liv Area lệch >10% → gán lại = tổng các tầng")
        df.loc[mask, 'Gr Liv Area'] = sum_floors[mask]
    else:
        print("  2.1. Gr Liv Area: OK")

    # 2.2. Total Bsmt SF = BsmtFin SF 1 + BsmtFin SF 2 + Bsmt Unf SF
    # Nếu không khớp thì điều chỉnh Bsmt Unf SF (phần chưa hoàn thiện là phần dư)
    # Chỉ kiểm tra trên các dòng có đủ cả 3 thành phần (không missing)
    mask_all_present = (
        df['BsmtFin SF 1'].notna() &
        df['BsmtFin SF 2'].notna() &
        df['Bsmt Unf SF'].notna()
    )
    sum_bsmt = df['BsmtFin SF 1'] + df['BsmtFin SF 2'] + df['Bsmt Unf SF']
    mask_bsmt = mask_all_present & (np.abs(df['Total Bsmt SF'] - sum_bsmt) > 1)
    if mask_bsmt.any():
        print(f"  2.2. {mask_bsmt.sum()} dòng Total Bsmt SF không khớp → điều chỉnh Bsmt Unf SF")
        df.loc[mask_bsmt, 'Bsmt Unf SF'] = (
            df.loc[mask_bsmt, 'Total Bsmt SF']
            - df.loc[mask_bsmt, 'BsmtFin SF 1']
            - df.loc[mask_bsmt, 'BsmtFin SF 2']
        ).clip(lower=0)
    else:
        print("  2.2. Total Bsmt SF: OK")

    # 2.3. Year Remod/Add không được nhỏ hơn Year Built
    mask_remod = df['Year Remod/Add'] < df['Year Built']
    if mask_remod.any():
        print(f"  2.3. {mask_remod.sum()} dòng Year Remod < Year Built → gán = Year Built")
        df.loc[mask_remod, 'Year Remod/Add'] = df.loc[mask_remod, 'Year Built']
    else:
        print("  2.3. Year Remod/Add: OK")

    # 2.4. Age_At_Sale âm (Year Built > Yr Sold — lỗi nhập liệu)
    age_check = df['Yr Sold'] - df['Year Built']
    mask_age = age_check < 0
    if mask_age.any():
        print(f"  2.4. {mask_age.sum()} dòng Year Built > Yr Sold → gán Year Built = Yr Sold")
        df.loc[mask_age, 'Year Built'] = df.loc[mask_age, 'Yr Sold']
    else:
        print("  2.4. Age_At_Sale: OK")

    # 2.5. Garage Attchd không thể có năm xây trước năm xây nhà
    mask_attchd_err = (
        df['Garage Type'].notna() &
        (df['Garage Type'] == 'Attchd') &
        (df['Garage Yr Blt'].notna()) &
        (df['Garage Yr Blt'] < df['Year Built'])
    )
    if mask_attchd_err.any():
        print(f"  2.5. {mask_attchd_err.sum()} dòng Garage Attchd có Yr Blt < Year Built → gán = Year Built")
        df.loc[mask_attchd_err, 'Garage Yr Blt'] = df.loc[mask_attchd_err, 'Year Built']
    else:
        print("  2.5. Garage Attchd Yr Blt: OK")

    # GHI CHÚ: KHÔNG sửa các trường hợp Garage Yr Blt < Year Built - 5 nếu không phải Attchd
    # (nhà cũ có thể mua đất đã có garage)

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 3: XỬ LÝ MISSING CÓ CẤU TRÚC
    # ══════════════════════════════════════════════════════════════════════
    print("\nBƯỚC 3: XỬ LÝ MISSING CÓ CẤU TRÚC")

    # Đảm bảo các cột phân loại thuộc nhóm structural-missing có thể gán an toàn nhãn chuỗi như "None".
    # Với bộ test tổng hợp rất nhỏ, pandas có thể suy luận các cột này thành float khi toàn bộ giá trị quan sát đều là NaN.
    structural_cat_cols = [
        'Mas Vnr Type', 'Fireplace Qu',
        'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond'
    ]
    for col in structural_cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)
    # ── 3.1. Mas Vnr (Lớp ốp đá/gạch) ──────────────────────────────────
    # Anchor: Mas Vnr Area > 0 HOẶC Type không NaN và khác 'None' => có ốp đá
    # Không thể dùng chỉ một trong hai vì có thể một bên bị bỏ sót
    # GHI CHÚ: 'None' là category hợp lệ theo Ames Data Dictionary — trong CSV gốc
    # nó được lưu dưới dạng NaN do export. Code bên dưới chuẩn hóa NaN → 'None'
    # để tường minh hóa "không có ốp đá", thống nhất convention với Garage và Basement.
    print("  3.1. Mas Vnr...")
    has_masonry = (df['Mas Vnr Area'].fillna(0) > 0) | (
        df['Mas Vnr Type'].notna() & (df['Mas Vnr Type'] != 'None')
    )
    no_masonry = ~has_masonry
    df.loc[no_masonry, 'Mas Vnr Type'] = 'None'
    df.loc[no_masonry, 'Mas Vnr Area'] = 0

    # Nhà có ốp đá nhưng thiếu loại: dùng mode theo nhóm diện tích
    # (diện tích nhỏ thường là BrkFace, lớn thường là Stone)
    mask_type_missing = has_masonry & df['Mas Vnr Type'].isna()
    if mask_type_missing.any():
        df['_area_group'] = pd.cut(
            df['Mas Vnr Area'], bins=[0, 100, 300, 10000],
            labels=['small', 'medium', 'large']
        )
        mode_by_area = (
            df[df['Mas Vnr Type'].notna() & (df['Mas Vnr Type'] != 'None')]
            .groupby('_area_group', observed=True)['Mas Vnr Type']
            .agg(lambda x: x.mode()[0] if not x.mode().empty else 'BrkFace')
        )
        df.loc[mask_type_missing, 'Mas Vnr Type'] = (
            df.loc[mask_type_missing, '_area_group']
            .map(mode_by_area)
            .fillna('BrkFace')
        )
        df.drop(columns='_area_group', inplace=True)
        print(f"    {mask_type_missing.sum()} dòng có ốp đá nhưng thiếu loại → điền mode theo nhóm diện tích")

    # Nhà có ốp đá nhưng thiếu diện tích: median theo loại ốp đá
    mask_area_missing = has_masonry & df['Mas Vnr Area'].isna()
    if mask_area_missing.any():
        median_by_type = (
            df[df['Mas Vnr Area'] > 0]
            .groupby('Mas Vnr Type')['Mas Vnr Area'].median()
        )
        df.loc[mask_area_missing, 'Mas Vnr Area'] = (
            df.loc[mask_area_missing, 'Mas Vnr Type']
            .map(median_by_type)
            .fillna(0)
        )
        print(f"    {mask_area_missing.sum()} dòng có ốp đá nhưng thiếu diện tích → median theo loại")

    # ── 3.2. Fireplace ──────────────────────────────────────────────────
    # Quan hệ nhân-quả: Fireplaces (số lượng) là anchor
    # Fireplaces = 0 thì chắc chắn không có lò → Fireplace Qu = 'None'
    # Fireplaces > 0 nhưng thiếu Qu → dùng mode theo Overall Qual
    # (nhà chất lượng cao thường có lò sưởi tốt hơn)
    print("  3.2. Fireplace Qu...")
    df.loc[df['Fireplaces'] == 0, 'Fireplace Qu'] = 'None'
    mask_fire = (df['Fireplaces'] > 0) & df['Fireplace Qu'].isna()
    if mask_fire.any():
        # Nhóm theo Overall Qual và Bldg Type vì chất lượng lò phụ thuộc loại nhà và chất lượng tổng thể
        mode_by_group = (
            df[df['Fireplace Qu'].notna()]
            .groupby(['Overall Qual', 'Bldg Type'], observed=True)['Fireplace Qu']
            .agg(lambda x: x.mode()[0] if not x.mode().empty else 'TA')
        )
        fill_idx = pd.MultiIndex.from_frame(df.loc[mask_fire, ['Overall Qual', 'Bldg Type']])
        df.loc[mask_fire, 'Fireplace Qu'] = mode_by_group.reindex(fill_idx, fill_value='TA').values
        print(f"    {mask_fire.sum()} dòng có lò sưởi nhưng thiếu Qu → mode theo (Overall Qual, Bldg Type)")
    else:
        print("    OK")

    # ── 3.3. Garage ──────────────────────────────────────────────────────
    print("  3.3. Garage...")
    # Nhà có garage — xử lý từng biến theo logic riêng:
    has_g = df['Has_Garage'].astype(bool)
    no_garage = ~has_g

    # Nhà không có garage: tất cả biến garage = giá trị "không tồn tại"
    for col in ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']:
        df.loc[no_garage, col] = 'None'
    df.loc[no_garage, ['Garage Cars', 'Garage Area']] = 0

    # Garage Yr Blt: điền theo median của nhóm (Year Built ÷ 10) × (Garage Type)
    # Lý do: năm xây garage gắn chặt với năm xây nhà VÀ loại garage
    # Nhà xây 1960 + Detchd sẽ khác với nhà xây 2000 + Attchd
    mask_yr = has_g & df['Garage Yr Blt'].isna()
    if mask_yr.any():
        df['_decade'] = (df['Year Built'] // 10) * 10
        median_yr = (
            df[df['Garage Yr Blt'].notna()]
            .groupby(['_decade', 'Garage Type'])['Garage Yr Blt'].median()
        )
        # Fallback 1: median theo riêng Garage Type (không xét thập niên)
        median_by_type = df[df['Garage Yr Blt'].notna()].groupby('Garage Type')['Garage Yr Blt'].median()
        # Fallback 2: median toàn cục (dùng nếu vẫn không có)
        global_median = df[df['Garage Yr Blt'].notna()]['Garage Yr Blt'].median()

        missing = df.loc[mask_yr, ['_decade', 'Garage Type']].copy()
        missing['Garage Type'] = missing['Garage Type'].fillna('Attchd')

        med1 = median_yr.reset_index(name='val1')
        med2 = median_by_type.reset_index(name='val2')
        missing = missing.merge(med1, on=['_decade', 'Garage Type'], how='left')
        missing = missing.merge(med2, on='Garage Type', how='left')
        result = missing['val1'].fillna(missing['val2']).fillna(global_median)

        df.loc[mask_yr, 'Garage Yr Blt'] = result.values
        df.drop(columns='_decade', inplace=True)
        print(f"    {mask_yr.sum()} dòng thiếu Garage Yr Blt → median theo thập niên × loại garage (fallback: loại → toàn cục)")

    # Garage Area: điền theo median của cùng Garage Type
    # Lý do: diện tích phụ thuộc nhiều nhất vào kiểu garage
    # (Detchd thường nhỏ hơn Attchd; CarPort nhỏ nhất)
    mask_area = has_g & df['Garage Area'].isna()
    if mask_area.any():
        median_area_by_type = (
            df[df['Garage Area'] > 0]
            .groupby('Garage Type')['Garage Area'].median()
        )
        df.loc[mask_area, 'Garage Area'] = (
            df.loc[mask_area, 'Garage Type']
            .map(median_area_by_type)
            .fillna(400)
        )
        print(f"    {mask_area.sum()} dòng thiếu Garage Area → median theo Garage Type")

    # Garage Cars: điền theo median của Garage Area
    mask_cars = has_g & df['Garage Cars'].isna()
    if mask_cars.any():
        # Chỉ lấy các dòng có Garage Cars > 0 và Garage Area > 0 để tính bins và median
        valid_cars = df[(df['Garage Cars'] > 0) & (df['Garage Area'] > 0)]
        if len(valid_cars) > 0:
            # Tạo bins dựa trên diện tích của valid_cars
            area_bins_valid = pd.cut(valid_cars['Garage Area'], bins=[0, 300, 500, 1000, 2000])
            median_cars_by_area = valid_cars.groupby(area_bins_valid, observed=True)['Garage Cars'].median()
            # Dùng toàn bộ df để gán bin cho từng dòng cần impute
            area_bins_all = pd.cut(df['Garage Area'], bins=[0, 300, 500, 1000, 2000])
            df.loc[mask_cars, 'Garage Cars'] = area_bins_all[mask_cars].map(median_cars_by_area).fillna(2)
        else:
            df.loc[mask_cars, 'Garage Cars'] = 2
        print(f"    {mask_cars.sum()} dòng thiếu Garage Cars → median theo Garage Area (chỉ trên dữ liệu có xe >0)")

    # Garage Finish, Qual, Cond: điền mode theo nhóm (Garage Type, Overall Qual)
    df['_qual_group'] = pd.cut(df['Overall Qual'], bins=[0, 4, 6, 10], labels=['low', 'mid', 'high'])

    for col in ['Garage Finish', 'Garage Qual', 'Garage Cond']:
        mask_cat = has_g & df[col].isna()
        if mask_cat.any():
            valid = df[col].notna() & (df[col] != 'None')
            mode1 = df[valid].groupby(['Garage Type', '_qual_group'], observed=True)[col].agg(
                lambda x: x.mode()[0] if not x.mode().empty else 'TA'
            )
            mode1_df = mode1.reset_index(name='mode1')
            mode2 = df[valid].groupby('Garage Type')[col].agg(
                lambda x: x.mode()[0] if not x.mode().empty else 'TA'
            )
            mode2_df = mode2.reset_index(name='mode2')

            missing = df.loc[mask_cat, ['Garage Type', '_qual_group']].copy()
            missing = missing.merge(mode1_df, on=['Garage Type', '_qual_group'], how='left')
            missing = missing.merge(mode2_df, on='Garage Type', how='left')
            filled = missing['mode1'].fillna(missing['mode2']).fillna('TA')
            df.loc[mask_cat, col] = filled.values

            print(f"    {mask_cat.sum()} dòng thiếu {col} → mode theo nhóm Garage Type và Overall Qual")
    df.drop(columns='_qual_group', inplace=True)

    # Garage Type: chỉ missing khi has_garage=True do Area/Cars > 0 nhưng Type bỏ trống
    mask_type = has_g & df['Garage Type'].isna()
    if mask_type.any():
        valid_garage = df[df['Garage Type'].notna() & (df['Garage Area'] > 0)]
        area_cuts_valid = pd.cut(valid_garage['Garage Area'], bins=[0, 300, 500, 1000])
        mode_by_area = (
            valid_garage.groupby(area_cuts_valid, observed=True)['Garage Type']
            .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Attchd')
        )
        area_cuts_all = pd.cut(df['Garage Area'], bins=[0, 300, 500, 1000])
        df.loc[mask_type, 'Garage Type'] = area_cuts_all[mask_type].map(mode_by_area).fillna('Attchd')
        print(f"    {mask_type.sum()} dòng thiếu Garage Type → mode theo nhóm diện tích")

    if not (mask_yr.any() or mask_area.any() or mask_cars.any() or mask_type.any()):
        print("    OK")

    # ── 3.4. Basement ────────────────────────────────────────────────────
    # Anchor: Total Bsmt SF > 0 HOẶC tổng diện tích các phần > 0 => có tầng hầm
    print("  3.4. Basement...")
    has_basement = (
        (df['Total Bsmt SF'].fillna(0) > 0) |
        (df['BsmtFin SF 1'].fillna(0) + df['BsmtFin SF 2'].fillna(0) + df['Bsmt Unf SF'].fillna(0) > 0)
    )
    no_basement = ~has_basement

    for col in ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']:
        df.loc[no_basement, col] = 'None'
    df.loc[no_basement, ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF',
                          'Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']] = 0

    has_b = has_basement

    # Đảm bảo nhất quán: nếu diện tích = 0 thì loại hoàn thiện phải là 'Unf'
    mask_type1_zero = has_b & (df['BsmtFin SF 1'].fillna(0) == 0) & (df['BsmtFin Type 1'].isna())
    if mask_type1_zero.any():
        df.loc[mask_type1_zero, 'BsmtFin Type 1'] = 'Unf'
        print(f"    {mask_type1_zero.sum()} dòng BsmtFin SF 1 = 0, gán BsmtFin Type 1 = 'Unf'")

    mask_type2_zero = has_b & (df['BsmtFin SF 2'].fillna(0) == 0) & (df['BsmtFin Type 2'].isna())
    if mask_type2_zero.any():
        df.loc[mask_type2_zero, 'BsmtFin Type 2'] = 'Unf'
        print(f"    {mask_type2_zero.sum()} dòng BsmtFin SF 2 = 0, gán BsmtFin Type 2 = 'Unf'")

    # Trường hợp 1: biết loại hoàn thiện (Type), thiếu diện tích (SF) → median theo Type
    for typ_col, sf_col in [('BsmtFin Type 1', 'BsmtFin SF 1'), ('BsmtFin Type 2', 'BsmtFin SF 2')]:
        mask_missing_sf = has_b & df[sf_col].isna()
        if mask_missing_sf.any():
            median_by_type = df[df[sf_col].notna() & (df[typ_col] != 'None')].groupby(typ_col)[sf_col].median()
            df.loc[mask_missing_sf, sf_col] = df.loc[mask_missing_sf, typ_col].map(median_by_type).fillna(0)
            print(f"    {mask_missing_sf.sum()} dòng thiếu {sf_col} → median theo {typ_col}")

    # Trường hợp 2: biết diện tích (SF), thiếu loại hoàn thiện (Type) → mode theo nhóm diện tích
    for typ_col, sf_col in [('BsmtFin Type 1', 'BsmtFin SF 1'), ('BsmtFin Type 2', 'BsmtFin SF 2')]:
        mask_missing_type = has_b & df[typ_col].isna() & (df[sf_col] > 0)
        if mask_missing_type.any():
            sf_valid = df.loc[has_b & (df[sf_col] > 0), sf_col]
            if len(sf_valid) >= 3:
                bins = pd.qcut(sf_valid, q=3, retbins=True, duplicates='drop')[1]
                sf_bins = pd.cut(df[sf_col], bins=bins)
                mode_by_bin = (
                    df[df[typ_col].notna() & (df[typ_col] != 'None') & has_b]
                    .groupby(sf_bins, observed=True)[typ_col]
                    .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unf')
                )
                df.loc[mask_missing_type, typ_col] = sf_bins[mask_missing_type].map(mode_by_bin).fillna('Unf')
            else:
                mode_val = df.loc[df[sf_col] > 0, typ_col].mode()
                mode_val = mode_val[0] if not mode_val.empty else 'Unf'
                df.loc[mask_missing_type, typ_col] = mode_val
            print(f"    {mask_missing_type.sum()} dòng thiếu {typ_col} → mode theo nhóm diện tích {sf_col}")

    # Total Bsmt SF: tính lại từ các phần nếu thiếu
    mask_total = has_b & df['Total Bsmt SF'].isna()
    if mask_total.any():
        df.loc[mask_total, 'Total Bsmt SF'] = (
            df.loc[mask_total, 'BsmtFin SF 1'].fillna(0)
            + df.loc[mask_total, 'BsmtFin SF 2'].fillna(0)
            + df.loc[mask_total, 'Bsmt Unf SF'].fillna(0)
        )
        print(f"    {mask_total.sum()} dòng thiếu Total Bsmt SF → Fin1 + Fin2 + Unf")

    # Bsmt Unf SF: phần dư = Total - Fin1 - Fin2
    mask_unf = has_b & df['Bsmt Unf SF'].isna()
    if mask_unf.any():
        df.loc[mask_unf, 'Bsmt Unf SF'] = (
            df.loc[mask_unf, 'Total Bsmt SF']
            - df.loc[mask_unf, 'BsmtFin SF 1'].fillna(0)
            - df.loc[mask_unf, 'BsmtFin SF 2'].fillna(0)
        ).clip(lower=0)
        print(f"    {mask_unf.sum()} dòng thiếu Bsmt Unf SF → Total - Fin1 - Fin2")

    # Bsmt Full Bath / Half Bath
    for col in ['Bsmt Full Bath', 'Bsmt Half Bath']:
        mask = has_b & df[col].isna()
        if mask.any():
            df.loc[mask, col] = 0
            print(f"    {mask.sum()} dòng thiếu {col} → 0 (tầng hầm không bắt buộc có phòng tắm)")

    # Bsmt Exposure
    mask_exp = has_b & df['Bsmt Exposure'].isna()
    if mask_exp.any():
        mode_by_type = (
            df[has_b & df['Bsmt Exposure'].notna()]
            .groupby('BsmtFin Type 1', observed=True)['Bsmt Exposure']
            .agg(lambda x: x.mode()[0] if not x.mode().empty else 'No')
        )
        global_mode = df.loc[has_b & df['Bsmt Exposure'].notna(), 'Bsmt Exposure'].mode()[0]
        df.loc[mask_exp, 'Bsmt Exposure'] = df.loc[mask_exp, 'BsmtFin Type 1'].map(mode_by_type).fillna(global_mode)
        print(f"    {mask_exp.sum()} dòng thiếu Bsmt Exposure (có tầng hầm) → mode theo BsmtFin Type 1")
    else:
        print("    OK")

    # Tạo biến nhóm diện tích (quantile) và nhóm Overall Qual cho các biến còn lại
    missing_exists = False
    for col in ['Bsmt Qual', 'Bsmt Cond', 'BsmtFin Type 1', 'BsmtFin Type 2']:
        if (has_b & df[col].isna()).any():
            missing_exists = True
            break

    if missing_exists:
        # Bsmt Qual
        mask = has_b & df['Bsmt Qual'].isna()
        if mask.any():
            sf_valid = df.loc[has_b & df['Total Bsmt SF'].notna() & (df['Total Bsmt SF'] > 0), 'Total Bsmt SF']
            bins = pd.qcut(sf_valid, q=4, duplicates='drop', retbins=True)[1]

            valid_rows = df[
                df['Bsmt Qual'].notna() &
                (df['Bsmt Qual'] != 'None') &
                has_b &
                df['Total Bsmt SF'].notna() &
                (df['Total Bsmt SF'] > 0)
            ]
            valid_cuts = pd.cut(valid_rows['Total Bsmt SF'], bins=bins)
            mode_by_sf = valid_rows.groupby(valid_cuts, observed=True)['Bsmt Qual'].agg(
                lambda x: x.mode()[0] if not x.mode().empty else 'TA'
            )

            sf_cuts = pd.cut(df['Total Bsmt SF'], bins=bins)
            df.loc[mask, 'Bsmt Qual'] = sf_cuts[mask].map(mode_by_sf).fillna('TA')
            print(f"    {mask.sum()} dòng thiếu Bsmt Qual → mode theo nhóm Total Bsmt SF quantile (has_basement only)")

        # Bsmt Cond
        mask = has_b & df['Bsmt Cond'].isna()
        if mask.any():
            df['_decade'] = (df['Year Built'] // 10) * 10
            mode_bsmt_cond = (
                df[has_b & df['Bsmt Cond'].notna()]
                .groupby('_decade', observed=True)['Bsmt Cond']
                .agg(lambda x: x.mode()[0] if not x.mode().empty else 'TA')
            )
            df.loc[mask, 'Bsmt Cond'] = df.loc[mask, '_decade'].map(mode_bsmt_cond).fillna('TA')
            df.drop(columns='_decade', inplace=True)
            print(f"    {mask.sum()} dòng thiếu Bsmt Cond → mode theo thập niên Year Built (fallback: 'TA')")

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 4: IMPUTATION LOT FRONTAGE VÀ ELECTRICAL
    # ══════════════════════════════════════════════════════════════════════
    print("\nBƯỚC 4: IMPUTATION LOT FRONTAGE VÀ ELECTRICAL")
    # Lot Frontage (~16.7% missing): MAR — grouped median theo Neighborhood
    print("  4.1. Lot Frontage → grouped median theo Neighborhood")
    median_by_nbhd = df.groupby('Neighborhood')['Lot Frontage'].transform('median')
    df['Lot Frontage'] = df['Lot Frontage'].fillna(median_by_nbhd)
    if df['Lot Frontage'].isna().any():
        missing_count = df['Lot Frontage'].isna().sum()
        df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median())
        print(f"    Cảnh báo: {missing_count} dòng missing sau grouped median → fallback global median")
    else:
        print("    OK (không cần fallback)")

    # Electrical (1 hàng): MAR đơn lẻ → mode (SBrkr chiếm 98%)
    print("  4.2. Electrical → mode")
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 5: TẠO BIẾN MỚI, XÓA BIẾN GỐC
    # ══════════════════════════════════════════════════════════════════════
    print("\nBƯỚC 5: TẠO BIẾN MỚI, XÓA BIẾN GỐC")

    def fix_garage_year_blt(df):
        mask_future = df['Garage Yr Blt'] > df['Yr Sold']
        if mask_future.any():
            print(f"[CẢNH BÁO] {mask_future.sum()} dòng có Garage Yr Blt > Yr Sold. "
                  f"Đã thay thế bằng Year Built.")
            df.loc[mask_future, 'Garage Yr Blt'] = df.loc[mask_future, 'Year Built']
        return df

    df = fix_garage_year_blt(df)
    mask_garage_check = df['Has_Garage'] == 1
    if not (df.loc[mask_garage_check, 'Garage Yr Blt'] <= df.loc[mask_garage_check, 'Yr Sold']).all():
        raise ValueError("Phát hiện Garage Yr Blt > Yr Sold sau khi sửa lỗi!")

    if ((df['Has_Garage'] == 1) & (df['Garage Yr Blt'].isna())).any():
        raise ValueError("Vẫn còn Garage Yr Blt = NaN cho nhà có garage. Cần kiểm tra lại Bước 3.")

    df['Garage_Age'] = np.where(
        df['Has_Garage'] == 1,
        df['Yr Sold'] - df['Garage Yr Blt'],
        0
    ).astype(int)
    print("  - Garage_Age (tuổi garage, =0 nếu không có garage)")

    df.drop(columns=['Garage Yr Blt'], inplace=True, errors='ignore')
    print("  - Đã xóa Garage Yr Blt (thay bằng Garage_Age và Has_Garage)")

    # Biến thời gian
    df['Age_At_Sale'] = df['Yr Sold'] - df['Year Built']
    df['Remod_Age'] = df['Yr Sold'] - df['Year Remod/Add']
    df['Was_Remodeled'] = (df['Year Remod/Add'] > df['Year Built']).astype(int)

    df.drop(columns=['Year Built', 'Year Remod/Add', 'Yr Sold'], inplace=True)
    print("  - Age_At_Sale, Remod_Age, Was_Remodeled → drop Year Built + Year Remod/Add")

    # Xử lý Low Qual Fin SF
    low_qual_zero_pct = (df['Low Qual Fin SF'] == 0).mean()
    if low_qual_zero_pct > 0.9:
        df['Has_Low_Qual_Fin'] = (df['Low Qual Fin SF'] > 0).astype(int)
        df.drop(columns=['Low Qual Fin SF'], inplace=True)
        print(f"  - Low Qual Fin SF ({low_qual_zero_pct:.1%} = 0) → chuyển thành cờ Has_Low_Qual_Fin")
    else:
        print(f"  - Low Qual Fin SF giữ nguyên (tỷ lệ >0 = {1-low_qual_zero_pct:.1%})")

    # Is_Normal_Sale
    df['Is_Normal_Sale'] = (
        (df['Sale Type'] == 'WD') & (df['Sale Condition'] == 'Normal')
    ).astype(int)
    print("  - Is_Normal_Sale = (Sale Type='WD' và Sale Condition='Normal')")

    negative_conditions = ['Artery', 'RRNn', 'RRAn', 'RRNe', 'RRAe']
    df['Has_Negative_Condition'] = (
        df['Condition 1'].isin(negative_conditions) |
        df['Condition 2'].isin(negative_conditions)
    ).astype(int)
    print("  - Has_Negative_Condition (gần đường sắt/đường lớn)")

    df.drop(columns=['Condition 1', 'Condition 2'], inplace=True)
    print("  - Đã xóa Condition 1, Condition 2 (đã có cờ Has_Negative_Condition)")

    # Tổng diện tích hiên
    df['Total_Porch'] = df['Open Porch SF'] + df['Enclosed Porch'] + df['3Ssn Porch'] + df['Screen Porch']
    df.drop(columns=['Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch'], inplace=True)
    print("  - Total_Porch → drop 4 biến porch gốc")

    # Tổng phòng tắm
    df['Total_Bath'] = df['Full Bath'] + 0.5*df['Half Bath'] + df['Bsmt Full Bath'] + 0.5*df['Bsmt Half Bath']
    df.drop(columns=['Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath'], inplace=True)
    print("  - Total_Bath → drop 4 biến bath gốc")

    # ══════════════════════════════════════════════════════════════════════
    # BƯỚC 6: KIỂM TRA LẦN CUỐI
    # ══════════════════════════════════════════════════════════════════════
    print("\nBƯỚC 6: KIỂM TRA LẦN CUỐI")
    print(f"  Shape: {df.shape[0]} dòng × {df.shape[1]} cột")

    missing_final = df.isnull().sum().sum()
    if missing_final == 0:
        print("  Không còn missing values. Dữ liệu sẵn sàng.")
    else:
        still_missing = df.isnull().sum()[df.isnull().sum() > 0]
        raise ValueError(
            f"Pipeline chưa xử lý hết missing values!\n"
            f"Các cột còn missing:\n{still_missing}\n"
            f"Kiểm tra lại logic từng cột trước khi thêm xử lý."
        )

    return df


if __name__ == "__main__":
    from pathlib import Path
    
    # Lấy thư mục chứa file clean_data.py hiện tại (chính là thư mục part2)
    current_dir = Path(__file__).parent
    
    # Tạo đường dẫn an toàn cho file input và output (sửa lỗi comment 3)
    input_file_path = current_dir / "data" / "AmesHousing.csv"
    output_file_path = current_dir / "data" / "AmesHousing_cleaned.csv"
    
    # Gọi hàm và truyền path vào
    df_clean = clean_data(input_file_path)
    
    # Lưu file
    df_clean.to_csv(output_file_path, index=False)
    print(f"\nĐã lưu file cleaned: {output_file_path}")
