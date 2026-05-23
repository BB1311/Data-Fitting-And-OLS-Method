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

import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# BƯỚC 0: ĐỌC DỮ LIỆU
# ══════════════════════════════════════════════════════════════════════
df = pd.read_csv('data/AmesHousing.csv')
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
df['MS SubClass'] = df['MS SubClass'].astype(str)

# Mo Sold: dtype int64 nhưng mang tính mùa vụ (không có tính tuyến tính) -> convert sang string để One-hot
df['Mo Sold'] = df['Mo Sold'].astype(str)

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
        # Bins [0, 300, 500, 1000, 2000] được chọn dựa trên phân phối Garage Area:
        #   - <300 sq.ft  : garage nhỏ (CarPort, Detchd 1 xe) → thường 1 xe
        #   - 300–500     : garage tiêu chuẩn (Attchd 1–2 xe) → thường 2 xe
        #   - 500–1000    : garage lớn (Attchd/BuiltIn 2–3 xe) → thường 2 xe
        #   - >1000       : garage rất lớn → thường 3 xe
        # Nếu thêm dữ liệu từ nguồn khác, nên thay bằng quantile-based bins.
        area_bins_all = pd.cut(df['Garage Area'], bins=[0, 300, 500, 1000, 2000])
        df.loc[mask_cars, 'Garage Cars'] = area_bins_all[mask_cars].map(median_cars_by_area).fillna(2)
    else:
        # Fallback: không có dữ liệu valid, gán 2 (mặc định)
        df.loc[mask_cars, 'Garage Cars'] = 2
    print(f"    {mask_cars.sum()} dòng thiếu Garage Cars → median theo Garage Area (chỉ trên dữ liệu có xe >0)")

# Garage Finish, Qual, Cond: điền mode theo nhóm (Garage Type, Overall Qual)
# Lý do: chất lượng hoàn thiện phụ thuộc vào kiểu garage và chất lượng tổng thể của ngôi nhà
# Tạo nhóm chất lượng tổng thể
df['_qual_group'] = pd.cut(df['Overall Qual'], bins=[0, 4, 6, 10], labels=['low', 'mid', 'high'])

for col in ['Garage Finish', 'Garage Qual', 'Garage Cond']:
    mask_cat = has_g & df[col].isna()
    if mask_cat.any():
        valid = df[col].notna() & (df[col] != 'None')
        # Mode theo (Garage Type, _qual_group)
        mode1 = df[valid].groupby(['Garage Type', '_qual_group'], observed=True)[col].agg(
            lambda x: x.mode()[0] if not x.mode().empty else 'TA'
        )
        mode1_df = mode1.reset_index(name='mode1')
        # Mode chỉ theo Garage Type (fallback)
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
# Sau vòng lặp, xóa cột _qual_group
df.drop(columns='_qual_group', inplace=True)

# Garage Type: chỉ missing khi has_garage=True do Area/Cars > 0 nhưng Type bỏ trống
mask_type = has_g & df['Garage Type'].isna()
if mask_type.any():
    # Chỉ dùng các dòng có Garage Type đã biết để tính bins VÀ mode
    # → đảm bảo index đồng nhất giữa groupby và tra cứu
    valid_garage = df[df['Garage Type'].notna() & (df['Garage Area'] > 0)]
    area_cuts_valid = pd.cut(valid_garage['Garage Area'], bins=[0, 300, 500, 1000])
    mode_by_area = (
        valid_garage.groupby(area_cuts_valid, observed=True)['Garage Type']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Attchd')
    )
    # Dùng toàn bộ df để gán bin cho từng hàng cần impute
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

# Đảm bảo nhất quán: nếu diện tích = 0 thì loại hoàn thiện phải là 'Unf' (xử lý cả NaN)
mask_type1_zero = has_b & (df['BsmtFin SF 1'].fillna(0) == 0) & (df['BsmtFin Type 1'].isna())
if mask_type1_zero.any():
    df.loc[mask_type1_zero, 'BsmtFin Type 1'] = 'Unf'
    print(f"    {mask_type1_zero.sum()} dòng BsmtFin SF 1 = 0, gán BsmtFin Type 1 = 'Unf'")

mask_type2_zero = has_b & (df['BsmtFin SF 2'].fillna(0) == 0) & (df['BsmtFin Type 2'].isna())
if mask_type2_zero.any():
    df.loc[mask_type2_zero, 'BsmtFin Type 2'] = 'Unf'
    print(f"    {mask_type2_zero.sum()} dòng BsmtFin SF 2 = 0, gán BsmtFin Type 2 = 'Unf'")

# Trường hợp 1: biết loại hoàn thiện (Type), thiếu diện tích (SF) → median theo Type ---
for typ_col, sf_col in [('BsmtFin Type 1', 'BsmtFin SF 1'), ('BsmtFin Type 2', 'BsmtFin SF 2')]:
    mask_missing_sf = has_b & df[sf_col].isna()
    if mask_missing_sf.any():
        # Tính median của SF cho từng loại Type (chỉ dùng các dòng có SF > 0)
        median_by_type = df[df[sf_col].notna() & (df[typ_col] != 'None')].groupby(typ_col)[sf_col].median()
        df.loc[mask_missing_sf, sf_col] = df.loc[mask_missing_sf, typ_col].map(median_by_type).fillna(0)
        print(f"    {mask_missing_sf.sum()} dòng thiếu {sf_col} → median theo {typ_col}")

# Trường hợp 2: biết diện tích (SF), thiếu loại hoàn thiện (Type) → mode theo nhóm diện tích ---
for typ_col, sf_col in [('BsmtFin Type 1', 'BsmtFin SF 1'), ('BsmtFin Type 2', 'BsmtFin SF 2')]:
    mask_missing_type = has_b & df[typ_col].isna() & (df[sf_col] > 0)
    if mask_missing_type.any():
        # Chỉ lấy các dòng có SF > 0 để tính các khoảng phân vị (bins)
        sf_valid = df.loc[has_b & (df[sf_col] > 0), sf_col]
        if len(sf_valid) >= 3:
            # Tính 3 khoảng phân vị dựa trên SF (chỉ từ các dòng có hầm và SF>0)
            bins = pd.qcut(sf_valid, q=3, retbins=True, duplicates='drop')[1]
            # Gán mỗi giá trị SF vào một khoảng (bin)
            sf_bins = pd.cut(df[sf_col], bins=bins)
            # Tính mode của Type trong mỗi bin (chỉ dùng các dòng có Type biết)
            mode_by_bin = (
                df[df[typ_col].notna() & (df[typ_col] != 'None') & has_b]
                .groupby(sf_bins, observed=True)[typ_col]
                .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unf')
            )
            df.loc[mask_missing_type, typ_col] = sf_bins[mask_missing_type].map(mode_by_bin).fillna('Unf')
        else:
            # Fallback: nếu không đủ dữ liệu để chia nhóm, dùng mode chung của các dòng có SF>0
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

# Bsmt Full Bath / Half Bath: 2 hàng MAR (có tầng hầm nhưng bỏ sót)
# Điền 0 là hợp lý vì tầng hầm không nhất thiết phải có phòng tắm
for col in ['Bsmt Full Bath', 'Bsmt Half Bath']:
    mask = has_b & df[col].isna()
    if mask.any():
        df.loc[mask, col] = 0
        print(f"    {mask.sum()} dòng thiếu {col} → 0 (tầng hầm không bắt buộc có phòng tắm)")

# Bsmt Exposure: 80 MNAR (no_basement) + 3 MAR (có tầng hầm nhưng bỏ sót)
# Phụ thuộc vào BsmtFin Type 1 (loại hoàn thiện ảnh hưởng đến lộ sáng)
mask_exp = has_b & df['Bsmt Exposure'].isna()
if mask_exp.any():
    # Tính mode theo từng loại BsmtFin Type 1 (chỉ dùng các dòng có Exposure biết)
    mode_by_type = (
        df[has_b & df['Bsmt Exposure'].notna()]
        .groupby('BsmtFin Type 1', observed=True)['Bsmt Exposure']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else 'No')
    )
    # Fallback toàn cục (nếu loại Type không có trong mode_by_type)
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
    # Bsmt Qual: đánh giá chiều cao tầng hầm → phụ thuộc Total Bsmt SF
    mask = has_b & df['Bsmt Qual'].isna()
    if mask.any():
        # Dùng điều kiện notna() & > 0 thay vì chỉ has_b để đảm bảo sf_valid
        # không chứa NaN — tránh index mismatch với groupby nếu has_basement
        # được mở rộng sau này với điều kiện khác ngoài Total Bsmt SF > 0.
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

    # Bsmt Cond: tình trạng bảo trì tầng hầm
    # Dù global mode là 'TA' (91.8%), nhà trước 1900 chỉ có 67–77% là 'TA'
    # và tỉ lệ 'Fa' lên đến 28–33% — điền global mode sẽ sai lệch đáng kể.
    # Dùng mode theo thập niên xây dựng, fallback về 'TA' nếu thập niên không có dữ liệu.
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
# Lot Frontage (~16.7% missing): MAR —  missing rate biến thiên mạnh theo Neighborhood
# (BrDale/Blueste: 0%, GrnHill/Landmrk: 100%, CulDSac: 48.9%) cho thấy missing
# phụ thuộc vào Neighborhood → MAR
# Grouped median theo Neighborhood vì mặt tiền phụ thuộc mạnh vào khu vực
# (dao động 21–92 ft theo neighborhood, khác rất nhiều so với global median 68 ft)
print("  4.1. Lot Frontage → grouped median theo Neighborhood")
median_by_nbhd = df.groupby('Neighborhood')['Lot Frontage'].transform('median')
df['Lot Frontage'] = df['Lot Frontage'].fillna(median_by_nbhd)
# Fallback: Neighborhood quá ít mẫu (GrnHill, Landmrk) → global median
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
# Kiếm tra Garage Yr Blt > Yr Sold trước khi tạo biến Garage_Age
def fix_garage_year_blt(df):
    """
    Xử lý các giá trị Garage Yr Blt bất hợp lý:
    - Nếu > Yr Sold => lỗi nhập liệu => thay thế bằng Year Built.
    - Nếu < Year Built - 5 và Garage không phải Attchd => giữ nguyên (nhà mua đất có garage cũ).
    """
    # Sửa các giá trị vượt quá năm bán (lỗi nhập liệu rõ ràng)
    mask_future = df['Garage Yr Blt'] > df['Yr Sold']
    if mask_future.any():
        print(f"[CẢNH BÁO] {mask_future.sum()} dòng có Garage Yr Blt > Yr Sold. "
              f"Đã thay thế bằng Year Built.")
        df.loc[mask_future, 'Garage Yr Blt'] = df.loc[mask_future, 'Year Built']
    return df

# Kiểm tra an toàn chỉ trên nhà có garage
df = fix_garage_year_blt(df)
mask_garage_check = df['Has_Garage'] == 1
if not (df.loc[mask_garage_check, 'Garage Yr Blt'] <= df.loc[mask_garage_check, 'Yr Sold']).all():
    raise ValueError("Phát hiện Garage Yr Blt > Yr Sold sau khi sửa lỗi!")

# Kiểm tra an toàn: nếu vẫn còn NaN ở Garage Yr Blt cho nhà có garage -> báo lỗi
if ((df['Has_Garage'] == 1) & (df['Garage Yr Blt'].isna())).any():
    raise ValueError("Vẫn còn Garage Yr Blt = NaN cho nhà có garage. Cần kiểm tra lại Bước 3.")

# Tạo tuổi garage chỉ cho nhà có garage (dùng sentinel 0 cho nhà không có)
df['Garage_Age'] = np.where(
    df['Has_Garage'] == 1,
    df['Yr Sold'] - df['Garage Yr Blt'],
    0   # sentinel: nhà không có garage thì tuổi = 0
).astype(int)
print("  - Garage_Age (tuổi garage, =0 nếu không có garage)")

# Nếu muốn giữ Garage Yr Blt gốc, có thể thay NaN bằng sentinel -1
# (nhưng không khuyến khích vì sẽ ảnh hưởng đến các biến tương tác)
# df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(-1)

# Sau khi tạo Garage_Age, xóa Garage Yr Blt gốc
df.drop(columns=['Garage Yr Blt'], inplace=True, errors='ignore')
print("  - Đã xóa Garage Yr Blt (thay bằng Garage_Age và Has_Garage)")

# Biến thời gian
df['Age_At_Sale'] = df['Yr Sold'] - df['Year Built']
df['Remod_Age'] = df['Yr Sold'] - df['Year Remod/Add']
df['Was_Remodeled'] = (df['Year Remod/Add'] > df['Year Built']).astype(int)

# Drop các biến gốc đã được gom
df.drop(columns=['Year Built', 'Year Remod/Add', 'Yr Sold'], inplace=True)
print("  - Age_At_Sale, Remod_Age, Was_Remodeled → drop Year Built + Year Remod/Add")

# Xử lý Low Qual Fin SF (chuyển thành cờ nếu tỷ lệ 0 quá cao)
low_qual_zero_pct = (df['Low Qual Fin SF'] == 0).mean()
if low_qual_zero_pct > 0.9:
    df['Has_Low_Qual_Fin'] = (df['Low Qual Fin SF'] > 0).astype(int)
    df.drop(columns=['Low Qual Fin SF'], inplace=True)
    print(f"  - Low Qual Fin SF ({low_qual_zero_pct:.1%} = 0) → chuyển thành cờ Has_Low_Qual_Fin")
else:
    print(f"  - Low Qual Fin SF giữ nguyên (tỷ lệ >0 = {1-low_qual_zero_pct:.1%})")

# Is_Normal_Sale: giao dịch chuẩn (Warranty Deed + Normal Condition)
df['Is_Normal_Sale'] = (
    (df['Sale Type'] == 'WD') & (df['Sale Condition'] == 'Normal')
).astype(int)
print("  - Is_Normal_Sale = (Sale Type='WD' và Sale Condition='Normal')")
# Giữ lại Sale Type và Sale Condition vì chúng còn chứa thông tin khác:
#   Sale Type = 'New' (nhà mới xây), Sale Condition = 'Partial' (bán chưa hoàn thiện)
# Nếu muốn xóa để tránh trùng lặp, bỏ comment dòng dưới:
# df.drop(columns=['Sale Type', 'Sale Condition'], inplace=True)

negative_conditions = ['Artery', 'RRNn', 'RRAn', 'RRNe', 'RRAe']
df['Has_Negative_Condition'] = (
    df['Condition 1'].isin(negative_conditions) |
    df['Condition 2'].isin(negative_conditions)
).astype(int)
print("  - Has_Negative_Condition (gần đường sắt/đường lớn)")

# Có thể drop Condition 1 và Condition 2 nếu không cần giữ chi tiết
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

    X_train_clean = pipe.fit_transform(X_train, y_train)
    X_test_clean  = pipe.transform(X_test)

    vif_df = run_vif_check(X_train_clean, threshold=10)
    X_train_no_mc, dropped = pipe.drop_high_vif(X_train_clean, threshold=10)
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize


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
# 1. HÀM TIỆN ÍCH — VIF
# ══════════════════════════════════════════════════════════════════════
def run_vif_check(X: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    Tính VIF cho tất cả các cột số trong X.

    VIF_j = 1 / (1 - R²_j), trong đó R²_j là R² khi hồi quy cột j
    theo tất cả các cột còn lại.

    Parameters
    ----------
    X         : DataFrame chứa toàn bộ features (đã encode và scale)
    threshold : Ngưỡng cảnh báo đa cộng tuyến (thường 5 hoặc 10)

    Returns
    -------
    DataFrame với cột 'feature' và 'VIF', sắp xếp giảm dần theo VIF.
    """
    from numpy.linalg import lstsq

    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num = X_num.dropna(axis=1)
    cols = X_num.columns.tolist()

    vifs = []
    for j, col in enumerate(cols):
        y_j = X_num[col].values
        X_others = X_num.drop(columns=[col]).values

        # Thêm intercept
        X_others_int = np.column_stack([np.ones(len(X_others)), X_others])
        beta, _, _, _ = lstsq(X_others_int, y_j, rcond=None)
        y_pred = X_others_int @ beta

        ss_res = np.sum((y_j - y_pred) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        vif = 1 / (1 - r2) if r2 < 1 - 1e-10 else np.inf
        vifs.append({'feature': col, 'VIF': round(vif, 2)})

    vif_df = pd.DataFrame(vifs).sort_values('VIF', ascending=False).reset_index(drop=True)

    high_vif = vif_df[vif_df['VIF'] > threshold]
    if not high_vif.empty:
        print(f"\n[VIF] {len(high_vif)} cột có VIF > {threshold} (đa cộng tuyến cao):")
        print(high_vif.to_string(index=False))
    else:
        print(f"\n[VIF] Không có cột nào có VIF > {threshold}. OK.")

    return vif_df


# ══════════════════════════════════════════════════════════════════════
# 2. CLASS DataPipeline
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
        self._ordinal_medians: dict = {}      # Lưu median để tránh leakage
        self._skewed_cols_to_log: list = []   # Lưu cột cần log
        self._onehot_cols_: list   = []   
        self._ordinal_cols_: list  = []   
        self._dummy_cols_: list    = []   
        self._feature_names_: list = []   
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
        valid_cols = [c for c in num_cols if X[c].min() >= 0] # Chỉ lấy cột không âm
        skewness = X[valid_cols].skew()
        
        self._skewed_cols_to_log = skewness[skewness > 0.75].index.tolist()
        self._skewed_cols_to_log = [c for c in self._skewed_cols_to_log if c not in self._ordinal_cols_]

    def _apply_log_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ép hàm log1p lên các cột đã phát hiện"""
        if not self.log_skewed_features: return X
        X = X.copy()
        for col in self._skewed_cols_to_log:
            if col in X.columns:
                X[col] = np.log1p(X[col])
        return X
    

    # ------------------------------------------------------------------
    # PHẦN A: Outlier
    # ------------------------------------------------------------------
    def _detect_outlier_cols(self, X: pd.DataFrame) -> list:
        """Tự động chọn các cột số liên tục để xử lý outlier."""
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # Bỏ qua cột nhị phân (chỉ có 0/1) và cột đếm nhỏ (Garage Cars, v.v.)
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
            # Xử lý cả target nếu được truyền vào
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
            # Trên train: hàng nằm ngoài sẽ bị xóa
            # Trên test: clip thay vì xóa để không mất hàng
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
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

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

    def _apply_ordinal(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self._ordinal_cols_:
            if col not in X.columns:
                continue
            if col in ORDINAL_MAPS:
                X[col] = X[col].map(ORDINAL_MAPS[col])
                if X[col].isna().any():
                    # --- FIX LEAKAGE: Lấy median đã học từ Train ra xài ---
                    med_val = self._ordinal_medians.get(col, 0)
                    X[col] = X[col].fillna(med_val)
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
    # PHẦN D: fit / transform / fit_transform
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> 'DataPipeline':
        print("=== DataPipeline.fit() ===")
        # Log target
        _y = np.log1p(y) if (self.log_target and y is not None) else y

        X_temp = self._create_new_features(X)
        
        if self.outlier_method:
            print(f"  [Outlier] Học tham số outlier (method='{self.outlier_method}')...")
            self._fit_outlier(X_temp, _y)

        print("  [Encoding] Phân loại cột categorical...")
        self._fit_encoding(X_temp)
        X_temp = self._apply_ordinal(X_temp)
        
        X_temp = self._create_interactions(X_temp)
        self._fit_log_features(X_temp)
        X_temp = self._apply_log_features(X_temp)

        self._fit_onehot(X_temp)
        if self.scale:
            X_enc = self._apply_onehot(X_temp)
            print(f"  [Scale] Học tham số z-score...")
            self._fit_scale(X_enc)

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
        if self.log_target and y is not None:
            print("  [Target] Áp dụng np.log1p(y)")
            y = np.log1p(y)
            
        self.fit(X, y)

        X = self._create_new_features(X)
        
        if self.outlier_method == 'remove' and y is not None:
            X, y = self._remove_outlier_rows(X, y)
        elif self.outlier_method == 'winsorize':
            X = self._transform_outlier_X(X)
            if y is not None and '__target__' in self._winsor_limits:
                lo, hi = self._winsor_limits['__target__']
                y = y.clip(lower=lo, upper=hi)
        else:
            X = self._transform_outlier_X(X)

        X = self._apply_ordinal(X)
        X = self._create_interactions(X)
        X = self._apply_log_features(X)
        X = self._apply_onehot(X)
        
        if self.scale:
            X = self._apply_scale(X)

        self._feature_names_ = X.columns.tolist()
        print(f"  Shape sau pipeline: {X.shape}")
        return X, y
    # ------------------------------------------------------------------
    # PHẦN E: VIF — loại cột đa cộng tuyến
    # ------------------------------------------------------------------
    def drop_high_vif(
        self,
        X: pd.DataFrame,
        threshold: float = 10.0,
        max_iter: int = 20,
    ) -> tuple[pd.DataFrame, list]:
        """
        Lặp loại bỏ cột có VIF cao nhất cho đến khi tất cả VIF <= threshold.

        Returns
        -------
        X_reduced : DataFrame không còn cột đa cộng tuyến cao
        dropped   : list tên cột đã bị loại
        """
        dropped = []
        for iteration in range(max_iter):
            vif_df = run_vif_check(X, threshold=threshold)
            worst = vif_df.iloc[0]
            if worst['VIF'] <= threshold or np.isinf(worst['VIF']) is False and worst['VIF'] <= threshold:
                break
            if worst['VIF'] == np.inf or worst['VIF'] > threshold:
                print(f"  [VIF iter {iteration+1}] Loại '{worst['feature']}' (VIF={worst['VIF']})")
                X = X.drop(columns=[worst['feature']])
                dropped.append(worst['feature'])
            else:
                break
        print(f"\n[VIF] Đã loại {len(dropped)} cột: {dropped}")
        return X, dropped

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

