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

# Near-zero variance: gần như hằng số, không phân biệt được nhà
# Utilities: 99.9% = AllPub; Street: 99.6% = Pave
nzv_cols = ['Utilities', 'Street']

# MS SubClass: dtype int64 nhưng là MÃ LOẠI NHÀ, không có thứ tự số học
# (20 không phải "nhỏ hơn" 60). Convert sang string để one-hot encode đúng
df['MS SubClass'] = df['MS SubClass'].astype(str)

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
sum_bsmt = df['BsmtFin SF 1'].fillna(0) + df['BsmtFin SF 2'].fillna(0) + df['Bsmt Unf SF'].fillna(0)
mask_bsmt = df['Total Bsmt SF'].notna() & (np.abs(df['Total Bsmt SF'] - sum_bsmt) > 1)
if mask_bsmt.any():
    print(f"  2.2. {mask_bsmt.sum()} dòng Total Bsmt SF không khớp → điều chỉnh Bsmt Unf SF")
    df.loc[mask_bsmt, 'Bsmt Unf SF'] = (
        df.loc[mask_bsmt, 'Total Bsmt SF']
        - df.loc[mask_bsmt, 'BsmtFin SF 1'].fillna(0)
        - df.loc[mask_bsmt, 'BsmtFin SF 2'].fillna(0)
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
    for idx in df.index[mask_type_missing]:
        grp = df.loc[idx, '_area_group']
        df.loc[idx, 'Mas Vnr Type'] = (
            mode_by_area[grp] if grp in mode_by_area.index else 'BrkFace'
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
    for idx in df.index[mask_area_missing]:
        vtype = df.loc[idx, 'Mas Vnr Type']
        df.loc[idx, 'Mas Vnr Area'] = (
            median_by_type[vtype] if vtype in median_by_type.index else 0
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
    mode_by_qual = (
        df[df['Fireplace Qu'].notna()]
        .groupby('Overall Qual')['Fireplace Qu']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else 'TA')
    )
    for idx in df.index[mask_fire]:
        qual = df.loc[idx, 'Overall Qual']
        df.loc[idx, 'Fireplace Qu'] = (
            mode_by_qual[qual] if qual in mode_by_qual.index else 'TA'
        )
    print(f"    {mask_fire.sum()} dòng có lò sưởi nhưng thiếu Qu → mode theo Overall Qual")
else:
    print("    OK")

# ── 3.3. Garage ──────────────────────────────────────────────────────
# Anchor: đối chiếu chéo 3 cột để tránh quy kết ẩu
# Nhà có garage nếu: Garage Type notna HOẶC Garage Area > 0 HOẶC Garage Cars > 0
# (vì có thể bỏ sót Garage Type nhưng diện tích/số xe vẫn được ghi)
print("  3.3. Garage...")
has_garage = (
    (df['Garage Area'].fillna(0) > 0) |
    (df['Garage Cars'].fillna(0) > 0) |
    (df['Garage Type'].notna() & (df['Garage Type'] != 'None'))
)
no_garage = ~has_garage

# Nhà không có garage: tất cả biến garage = giá trị "không tồn tại"
for col in ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']:
    df.loc[no_garage, col] = 'None'
df.loc[no_garage, ['Garage Cars', 'Garage Area']] = 0

# Nhà có garage — xử lý từng biến theo logic riêng:
has_g = has_garage

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
    for idx in df.index[mask_yr]:
        dec = df.loc[idx, '_decade']
        gt  = df.loc[idx, 'Garage Type'] if pd.notna(df.loc[idx, 'Garage Type']) else 'Attchd'
        if (dec, gt) in median_yr.index:
            df.loc[idx, 'Garage Yr Blt'] = median_yr[(dec, gt)]
        else:
            df.loc[idx, 'Garage Yr Blt'] = df.loc[idx, 'Year Built']
    df.drop(columns='_decade', inplace=True)
    print(f"    {mask_yr.sum()} dòng thiếu Garage Yr Blt → median theo thập niên × loại garage")

# Garage Area: điền theo median của cùng Garage Type
# Lý do: diện tích phụ thuộc nhiều nhất vào kiểu garage
# (Detchd thường nhỏ hơn Attchd; CarPort nhỏ nhất)
mask_area = has_g & df['Garage Area'].isna()
if mask_area.any():
    median_area_by_type = (
        df[df['Garage Area'] > 0]
        .groupby('Garage Type')['Garage Area'].median()
    )
    for idx in df.index[mask_area]:
        gt = df.loc[idx, 'Garage Type']
        df.loc[idx, 'Garage Area'] = (
            median_area_by_type[gt] if gt in median_area_by_type.index else 400
        )
    print(f"    {mask_area.sum()} dòng thiếu Garage Area → median theo Garage Type")

# Garage Cars: điền theo median của Garage Area
mask_cars = has_g & df['Garage Cars'].isna()
if mask_cars.any():
    # Tạo nhóm diện tích
    area_bins = pd.cut(df['Garage Area'], bins=[0, 300, 500, 1000, 2000])
    median_cars_by_area = (
        df[df['Garage Cars'] > 0]
        .groupby(area_bins)['Garage Cars'].median()
    )
    for idx in df.index[mask_cars]:
        area = df.loc[idx, 'Garage Area']
        bin_ = pd.cut([area], bins=[0, 300, 500, 1000, 2000])[0]
        df.loc[idx, 'Garage Cars'] = (
            median_cars_by_area[bin_] if bin_ in median_cars_by_area.index else 2
        )
    print(f"    {mask_cars.sum()} dòng thiếu Garage Cars → median theo Garage Area")

# Garage Finish, Qual, Cond: điền mode theo nhóm (Garage Type, Overall Qual)
# Lý do: chất lượng hoàn thiện phụ thuộc vào kiểu garage và chất lượng tổng thể của ngôi nhà
# Tạo nhóm chất lượng tổng thể
df['_qual_group'] = pd.cut(df['Overall Qual'], bins=[0, 4, 6, 10], labels=['low', 'mid', 'high'])

for col in ['Garage Finish', 'Garage Qual', 'Garage Cond']:
    mask_cat = has_g & df[col].isna()
    if mask_cat.any():
        mode_by_type_qual = (
            df[df[col].notna() & (df[col] != 'None')]
            .groupby(['Garage Type', '_qual_group'], observed=True)[col]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else 'TA')
        )
        for idx in df.index[mask_cat]:
            gt = df.loc[idx, 'Garage Type']
            qg = df.loc[idx, '_qual_group']
            if (gt, qg) in mode_by_type_qual.index:
                df.loc[idx, col] = mode_by_type_qual[(gt, qg)]
            else:
                # fallback: mode theo Garage Type (bỏ qua qual_group)
                mode_by_type_only = (
                    df[df[col].notna() & (df[col] != 'None')]
                    .groupby('Garage Type')[col]
                    .agg(lambda x: x.mode()[0] if not x.mode().empty else 'TA')
                )
                df.loc[idx, col] = mode_by_type_only.get(gt, 'TA')

        print(f"    {mask_cat.sum()} dòng thiếu {col} → mode theo nhóm Garage Type và Overall Qual")
# Sau vòng lặp, xóa cột _qual_group
df.drop(columns='_qual_group', inplace=True)

# Garage Type: chỉ missing khi has_garage=True do Area/Cars > 0 nhưng Type bỏ trống
mask_type = has_g & df['Garage Type'].isna()
if mask_type.any():
    mode_by_area = (
        df[df['Garage Type'].notna()]
        .groupby(pd.cut(df['Garage Area'], bins=[0, 300, 500, 1000]))['Garage Type']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Attchd')
    )
    for idx in df.index[mask_type]:
        area = df.loc[idx, 'Garage Area']
        bin_ = pd.cut([area], bins=[0, 300, 500, 1000])[0]
        df.loc[idx, 'Garage Type'] = (
            mode_by_area[bin_] if bin_ in mode_by_area.index else 'Attchd'
        )
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
mask_type1_zero = has_b & (df['BsmtFin SF 1'] == 0) & (df['BsmtFin Type 1'].isna())
if mask_type1_zero.any():
    df.loc[mask_type1_zero, 'BsmtFin Type 1'] = 'Unf'
    print(f"    {mask_type1_zero.sum()} dòng BsmtFin SF 1 = 0, gán BsmtFin Type 1 = 'Unf'")

mask_type2_zero = has_b & (df['BsmtFin SF 2'] == 0) & (df['BsmtFin Type 2'].isna())
if mask_type2_zero.any():
    df.loc[mask_type2_zero, 'BsmtFin Type 2'] = 'Unf'
    print(f"    {mask_type2_zero.sum()} dòng BsmtFin SF 2 = 0, gán BsmtFin Type 2 = 'Unf'")

# BsmtFin SF 1/2: median theo loại hoàn thiện (BsmtFin Type 1/2)
for col, sf_col in [('BsmtFin Type 1', 'BsmtFin SF 1'), ('BsmtFin Type 2', 'BsmtFin SF 2')]:
    mask = has_b & df[col].isna()
    if mask.any():
        # Chỉ tính bins trên nhà CÓ tầng hầm và SF > 0
        # để tránh các giá trị 0 của nhà không có tầng hầm kéo lệch ranh giới nhóm
        valid = df[has_b & df[sf_col].notna() & (df[sf_col] > 0)]
        bins = pd.qcut(valid[sf_col], q=3, duplicates='drop', retbins=True)[1]

        mode_by_sf = (
            df[df[col].notna() & (df[col] != 'None') & has_b]
            .groupby(pd.cut(valid[sf_col], bins=bins), observed=True)[col]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unf')
        )
        sf_cuts = pd.cut(df[sf_col], bins=bins)
        for idx in df.index[mask]:
            sf_grp = sf_cuts[idx]
            df.loc[idx, col] = mode_by_sf.get(sf_grp, 'Unf')
        print(f"    {mask.sum()} dòng thiếu {col} → mode theo nhóm diện tích {sf_col} (has_basement only)")
# Bsmt Unf SF: phần dư = Total - Fin1 - Fin2
mask_unf = has_b & df['Bsmt Unf SF'].isna()
if mask_unf.any():
    df.loc[mask_unf, 'Bsmt Unf SF'] = (
        df.loc[mask_unf, 'Total Bsmt SF']
        - df.loc[mask_unf, 'BsmtFin SF 1']
        - df.loc[mask_unf, 'BsmtFin SF 2']
    ).clip(lower=0)
    print(f"    {mask_unf.sum()} dòng thiếu Bsmt Unf SF → Total - Fin1 - Fin2")

# Total Bsmt SF: tính lại từ các phần nếu thiếu
mask_total = has_b & df['Total Bsmt SF'].isna()
if mask_total.any():
    df.loc[mask_total, 'Total Bsmt SF'] = (
        df.loc[mask_total, 'BsmtFin SF 1']
        + df.loc[mask_total, 'BsmtFin SF 2']
        + df.loc[mask_total, 'Bsmt Unf SF']
    )
    print(f"    {mask_total.sum()} dòng thiếu Total Bsmt SF → Fin1 + Fin2 + Unf")

# Bsmt Full Bath / Half Bath: 2 hàng MAR (có tầng hầm nhưng bỏ sót)
# Điền 0 là hợp lý vì tầng hầm không nhất thiết phải có phòng tắm
for col in ['Bsmt Full Bath', 'Bsmt Half Bath']:
    mask = has_b & df[col].isna()
    if mask.any():
        df.loc[mask, col] = 0
        print(f"    {mask.sum()} dòng thiếu {col} → 0 (tầng hầm không bắt buộc có phòng tắm)")

# Bsmt Exposure: 80 MNAR (no_basement) + 3 MAR (có tầng hầm nhưng bỏ sót)
mask_exp = has_b & df['Bsmt Exposure'].isna()
if mask_exp.any():
    mode_exp = df.loc[has_b & df['Bsmt Exposure'].notna(), 'Bsmt Exposure'].mode()[0]
    df.loc[mask_exp, 'Bsmt Exposure'] = mode_exp
    print(f"    {mask_exp.sum()} dòng thiếu Bsmt Exposure (có tầng hầm) → mode = '{mode_exp}'")

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
        mode_by_sf = (
            df[df['Bsmt Qual'].notna() & (df['Bsmt Qual'] != 'None')]
            .groupby(pd.qcut(df['Total Bsmt SF'], q=4, duplicates='drop'), observed=True)['Bsmt Qual']
            .agg(lambda x: x.mode()[0] if not x.mode().empty else 'TA')
        )
        sf_bins = pd.qcut(df['Total Bsmt SF'], q=4, duplicates='drop')
        for idx in df.index[mask]:
            grp = sf_bins[idx]
            df.loc[idx, 'Bsmt Qual'] = mode_by_sf.get(grp, 'TA')
        print(f"    {mask.sum()} dòng thiếu Bsmt Qual → mode theo nhóm Total Bsmt SF quantile")

    # Bsmt Cond: tình trạng bảo trì tầng hầm, có 89% = 'TA', tương quan với Overall Cond chỉ 0.11
    # Phân nhóm theo Overall Cond không mang thêm thông tin.
    # Dù Year Built có tương quan cao nhất (r=0.22), 'TA' chiếm 83–94% ở mọi nhóm năm xây dựng.
    # Phân nhóm thêm không thay đổi kết quả → điền 'TA' trực tiếp.
    mask = has_b & df['Bsmt Cond'].isna()
    if mask.any():
        df.loc[mask, 'Bsmt Cond'] = 'TA'
        print(f"    {mask.sum()} dòng thiếu Bsmt Cond → 'TA' (chiếm 89% giá trị)")

    # BsmtFin Type 1: phụ thuộc diện tích hoàn thiện tương ứng
    # Loại hoàn thiện (GLQ, ALQ, Rec...) quyết định bởi diện tích Fin SF
    for col, sf_col in [('BsmtFin Type 1', 'BsmtFin SF 1'), ('BsmtFin Type 2', 'BsmtFin SF 2')]:
        mask = has_b & df[col].isna()
        if mask.any():
            mode_by_sf = (
                df[df[col].notna() & (df[col] != 'None')]
                .groupby(pd.qcut(df['BsmtFin SF 1'], q=3, duplicates='drop'), observed=True)[col]
                .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unf')
            )
            sf_bins = pd.qcut(df[sf_col], q=3, duplicates='drop')
            for idx in df.index[mask]:
                sf_grp = sf_bins[idx]
                df.loc[idx, col] = mode_by_sf.get(sf_grp, 'Unf')
            print(f"    {mask.sum()} dòng thiếu {col} → mode theo nhóm diện tích {sf_col}")

    # BsmtFin Type 2: chỉ 1 hàng missing, phân nhóm không có ý nghĩa
    # Điền mode của nhà có tầng hầm và có BsmtFin SF 2 > 0
    mask = has_b & df['BsmtFin Type 2'].isna()
    if mask.any():
        mode_type2 = df.loc[df['BsmtFin SF 2'] > 0, 'BsmtFin Type 2'].mode()[0]
        df.loc[mask, 'BsmtFin Type 2'] = mode_type2
        print(f"    {mask.sum()} dòng thiếu BsmtFin Type 2 → mode của nhà có BsmtFin SF 2 > 0")

# ══════════════════════════════════════════════════════════════════════
# BƯỚC 4: IMPUTATION LOT FRONTAGE VÀ ELECTRICAL
# ══════════════════════════════════════════════════════════════════════
print("\nBƯỚC 4: IMPUTATION LOT FRONTAGE VÀ ELECTRICAL")

# Lot Frontage (~16.7% missing): MAR — bị bỏ sót khi đo đạc hoặc MCAR — bị bỏ sót ngẫu nhiên trong quá trình thu thập.
# Grouped median theo Neighborhood vì mặt tiền phụ thuộc mạnh vào khu vực
# (dao động 21–92 ft theo neighborhood, khác rất nhiều so với global median 68 ft)
print("  4.1. Lot Frontage → grouped median theo Neighborhood")
median_by_nbhd = df.groupby('Neighborhood')['Lot Frontage'].transform('median')
df['Lot Frontage'] = df['Lot Frontage'].fillna(median_by_nbhd)
# Fallback: Neighborhood quá ít mẫu (GrnHill, Landmrk) → global median
df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median())

# Electrical (1 hàng): MAR đơn lẻ → mode (SBrkr chiếm 98%)
print("  4.2. Electrical → mode")
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# ══════════════════════════════════════════════════════════════════════
# BƯỚC 5: TẠO BIẾN MỚI, XÓA BIẾN GỐC
# ══════════════════════════════════════════════════════════════════════
print("\nBƯỚC 5: TẠO BIẾN MỚI, XÓA BIẾN GỐC")

# Biến cờ có garage (đã có từ bước 3, nhưng lưu lại)
df['Has_Garage'] = has_garage.astype(int)
print("  - Has_Garage (cờ có garage)")

# Tạo tuổi garage chỉ cho nhà có garage (dùng sentinel 0 cho nhà không có)
df['Garage_Age'] = np.where(
    df['Has_Garage'] == 1,
    df['Yr Sold'] - df['Garage Yr Blt'],
    0   # sentinel: nhà không có garage thì tuổi = 0
)
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