"""
test_clean_data.py
==================

Chiến lược:
  • _base_row(idx) tạo một dict đầy đủ cột tối thiểu, không lỗi, không missing.
  • Mỗi test chỉ override đúng trường cần kiểm tra.
  • _run(tmp_path, rows) ghi CSV tạm → gọi clean_data() → trả về DataFrame kết quả.
  • Sử dụng pytest built-in `tmp_path` để tránh phụ thuộc đường dẫn tuyệt đối.

Bất biến của _base_row (phải duy trì khi override):
  • Gr Liv Area      = 1st Flr SF + 2nd Flr SF + Low Qual Fin SF  (1400 = 800+600+0)
  • Total Bsmt SF    = BsmtFin SF 1 + BsmtFin SF 2 + Bsmt Unf SF  (600 = 400+0+200)
  • Year Remod/Add  >= Year Built                                   (2005 >= 2000)
  • Year Built      <= Yr Sold                                      (2000 <= 2010)
  • Garage Yr Blt   <= Yr Sold  (khi có garage)                     (2000 <= 2010)
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part2.clean_data import clean_data


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def _base_row(idx: int = 1) -> dict:
    """
    Dict chứa đầy đủ các cột tối thiểu mà clean_data() cần.
    Tất cả giá trị hợp lệ, không missing, không vi phạm logic.
    Dùng làm nền — mỗi test chỉ override đúng trường cần kiểm tra.
    """
    return {
        # ── Định danh (bị xóa trong Bước 1) ───────────────────────
        "Order": idx,
        "PID": 535300000 + idx,
        # ── Pool ───────────────────────────────────────────────────
        "Pool Area": 0,
        "Pool QC": np.nan,
        # ── Misc ───────────────────────────────────────────────────
        "Misc Feature": np.nan,
        "Misc Val": 0,
        # ── Alley / Fence ──────────────────────────────────────────
        "Alley": np.nan,
        "Fence": np.nan,
        # ── Near-zero variance (bị xóa) ────────────────────────────
        "Utilities": "AllPub",
        "Street": "Pave",
        # ── Loại nhà và tháng bán ──────────────────────────────────
        "MS SubClass": 60,
        "Mo Sold": 6,
        # ── Garage (đầy đủ, không missing) ─────────────────────────
        "Garage Type": "Attchd",
        "Garage Finish": "Fin",
        "Garage Qual": "TA",
        "Garage Cond": "TA",
        "Garage Cars": 2.0,
        "Garage Area": 500.0,
        "Garage Yr Blt": 2000.0,
        # ── Diện tích các tầng (nhất quán) ─────────────────────────
        "1st Flr SF": 800,
        "2nd Flr SF": 600,
        "Low Qual Fin SF": 0,
        "Gr Liv Area": 1400,        # = 800 + 600 + 0
        # ── Basement (đầy đủ, nhất quán) ───────────────────────────
        "BsmtFin SF 1": 400,
        "BsmtFin SF 2": 0,
        "Bsmt Unf SF": 200,
        "Total Bsmt SF": 600,       # = 400 + 0 + 200
        "Bsmt Qual": "TA",
        "Bsmt Cond": "TA",
        "Bsmt Exposure": "No",
        "BsmtFin Type 1": "GLQ",
        "BsmtFin Type 2": "Unf",
        "Bsmt Full Bath": 1.0,
        "Bsmt Half Bath": 0.0,
        # ── Phòng tắm ──────────────────────────────────────────────
        "Full Bath": 2,
        "Half Bath": 1,
        # ── Năm (nhất quán) ────────────────────────────────────────
        "Year Built": 2000,
        "Year Remod/Add": 2005,
        "Yr Sold": 2010,
        # ── Lớp ốp đá (không có) ───────────────────────────────────
        "Mas Vnr Type": np.nan,
        "Mas Vnr Area": 0.0,
        # ── Lò sưởi (không có) ─────────────────────────────────────
        "Fireplaces": 0,
        "Fireplace Qu": np.nan,
        "Overall Qual": 7,
        "Bldg Type": "1Fam",
        # ── Lô đất ─────────────────────────────────────────────────
        "Neighborhood": "NAmes",
        "Lot Frontage": 65.0,
        # ── Điện ───────────────────────────────────────────────────
        "Electrical": "SBrkr",
        # ── Giao dịch ──────────────────────────────────────────────
        "Sale Type": "WD",
        "Sale Condition": "Normal",
        "Condition 1": "Norm",
        "Condition 2": "Norm",
        # ── Hiên ───────────────────────────────────────────────────
        "Open Porch SF": 61,
        "Enclosed Porch": 0,
        "3Ssn Porch": 0,
        "Screen Porch": 0,
        # ── Giá mục tiêu ───────────────────────────────────────────
        "SalePrice": 208500,
    }


def _run(tmp_path, rows: list) -> pd.DataFrame:
    """Ghi list[dict] ra CSV tạm, gọi clean_data(), trả về DataFrame kết quả."""
    path = tmp_path / "ames_test.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return clean_data(str(path))


# ══════════════════════════════════════════════════════════════════════
# BƯỚC 1 — TẠO CỜ VÀ XÓA CỘT
# ══════════════════════════════════════════════════════════════════════

class TestStep1_FlagsAndDrops:
    """
    Kiểm tra Bước 1: tạo cờ Has_* và xóa các cột không cần thiết.
    """

    def test_has_pool_flag_from_pool_area(self, tmp_path):
        """
        Pool Area = 0 → Has_Pool = 0
        Pool Area > 0 → Has_Pool = 1
        """
        row_no  = {**_base_row(1), "Pool Area": 0}
        row_yes = {**_base_row(2), "Pool Area": 100}
        out = _run(tmp_path, [row_no, row_yes, _base_row(3)])
        assert out.iloc[0]["Has_Pool"] == 0, "Pool Area=0 phải cho Has_Pool=0"
        assert out.iloc[1]["Has_Pool"] == 1, "Pool Area=100 phải cho Has_Pool=1"

    def test_has_alley_and_fence_flags(self, tmp_path):
        """
        Alley=NaN, Fence=NaN → Has_Alley=0, Has_Fence=0
        Alley='Grvl', Fence='MnPrv' → Has_Alley=1, Has_Fence=1
        """
        row_no  = {**_base_row(1), "Alley": np.nan, "Fence": np.nan}
        row_yes = {**_base_row(2), "Alley": "Grvl", "Fence": "MnPrv"}
        out = _run(tmp_path, [row_no, row_yes, _base_row(3)])
        assert out.iloc[0]["Has_Alley"] == 0 and out.iloc[0]["Has_Fence"] == 0
        assert out.iloc[1]["Has_Alley"] == 1 and out.iloc[1]["Has_Fence"] == 1

    def test_has_misc_feature_flag(self, tmp_path):
        """
        Misc Feature = NaN → Has_Misc_Feature = 0
        Misc Feature = 'Gar2' → Has_Misc_Feature = 1
        """
        row_no  = {**_base_row(1), "Misc Feature": np.nan}
        row_yes = {**_base_row(2), "Misc Feature": "Gar2"}
        out = _run(tmp_path, [row_no, row_yes, _base_row(3)])
        assert out.iloc[0]["Has_Misc_Feature"] == 0
        assert out.iloc[1]["Has_Misc_Feature"] == 1

    def test_has_garage_flag_from_area(self, tmp_path):
        """
        Garage Area > 0 → Has_Garage = 1
        Garage Area = Cars = 0 và Type = NaN → Has_Garage = 0
        """
        row_yes = _base_row(1)   # Garage Area = 500 (default)
        row_no  = {
            **_base_row(2),
            "Garage Type": np.nan, "Garage Area": 0.0, "Garage Cars": 0.0,
            "Garage Finish": np.nan, "Garage Qual": np.nan, "Garage Cond": np.nan,
            "Garage Yr Blt": np.nan,
        }
        out = _run(tmp_path, [row_yes, row_no, _base_row(3)])
        assert out.iloc[0]["Has_Garage"] == 1, "Garage Area=500 phải cho Has_Garage=1"
        assert out.iloc[1]["Has_Garage"] == 0, "Không có garage phải cho Has_Garage=0"

    def test_id_and_nzv_cols_dropped(self, tmp_path):
        """
        'Order', 'PID' (định danh) phải bị xóa.
        'Utilities', 'Street' (near-zero variance) phải bị xóa.
        """
        out = _run(tmp_path, [_base_row(i) for i in range(1, 4)])
        for col in ["Order", "PID", "Utilities", "Street"]:
            assert col not in out.columns, f"'{col}' vẫn còn trong output"

    def test_high_missing_cols_replaced_by_flags(self, tmp_path):
        """
        'Pool QC', 'Misc Feature', 'Alley', 'Fence' bị xóa.
        Cờ 'Has_Pool', 'Has_Misc_Feature', 'Has_Alley', 'Has_Fence' có mặt trong output.
        """
        out = _run(tmp_path, [_base_row(i) for i in range(1, 4)])
        for col in ["Pool QC", "Misc Feature", "Alley", "Fence"]:
            assert col not in out.columns, f"'{col}' phải bị xóa"
        for flag in ["Has_Pool", "Has_Misc_Feature", "Has_Alley", "Has_Fence"]:
            assert flag in out.columns, f"Cờ '{flag}' thiếu trong output"

    def test_ms_subclass_and_mo_sold_converted_to_str(self, tmp_path):
        """
        MS SubClass (int=60) → str '60' trong output.
        Mo Sold (int=6) → str '6' trong output.
        """
        out = _run(tmp_path, [_base_row(i) for i in range(1, 4)])
        assert out["MS SubClass"].dtype == object, "MS SubClass phải là dtype object"
        assert out["Mo Sold"].dtype == object,     "Mo Sold phải là dtype object"
        assert out["MS SubClass"].iloc[0] == "60"
        assert out["Mo Sold"].iloc[0] == "6"


# ══════════════════════════════════════════════════════════════════════
# BƯỚC 2 — TÍNH NHẤT QUÁN
# ══════════════════════════════════════════════════════════════════════

class TestStep2_Consistency:
    """
    Kiểm tra Bước 2: sửa các mâu thuẫn giữa các biến (Gr Liv Area, năm, v.v.)
    """

    def test_gr_liv_area_corrected_when_lệch_over_10pct(self, tmp_path):
        """
        Gr Liv Area lệch >10% so với tổng tầng → được gán lại = tổng tầng.
        Input:  1st=800, 2nd=700, LQ=0 → sum=1500; Gr Liv Area=500 (lệch 67%)
        Output: Gr Liv Area = 1500
        """
        row = {**_base_row(), "1st Flr SF": 800, "2nd Flr SF": 700,
               "Low Qual Fin SF": 0, "Gr Liv Area": 500}
        out = _run(tmp_path, [row] * 3)
        assert out["Gr Liv Area"].iloc[0] == 1500

    def test_gr_liv_area_unchanged_when_consistent(self, tmp_path):
        """
        Gr Liv Area khớp chính xác với tổng tầng → không bị sửa.
        Input: 1st=800, 2nd=600, LQ=0 → sum=1400; Gr Liv Area=1400
        Output: Gr Liv Area = 1400 (không đổi)
        """
        row = {**_base_row(), "1st Flr SF": 800, "2nd Flr SF": 600,
               "Low Qual Fin SF": 0, "Gr Liv Area": 1400}
        out = _run(tmp_path, [row] * 3)
        assert out["Gr Liv Area"].iloc[0] == 1400

    def test_year_remod_corrected_when_before_year_built(self, tmp_path):
        """
        Year Remod/Add < Year Built → được gán bằng Year Built → Was_Remodeled = 0.
        Input: Year Built=2005, Year Remod/Add=2000 (lỗi nhập liệu)
        Output: Was_Remodeled = 0 (sau khi sửa: Remod = Built → không phải cải tạo)
        """
        row = {**_base_row(), "Year Built": 2005, "Year Remod/Add": 2000,
               "Yr Sold": 2015, "Garage Yr Blt": 2005.0}
        out = _run(tmp_path, [row] * 3)
        assert out["Was_Remodeled"].iloc[0] == 0

    def test_year_remod_valid_preserves_remod_age(self, tmp_path):
        """
        Year Remod/Add hợp lệ (> Year Built) → Remod_Age và Was_Remodeled đúng.
        Input: Year Built=1990, Year Remod/Add=2000, Yr Sold=2015
        Output: Remod_Age = 15, Was_Remodeled = 1
        """
        row = {**_base_row(), "Year Built": 1990, "Year Remod/Add": 2000,
               "Yr Sold": 2015, "Garage Yr Blt": 1990.0}
        out = _run(tmp_path, [row] * 3)
        assert out["Remod_Age"].iloc[0] == 15
        assert out["Was_Remodeled"].iloc[0] == 1


# ══════════════════════════════════════════════════════════════════════
# BƯỚC 3 — MISSING CÓ CẤU TRÚC (STRUCTURAL MISSING)
# ══════════════════════════════════════════════════════════════════════

class TestStep3_StructuralMissing:
    """
    Kiểm tra Bước 3: điền missing theo quan hệ nhân-quả (Mas Vnr, Fireplace, Garage, Basement).
    """

    def test_fireplace_qu_set_to_none_when_no_fireplace(self, tmp_path):
        """
        Fireplaces = 0, Fireplace Qu = NaN → Fireplace Qu = 'None' trong output.
        Lý do: Fireplaces = 0 là anchor — không thể có chất lượng lò khi không có lò.
        """
        row = {**_base_row(), "Fireplaces": 0, "Fireplace Qu": np.nan}
        out = _run(tmp_path, [row] * 3)
        assert out["Fireplace Qu"].iloc[0] == "None"

    def test_fireplace_qu_preserved_when_has_fireplace(self, tmp_path):
        """
        Fireplaces > 0, Fireplace Qu đã có giá trị → giá trị gốc được giữ nguyên.
        Input: Fireplaces=1, Fireplace Qu='Gd' → Output: Fireplace Qu='Gd'
        """
        row = {**_base_row(), "Fireplaces": 1, "Fireplace Qu": "Gd"}
        out = _run(tmp_path, [row] * 3)
        assert out["Fireplace Qu"].iloc[0] == "Gd"

    def test_masonry_set_to_none_when_area_is_zero(self, tmp_path):
        """
        Mas Vnr Area = 0, Mas Vnr Type = NaN → Type = 'None', Area = 0.
        Không có ốp đá → cả Type và Area phải được chuẩn hóa về 'None'/0.
        """
        row = {**_base_row(), "Mas Vnr Area": 0.0, "Mas Vnr Type": np.nan}
        out = _run(tmp_path, [row] * 3)
        assert out["Mas Vnr Type"].iloc[0] == "None"
        assert out["Mas Vnr Area"].iloc[0] == 0.0

    def test_masonry_preserved_when_present(self, tmp_path):
        """
        Mas Vnr Area > 0, Mas Vnr Type đã có → giữ nguyên, không bị ghi đè.
        Input: Area=200, Type='BrkFace' → Output: Area=200, Type='BrkFace'
        """
        row = {**_base_row(), "Mas Vnr Area": 200.0, "Mas Vnr Type": "BrkFace"}
        out = _run(tmp_path, [row] * 3)
        assert out["Mas Vnr Type"].iloc[0] == "BrkFace"
        assert out["Mas Vnr Area"].iloc[0] == 200.0

    def test_no_garage_categorical_cols_set_to_none(self, tmp_path):
        """
        Has_Garage = 0 → Garage Type/Finish/Qual/Cond = 'None'; Cars = Area = 0.
        Garage_Age phải = 0 (np.where trả về 0 khi không có garage).
        """
        row_no = {
            **_base_row(1),
            "Garage Type": np.nan, "Garage Area": 0.0, "Garage Cars": 0.0,
            "Garage Finish": np.nan, "Garage Qual": np.nan, "Garage Cond": np.nan,
            "Garage Yr Blt": np.nan,
        }
        out = _run(tmp_path, [row_no, _base_row(2), _base_row(3)])
        r = out.iloc[0]
        assert r["Has_Garage"] == 0
        assert r["Garage_Age"] == 0

    def test_no_garage_garage_age_is_zero_not_nan(self, tmp_path):
        """
        Garage_Age của nhà không có garage = 0 (int), không phải NaN.
        Đảm bảo np.where xử lý đúng khi Garage Yr Blt = NaN.
        """
        row_no = {
            **_base_row(1),
            "Garage Type": np.nan, "Garage Area": 0.0, "Garage Cars": 0.0,
            "Garage Finish": np.nan, "Garage Qual": np.nan, "Garage Cond": np.nan,
            "Garage Yr Blt": np.nan,
        }
        out = _run(tmp_path, [row_no, _base_row(2), _base_row(3)])
        val = out.iloc[0]["Garage_Age"]
        assert val == 0
        assert not np.isnan(float(val)), "Garage_Age không được là NaN"


# ══════════════════════════════════════════════════════════════════════
# BƯỚC 4 — IMPUTATION LOT FRONTAGE VÀ ELECTRICAL
# ══════════════════════════════════════════════════════════════════════

class TestStep4_Imputation:
    """
    Kiểm tra Bước 4: Lot Frontage điền theo grouped median; Electrical điền theo mode.
    """

    def test_lot_frontage_filled_by_neighborhood_median(self, tmp_path):
        """
        Lot Frontage NaN → điền median của cùng Neighborhood.
        Nhóm 'NAmes' có LF = [60, 70, 80, 90] → median = 75.
        Dòng thiếu trong cùng nhóm → Lot Frontage = 75.0.
        """
        rows = []
        for i, lf in enumerate([60.0, 70.0, 80.0, 90.0], start=1):
            rows.append({**_base_row(i), "Neighborhood": "NAmes", "Lot Frontage": lf})
        rows.append({**_base_row(5), "Neighborhood": "NAmes", "Lot Frontage": np.nan})
        out = _run(tmp_path, rows)
        assert abs(out.iloc[4]["Lot Frontage"] - 75.0) < 0.1, \
            f"Expected 75.0, got {out.iloc[4]['Lot Frontage']}"

    def test_lot_frontage_unchanged_when_present(self, tmp_path):
        """
        Lot Frontage đã có giá trị → không bị thay đổi bởi grouped median.
        Input: Lot Frontage = 88.0 → Output: Lot Frontage = 88.0
        """
        row = {**_base_row(), "Lot Frontage": 88.0}
        out = _run(tmp_path, [row] * 3)
        assert out["Lot Frontage"].iloc[0] == 88.0

    def test_electrical_missing_filled_with_mode(self, tmp_path):
        """
        Electrical = NaN (1 dòng) → điền mode = 'SBrkr' (chiếm đa số).
        2 dòng 'SBrkr', 1 dòng NaN → dòng NaN được điền 'SBrkr'.
        """
        rows = [
            {**_base_row(1), "Electrical": "SBrkr"},
            {**_base_row(2), "Electrical": "SBrkr"},
            {**_base_row(3), "Electrical": np.nan},
        ]
        out = _run(tmp_path, rows)
        assert out.iloc[2]["Electrical"] == "SBrkr"

    def test_electrical_non_mode_value_unchanged(self, tmp_path):
        """
        Electrical đã có giá trị khác mode → giữ nguyên, không bị ghi đè.
        1 dòng 'FuseA', 2 dòng 'SBrkr' → dòng 'FuseA' vẫn là 'FuseA'.
        """
        rows = [
            {**_base_row(1), "Electrical": "FuseA"},
            {**_base_row(2), "Electrical": "SBrkr"},
            {**_base_row(3), "Electrical": "SBrkr"},
        ]
        out = _run(tmp_path, rows)
        assert out.iloc[0]["Electrical"] == "FuseA"


# ══════════════════════════════════════════════════════════════════════
# BƯỚC 5 — TẠO BIẾN MỚI, XÓA BIẾN GỐC
# ══════════════════════════════════════════════════════════════════════

class TestStep5_FeatureEngineering:
    """
    Kiểm tra Bước 5: công thức tính các biến mới trên dữ liệu đã biết kết quả.
    """

    def test_age_at_sale_formula(self, tmp_path):
        """
        Age_At_Sale = Yr Sold - Year Built.
        Case 1: 2010 - 1990 = 20
        Case 2: 2008 - 2005 = 3
        """
        row1 = {**_base_row(1), "Year Built": 1990, "Yr Sold": 2010,
                "Year Remod/Add": 1990, "Garage Yr Blt": 1990.0}
        row2 = {**_base_row(2), "Year Built": 2005, "Yr Sold": 2008,
                "Year Remod/Add": 2005, "Garage Yr Blt": 2005.0}
        out = _run(tmp_path, [row1, row2, _base_row(3)])
        assert out.iloc[0]["Age_At_Sale"] == 20
        assert out.iloc[1]["Age_At_Sale"] == 3

    def test_total_bath_formula(self, tmp_path):
        """
        Total_Bath = Full Bath + 0.5×Half Bath + Bsmt Full Bath + 0.5×Bsmt Half Bath.
        Case 1: 2 + 0.5×1 + 1 + 0.5×0 = 3.5
        Case 2: 1 + 0.5×0 + 0 + 0.5×1 = 1.5
        """
        row1 = {**_base_row(1), "Full Bath": 2, "Half Bath": 1,
                "Bsmt Full Bath": 1.0, "Bsmt Half Bath": 0.0}   # 3.5
        row2 = {**_base_row(2), "Full Bath": 1, "Half Bath": 0,
                "Bsmt Full Bath": 0.0, "Bsmt Half Bath": 1.0}   # 1.5
        out = _run(tmp_path, [row1, row2, _base_row(3)])
        assert abs(out.iloc[0]["Total_Bath"] - 3.5) < 1e-9
        assert abs(out.iloc[1]["Total_Bath"] - 1.5) < 1e-9

    def test_total_porch_formula(self, tmp_path):
        """
        Total_Porch = Open Porch SF + Enclosed Porch + 3Ssn Porch + Screen Porch.
        Case 1: 50 + 20 + 10 + 5 = 85
        Case 2: 0 + 0 + 0 + 0 = 0
        """
        row1 = {**_base_row(1), "Open Porch SF": 50, "Enclosed Porch": 20,
                "3Ssn Porch": 10, "Screen Porch": 5}    # 85
        row2 = {**_base_row(2), "Open Porch SF": 0,  "Enclosed Porch": 0,
                "3Ssn Porch": 0,  "Screen Porch": 0}    # 0
        out = _run(tmp_path, [row1, row2, _base_row(3)])
        assert out.iloc[0]["Total_Porch"] == 85
        assert out.iloc[1]["Total_Porch"] == 0

    def test_is_normal_sale_flag(self, tmp_path):
        """
        Is_Normal_Sale = 1 khi Sale Type='WD' VÀ Sale Condition='Normal'.
        Is_Normal_Sale = 0 khi một trong hai điều kiện không thỏa mãn.
        """
        row_normal = {**_base_row(1), "Sale Type": "WD",  "Sale Condition": "Normal"}
        row_abnorm = {**_base_row(2), "Sale Type": "COD", "Sale Condition": "Abnorml"}
        out = _run(tmp_path, [row_normal, row_abnorm, _base_row(3)])
        assert out.iloc[0]["Is_Normal_Sale"] == 1
        assert out.iloc[1]["Is_Normal_Sale"] == 0

    def test_has_negative_condition_flag(self, tmp_path):
        """
        Condition 1 ∈ {'Artery','RRNn','RRAn','RRNe','RRAe'} → Has_Negative_Condition = 1.
        Condition 1 = Condition 2 = 'Norm' → Has_Negative_Condition = 0.
        """
        row_neg = {**_base_row(1), "Condition 1": "Artery", "Condition 2": "Norm"}
        row_ok  = {**_base_row(2), "Condition 1": "Norm",   "Condition 2": "Norm"}
        out = _run(tmp_path, [row_neg, row_ok, _base_row(3)])
        assert out.iloc[0]["Has_Negative_Condition"] == 1
        assert out.iloc[1]["Has_Negative_Condition"] == 0

    def test_garage_age_formula(self, tmp_path):
        """
        Garage_Age = Yr Sold - Garage Yr Blt (khi Has_Garage = 1).
        Case 1: 2010 - 1995 = 15
        Case 2: 2008 - 2000 = 8
        """
        row1 = {**_base_row(1), "Yr Sold": 2010, "Garage Yr Blt": 1995.0,
                "Year Built": 1995, "Year Remod/Add": 1995}
        row2 = {**_base_row(2), "Yr Sold": 2008, "Garage Yr Blt": 2000.0}
        out = _run(tmp_path, [row1, row2, _base_row(3)])
        assert out.iloc[0]["Garage_Age"] == 15
        assert out.iloc[1]["Garage_Age"] == 8

    def test_was_remodeled_flag(self, tmp_path):
        """
        Was_Remodeled = 1 khi Year Remod/Add > Year Built (đã cải tạo).
        Was_Remodeled = 0 khi Year Remod/Add == Year Built (chưa cải tạo).
        """
        row_yes = {**_base_row(1), "Year Built": 1990, "Year Remod/Add": 2000,
                   "Yr Sold": 2015, "Garage Yr Blt": 1990.0}
        row_no  = {**_base_row(2), "Year Built": 2000, "Year Remod/Add": 2000,
                   "Yr Sold": 2015, "Garage Yr Blt": 2000.0}
        out = _run(tmp_path, [row_yes, row_no, _base_row(3)])
        assert out.iloc[0]["Was_Remodeled"] == 1
        assert out.iloc[1]["Was_Remodeled"] == 0

    def test_has_low_qual_fin_flag_created_when_mostly_zero(self, tmp_path):
        """
        Low Qual Fin SF = 0 cho >90% dòng → cột bị đổi thành cờ Has_Low_Qual_Fin.
        10 dòng = 0 + 1 dòng = 200 → pct_zero = 10/11 ≈ 91% > 90% → flag được tạo.
        Dòng có Low Qual > 0 → Has_Low_Qual_Fin = 1; dòng = 0 → Has_Low_Qual_Fin = 0.
        """
        rows = []
        for i in range(1, 11):
            rows.append({**_base_row(i), "Low Qual Fin SF": 0,   "Gr Liv Area": 1400})
        rows.append(       {**_base_row(11), "Low Qual Fin SF": 200, "Gr Liv Area": 1600})
        out = _run(tmp_path, rows)
        assert "Has_Low_Qual_Fin" in out.columns,     "Thiếu cột Has_Low_Qual_Fin"
        assert "Low Qual Fin SF" not in out.columns,  "Low Qual Fin SF chưa bị xóa"
        assert out.iloc[0]["Has_Low_Qual_Fin"] == 0
        assert out.iloc[10]["Has_Low_Qual_Fin"] == 1


# ══════════════════════════════════════════════════════════════════════
# BƯỚC 6 — TÍNH HỢP LỆ CỦA OUTPUT CUỐI
# ══════════════════════════════════════════════════════════════════════

class TestOutputValidity:
    """
    Kiểm tra tính đúng đắn tổng thể của output: không missing, cột đúng.
    """

    def test_no_missing_values_in_output(self, tmp_path):
        """
        Pipeline phải trả về DataFrame hoàn toàn không còn giá trị NaN.
        Nếu còn missing → Bước 6 sẽ raise ValueError (đây là safety net thứ hai).
        """
        out = _run(tmp_path, [_base_row(i) for i in range(1, 4)])
        missing = out.isnull().sum().sum()
        assert missing == 0, f"Output còn {missing} giá trị NaN trong các cột: " \
                             f"{list(out.columns[out.isnull().any()])}"

    def test_component_cols_replaced_by_aggregate_features(self, tmp_path):
        """
        Các cột gốc bị xóa; các cột tổng hợp mới xuất hiện:
          • Porch gốc (4 cột) → Total_Porch
          • Bath gốc  (4 cột) → Total_Bath
          • Year cols (3 cột) → Age_At_Sale, Remod_Age, Was_Remodeled
          • Garage Yr Blt     → Garage_Age
          • Condition 1/2     → Has_Negative_Condition
        """
        out = _run(tmp_path, [_base_row(i) for i in range(1, 4)])
        dropped = [
            "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch",
            "Full Bath", "Half Bath", "Bsmt Full Bath", "Bsmt Half Bath",
            "Year Built", "Year Remod/Add", "Yr Sold", "Garage Yr Blt",
            "Condition 1", "Condition 2",
        ]
        created = [
            "Total_Porch", "Total_Bath", "Age_At_Sale", "Remod_Age",
            "Was_Remodeled", "Garage_Age", "Has_Negative_Condition",
        ]
        for col in dropped:
            assert col not in out.columns, f"Cột gốc '{col}' vẫn còn trong output"
        for col in created:
            assert col in out.columns, f"Cột mới '{col}' thiếu trong output"

    def test_all_flag_cols_present(self, tmp_path):
        """
        Tất cả cờ được tạo trong Bước 1 và Bước 5 phải có mặt trong output.
        """
        out = _run(tmp_path, [_base_row(i) for i in range(1, 4)])
        expected = [
            "Has_Pool", "Has_Misc_Feature", "Has_Alley", "Has_Fence",
            "Has_Garage", "Is_Normal_Sale", "Has_Negative_Condition",
            "Garage_Age", "Age_At_Sale", "Remod_Age", "Was_Remodeled",
            "Total_Porch", "Total_Bath",
        ]
        for col in expected:
            assert col in out.columns, f"Thiếu cột '{col}' trong output"