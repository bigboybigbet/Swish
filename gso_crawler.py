"""
gso_crawler.py
==============
Tự động tải file biểu hàng tháng từ nso.gov.vn và tổng hợp
dữ liệu XK + NK (sheet 14 & 15) theo phân ngành thành 1 file Excel.

Cách dùng:
    python gso_crawler.py                    # Tải tất cả (T1/2024 → tháng hiện tại)
    python gso_crawler.py --from 2025-01     # Tải từ T1/2025
    python gso_crawler.py --dry-run          # Chỉ in URL, không tải
    python gso_crawler.py --input-dir ./raw  # Dùng file đã có sẵn, bỏ qua download
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, date
from pathlib import Path

import openpyxl
import requests
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─────────────────────────────────────────────
# 1. CẤU HÌNH
# ─────────────────────────────────────────────

BASE_URL = "https://www.nso.gov.vn/wp-content/uploads"

# Pattern filename GSO: 02-Bieu-T{M}-{YYYY}-final.xlsx
# Một số tháng dùng dấu gạch dưới: 02_-Bieu-T{M}_{YYYY}-final.xlsx
# Script thử cả 2 variant
FILENAME_VARIANTS = [
    "02-Bieu-T{month}.{year}.xlsx",             # ✓ pattern xác nhận: T4.2026
    "02-Bieu-T{month:02d}.{year}.xlsx",          # zero-padded: T04.2026
    "02-Bieu-T{month}-{year}-final.xlsx",        # variant cũ phòng thủ
    "02_-Bieu-T{month}_{year}-final.xlsx",
    "02_-Bieu-T{month}.{year}.xlsx",
    "02.-Bieu-T{month}.{year}.xlsx",
]

# File upload lên server vào tháng kỳ + 1
# (tháng 5 năm 2026 → upload tháng 6/2026)
UPLOAD_MONTH_OFFSET = 1

# Sheet cần đọc
SHEET_XK = "14. XK"
SHEET_NK = "15. NK"

# Hàng dữ liệu cần lấy trong mỗi sheet
# Format: {row_index (1-based): tên chuẩn hiển thị}
# Dựa vào cấu trúc file T5/2026 (đã xác nhận giống hệt nhau)
XK_ROWS = {
    9:  "TỔNG TRỊ GIÁ",
    10: "Khu vực kinh tế trong nước",
    11: "Khu vực có vốn đầu tư nước ngoài",
    15: "Thủy sản",
    16: "Rau quả",
    17: "Hạt điều",
    18: "Cà phê",
    19: "Chè",
    20: "Hạt tiêu",
    21: "Gạo",
    22: "Sắn và sản phẩm của sắn",
    23: "Bánh kẹo và các sản phẩm từ ngũ cốc",
    24: "Thức ăn gia súc và nguyên liệu",
    25: "Clanhke và xi măng",
    27: "Xăng dầu",
    28: "Hóa chất",
    29: "Sản phẩm hóa chất",
    30: "Chất dẻo nguyên liệu",
    31: "Sản phẩm chất dẻo",
    32: "Cao su",
    33: "Sản phẩm từ cao su",
    34: "Túi xách, ví, va li, mũ, ô dù",
    35: "Gỗ và sản phẩm gỗ",
    36: "Giấy và các sản phẩm từ giấy",
    37: "Xơ, sợi dệt các loại",
    38: "Hàng dệt, may",
    39: "Giày dép",
    40: "Nguyên phụ liệu dệt, may, da, giày",
    41: "Thủy tinh và các sản phẩm từ thủy tinh",
    42: "Sắt thép",
    43: "Sản phẩm từ sắt thép",
    44: "Kim loại thường khác và sản phẩm",
    45: "Điện tử, máy tính và linh kiện",
    46: "Điện thoại các loại và linh kiện",
    47: "Máy ảnh, máy quay phim và linh kiện",
    48: "Máy móc, thiết bị, dụng cụ, phụ tùng khác",
    49: "Dây điện và cáp điện",
    50: "Phương tiện vận tải và phụ tùng",
    51: "Sản phẩm nội thất từ chất liệu khác gỗ",
    52: "Đồ chơi, dụng cụ thể thao và bộ phận",
}

NK_ROWS = {
    9:  "TỔNG TRỊ GIÁ",
    10: "Khu vực kinh tế trong nước",
    11: "Khu vực có vốn đầu tư nước ngoài",
    13: "Thủy sản",
    14: "Sữa và sản phẩm sữa",
    15: "Rau quả",
    16: "Hạt điều",
    17: "Lúa mỳ",
    18: "Ngô",
    19: "Đậu tương",
    20: "Dầu mỡ động thực vật",
    21: "Chế phẩm thực phẩm khác",
    22: "Thức ăn gia súc và NPL",
    23: "Quặng và khoáng sản khác",
    24: "Than đá",
    25: "Dầu thô",
    26: "Xăng dầu",
    27: "Khí đốt hóa lỏng",
    28: "Sản phẩm khác từ dầu mỏ",
    29: "Hóa chất",
    30: "Sản phẩm hoá chất",
    31: "Tân dược",
    32: "Phân bón",
    33: "Chất thơm, mỹ phẩm và chế phẩm vệ sinh",
    34: "Chất dẻo",
    35: "Sản phẩm chất dẻo",
    36: "Cao su",
    37: "Sản phẩm từ cao su",
    38: "Gỗ và sản phẩm gỗ",
    39: "Giấy các loại",
    40: "Sản phẩm từ giấy",
    41: "Bông",
    42: "Sợi dệt",
    43: "Vải",
    44: "Nguyên PL dệt, may, giày dép",
    45: "Thủy tinh và các sản phẩm từ thủy tinh",
    46: "Phế liệu sắt thép",
    47: "Sắt thép",
    48: "Sản phẩm từ sắt thép",
    49: "Kim loại thường khác",
    50: "Sản phẩm từ kim loại thường khác",
    51: "Điện tử, máy tính và linh kiện",
    52: "Hàng điện gia dụng và linh kiện",
    53: "Điện thoại các loại và linh kiện",
    54: "Máy ảnh, máy quay phim và linh kiện",
    55: "Máy móc, thiết bị, dụng cụ, phụ tùng khác",
    56: "Dây điện và cáp điện",
    57: "Ô tô",
    59: "Phương tiện vận tải khác và phụ tùng",
}

# Màu SSI
COLOR_HEADER_BG = "C8102E"    # Đỏ SSI
COLOR_HEADER_FG = "FFFFFF"    # Chữ trắng
COLOR_SUBHEADER  = "F2DCDB"   # Hồng nhạt cho nhóm
COLOR_TOTAL_BG   = "FCE4D6"   # Cam nhạt cho TỔNG
COLOR_ODD_ROW    = "FFFFFF"
COLOR_EVEN_ROW   = "FFF2F2"   # Hồng rất nhạt
COLOR_BORDER     = "D9D9D9"


# ─────────────────────────────────────────────
# 2. URL BUILDER
# ─────────────────────────────────────────────

def build_urls(month: int, year: int) -> list[str]:
    """Trả về danh sách URL candidate cho 1 tháng/năm."""
    upload_month = month + UPLOAD_MONTH_OFFSET
    upload_year = year
    if upload_month > 12:
        upload_month -= 12
        upload_year += 1

    urls = []
    for tpl in FILENAME_VARIANTS:
        filename = tpl.format(month=month, year=year)
        url = f"{BASE_URL}/{upload_year}/{upload_month:02d}/{filename}"
        urls.append(url)
    return urls


def months_range(from_ym: tuple, to_ym: tuple) -> list[tuple]:
    """Sinh danh sách (month, year) từ from đến to."""
    result = []
    y, m = from_ym
    ey, em = to_ym
    while (y, m) <= (ey, em):
        result.append((m, y))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return result


# ─────────────────────────────────────────────
# 3. DOWNLOADER
# ─────────────────────────────────────────────

def download_file(month: int, year: int, save_dir: Path,
                  dry_run: bool = False) -> Path | None:
    """
    Thử tải file xlsx từ nso.gov.vn.
    Thử lần lượt các URL variant cho đến khi thành công.
    Trả về Path đến file đã lưu, hoặc None nếu thất bại.
    """
    urls = build_urls(month, year)
    label = f"T{month}/{year}"

    if dry_run:
        print(f"  [DRY-RUN] {label}: sẽ thử {len(urls)} URL:")
        for u in urls:
            print(f"    {u}")
        return None

    save_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.nso.gov.vn/",
        "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*",
    }

    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=20, stream=True)
            if resp.status_code == 200:
                # Kiểm tra đây là xlsx thật (magic bytes PK)
                content = b""
                for chunk in resp.iter_content(8192):
                    content += chunk
                if content[:2] == b"PK":
                    # Lưu file
                    fname = url.split("/")[-1]
                    out = save_dir / fname
                    out.write_bytes(content)
                    print(f"  ✓ {label}: {fname} ({len(content)//1024} KB)")
                    return out
                else:
                    # HTML hoặc error page
                    continue
            elif resp.status_code == 404:
                continue
            else:
                print(f"  ! {label}: HTTP {resp.status_code} — {url}")
        except requests.RequestException as e:
            print(f"  ! {label}: lỗi kết nối — {e}")
            time.sleep(1)

    print(f"  ✗ {label}: không tìm thấy file (thử {len(urls)} URL)")
    return None


# ─────────────────────────────────────────────
# 4. EXTRACTOR
# ─────────────────────────────────────────────

def extract_month_data(xlsx_path: Path,
                       sheet_name: str,
                       row_map: dict) -> dict | None:
    """
    Đọc 1 file xlsx, lấy cột D (trị giá tháng) cho các row trong row_map.
    Trả về {row_index: value} hoặc None nếu lỗi.
    """
    try:
        wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
        if sheet_name not in wb.sheetnames:
            # Thử tìm sheet gần giống (phòng GSO đổi tên nhẹ)
            alt = next(
                (s for s in wb.sheetnames if "XK" in s or "NK" in s
                 if ("XK" in sheet_name and "XK" in s)
                 or ("NK" in sheet_name and "NK" in s)),
                None
            )
            if alt:
                sheet_name = alt
            else:
                print(f"    ⚠ Không tìm thấy sheet '{sheet_name}' trong {xlsx_path.name}")
                print(f"      Các sheet có: {wb.sheetnames}")
                return None

        ws = wb[sheet_name]
        rows_data = list(ws.iter_rows(values_only=True))

        result = {}
        for row_idx, label in row_map.items():
            if row_idx - 1 < len(rows_data):
                row = rows_data[row_idx - 1]
                val = row[3] if len(row) > 3 else None   # cột D = index 3
                result[row_idx] = float(val) if val is not None else None
            else:
                result[row_idx] = None
        return result

    except Exception as e:
        print(f"    ✗ Lỗi đọc {xlsx_path.name}: {e}")
        return None


# ─────────────────────────────────────────────
# 5. OUTPUT BUILDER
# ─────────────────────────────────────────────

def build_excel(
    data_xk: dict,   # {(month, year): {row_idx: value}}
    data_nk: dict,
    sorted_months: list[tuple],
    output_path: Path
) -> None:
    """Tạo file Excel output 2 sheet XK + NK."""

    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # xóa sheet mặc định

    for sheet_tag, row_map, data in [
        ("XK - Hàng hóa xuất khẩu", XK_ROWS, data_xk),
        ("NK - Hàng hóa nhập khẩu", NK_ROWS, data_nk),
    ]:
        ws = wb.create_sheet(title=sheet_tag)
        _write_sheet(ws, sheet_tag, row_map, data, sorted_months)

    wb.save(output_path)
    print(f"\n✅ Đã lưu: {output_path}")


def _make_border():
    thin = Side(style="thin", color=COLOR_BORDER)
    return Border(left=thin, right=thin, top=thin, bottom=thin)


def _write_sheet(ws, title: str, row_map: dict, data: dict,
                 sorted_months: list[tuple]) -> None:
    """Ghi 1 sheet."""

    border = _make_border()

    # ── Header row 1: tiêu đề sheet ──
    ws.merge_cells(f"A1:{get_column_letter(1 + len(sorted_months))}1")
    h1 = ws.cell(1, 1, f"BẢNG SỐ LIỆU {title.upper()}")
    h1.font = Font(bold=True, color=COLOR_HEADER_FG, size=13,
                   name="Times New Roman")
    h1.fill = PatternFill("solid", fgColor=COLOR_HEADER_BG)
    h1.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 26

    # ── Header row 2: đơn vị ──
    ws.cell(2, 1, "Đơn vị: Triệu USD").font = Font(italic=True, size=10,
                                                     name="Times New Roman")
    ws.row_dimensions[2].height = 16

    # ── Header row 3: tên cột ──
    ws.cell(3, 1, "Mặt hàng")
    _style_header_cell(ws.cell(3, 1), border)

    for col_i, (m, y) in enumerate(sorted_months, start=2):
        label = f"T{m}/{y}"
        cell = ws.cell(3, col_i, label)
        _style_header_cell(cell, border)

    ws.row_dimensions[3].height = 20
    ws.column_dimensions["A"].width = 44

    for col_i in range(2, 2 + len(sorted_months)):
        ws.column_dimensions[get_column_letter(col_i)].width = 11

    # ── Dữ liệu ──
    group_rows = {9, 10, 11, 14}   # hàng nhóm (header)
    total_rows = {9}                # hàng tổng

    excel_row = 4
    for row_idx, label in row_map.items():
        is_total = row_idx in total_rows
        is_group = row_idx in group_rows

        # Tên mặt hàng
        name_cell = ws.cell(excel_row, 1, label)
        name_cell.border = border
        name_cell.font = Font(
            bold=is_total or is_group,
            size=10,
            name="Times New Roman"
        )
        name_cell.alignment = Alignment(
            horizontal="left",
            vertical="center",
            indent=0 if is_group or is_total else 1
        )
        if is_total:
            name_cell.fill = PatternFill("solid", fgColor=COLOR_TOTAL_BG)
        elif is_group:
            name_cell.fill = PatternFill("solid", fgColor=COLOR_SUBHEADER)
        elif excel_row % 2 == 0:
            name_cell.fill = PatternFill("solid", fgColor=COLOR_EVEN_ROW)

        # Giá trị từng tháng
        for col_i, (m, y) in enumerate(sorted_months, start=2):
            val = data.get((m, y), {}).get(row_idx)
            cell = ws.cell(excel_row, col_i)
            cell.border = border
            cell.font = Font(size=10, name="Times New Roman",
                             bold=is_total)
            cell.alignment = Alignment(horizontal="right", vertical="center")

            if val is not None:
                cell.value = round(val, 1)
                cell.number_format = '#,##0.0'
            else:
                cell.value = "n/a"
                cell.font = Font(size=9, color="AAAAAA",
                                 name="Times New Roman")

            if is_total:
                cell.fill = PatternFill("solid", fgColor=COLOR_TOTAL_BG)
            elif is_group:
                cell.fill = PatternFill("solid", fgColor=COLOR_SUBHEADER)
            elif excel_row % 2 == 0:
                cell.fill = PatternFill("solid", fgColor=COLOR_EVEN_ROW)

        ws.row_dimensions[excel_row].height = 16
        excel_row += 1

    # ── Ghi chú cuối ──
    ws.cell(excel_row + 1, 1,
            "Nguồn: Tổng cục Thống kê (nso.gov.vn) — Báo cáo tình hình KTXH hàng tháng"
            ).font = Font(italic=True, size=9, color="666666",
                          name="Times New Roman")

    # Freeze panes
    ws.freeze_panes = "B4"


def _style_header_cell(cell, border):
    cell.font = Font(bold=True, color=COLOR_HEADER_FG, size=10,
                     name="Times New Roman")
    cell.fill = PatternFill("solid", fgColor=COLOR_HEADER_BG)
    cell.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
    cell.border = border


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def parse_ym(s: str) -> tuple:
    """Parse '2024-01' → (2024, 1)"""
    m = re.match(r"(\d{4})-(\d{1,2})$", s)
    if not m:
        raise ValueError(f"Định dạng phải là YYYY-MM, ví dụ 2024-01, nhận được: {s!r}")
    return int(m.group(1)), int(m.group(2))


def main():
    parser = argparse.ArgumentParser(
        description="Cào dữ liệu XK/NK từ nso.gov.vn và xuất Excel."
    )
    parser.add_argument(
        "--from", dest="from_ym", default="2024-01",
        help="Bắt đầu từ tháng nào, định dạng YYYY-MM (mặc định: 2024-01)"
    )
    parser.add_argument(
        "--to", dest="to_ym", default=None,
        help="Kết thúc tháng nào, định dạng YYYY-MM (mặc định: tháng hiện tại)"
    )
    parser.add_argument(
        "--output", default="XK_NK_lich_su.xlsx",
        help="Tên file output (mặc định: XK_NK_lich_su.xlsx)"
    )
    parser.add_argument(
        "--raw-dir", default="./raw_gso",
        help="Thư mục lưu file xlsx gốc tải về (mặc định: ./raw_gso)"
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Nếu đã có file sẵn, chỉ định thư mục chứa file (bỏ qua download)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ in URL sẽ thử, không tải"
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Thời gian chờ giữa các request (giây, mặc định: 1.5)"
    )
    args = parser.parse_args()

    # Parse khoảng thời gian
    from_ym = parse_ym(args.from_ym)
    if args.to_ym:
        to_ym = parse_ym(args.to_ym)
    else:
        now = date.today()
        to_ym = (now.year, now.month)
        # Chỉ lấy đến tháng trước (tháng hiện tại thường chưa có file)
        m, y = to_ym
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        to_ym = (y, m)

    months = months_range(from_ym, to_ym)
    print(f"📅 Phạm vi: T{from_ym[1]}/{from_ym[0]} → T{to_ym[1]}/{to_ym[0]}")
    print(f"   Tổng cộng: {len(months)} tháng\n")

    raw_dir = Path(args.raw_dir)
    data_xk: dict = {}
    data_nk: dict = {}
    success_months = []

    # ── Bước 1: Thu thập file ──
    print("═" * 55)
    print("BƯỚC 1: TẢI FILE")
    print("═" * 55)

    for month, year in months:
        label = f"T{month}/{year}"

        # Nếu có --input-dir, tìm file trong đó
        if args.input_dir:
            input_dir = Path(args.input_dir)
            candidates = list(input_dir.glob(f"*Bieu*T{month}*{year}*.xlsx")) + \
                         list(input_dir.glob(f"*Bieu*T{month:02d}*{year}*.xlsx"))
            if candidates:
                local_file = candidates[0]
                print(f"  📂 {label}: dùng file local — {local_file.name}")
            else:
                print(f"  ✗ {label}: không tìm thấy file trong {args.input_dir}")
                continue
        else:
            # Kiểm tra đã tải chưa
            existing = list(raw_dir.glob(f"*Bieu*T{month}*{year}*.xlsx")) + \
                       list(raw_dir.glob(f"*Bieu*T{month:02d}*{year}*.xlsx"))
            if existing:
                local_file = existing[0]
                print(f"  ⏭ {label}: đã có — {local_file.name}")
            else:
                local_file = download_file(month, year, raw_dir,
                                            dry_run=args.dry_run)
                if local_file is None:
                    continue
                time.sleep(args.delay)

        if args.dry_run:
            continue

        # ── Bước 2: Extract ──
        xk = extract_month_data(local_file, SHEET_XK, XK_ROWS)
        nk = extract_month_data(local_file, SHEET_NK, NK_ROWS)

        if xk is not None:
            data_xk[(month, year)] = xk
        if nk is not None:
            data_nk[(month, year)] = nk

        if xk is not None and nk is not None:
            success_months.append((month, year))

    if args.dry_run:
        print("\n[DRY-RUN] Kết thúc. Chạy lại không có --dry-run để tải thật.")
        return

    if not success_months:
        print("\n⚠ Không có tháng nào đọc được dữ liệu. Kiểm tra kết nối và URL pattern.")
        return

    # ── Bước 3: Xuất Excel ──
    print(f"\n{'═'*55}")
    print("BƯỚC 3: XUẤT EXCEL")
    print("═" * 55)
    print(f"  Số tháng có dữ liệu: {len(success_months)}")

    sorted_months = sorted(success_months, key=lambda t: (t[1], t[0]))
    output_path = Path(args.output)
    build_excel(data_xk, data_nk, sorted_months, output_path)
    print(f"  Mở file: {output_path.resolve()}")


if __name__ == "__main__":
    main()
