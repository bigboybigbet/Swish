# -*- coding: utf-8 -*-
"""
VAMA Light Trucks Sales Scraper
Tu dong tai PDF tu vama.org.vn va trich xuat du lieu doanh so Trucks (2.1) vao Excel.

Du lieu lay tu file Summary cua VAMA - muc "2.1 Trucks" = tong doanh so xe tai thang do.
"""

import sys
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd

# Fix Unicode output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# CAU HINH
# ============================================================
BASE_URL = "http://vama.org.vn"
DOWNLOAD_DIR = Path("vama_pdfs")
OUTPUT_EXCEL = "vama_light_trucks_sales.xlsx"
REQUEST_DELAY = 0.8

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": BASE_URL,
}

# Trang chinh chua danh sach bao cao ban hang cua VAMA
REPORTS_URL = "http://vama.org.vn/vn/bao-cao-ban-hang.html"

MONTH_MAP = {
    "january": 1,  "jan": 1,
    "february": 2, "feb": 2,
    "march": 3,    "mar": 3,
    "april": 4,    "apr": 4,
    "may": 5,
    "june": 6,     "jun": 6,
    "july": 7,     "jul": 7,
    "august": 8,   "aug": 8,
    "september": 9,"sep": 9,
    "october": 10, "oct": 10,
    "november": 11,"nov": 11,
    "december": 12,"dec": 12,
}


# ============================================================
# BUOC 1 - Tim link PDF
# ============================================================
def find_pdf_links(session: requests.Session) -> list[dict]:
    """
    Tim tat ca link PDF Summary tren trang bao cao ban hang cua VAMA.
    Tu dong xu ly phan trang neu co.
    """
    seen: set[str] = set()
    links: list[dict] = []

    # Thu cac URL co the chua danh sach bao cao (trang hien tai + phan trang)
    candidate_urls = [REPORTS_URL]
    # Thu them trang 2, 3 (neu co phan trang)
    for page_num in range(2, 6):
        candidate_urls.append(REPORTS_URL.replace(".html", f"-{page_num}.html"))

    for page_url in candidate_urls:
        try:
            resp = session.get(page_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                break  # Khong con trang tiep theo

            soup = BeautifulSoup(resp.text, "lxml")
            found_on_page = 0

            for a in soup.find_all("a", href=True):
                href: str = a["href"]
                if ".pdf" in href.lower():
                    url = urljoin(BASE_URL, href)
                    if url not in seen:
                        seen.add(url)
                        links.append({"url": url, "text": a.get_text(strip=True)})
                        found_on_page += 1

            print(f"  [ok] {page_url} -> {found_on_page} PDF moi")
            if found_on_page == 0:
                break  # Trang trong, dung lai
            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"  [!] Loi truy cap {page_url}: {e}")
            break

    print(f"  => Tong cong {len(links)} PDF tim thay.")
    return links


# ============================================================
# BUOC 2 - Tai PDF
# ============================================================
def download_pdf(session: requests.Session, url: str, save_dir: Path) -> Path | None:
    filename = unquote(Path(urlparse(url).path).name)
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    save_path = save_dir / filename
    if save_path.exists():
        return save_path  # Da co san, bo qua

    try:
        resp = session.get(url, headers=HEADERS, timeout=30, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  [+] Da tai: {filename}")
        return save_path
    except Exception as e:
        print(f"  [!] Loi tai {url}: {e}")
        return None


# ============================================================
# PARSE NGAY - lay thang/nam tu ten file
# ============================================================
def parse_date(filename: str) -> tuple[int | None, int | None]:
    """
    Lay (thang, nam) tu ten file da URL-decode.
    Vi du: 'VAMA sales report April 2024 - Summary.pdf' -> (4, 2024)
    """
    name = unquote(filename).lower()

    # Tim ten thang (tieng Anh)
    month = None
    for name_str, num in MONTH_MAP.items():
        if re.search(r'\b' + name_str + r'\b', name):
            month = num
            break

    # Tim nam (4 chu so)
    year_match = re.search(r'\b(20\d{2})\b', name)
    year = int(year_match.group(1)) if year_match else None

    return month, year


# ============================================================
# BUOC 3 - Trich xuat "2.1 Trucks" tu Summary PDF
# ============================================================
def extract_trucks_from_summary(pdf_path: Path) -> int | None:
    """
    Doc Summary PDF va tim gia tri doanh so 'Trucks' (hang 2.1) cua thang.
    Tra ve so nguyen (don vi: chiec) hoac None neu khong tim thay.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    for row in table:
                        if not row:
                            continue
                        cells = [str(c).strip() if c else "" for c in row]
                        # Tim hang "2.1 Trucks"
                        if cells and re.match(r"2\.1\s*trucks?", cells[0], re.IGNORECASE):
                            # Lay so dau tien hop le (= doanh so thang hien tai)
                            for cell in cells[1:]:
                                clean = cell.replace(",", "").replace(" ", "")
                                try:
                                    val = int(clean)
                                    if val > 0:
                                        return val
                                except ValueError:
                                    continue
    except Exception as e:
        print(f"  [!] Loi doc PDF {pdf_path.name}: {e}")

    return None


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print(" VAMA Trucks Sales Scraper")
    print("=" * 60)

    DOWNLOAD_DIR.mkdir(exist_ok=True)
    session = requests.Session()

    # Buoc 1: Tim PDF
    print("\n[1/4] Dang tim PDF tren vama.org.vn ...")
    all_links = find_pdf_links(session)

    # Chi lay file Summary (co du lieu tong hop can thiet)
    summary_links = [
        link for link in all_links
        if "summary" in link["url"].lower()
        and "bmw" not in link["url"].lower()
        and "lexus" not in link["url"].lower()
    ]
    print(f"  => {len(summary_links)} file Summary (se xu ly).")

    if not summary_links:
        print("  [!] Khong tim thay Summary PDF. Xu ly cac PDF co san trong thu muc.")
        _process_local_summaries()
        return

    # Buoc 2: Tai PDF
    print("\n[2/4] Dang tai PDF ...")
    for item in summary_links:
        download_pdf(session, item["url"], DOWNLOAD_DIR)
        time.sleep(REQUEST_DELAY)

    # Xu ly TAT CA Summary PDF trong thu muc (ca cu lan moi)
    all_summaries = [
        p for p in DOWNLOAD_DIR.glob("*.pdf")
        if "summary" in unquote(p.name).lower()
        and "bmw" not in p.name.lower()
        and "lexus" not in p.name.lower()
    ]
    print(f"  => {len(all_summaries)} Summary PDF tong cong (ca cu + moi).")

    _extract_and_save(all_summaries)


def _process_local_summaries():
    """Fallback: xu ly cac Summary PDF co san trong thu muc."""
    pdfs = [p for p in DOWNLOAD_DIR.glob("*.pdf") if "summary" in p.name.lower()]
    if not pdfs:
        print(f"  [!] Khong co Summary PDF nao trong '{DOWNLOAD_DIR}'. Ket thuc.")
        return
    print(f"  => Tim thay {len(pdfs)} Summary PDF cuc bo.")
    _extract_and_save(pdfs)


def _extract_and_save(pdf_paths: list[Path]):
    # Buoc 3: Trich xuat
    print("\n[3/4] Trich xuat du lieu Trucks tu Summary PDF ...")
    records: list[dict] = []

    for pdf_path in pdf_paths:
        month, year = parse_date(pdf_path.stem)
        value = extract_trucks_from_summary(pdf_path)

        status = f"{value:,}" if value else "KHONG TIM THAY"
        month_str = f"{month:02d}" if month else "xx"
        print(f"  {pdf_path.name[:55]:55} -> {year}/{month_str} | {status}")

        if value is not None:
            records.append({
                "year": year,
                "month": month,
                "trucks_sales": value,
                "source_file": pdf_path.name,
            })

    if not records:
        print("\n  [!] Khong trich xuat duoc du lieu nao.")
        return

    # Buoc 4: Ghi Excel
    print(f"\n[4/4] Ghi ra Excel: {OUTPUT_EXCEL}")
    df = pd.DataFrame(records)
    df = df.sort_values(["year", "month"], na_position="last")
    # Xoa ban ghi trung lap cung thang/nam (giu ban ghi dau tien)
    df = df.drop_duplicates(subset=["year", "month"], keep="first").reset_index(drop=True)
    df.columns = ["Nam", "Thang", "Doanh so Trucks (chiec)", "File PDF nguon"]

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="VAMA Trucks Sales")
        ws = writer.sheets["VAMA Trucks Sales"]
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

    print(f"  => Da luu {len(df)} ban ghi.\n")
    print(df.to_string(index=False))
    print("\nHoan tat!")


if __name__ == "__main__":
    main()
