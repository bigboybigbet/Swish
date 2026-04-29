import random
import time
import pandas as pd
import matplotlib.pyplot as plt

# --- HÀM HỖ TRỢ LOGIC ---


def gieo_xuc_xac_ngau_nhien():
    """Gieo xúc xắc hoàn toàn ngẫu nhiên"""
    return [random.randint(1, 6) for _ in range(3)]


def cuong_che_thua(lua_chon_nguoi_choi):
    """Tính toán bộ xúc xắc để người chơi chắc chắn thua"""
    if lua_chon_nguoi_choi == 'T':  # Người chơi chọn Tài -> Phải ra Xỉu (4-10)
        tong_muc_tieu = random.randint(4, 10)
    else:  # Người chơi chọn Xỉu -> Phải ra Tài (11-17)
        tong_muc_tieu = random.randint(11, 17)

    # Chia tổng điểm thành 3 con xúc xắc
    while True:
        s1 = random.randint(1, 6)
        s2 = random.randint(1, 6)
        s3 = tong_muc_tieu - s1 - s2
        if 1 <= s3 <= 6:
            return [s1, s2, s3]


def kiem_tra_ket_qua(danh_sach_diem):
    tong = sum(danh_sach_diem)
    if danh_sach_diem[0] == danh_sach_diem[1] == danh_sach_diem[2]:
        return "BO BA", tong
    elif 4 <= tong <= 10:
        return "XIU", tong
    else:
        return "TAI", tong

# --- HÀM HIỂN THỊ DỮ LIỆU ---


def hien_thi_va_ve_bieu_do(lich_su_choi):
    if not lich_su_choi:
        print("\n[!] Chưa có dữ liệu ván chơi nào.")
        return

    # Chuyển dữ liệu sang DataFrame của Pandas
    df = pd.DataFrame(lich_su_choi)
    df.index = df.index + 1
    df.index.name = 'Ván'

    print("\n" + "="*60)
    print("--- LỊCH SỬ CHI TIẾT VÀ THỐNG KÊ ---")
    print(df[['lua_chon', 'xuc_xac', 'tong', 'ket_qua', 'so_du']])

    # Tính toán thống kê
    tong_van = len(df)
    so_van_thang = len(df[df['ket_qua'] == 'THẮNG'])
    so_van_thua = tong_van - so_van_thang
    print(
        f"\n=> TỔNG KẾT: Chơi {tong_van} ván | Thắng {so_van_thang} | Thua {so_van_thua}")
    print(f"=> Tỷ lệ thắng: {(so_van_thang/tong_van)*100:.1f}%")
    print("="*60 + "\n")

    # Vẽ biểu đồ bằng Matplotlib
    plt.figure(figsize=(12, 5))

    # Biểu đồ 1: Tỷ lệ thắng thua
    plt.subplot(1, 2, 1)
    plt.pie([so_van_thang, so_van_thua], labels=['Thắng', 'Thua'],
            autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#F44336'])
    plt.title('Tỷ lệ Thắng/Thua')

    # Biểu đồ 2: Biến động số dư
    plt.subplot(1, 2, 2)
    plt.plot(df.index, df['so_du'], marker='o', linestyle='-', color='b')
    plt.axhline(y=1000000, color='r', linestyle='--', label='Vốn ban đầu')
    plt.title('Biến động số dư qua từng ván')
    plt.xlabel('Ván số')
    plt.ylabel('Số dư ($)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# --- LUỒNG TRÒ CHƠI CHÍNH ---


def bat_dau_tro_choi():
    so_du = 1000000
    lich_su_choi = []

    print("Welcome to Python Casino - Tài Xỉu High-Tech")

    while so_du > 0:
        print(f"\n--- SỐ DƯ HIỆN TẠI: {so_du:,}$ ---")

        # Nhận lệnh người chơi
        lua_chon = ""
        while lua_chon not in ['T', 'X', 'H', 'Q']:
            lua_chon = input(
                "Chọn Tài (T), Xỉu (X), Xem lịch sử (H), hoặc Thoát (Q): ").upper()

        if lua_chon == 'Q':
            break
        if lua_chon == 'H':
            hien_thi_va_ve_bieu_do(lich_su_choi)
            continue

        # Nhập tiền cược
        muc_cuoc = 0
        while muc_cuoc <= 0 or muc_cuoc > so_du:
            try:
                muc_cuoc = int(input(f"Nhập tiền cược (1 - {so_du}): "))
            except ValueError:
                print("Lỗi: Vui lòng nhập số nguyên.")

        # KIỂM TRA LOGIC THẮNG 3 LẦN LIÊN TIẾP
        kiem_tra_bi_soi = False
        if len(lich_su_choi) >= 3:
            ba_van_cuoi = lich_su_choi[-3:]
            if all(v['ket_qua'] == "THẮNG" for v in ba_van_cuoi):
                kiem_tra_bi_soi = True

        # Thực hiện lắc xúc xắc
        print("\nĐang lắc...")
        time.sleep(1)

        if kiem_tra_bi_soi:
            # Can thiệp kết quả nếu thắng quá nhiều
            ket_qua_xuc_xac = cuong_che_thua(lua_chon)
            print("[Hệ thống] Phát hiện bất thường... đang điều chỉnh độ khó.")
        else:
            ket_qua_xuc_xac = gieo_xuc_xac_ngau_nhien()

        chu, tong = kiem_tra_ket_qua(ket_qua_xuc_xac)
        print(f"==> KẾT QUẢ: {ket_qua_xuc_xac} | Tổng: {tong} ({chu})")

        # Xác định thắng thua
        is_thang = False
        if chu != "BO BA":
            if (lua_chon == 'T' and chu == "TAI") or (lua_chon == 'X' and chu == "XIU"):
                is_thang = True

        if is_thang:
            print(f"CHÚC MỪNG! Bạn nhận được +{muc_cuoc}$")
            so_du += muc_cuoc
            trang_thai = "THẮNG"
        else:
            print(f"RẤT TIẾC! Bạn mất -{muc_cuoc}$")
            so_du -= muc_cuoc
            trang_thai = "THUA"

        # Lưu vào lịch sử
        lich_su_choi.append({
            "lua_chon": "Tài" if lua_chon == 'T' else "Xỉu",
            "xuc_xac": str(ket_qua_xuc_xac),
            "tong": tong,
            "ket_qua": trang_thai,
            "so_du": so_du
        })

        if so_du <= 0:
            print("\n[!] BẠN ĐÃ CHÁY TÚI!")
            hien_thi_va_ve_bieu_do(lich_su_choi)
            break


if __name__ == "__main__":
    bat_dau_tro_choi()
