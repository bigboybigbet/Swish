def is_prime(n):
    """
    Kiểm tra xem một số nguyên n có phải là số nguyên tố hay không.
    
    Parameters:
        n (int): Số nguyên cần kiểm tra
    
    Returns:
        bool: True nếu n là số nguyên tố, False nếu không
    """
    # Trường hợp đặc biệt
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Kiểm tra các số lẻ từ 3 đến sqrt(n)
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    
    return True

# Ví dụ sử dụng hàm
numbers_to_check = [1, 2, 3, 4, 5, 17, 25, 97]
for num in numbers_to_check:
    print(f"{num} là số nguyên tố: {is_prime(num)}")
