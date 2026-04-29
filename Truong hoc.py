import uuid


class User:
    def __init__(self, name, address, phone_number):
        self.id = str(uuid.uuid4())
        self.name = name
        self.address = address                            # this is the main class
        self.phone_number = phone_number

    def get_info(self):
        return f"ID:{self.id}   /  Ten: {self.name}  /  Dia chi: {self.address}  /  SDT: {self.phone_number}"


class Staff(User):
    def __init__(self, name, address, phone_number, salary):
        super().__init__(name, address, phone_number)
        self.salary = salary

    def get_salary(self):
        return self.salary


class Teacher(Staff):
    def __init__(self, name, address, phone_number, salary, classes):
        super().__init__(name, address, phone_number, salary)
        self.classes = classes

    def get_classes(self):
        return self.classes


class Student(User):
    def __init__(self, name, address, phone_number, subject):
        super().__init__(name, address, phone_number)
        self.subject = subject

    def get_subject(self):
        return self.subject


Student_list = []
Teacher_list = []
Staff_list = []


def set_student():
    print("MOI NHAP THONG TIN")
    name = input("TEN: ")
    address = input("DIA CHI: ")
    phone_number = input("SO DIEN THOAI: ")
    subject = input("MON HOC: ")

    sv = Student(name, address, phone_number, subject)
    Student_list.append(sv)
    print("SUCCESS!, NOW RETURN")


def set_teacher():
    print("MOI NHAP THONG TIN")
    name = input("TEN: ")
    address = input("DIA CHI: ")
    phone_number = input("SO DIEN THOAI: ")
    salary = input("LUONG: ")
    classes = input("MON DAY: ")

    gv = Teacher(name, address, phone_number, salary, classes)
    Teacher_list.append(gv)
    print("SUCCESS!, NOW RETURN")


def set_staff():
    print("MOI NHAP THONG TIN")
    name = input("TEN: ")
    address = input("DIA CHI: ")
    phone_number = input("SO DIEN THOAI: ")
    subject = input("MON HOC: ")
    salary = input("LUONG: ")

    nv = Staff(name, address, phone_number, salary)
    Staff_list.append(nv)
    print("SUCCESS!, NOW RETURN")


def print_student():
    print("=== DANH SACH HOC SINH ===")
    for sv in Student_list:
        print(f"{sv.get_info()}  /  Mon hoc:{sv.get_subject()}")


def print_teacher():
    print("=== DANH SACH GIAO VIEN ===")
    for gv in Teacher_list:
        print(
            f"{gv.get_info()}  /  Luong:{gv.get_salary()}  /  Lop day:{gv.get_classes()}"
        )


def print_staff():
    print("=== DANH SACH NHAN VIEN ===")
    for nv in Staff_list:
        print(
            f"{nv.get_info()}  /  Luong:{nv.get_salary()}"
        )


def print_all():
    print("=== DANH SACH TOAN BO ===")
    print_student()
    print_teacher()
    print_staff()


def main():

    while True:
        print("====MENU====")
        print("1. Them hoc sinh")
        print("2. Them giao vien")
        print("3. Them nhan vien")
        print("4. In danh sach hoc sinh")
        print("5. In danh sach giao vien")
        print("6. In danh sach nhan vien")
        print("7. In tat ca")
        print("8. Exit")

        choice = input("Moi chon tu 1 den 8: ")

        if choice == '1':
            set_student()
        elif choice == '2':
            set_teacher()
        elif choice == '3':
            set_staff()
        elif choice == '4':
            print_student()
        elif choice == '5':
            print_teacher()
        elif choice == '6':
            print_staff()
            print_teacher()
        elif choice == '7':
            print_all()
        elif choice == '8':
            print("DONG CHUONG TRINH")
            break
        else:
            print("DMM BI NGU A?")


if __name__ == "__main__":
    main()
