import math


def so_nguyen_to(a):
    if a <= 1:
        return False

    for i in range(2, a):
        if a % i == 0:
            return False

    return True


print(so_nguyen_to(128127482164753983798738579843758437598735987398573985398568736535874365874365984375984375843754365984365986435))


"""
boolean
thiet ke sudocode csdl cho 1 truong hoc
- luu dc thong tin ng dung: ten, dchi, sdt
- 3 nhom ng: hoc sinh, giao vien, nv truong (giao vien also nv truong, nv truong may not gvien)
- deu phai lay ra dc thong tin ve ten, dchi, sdt
- hoc sinh thi can phai luu them dang hoc mon gi
- giao vien can phai luu them thong tin la dang day nhung lop nao
- nv phai lay dc ket qua la luong hien tai la bao nhieu

class User:
    init(self):
      name:
      address:
      phone_number
    
    def get_name(self):
        return self.name
    def get_address(self):
        return self.address
    def get_phone_number(self):
        return self.phone_number

    def set_name(self, name):
        self.name = name
    def set_address(self, address):
        self.address = address

class Teacher(User):
    init(self):
      super().__init__(self)
      self.name = 'MA'
      self.classes = []

    def get_name(self):
        return self.address
    function add_class(self, class_name):
      self.classes.append(class_name)

    function get_classes(self):
      return self.classes


class Student():
    init(self):
      super().__init__(self)
      self.classes = []

class Principal(User):
    init(self):
      super().__init__(self)



Bai tap 2
thiet ke du lieu cho game onl:
- player (mon phai(nga mi, vo dang, thieu lam), )
if nga mi -> nu
- nga mi chieu a1: dam =5
- vo dang chieu a2: dam =0,5*tuvi
- thieu lam chieu a3: dam to nga mi =0, dam to vo dang = tuvi*2
- user.player (name, age, so ngay choi)

class Gender:
    self.male = 'male'
    self.female = 'female'

def tu_vi(a):
    if a < 0:
        return
    if a <= 100
        return a*1
    if 100 < a < 200
        return a*
    
class skill:
    init(self):
        skill_name
        damage

class Guild:
    init(self, gender):
        self.gender = gender
        self.tu_vi = None
    def set_tu_vi(self)
        if a < 0
    def cast_skill(self)
        pass        

    
class nga_mi
    super_init(self)
        self.gender = Gender.female

    def set_tu_vi(self):
        
    def cast_skill(self):
        self.skill_name = 'a1'
        self.damage = 5


class thieu_lam
    super_init(self)
    

    def cast_skill(self):
        self.skill_name = 'a2"
        self.damage = 0.5*tu_vi(self.tu_vi)


Bai tap: 
- viet lai bai thiet ke bang hoc sinh
- chay dc script a python 3 lua chon tao, 3 to 4 lua chon lay du lieu

"""
