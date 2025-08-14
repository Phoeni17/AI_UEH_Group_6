import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ===== Các biến vũ trụ =====
distance = ctrl.Antecedent(np.arange(20, 201, 1), 'distance')         # cm
direction = ctrl.Antecedent(np.arange(-90, 91, 1), 'direction')       # độ, - = Trái, + = Phải
speed     = ctrl.Consequent(np.arange(0, 101, 1), 'speed')            # 0..100 (%)
steer     = ctrl.Consequent(np.arange(-45, 46, 1), 'steer')           # độ, - = Trái, + = Phải

# ===== Hàm thành viên =====
# Khoảng cách
distance['gần']   = fuzz.trapmf(distance.universe, [0, 0, 20, 80])
distance['trung bình'] = fuzz.trimf(distance.universe, [50, 120, 190])
distance['xa']    = fuzz.trapmf(distance.universe, [160, 240, 300, 300])

distance.view()
# Hướng (hướng mục tiêu)
direction['trái']   = fuzz.trapmf(direction.universe, [-90, -90, -60, -10])
direction['trung tâm'] = fuzz.trimf(direction.universe,  [-20, 0, 20])
direction['phải']  = fuzz.trapmf(direction.universe, [10, 60, 90, 90])

direction.view()
# Tốc độ
speed['chậm']   = fuzz.trapmf(speed.universe, [0, 0, 20, 40])
speed['trung bình'] = fuzz.trimf(speed.universe, [30, 50, 70])
speed['nhanh']   = fuzz.trapmf(speed.universe, [60, 100, 101, 102])

speed.view()
# Góc lái
steer['rẽ gắt trái']  = fuzz.trapmf(steer.universe, [-45, -45, -40, -20])
steer['rẽ nhẹ trái'] = fuzz.trimf(steer.universe,  [-30, -15, 0])
steer['đi thẳng']    = fuzz.trimf(steer.universe,  [-5, 0, 5])
steer['rẽ nhẹ phải']= fuzz.trimf(steer.universe,  [0, 15, 30])
steer['rẽ gắt phải'] = fuzz.trapmf(steer.universe, [20, 40, 45, 45])

steer.view()
# ===== Quy tắc =====
# Tốc độ chủ yếu theo khoảng cách
r_s1 = ctrl.Rule(distance['gần'],   speed['chậm'])
r_s2 = ctrl.Rule(distance['trung bình'], speed['trung bình'])
r_s3 = ctrl.Rule(distance['xa'],    speed['nhanh'])

# Lái theo hướng mục tiêu; cường độ thay đổi theo độ gần
r1 = ctrl.Rule(direction['trái'] | distance['gần'],   steer['rẽ gắt phải'])
r2 = ctrl.Rule(direction['trái'] | distance['trung bình'], steer['rẽ nhẹ phải'])
r3 = ctrl.Rule(direction['trái'] | distance['xa'],    steer['đi thẳng'])

r4 = ctrl.Rule(direction['trung tâm'] | distance['xa'],    steer['đi thẳng'])
r5 = ctrl.Rule(direction['trung tâm'] | distance['trung bình'], steer['rẽ nhẹ trái'])
r6 = ctrl.Rule(direction['trung tâm'] | distance['gần'],   steer['rẽ gắt trái'])

r7 = ctrl.Rule(direction['phải'] | distance['gần'],   steer['rẽ gắt trái'])
r8 = ctrl.Rule(direction['phải'] | distance['trung bình'], steer['rẽ nhẹ trái'])
r9 = ctrl.Rule(direction['phải'] | distance['xa'],    steer['đi thẳng'])

# Xây dựng hệ thống
control_system = ctrl.ControlSystem([r_s1, r_s2, r_s3, r1, r2, r3, r4, r5, r6, r7, r8, r9])
sys = ctrl.ControlSystemSimulation(control_system)

sys.input['distance'] = int(input("Khoảng cách (Từ 20cm -> 200cm) (Nhập số thôi): "))
sys.input['direction'] = int(input("Hướng của vật (Từ -90 độ -> 90 độ) (Nhập số thôi): "))


sys.compute()

print(sys.output['speed'])
print(sys.output['steer'])

speed.view(sim=sys)
steer.view(sim=sys)
