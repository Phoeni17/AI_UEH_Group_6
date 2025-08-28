import folium
import osmnx as ox
import networkx as nx
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ==============================
# 1. Thiết lập hệ thống fuzzy
# ==============================
distance = ctrl.Antecedent(np.arange(0, 21, 1), 'distance')   # km
traffic = ctrl.Antecedent(np.arange(0, 11, 1), 'traffic')     # 0-10 (kẹt xe)
weather = ctrl.Antecedent(np.arange(0, 2, 1), 'weather')      # 0=good, 1=bad
time = ctrl.Antecedent(np.arange(0, 2, 1), 'time')            # 0=off-peak, 1=peak
fare = ctrl.Consequent(np.arange(0, 301, 1), 'fare')          # 0-300k VND

# Membership functions
distance['short'] = fuzz.trimf(distance.universe, [0, 0, 5])
distance['medium'] = fuzz.trimf(distance.universe, [3, 10, 15])
distance['long'] = fuzz.trimf(distance.universe, [10, 20, 20])

traffic['low'] = fuzz.trimf(traffic.universe, [0, 0, 3])
traffic['medium'] = fuzz.trimf(traffic.universe, [2, 5, 8])
traffic['high'] = fuzz.trimf(traffic.universe, [7, 10, 10])

weather['good'] = fuzz.trimf(weather.universe, [0, 0, 1])
weather['bad'] = fuzz.trimf(weather.universe, [0, 1, 1])

time['offpeak'] = fuzz.trimf(time.universe, [0, 0, 1])
time['peak'] = fuzz.trimf(time.universe, [0, 1, 1])

fare['cheap'] = fuzz.trimf(fare.universe, [0, 0, 100])
fare['normal'] = fuzz.trimf(fare.universe, [50, 150, 200])
fare['expensive'] = fuzz.trimf(fare.universe, [150, 300, 300])

# Rule base
rules = [
    ctrl.Rule(distance['short'] & traffic['low'] & weather['good'] & time['offpeak'], fare['cheap']),
    ctrl.Rule(distance['short'] & traffic['low'] & weather['good'] & time['peak'], fare['normal']),
    ctrl.Rule(distance['short'] & traffic['low'] & weather['bad'] , fare['normal']),
    ctrl.Rule(distance['short'] & traffic['medium'] & weather['good'] & time['offpeak'], fare['cheap']),
    ctrl.Rule(distance['short'] & traffic['medium'] & weather['good'] & time['peak'], fare['normal']),
    ctrl.Rule(distance['short'] & traffic['medium'] & weather['bad'], fare['normal']),
    ctrl.Rule(distance['short'] & traffic['high'] & weather['good'], fare['normal']),
    ctrl.Rule(distance['short'] & traffic['high'] & weather['bad'] & time['offpeak'], fare['normal']),
    ctrl.Rule(distance['short'] & traffic['high'] & weather['bad'] & time['peak'], fare['expensive']),
    ctrl.Rule(distance['medium'] & traffic['low'] & weather['good'], fare['normal']),
    ctrl.Rule(distance['medium'] & traffic['low'] & weather['bad'], fare['normal']),
    ctrl.Rule(distance['medium'] & traffic['medium'] & weather['good'], fare['normal']),
    ctrl.Rule(distance['medium'] & traffic['medium'] & weather['bad'] & time['offpeak'], fare['normal']),
    ctrl.Rule(distance['medium'] & traffic['medium'] & weather['bad'] & time['peak'], fare['expensive']),
    ctrl.Rule(distance['medium'] & traffic['high'] & weather['good'] & time['offpeak'], fare['normal']),
    ctrl.Rule(distance['medium'] & traffic['high'] & weather['good'] & time['peak'], fare['expensive']),
    ctrl.Rule(distance['medium'] & traffic['high'] & weather['bad'], fare['expensive']),
    ctrl.Rule(distance['long'] & traffic['low'] & weather['good'], fare['normal']),
    ctrl.Rule(distance['long'] & traffic['low'] & weather['bad'] & time['offpeak'], fare['normal']),
    ctrl.Rule(distance['long'] & traffic['low'] & weather['bad'] & time['peak'], fare['expensive']),
    ctrl.Rule(distance['long'] & traffic['medium'] & weather['good'] & time['offpeak'], fare['normal']),
    ctrl.Rule(distance['long'] & traffic['medium'] & weather['good'] & time['peak'], fare['expensive']),
    ctrl.Rule(distance['long'] & traffic['medium'] & weather['bad'], fare['expensive']),
    ctrl.Rule(distance['long'] & traffic['high'] & weather['good'], fare['expensive']),
    ctrl.Rule(distance['long'] & traffic['high'] & weather['bad'], fare['expensive']),
    # Thêm tác động của thời tiết và giờ cao điểm
    ctrl.Rule(weather['bad'] | time['peak'], fare['expensive']),
]

fare_ctrl = ctrl.ControlSystem(rules)
fare_sim = ctrl.ControlSystemSimulation(fare_ctrl)

# ==============================
# 2. Tải dữ liệu OSM & UI
# ==============================
place = "District 3, Ho Chi Minh City, Vietnam"
G = ox.graph_from_place(place, network_type="drive")

tags = {"amenity": "restaurant"}
rest = ox.features_from_place(place, tags)
restaurant_names = rest.get("name").dropna().unique().tolist()

start_dropdown = widgets.Dropdown(options=restaurant_names, description='Start:')
end_dropdown = widgets.Dropdown(options=restaurant_names, description='End:')
traffic_slider = widgets.IntSlider(value=5, min=0, max=10, step=1, description='Traffic:')
weather_dropdown = widgets.Dropdown(options=["Good", "Bad"], description='Weather:')
time_dropdown = widgets.Dropdown(options=["Off-peak", "Peak"], description='Time:')

display_map_button = widgets.Button(description="Display Route")
output_map = widgets.Output()

# ==============================
# 3. Hàm xử lý
# ==============================
def display_selected_restaurants_map(b):
    with output_map:
        clear_output(wait=True)
        start_name = start_dropdown.value
        end_name = end_dropdown.value

        start_point = rest[rest['name'] == start_name].iloc[0].geometry
        end_point = rest[rest['name'] == end_name].iloc[0].geometry

        start_node = ox.distance.nearest_nodes(G, start_point.x, start_point.y)
        end_node = ox.distance.nearest_nodes(G, end_point.x, end_point.y)

        route = nx.shortest_path(G, start_node, end_node, weight="length")
        length_m = nx.shortest_path_length(G, start_node, end_node, weight="length")
        length_km = length_m / 1000

        # Tính tiền bằng fuzzy
        traffic_level = traffic_slider.value
        weather_val = 0 if weather_dropdown.value == "Good" else 1
        time_val = 0 if time_dropdown.value == "Off-peak" else 1

        fare_sim.input['distance'] = length_km
        fare_sim.input['traffic'] = traffic_level
        fare_sim.input['weather'] = weather_val
        fare_sim.input['time'] = time_val
        fare_sim.compute()
        price = fare_sim.output['fare'] * 1000   # quy đổi ra VND

        # Tạo map
        m = folium.Map(location=[start_point.y, start_point.x], zoom_start=15)
        route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
        folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.8).add_to(m)

        folium.Marker([start_point.y, start_point.x], popup=f"Start: {start_name}", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker([end_point.y, end_point.x], popup=f"End: {end_name}", icon=folium.Icon(color="red")).add_to(m)

        mid_idx = len(route_coords) // 2
        folium.Marker(
            location=route_coords[mid_idx],
            popup=f"Distance: {length_km:.2f} km\nTraffic: {traffic_level}\nWeather: {weather_dropdown.value}\nTime: {time_dropdown.value}\nPrice: {price:,.0f} VND",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

        print(f"Khoảng cách: {length_km:.2f} km")
        print(f"Kẹt xe (0-10): {traffic_level}")
        print(f"Thời tiết: {weather_dropdown.value}")
        print(f"Thời gian: {time_dropdown.value}")
        print(f"Giá tiền fuzzy: {price:,.0f} VND")

        display(m)

# ==============================
# 4. Chạy app
# ==============================
display_map_button.on_click(display_selected_restaurants_map)
display(start_dropdown, end_dropdown, traffic_slider, weather_dropdown, time_dropdown, display_map_button, output_map)