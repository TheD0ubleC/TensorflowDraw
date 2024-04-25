import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk
import math
import time
import random
def set_theme(root):
    style = ttk.Style(root)
    style.theme_use('clam')
script_dir = os.path.dirname(os.path.abspath(__file__))
def get_model_options():
    model_dir = os.path.join(script_dir, 'models')
    return [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
def load_model(model_name):
    global current_model, detect_fn, font
    model_path = os.path.join(script_dir, 'models', model_name, 'saved_model')
    if not os.path.exists(model_path):
        print("模型路径不存在：", model_path)
        return
    current_model = tf.saved_model.load(model_path)
    detect_fn = current_model.signatures['serving_default']
    print(f"模型 {model_name} 已加载成功")
    update_labels_and_excludes(model_name)
    font_path = os.path.join(script_dir, 'models', "SourceHanSansCN-Light.otf")
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)
def update_labels_and_excludes(model_name):
    global LABEL_MAP, EXCLUDE_LIST
    label_path = os.path.join(script_dir, 'models', model_name, 'labels.txt')
    exclude_path = os.path.join(script_dir, 'models', model_name, 'exclude_list.txt')
    LABEL_MAP = load_labels(label_path)
    EXCLUDE_LIST = load_exclude_list(exclude_path)
def load_labels(label_path):
    labels = {}
    try:
        with open(label_path, 'r', encoding='utf-8') as file:
            for line in file:
                index, label = line.strip().split(':')
                labels[int(index)] = label
    except FileNotFoundError:
        print("未找到标签文件：", label_path)
    return labels
def load_exclude_list(exclude_path):
    exclude_set = set()
    try:
        with open(exclude_path, 'r') as file:
            exclude_set = {int(line.strip()) for line in file}
    except FileNotFoundError:
        print("未找到排除列表文件：", exclude_path)
    return exclude_set
def detect_objects(frame, threshold, red, green, blue, width, dynamic_color):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections['num_detections'])
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)
    t = time.time()
    if dynamic_color:
        red = int((math.sin(t * 0.5) + 1) * 127.5)
        green = int((math.sin(t * 0.5 + 2) + 1) * 127.5)
        blue = int((math.sin(t * 0.5 + 4) + 1) * 127.5)
    color = (int(red), int(green), int(blue))
    border_width = width
    for i in range(num_detections):
        if scores[i] >= threshold and classes[i] not in EXCLUDE_LIST:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            draw.rectangle([(left, top), (right, bottom)], outline=color, width=border_width)
            label = LABEL_MAP.get(classes[i], '未知')
            draw.text((left, top - 30), f"{label}: {int(scores[i] * 100)}%", fill=color, font=font)
    update_prank(frame.shape, red, green, blue, width)
    draw_prank(draw, frame.shape)
    return ImageTk.PhotoImage(image)
def main():
    global red_slider, green_slider, blue_slider, threshold_slider, width_slider
    global current_model, detect_fn, font, LABEL_MAP, EXCLUDE_LIST, cap
    cap = None
    root = tk.Tk()
    root.title("Tensorflow模型[COCO数据集]加载器")
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=3)
    root.rowconfigure(0, weight=1)
    control_frame = ttk.Frame(root, padding="4 4 12 12")
    control_frame.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
    control_frame.rowconfigure(0, weight=1)
    video_frame = ttk.Frame(root, padding="4 4 12 12")
    video_frame.grid(column=1, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
    video_frame.rowconfigure(0, weight=1)
    video_frame.columnconfigure(0, weight=1)
    video_label = tk.Label(video_frame)
    video_label.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    settings_frame = ttk.Frame(control_frame, padding="3 3 12 12")
    settings_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)
    create_control_widgets(settings_frame)
    setup_camera_and_model(video_label)
    root.mainloop()
def create_control_widgets(settings_frame):
    global red_slider, green_slider, blue_slider, threshold_slider, width_slider
    global model_combobox
    global color_dynamics_check
    color_dynamics_label = ttk.Label(settings_frame, text="启用颜色缓动:")
    color_dynamics_label.grid(column=0, row=6, sticky=tk.W)
    color_dynamics_check = ttk.Checkbutton(settings_frame, text="启用")
    color_dynamics_check.grid(column=1, row=6, sticky=tk.W)
    red_label = ttk.Label(settings_frame, text="红 (R):")
    red_label.grid(column=0, row=0, sticky=tk.W)
    red_slider = ttk.Scale(settings_frame, from_=0, to=255, orient="horizontal")
    red_slider.grid(column=1, row=0, sticky=(tk.W, tk.E))
    red_slider.set(255)
    red_value_label = ttk.Label(settings_frame, text="255")
    red_value_label.grid(column=2, row=0, sticky=tk.W)
    red_slider['command'] = lambda v: red_value_label.config(text=f"{int(float(v))}")
    green_label = ttk.Label(settings_frame, text="绿 (G):")
    green_label.grid(column=0, row=1, sticky=tk.W)
    green_slider = ttk.Scale(settings_frame, from_=0, to=255, orient="horizontal")
    green_slider.grid(column=1, row=1, sticky=(tk.W, tk.E))
    green_slider.set(255)
    green_value_label = ttk.Label(settings_frame, text="255")
    green_value_label.grid(column=2, row=1, sticky=tk.W)
    green_slider['command'] = lambda v: green_value_label.config(text=f"{int(float(v))}")
    blue_label = ttk.Label(settings_frame, text="蓝 (B):")
    blue_label.grid(column=0, row=2, sticky=tk.W)
    blue_slider = ttk.Scale(settings_frame, from_=0, to=255, orient="horizontal")
    blue_slider.grid(column=1, row=2, sticky=(tk.W, tk.E))
    blue_slider.set(255)
    blue_value_label = ttk.Label(settings_frame, text="255")
    blue_value_label.grid(column=2, row=2, sticky=tk.W)
    blue_slider['command'] = lambda v: blue_value_label.config(text=f"{int(float(v))}")
    threshold_label = ttk.Label(settings_frame, text="置信度 (%):")
    threshold_label.grid(column=0, row=3, sticky=tk.W)
    threshold_slider = ttk.Scale(settings_frame, from_=0, to=100, orient="horizontal")
    threshold_slider.grid(column=1, row=3, sticky=(tk.W, tk.E))
    threshold_slider.set(50)
    threshold_value_label = ttk.Label(settings_frame, text="50%")
    threshold_value_label.grid(column=2, row=3, sticky=tk.W)
    threshold_slider['command'] = lambda v: threshold_value_label.config(text=f"{int(float(v))}%")
    width_label = ttk.Label(settings_frame, text="边框宽度:")
    width_label.grid(column=0, row=4, sticky=tk.W)
    width_slider = ttk.Scale(settings_frame, from_=1, to=10, orient="horizontal")
    width_slider.grid(column=1, row=4, sticky=(tk.W, tk.E))
    width_slider.set(1)
    width_value_label = ttk.Label(settings_frame, text="1")
    width_value_label.grid(column=2, row=4, sticky=tk.W)
    width_slider['command'] = lambda v: width_value_label.config(text=f"{int(float(v))}")
    model_label = ttk.Label(settings_frame, text="选择模型:")
    model_label.grid(column=0, row=5, sticky=tk.W)
    model_options = get_model_options()
    model_combobox = ttk.Combobox(settings_frame, values=model_options, state="readonly")
    model_combobox.grid(column=1, row=5, sticky=(tk.W, tk.E))
    model_combobox.current(0)
    model_combobox.bind("<<ComboboxSelected>>", lambda event: load_model(model_combobox.get()))
    toggle_prank_button = ttk.Button(settings_frame, text="恶作剧开关", command=toggle_prank)
    toggle_prank_button.grid(column=0, row=7, sticky=(tk.W, tk.E))
prank_active = False
prank_box = None
prank_label = "未知生物"
prank_timer = 0
def toggle_prank():
    global prank_active
    prank_active = not prank_active
def update_prank(frame_shape, red_value, green_value, blue_value, border_width):
    global prank_box, prank_timer, prank_color, prank_confidence, prank_width
    if prank_active:
        if prank_timer <= 0:
            move_x = random.randint(-20, 20)
            move_y = random.randint(-20, 20)
            adjust_width = random.randint(-20, 20)
            adjust_height = random.randint(-20, 20)
            min_width = 50 
            min_height = 50 
            if prank_box is None:
                x = random.randint(100, frame_shape[1] - 200)
                y = random.randint(100, frame_shape[0] - 200)
                width = max(min_width, random.randint(100, 200))
                height = max(min_height, random.randint(100, 200))
                prank_box = [y / frame_shape[0], x / frame_shape[1], (y + height) / frame_shape[0], (x + width) / frame_shape[1]]
            else:
                y_min, x_min, y_max, x_max = prank_box
                new_y_min = max(0, y_min + move_y / frame_shape[0])
                new_x_min = max(0, x_min + move_x / frame_shape[1])
                new_y_max = min(1, y_max + (move_y + adjust_height) / frame_shape[0])
                new_x_max = min(1, x_max + (move_x + adjust_width) / frame_shape[1])
                if (new_x_max - new_x_min) * frame_shape[1] < min_width:
                    new_x_max = new_x_min + min_width / frame_shape[1]
                if (new_y_max - new_y_min) * frame_shape[0] < min_height:
                    new_y_max = new_y_min + min_height / frame_shape[0]
                prank_box = [new_y_min, new_x_min, new_y_max, new_x_max]

            prank_color = (int(red_value), int(green_value), int(blue_value))
            prank_width = border_width
            prank_confidence = random.randint(55, 89)
            prank_timer = random.randint(1, 5)
        else:
            prank_timer -= 1
    else:
        prank_box = None
def draw_prank(draw, frame_shape):
    if prank_box and prank_active:
        ymin, xmin, ymax, xmax = prank_box
        jitter_x = random.randint(-5, 5)
        jitter_y = random.randint(-5, 5)
        left = min(xmin * frame_shape[1] + jitter_x, xmax * frame_shape[1] + jitter_x)
        right = max(xmin * frame_shape[1] + jitter_x, xmax * frame_shape[1] + jitter_x)
        top = min(ymin * frame_shape[0] + jitter_y, ymax * frame_shape[0] + jitter_y)
        bottom = max(ymin * frame_shape[0] + jitter_y, ymax * frame_shape[0] + jitter_y)
        draw.rectangle([(left, top), (right, bottom)], outline=prank_color, width=prank_width)
        draw.text((left, top - 30), f"{prank_label}: {prank_confidence}%", fill=prank_color, font=font)
def setup_camera_and_model(video_label):
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    load_model(get_model_options()[0])
    def update_frame():
        ret, frame = cap.read()
        if ret:
            dynamic_color = color_dynamics_check.instate(['selected'])
            if dynamic_color:
                t = time.time()
                red_value = int((math.sin(t * 0.5) + 1) * 127.5)
                green_value = int((math.sin(t * 0.5 + 2) + 1) * 127.5)
                blue_value = int((math.sin(t * 0.5 + 4) + 1) * 127.5)
            else:
                red_value = red_slider.get()
                green_value = green_slider.get()
                blue_value = blue_slider.get()
            red_slider.set(red_value)
            green_slider.set(green_value)
            blue_slider.set(blue_value)
            photo = detect_objects(frame, threshold_slider.get() / 100.0,
                                red_value, green_value, blue_value,
                                int(width_slider.get()), dynamic_color)
            video_label.config(image=photo)
            video_label.image = photo
        video_label.after(5, update_frame)
    update_frame()
if __name__ == "__main__":
    main()