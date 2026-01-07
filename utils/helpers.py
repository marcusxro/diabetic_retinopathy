import cv2
import numpy as np
import math
from PIL import Image, ImageTk
import tkinter as tk
from config import SEVERITY_COLORS

def resize_for_display(img, canvas_width, canvas_height, zoom_scale=1.0):
    if canvas_width <= 1 or canvas_height <= 1:
        canvas_width, canvas_height = 800, 600
    
    h, w = img.shape[:2]
    base_scale = min(canvas_width / w, canvas_height / h) * 0.95
    final_scale = base_scale * zoom_scale
    new_w, new_h = int(w * final_scale), int(h * final_scale)
    
    if new_w > 0 and new_h > 0:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def resize_for_preview(img, max_size):
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def cv2_to_tkimage(cv2_img):
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(image=im_pil)

def add_severity_label(img, severity, confidence):
    h, w = img.shape[:2]
    severity_text = f"{severity} ({confidence:.1%})"
    font_scale = max(1.2, w / 800)
    thickness = max(2, int(font_scale * 2))
    
    text_size = cv2.getTextSize(severity_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    bg_x1, bg_y1 = 20, 20
    bg_x2, bg_y2 = bg_x1 + text_size[0] + 30, bg_y1 + text_size[1] + 30
    
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                  SEVERITY_COLORS.get(severity, (255, 255, 255)), -1)
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), 3)
    
    cv2.putText(img, severity_text, 
                (bg_x1 + 15, bg_y1 + text_size[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                (0, 0, 0), thickness, cv2.LINE_AA)

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_contour_measurements(contour, pixels_per_micrometer):
    if len(contour) == 0:
        return 0, 0, 0
    
    area_pixels = cv2.contourArea(contour)
    
    if pixels_per_micrometer > 0:
        area_micrometers = area_pixels / (pixels_per_micrometer ** 2)
    else:
        area_micrometers = 0
    
    rect = cv2.minAreaRect(contour)
    width_pixels = rect[1][0]
    height_pixels = rect[1][1]
    
    if pixels_per_micrometer > 0:
        width_micrometers = width_pixels / pixels_per_micrometer
        height_micrometers = height_pixels / pixels_per_micrometer
    else:
        width_micrometers = 0
        height_micrometers = 0
    
    return area_micrometers, width_micrometers, height_micrometers

def draw_measurement_lines(img, rect, color=(255, 255, 0)):
    box_points = cv2.boxPoints(rect)
    box_points = box_points.astype(int)
    
    cv2.drawContours(img, [box_points], 0, color, 2)
    
    mid_y = int(rect[0][1])
    left_x = int(rect[0][0] - rect[1][0]/2)
    right_x = int(rect[0][0] + rect[1][0]/2)
    cv2.line(img, (left_x, mid_y-10), (right_x, mid_y-10), color, 2)
    
    mid_x = int(rect[0][0])
    top_y = int(rect[0][1] - rect[1][1]/2)
    bottom_y = int(rect[0][1] + rect[1][1]/2)
    cv2.line(img, (mid_x+10, top_y), (mid_x+10, bottom_y), color, 2)
    
    return box_points

def create_lesion_thumbnail(lesion_roi, lesion_type):
    """Create thumbnail for lesion with measurements."""
    gray = cv2.cvtColor(lesion_roi, cv2.COLOR_BGR2GRAY)
    
    if 'hemorrhage' in lesion_type.lower() or 'blood' in lesion_type.lower():
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    elif 'exudate' in lesion_type.lower() or 'bright' in lesion_type.lower():
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    else:
        binary = cv2.adaptiveThreshold(gray, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_roi = lesion_roi.copy()
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(vis_roi, [largest_contour], -1, (0, 255, 0), 2)
    
    return vis_roi, contours