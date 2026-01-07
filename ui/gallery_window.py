import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from utils.helpers import resize_for_preview, calculate_distance
from ui.components import ControlButton
from utils.constants import UI_COLORS, GALLERY_COLS, MAX_GALLERY_IMAGE_SIZE

class LesionGalleryWindow:
    def __init__(self, parent, uploaded_img, lesions, macula_center=None, optic_disc_diameter_pixels=0):
        self.parent = parent
        self.uploaded_img = uploaded_img
        self.lesions = lesions
        self.macula_center = macula_center
        self.optic_disc_diameter_pixels = optic_disc_diameter_pixels
        
        if not lesions:
            messagebox.showinfo("No Lesions", "No lesions detected in the current image.")
            return
        
        self.create_window()
    
    def create_window(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title("Lesion Gallery - Detailed Analysis")
        self.window.geometry("1200x800")
        self.window.configure(bg=UI_COLORS['bg_dark'])
        
        title_label = tk.Label(
            self.window, 
            text=f"Lesion Gallery ({len(self.lesions)} lesions detected)",
            bg=UI_COLORS['bg_dark'], 
            fg='white', 
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=15)
        
        self.create_scrollable_frame()
        
        self.create_close_button()
    
    def create_scrollable_frame(self):
        main_frame = tk.Frame(self.window, bg=UI_COLORS['bg_dark'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame, bg=UI_COLORS['bg_dark'], highlightthickness=0)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=UI_COLORS['bg_dark'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        if self.optic_disc_diameter_pixels > 0:
            pixels_per_micrometer = self.optic_disc_diameter_pixels / 1500.0
        else:
            if self.uploaded_img is not None:
                h, w = self.uploaded_img.shape[:2]
                pixels_per_micrometer = w / 15000.0  
            else:
                pixels_per_micrometer = 0.1
        
        self.create_lesion_thumbnails(scrollable_frame, pixels_per_micrometer)
        
        self.add_summary_statistics(scrollable_frame)
    
    def create_lesion_thumbnails(self, parent_frame, pixels_per_micrometer):
        lesion_images = []
        lesion_measurements = []
        
        for idx, lesion in enumerate(self.lesions):
            x1, y1, x2, y2 = lesion["box"]
            
            padding = 20
            h, w = self.uploaded_img.shape[:2]
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            lesion_roi = self.uploaded_img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            
            if lesion_roi.size == 0:
                continue
            
            vis_roi = self.enhance_lesion_display(lesion_roi, lesion['class'])
            
            measurement_text = self.calculate_lesion_measurements(vis_roi, lesion['class'], pixels_per_micrometer)
            
            lesion_images.append(vis_roi)
            lesion_measurements.append(measurement_text)
        
        num_cols = GALLERY_COLS
        num_rows = (len(lesion_images) + num_cols - 1) // num_cols
        
        for row in range(num_rows):
            row_frame = tk.Frame(parent_frame, bg=UI_COLORS['bg_dark'])
            row_frame.pack(fill='x', padx=10, pady=10)
            
            for col in range(num_cols):
                idx = row * num_cols + col
                if idx >= len(lesion_images):
                    break
                
                self.create_lesion_frame(row_frame, idx, lesion_images[idx], lesion_measurements[idx])
    
    def enhance_lesion_display(self, lesion_roi, lesion_type):
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
            try:
                largest_contour = max(contours, key=cv2.contourArea)
                
                cv2.drawContours(vis_roi, [largest_contour], -1, (0, 255, 0), 2)
                
                rect = cv2.minAreaRect(largest_contour)
                box_points = cv2.boxPoints(rect)
                box_points = box_points.astype(int)
                
                cv2.drawContours(vis_roi, [box_points], 0, (255, 0, 0), 2)
                
                mid_y = int(rect[0][1])
                left_x = int(rect[0][0] - rect[1][0]/2)
                right_x = int(rect[0][0] + rect[1][0]/2)
                cv2.line(vis_roi, (left_x, mid_y-10), (right_x, mid_y-10), (255, 255, 0), 2)
                
                mid_x = int(rect[0][0])
                top_y = int(rect[0][1] - rect[1][1]/2)
                bottom_y = int(rect[0][1] + rect[1][1]/2)
                cv2.line(vis_roi, (mid_x+10, top_y), (mid_x+10, bottom_y), (255, 255, 0), 2)
                
            except Exception as e:
                print(f"Error drawing lesion {e}")
        
        return vis_roi
    
    def calculate_lesion_measurements(self, vis_roi, lesion_type, pixels_per_micrometer):
        gray = cv2.cvtColor(vis_roi, cv2.COLOR_BGR2GRAY)
        
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
        
        if contours:
            try:
                largest_contour = max(contours, key=cv2.contourArea)
                area_pixels = cv2.contourArea(largest_contour)
                
                if pixels_per_micrometer > 0:
                    area_micrometers = area_pixels / (pixels_per_micrometer ** 2)
                else:
                    area_micrometers = 0
                
                rect = cv2.minAreaRect(largest_contour)
                width_pixels = rect[1][0]
                height_pixels = rect[1][1]
                
                if pixels_per_micrometer > 0:
                    width_micrometers = width_pixels / pixels_per_micrometer
                    height_micrometers = height_pixels / pixels_per_micrometer
                else:
                    width_micrometers = 0
                    height_micrometers = 0
                
                return f"Area: {area_micrometers:.1f}µm² | Size: {width_micrometers:.1f}×{height_micrometers:.1f}µm"
            except Exception as e:
                return f"Measurement error: {str(e)}"
        else:
            return "No contour found for measurement"
    
    def create_lesion_frame(self, parent_frame, idx, lesion_img, measurement_text):
        lesion = self.lesions[idx]
        
        lesion_frame = tk.Frame(parent_frame, bg=UI_COLORS['bg_medium'], relief=tk.RAISED, bd=2)
        lesion_frame.pack(side=tk.LEFT, padx=10, pady=10, fill='both', expand=True)
        
        header_text = f"Lesion {idx+1}: {lesion['class']} ({lesion['confidence']:.1%})"
        header_label = tk.Label(lesion_frame, text=header_text,
                               bg='#34495e', fg='white',
                               font=('Arial', 10, 'bold'))
        header_label.pack(fill='x', padx=5, pady=5)
        
        img_rgb = cv2.cvtColor(lesion_img, cv2.COLOR_BGR2RGB)
        
        h, w = img_rgb.shape[:2]
        scale = min(MAX_GALLERY_IMAGE_SIZE/w, MAX_GALLERY_IMAGE_SIZE/h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        im_pil = Image.fromarray(img_resized)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        
        img_label = tk.Label(lesion_frame, image=imgtk, bg=UI_COLORS['bg_medium'])
        img_label.image = imgtk
        img_label.pack(padx=10, pady=10)
        
        meas_label = tk.Label(lesion_frame, text=measurement_text,
                             bg=UI_COLORS['bg_medium'], fg='#ecf0f1',
                             font=('Courier', 9))
        meas_label.pack(padx=5, pady=5)
        
        if self.macula_center is not None and self.optic_disc_diameter_pixels > 0:
            x1, y1, x2, y2 = lesion["box"]
            lesion_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance_pixels = calculate_distance(lesion_center, self.macula_center)
            distance_DD = distance_pixels / self.optic_disc_diameter_pixels
            
            location_label = tk.Label(lesion_frame,
                                    text=f"Distance from macula: {distance_DD:.2f} DD",
                                    bg=UI_COLORS['bg_medium'], fg='#bdc3c7',
                                    font=('Arial', 8))
            location_label.pack(padx=5, pady=2)
    
    def add_summary_statistics(self, parent_frame):
        if not self.lesions:
            return
        
        summary_frame = tk.Frame(parent_frame, bg='#34495e', relief=tk.RAISED, bd=3)
        summary_frame.pack(fill='x', padx=20, pady=20)
        
        lesion_types = {}
        for lesion in self.lesions:
            lesion_type = lesion['class']
            lesion_types[lesion_type] = lesion_types.get(lesion_type, 0) + 1
        
        summary_text = "SUMMARY STATISTICS:\n"
        summary_text += f"Total Lesions: {len(self.lesions)}\n"
        
        for lesion_type, count in lesion_types.items():
            summary_text += f"{lesion_type}: {count} lesions\n"
        
        if self.optic_disc_diameter_pixels > 0:
            summary_text += f"\nConversion: 1 DD = {self.optic_disc_diameter_pixels}px ≈ 1500µm"
        
        summary_label = tk.Label(summary_frame, text=summary_text,
                                bg='#34495e', fg='white',
                                font=('Courier', 10), justify=tk.LEFT)
        summary_label.pack(padx=10, pady=10)
    
    def create_close_button(self):
        close_frame = tk.Frame(self.window, bg=UI_COLORS['bg_dark'])
        close_frame.pack(pady=20)
        
        close_btn = ControlButton(
            close_frame, 
            text="Close Gallery", 
            command=self.window.destroy,
            color=UI_COLORS['accent_red']
        )
        close_btn.pack()