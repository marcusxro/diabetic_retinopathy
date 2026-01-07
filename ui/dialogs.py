import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from ui.components import ControlButton, SettingsSlider, ColorPreview
from utils.constants import UI_COLORS
from utils.helpers import resize_for_preview

class ImageDialog:
    @staticmethod
    def load_image():
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp *.tiff *.webp")]
        )
        return file_path

class VesselSettingsDialog:
    def __init__(self, parent, vessel_processor, update_callback):
        self.parent = parent
        self.vessel_processor = vessel_processor
        self.update_callback = update_callback
        self.window = None
        
    def show(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title("Blood Vessel Detection Settings")
        self.window.geometry("550x800")
        self.window.configure(bg=UI_COLORS['bg_medium'])
        
        self.window.transient(self.parent)
        self.window.grab_set()
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = tk.Frame(self.window, bg=UI_COLORS['bg_medium'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame, bg=UI_COLORS['bg_medium'], highlightthickness=0)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=UI_COLORS['bg_medium'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        tk.Label(scrollable_frame, text="Blood Vessel Detection Settings", 
                bg=UI_COLORS['bg_medium'], fg='white', font=('Arial', 16, 'bold')).pack(pady=15)
        
        self.create_method_section(scrollable_frame)
        
        self.create_enhancement_section(scrollable_frame)
        
        self.create_visualization_section(scrollable_frame)
        
        self.create_action_buttons(scrollable_frame)
    
    def create_method_section(self, parent):
        """Create segmentation method section."""
        section = tk.LabelFrame(parent, text="Segmentation Method", 
                               bg='#34495e', fg='white', font=('Arial', 12, 'bold'),
                               padx=10, pady=10)
        section.pack(fill='x', padx=10, pady=10)
        
        settings = self.vessel_processor.get_settings()
        method_color = '#2ecc71' if settings['use_unet'] else '#f39c12'
        method_text = "UNet (Trained Model)" if settings['use_unet'] else "Traditional"
        
        self.method_label = tk.Label(section, text=f"Current: {method_text}", 
                           bg='#34495e', fg=method_color, font=('Arial', 11, 'bold'))
        self.method_label.pack(pady=5)
        
        def toggle_method():
            settings = self.vessel_processor.get_settings()
            settings['use_unet'] = not settings['use_unet']
            self.vessel_processor.update_setting('use_unet', settings['use_unet'])
            
            new_text = "UNet (Trained Model)" if settings['use_unet'] else "Traditional"
            self.method_label.config(text=f"Current: {new_text}", 
                           fg='#2ecc71' if settings['use_unet'] else '#f39c12')
            self.method_btn.config(text=f"Switch to {'Traditional' if settings['use_unet'] else 'UNet'}")
            
            if self.update_callback:
                self.update_callback()
        
        self.method_btn = ControlButton(section, 
                          text=f"Switch to {'Traditional' if settings['use_unet'] else 'UNet'}",
                          command=toggle_method,
                          color='#3498db')
        self.method_btn.pack(pady=10)
    
    def create_enhancement_section(self, parent):
        section = tk.LabelFrame(parent, text="Image Enhancement for UNet", 
                               bg='#34495e', fg='white', font=('Arial', 12, 'bold'),
                               padx=10, pady=10)
        section.pack(fill='x', padx=10, pady=10)
        
        settings = self.vessel_processor.get_settings()
        
        self.brightness_slider = SettingsSlider(
            section, "Brightness (0.5 - 3.0):", 0.5, 3.0, settings['enhance_brightness'], 0.1,
            lambda v: [self.vessel_processor.update_setting('enhance_brightness', float(v)), 
                      self.update_callback() if self.update_callback else None]
        )
        
        self.contrast_slider = SettingsSlider(
            section, "Contrast (0.5 - 3.0):", 0.5, 3.0, settings['enhance_contrast'], 0.1,
            lambda v: [self.vessel_processor.update_setting('enhance_contrast', float(v)), 
                      self.update_callback() if self.update_callback else None]
        )
        
        self.gamma_slider = SettingsSlider(
            section, "Gamma Correction (0.5 - 2.0):", 0.5, 2.0, settings['enhance_gamma'], 0.1,
            lambda v: [self.vessel_processor.update_setting('enhance_gamma', float(v)), 
                      self.update_callback() if self.update_callback else None]
        )
        
        self.clahe_slider = SettingsSlider(
            section, "CLAHE Clip Limit (1.0 - 5.0):", 1.0, 5.0, settings['clahe_clip'], 0.5,
            lambda v: [self.vessel_processor.update_setting('clahe_clip', float(v)), 
                      self.update_callback() if self.update_callback else None]
        )
        
        self.green_slider = SettingsSlider(
            section, "Green Channel Boost (0.5 - 3.0):", 0.5, 3.0, settings['green_boost'], 0.1,
            lambda v: [self.vessel_processor.update_setting('green_boost', float(v)), 
                      self.update_callback() if self.update_callback else None]
        )
        
        self.denoise_slider = SettingsSlider(
            section, "Denoise Strength (0-20):", 0, 20, settings['denoise_strength'], 1,
            lambda v: [self.vessel_processor.update_setting('denoise_strength', float(v)), 
                      self.update_callback() if self.update_callback else None]
        )
        
        toggle_frame = tk.Frame(section, bg='#34495e')
        toggle_frame.pack(pady=15)
        
        def toggle_invert():
            settings = self.vessel_processor.get_settings()
            new_value = not settings['invert_image']
            self.vessel_processor.update_setting('invert_image', new_value)
            self.invert_btn.config(text=f"Invert: {'ON' if new_value else 'OFF'}",
                                 bg='#e74c3c' if new_value else '#95a5a6')
            if self.update_callback:
                self.update_callback()
        
        self.invert_btn = ControlButton(toggle_frame, 
                          text=f"Invert: {'ON' if settings['invert_image'] else 'OFF'}",
                          command=toggle_invert,
                          color='#e74c3c' if settings['invert_image'] else '#95a5a6',
                          font=('Arial', 10))
        self.invert_btn.pack(side=tk.LEFT, padx=5)
        
        def toggle_equalize():
            settings = self.vessel_processor.get_settings()
            new_value = not settings['equalize_hist']
            self.vessel_processor.update_setting('equalize_hist', new_value)
            self.equalize_btn.config(text=f"Hist Eq: {'ON' if new_value else 'OFF'}",
                                   bg='#27ae60' if new_value else '#95a5a6')
            if self.update_callback:
                self.update_callback()
        
        self.equalize_btn = ControlButton(section, 
                            text=f"Histogram Equalization: {'ON' if settings['equalize_hist'] else 'OFF'}",
                            command=toggle_equalize,
                            color='#27ae60' if settings['equalize_hist'] else '#95a5a6',
                            font=('Arial', 10))
        self.equalize_btn.pack(pady=10)
    
    def create_visualization_section(self, parent):
        section = tk.LabelFrame(parent, text="Visualization Settings", 
                               bg='#34495e', fg='white', font=('Arial', 12, 'bold'),
                               padx=10, pady=10)
        section.pack(fill='x', padx=10, pady=10)
        
        settings = self.vessel_processor.get_settings()
        
        self.threshold_slider = SettingsSlider(
            section, "Detection Threshold (0.1 - 0.9):", 0.1, 0.9, settings['threshold'], 0.05,
            lambda v: [self.vessel_processor.update_setting('threshold', float(v)), 
                      self.update_callback() if self.update_callback else None]
        )
        
        self.opacity_slider = SettingsSlider(
            section, "Overlay Opacity (0.1 - 0.9):", 0.1, 0.9, settings['overlay_opacity'], 0.05,
            lambda v: [self.vessel_processor.update_setting('overlay_opacity', float(v)), 
                      self.update_callback() if self.update_callback else None]
        )
        
        tk.Label(section, text="Vessel Color:", 
                bg='#34495e', fg='white', font=('Arial', 10, 'bold')).pack(pady=(15, 5))
        
        color_frame = tk.Frame(section, bg='#34495e')
        color_frame.pack(pady=5)
        
        self.color_preview = ColorPreview(color_frame, 
                                         settings['color_r'], 
                                         settings['color_g'], 
                                         settings['color_b'])
        self.color_preview.pack(side=tk.LEFT, padx=5)
        
        def update_color_preview():
            settings = self.vessel_processor.get_settings()
            self.color_preview.set_color(settings['color_r'], 
                                        settings['color_g'], 
                                        settings['color_b'])
        
        tk.Label(section, text="Red:", bg='#34495e', fg='#ff6b6b').pack()
        self.red_slider = tk.Scale(section, from_=0, to=255, resolution=1,
                         orient=tk.HORIZONTAL, length=450,
                         bg='#34495e', fg='white',
                         command=lambda v: [
                             self.vessel_processor.update_setting('color_r', int(v)), 
                             update_color_preview(),
                             self.update_callback() if self.update_callback else None
                         ])
        self.red_slider.set(settings['color_r'])
        self.red_slider.pack(pady=5)
        
        tk.Label(section, text="Green:", bg='#34495e', fg='#6bff6b').pack()
        self.green_slider = tk.Scale(section, from_=0, to=255, resolution=1,
                           orient=tk.HORIZONTAL, length=450,
                           bg='#34495e', fg='white',
                           command=lambda v: [
                               self.vessel_processor.update_setting('color_g', int(v)), 
                               update_color_preview(),
                               self.update_callback() if self.update_callback else None
                           ])
        self.green_slider.set(settings['color_g'])
        self.green_slider.pack(pady=5)
        
        tk.Label(section, text="Blue:", bg='#34495e', fg='#6b6bff').pack()
        self.blue_slider = tk.Scale(section, from_=0, to=255, resolution=1,
                          orient=tk.HORIZONTAL, length=450,
                          bg='#34495e', fg='white',
                          command=lambda v: [
                              self.vessel_processor.update_setting('color_b', int(v)), 
                              update_color_preview(),
                              self.update_callback() if self.update_callback else None
                          ])
        self.blue_slider.set(settings['color_b'])
        self.blue_slider.pack(pady=5)
        
        def toggle_post_process():
            settings = self.vessel_processor.get_settings()
            new_value = not settings['post_process']
            self.vessel_processor.update_setting('post_process', new_value)
            self.post_btn.config(text=f"Post-processing: {'ON' if new_value else 'OFF'}",
                               bg='#e74c3c' if new_value else '#95a5a6')
            if self.update_callback:
                self.update_callback()
        
        self.post_btn = ControlButton(section, 
                        text=f"Post-processing: {'ON' if settings['post_process'] else 'OFF'}",
                        command=toggle_post_process,
                        color='#e74c3c' if settings['post_process'] else '#95a5a6',
                        font=('Arial', 10))
        self.post_btn.pack(pady=15)
    
    def create_action_buttons(self, parent):
        section = tk.Frame(parent, bg=UI_COLORS['bg_medium'])
        section.pack(pady=20)
        
        btn_frame = tk.Frame(section, bg=UI_COLORS['bg_medium'])
        btn_frame.pack(pady=10)
        
        preview_btn = ControlButton(btn_frame, text="Preview Enhancement", 
                                   command=self.show_enhanced_preview,
                                   color=UI_COLORS['accent_purple'])
        preview_btn.pack(side=tk.LEFT, padx=5)
        
        reset_btn = ControlButton(btn_frame, text="Reset All", 
                                 command=self.reset_settings,
                                 color=UI_COLORS['accent_red'])
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        apply_btn = ControlButton(btn_frame, text="Apply & Close", 
                                 command=self.window.destroy,
                                 color=UI_COLORS['accent_green'])
        apply_btn.pack(side=tk.LEFT, padx=5)
    
    def show_enhanced_preview(self):
        """Show enhancement preview dialog."""
        from processing.image_processor import ImageProcessor  
        
        messagebox.showinfo("Preview", "Enhancement preview requires an uploaded image.")
    
    def reset_settings(self):
        vessel_available = self.vessel_processor.vessel_model is not None
        
        self.vessel_processor.reset_settings(vessel_available)
        
        settings = self.vessel_processor.get_settings()
        
        self.brightness_slider.set_value(settings['enhance_brightness'])
        self.contrast_slider.set_value(settings['enhance_contrast'])
        self.gamma_slider.set_value(settings['enhance_gamma'])
        self.clahe_slider.set_value(settings['clahe_clip'])
        self.green_slider.set_value(settings['green_boost'])
        self.denoise_slider.set_value(settings['denoise_strength'])
        self.threshold_slider.set_value(settings['threshold'])
        self.opacity_slider.set_value(settings['overlay_opacity'])
        self.red_slider.set(settings['color_r'])
        self.green_slider.set(settings['color_g'])
        self.blue_slider.set(settings['color_b'])
        
        self.invert_btn.config(text="Invert: OFF", bg='#95a5a6')
        self.equalize_btn.config(text="Histogram Equalization: ON", bg='#27ae60')
        self.post_btn.config(text="Post-processing: ON", bg='#e74c3c')
        
        method_color = '#2ecc71' if settings['use_unet'] else '#f39c12'
        method_text = "UNet (Trained Model)" if settings['use_unet'] else "Traditional"
        self.method_label.config(text=f"Current: {method_text}", fg=method_color)
        self.method_btn.config(text=f"Switch to {'Traditional' if settings['use_unet'] else 'UNet'}")
        
        self.color_preview.set_color(settings['color_r'], 
                                    settings['color_g'], 
                                    settings['color_b'])
        
        if self.update_callback:
            self.update_callback()

class EnhancedPreviewDialog:
    def __init__(self, parent, original_img, enhanced_img, enhancement_summary):
        self.parent = parent
        self.original_img = original_img
        self.enhanced_img = enhanced_img
        self.enhancement_summary = enhancement_summary
        
        self.create_window()
    
    def create_window(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title("Image Enhancement Preview")
        self.window.geometry("1000x600")
        
        tk.Label(self.window, text="Image Enhancement Preview", 
                font=('Arial', 14, 'bold')).pack(pady=10)
        
        comparison_frame = tk.Frame(self.window)
        comparison_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        orig_frame = tk.Frame(comparison_frame, relief=tk.RAISED, bd=2)
        orig_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5)
        
        tk.Label(orig_frame, text="ORIGINAL", font=('Arial', 12, 'bold')).pack(pady=5)
        
        if self.original_img is not None:
            orig_resized = resize_for_preview(self.original_img, 400)
            orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
            orig_pil = Image.fromarray(orig_rgb)
            self.orig_photo = ImageTk.PhotoImage(orig_pil)
            
            orig_label = tk.Label(orig_frame, image=self.orig_photo)
            orig_label.image = self.orig_photo
            orig_label.pack(padx=10, pady=10)
        
        enh_frame = tk.Frame(comparison_frame, relief=tk.RAISED, bd=2)
        enh_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=5)
        
        tk.Label(enh_frame, text="ENHANCED (for UNet)", font=('Arial', 12, 'bold')).pack(pady=5)
        
        if self.enhanced_img is not None:
            enh_resized = resize_for_preview(self.enhanced_img, 400)
            enh_rgb = cv2.cvtColor(enh_resized, cv2.COLOR_BGR2RGB)
            enh_pil = Image.fromarray(enh_rgb)
            self.enh_photo = ImageTk.PhotoImage(enh_pil)
            
            enh_label = tk.Label(enh_frame, image=self.enh_photo)
            enh_label.image = self.enh_photo
            enh_label.pack(padx=10, pady=10)
        
        tk.Label(self.window, text=self.enhancement_summary, font=('Courier', 9), 
                justify=tk.LEFT).pack(pady=10, padx=20)
        
        close_btn = ControlButton(self.window, text="Close", 
                                command=self.window.destroy,
                                color=UI_COLORS['accent_green'])
        close_btn.pack(pady=10)