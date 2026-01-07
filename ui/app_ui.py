import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import threading
from PIL import Image, ImageTk

from ui.components import (
    ChatDisplay, AnalysisDisplay, ControlButton, 
    ImageCanvas, StatusLabel
)
from ui.dialogs import ImageDialog, VesselSettingsDialog, EnhancedPreviewDialog
from ui.gallery_window import LesionGalleryWindow
from utils.helpers import cv2_to_tkimage, resize_for_display, add_severity_label
from utils.constants import UI_COLORS
from config import SEVERITY_COLORS

class RetinaAnalyzerUI:
    def __init__(self, root, image_processor, vessel_processor, api_client, lesion_analyzer):
        self.root = root
        self.image_processor = image_processor
        self.vessel_processor = vessel_processor
        self.api_client = api_client
        self.lesion_analyzer = lesion_analyzer
        
        self.current_state = image_processor.current_state
        self.image_tk = None
        
        self.setup_ui()
        self.bind_events()
    
    def setup_ui(self):
        self.root.title("Retina AI Analyzer")
        self.root.geometry("1400x900")
        self.root.configure(bg=UI_COLORS['bg_dark'])
        
        self.root.grid_columnconfigure(0, weight=3)  
        self.root.grid_columnconfigure(1, weight=1) 
        self.root.grid_rowconfigure(0, weight=10)    
        self.root.grid_rowconfigure(1, weight=0)     
        
        self.setup_image_panel()
        
        self.setup_control_panel()
        

        self.setup_analysis_panel()
    
    def setup_image_panel(self):

        left_frame = tk.Frame(self.root, bg=UI_COLORS['bg_medium'], relief=tk.RAISED, bd=3)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        canvas_frame = tk.Frame(left_frame, bg=UI_COLORS['bg_light'])
        canvas_frame.grid(row=0, column=0, sticky='nsew')
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.image_canvas = ImageCanvas(canvas_frame)
        self.image_canvas.grid(row=0, column=0, sticky='nsew')
        
        self.image_label = tk.Label(self.image_canvas, 
                                   text="Upload a retina scan to begin analysis",
                                   bg=UI_COLORS['bg_light'],
                                   fg='white',
                                   font=('Arial', 14))
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def setup_control_panel(self):
        control_frame = tk.Frame(self.root, bg=UI_COLORS['bg_medium'], relief=tk.RAISED, bd=3)
        control_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        control_frame.grid_propagate(False)
        control_frame.config(height=80)
        
        btn_frame = tk.Frame(control_frame, bg=UI_COLORS['bg_medium'])
        btn_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.create_control_buttons(btn_frame)
        
        self.status_label = StatusLabel(control_frame)
        self.status_label.pack(side=tk.RIGHT, padx=10)
    
    def create_control_buttons(self, parent):
        self.buttons = {}
        
        button_configs = [
            ("upload", "Upload Retina Scan", self.load_image, UI_COLORS['accent_blue']),
            ("zoom_in", "Zoom In", self.zoom_in, UI_COLORS['accent_blue']),
            ("zoom_out", "Zoom Out", self.zoom_out, UI_COLORS['accent_blue']),
            ("heatmap", "Heatmap: OFF", self.toggle_heatmap, UI_COLORS['accent_red']),
            ("lesions", "Lesions: ON", self.toggle_lesion_boxes, UI_COLORS['accent_green']),
            ("macula", "Macula/Disc: ON", self.toggle_macula_disc, UI_COLORS['accent_orange']),
            ("vessels", "Vessels: OFF", self.toggle_vessel_overlay, UI_COLORS['accent_purple']),
            ("vessel_settings", "Vessel Settings", self.show_vessel_settings, UI_COLORS['accent_purple']),
            ("gallery", "Lesion Gallery", self.show_lesion_gallery, UI_COLORS['accent_teal']),
        ]
        
        for key, text, command, color in button_configs:
            btn = ControlButton(parent, text=text, command=command, color=color)
            btn.pack(side=tk.LEFT, padx=5)
            self.buttons[key] = btn
    
    def setup_analysis_panel(self):
        right_frame = tk.Frame(self.root, bg=UI_COLORS['bg_medium'], relief=tk.RAISED, bd=3)
        right_frame.grid(row=0, column=1, rowspan=2, sticky='nsew', padx=5, pady=5)
        right_frame.grid_rowconfigure(0, weight=1)  
        right_frame.grid_rowconfigure(1, weight=1)  
        right_frame.grid_columnconfigure(0, weight=1)
        
        report_frame = tk.Frame(right_frame, bg=UI_COLORS['bg_dark'])
        report_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=(5, 2))
        report_frame.grid_rowconfigure(0, weight=1)
        
        self.analysis_text = AnalysisDisplay(report_frame)
        self.analysis_text.pack(fill='both', expand=True)
        
        chat_frame = tk.Frame(right_frame, bg=UI_COLORS['bg_dark'])
        chat_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=(2, 5))
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)
        
        self.chat_display = ChatDisplay(chat_frame)
        self.chat_display.pack(fill='both', expand=True)
        
        self.chat_display.add_ai_message(
            "Hi! I'm RetinaExpert, your AI ophthalmology assistant. "
            "Upload a retinal scan and I'll help you analyze it!"
        )
        
        self.setup_chat_input(chat_frame)
    
    def setup_chat_input(self, parent):
        input_frame = tk.Frame(parent, bg=UI_COLORS['chat_bg'])
        input_frame.pack(fill='x', padx=5, pady=5)
        
        self.chat_input = tk.Entry(
            input_frame, 
            bg='white', 
            fg=UI_COLORS['text_dark'], 
            font=('Arial', 11),
            relief=tk.FLAT,
            borderwidth=2
        )
        self.chat_input.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        self.chat_input.bind("<Return>", lambda e: self.send_message())
        
        self.send_btn = ControlButton(
            input_frame, 
            text="Send", 
            command=self.send_message,
            color=UI_COLORS['accent_blue'],
            font=('Arial', 11, 'bold'),
            width=6
        )
        self.send_btn.pack(side=tk.RIGHT)
    
    def bind_events(self):
        self.image_canvas.bind("<Configure>", self.on_canvas_resize)
        self.root.bind("<Control-o>", lambda e: self.load_image())
        self.root.bind("<Control-plus>", lambda e: self.zoom_in())
        self.root.bind("<Control-minus>", lambda e: self.zoom_out())
    
    def on_canvas_resize(self, event):
        """Handle canvas resize event."""
        if self.current_state['uploaded_img'] is not None:
            self.update_display()
    
    
    def load_image(self):
        file_path = ImageDialog.load_image()
        if not file_path:
            return
        
        try:
            uploaded_img = cv2.imread(file_path)
            if uploaded_img is None:
                messagebox.showerror("Error", f"Could not load image: {file_path}")
                return
            
            self.image_processor.set_image(uploaded_img)
            
            self.current_state['show_vessels_only'] = False
            self.current_state['show_original_with_vessels'] = False
            self.current_state['zoom_scale'] = 1.0
            self.buttons['vessels'].config(text="Vessels: OFF", bg='#95a5a6')
            
            self.image_label.place_forget()
            
            self.analyze_image()
            self.update_display()
            self.update_status(f"Loaded: {os.path.basename(file_path)}")
            
            self.auto_send_analysis()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def analyze_image(self):
        """Analyze loaded image."""
        self.update_status("Analyzing image...")
        
        report = self.image_processor.analyze_image()
        self.analysis_text.set_report(report)
        
        if self.current_state['uploaded_img'] is not None:
            vessel_overlay, vessel_density = self.vessel_processor.segment_vessels(
                self.current_state['uploaded_img']
            )
            self.current_state['vessel_mask'] = vessel_overlay
            self.current_state['vessel_density'] = vessel_density
        
        self.update_status("Analysis complete")
    
    def update_display(self):
        if self.current_state['uploaded_img'] is None:
            return
        

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        display_img = self.create_display_image()
        
        display_resized = resize_for_display(
            display_img, 
            canvas_width, 
            canvas_height, 
            self.current_state['zoom_scale']
        )
        
        self.image_tk = cv2_to_tkimage(display_resized)
        

        self.image_canvas.display_image(self.image_tk)
    
    def create_display_image(self):
        if self.current_state['show_vessels_only'] and self.current_state['vessel_mask'] is not None:
            return self.vessel_processor.create_vessel_only_image(
                self.current_state['vessel_mask'],
                self.current_state['vessel_density']
            )
        
        elif self.current_state['show_original_with_vessels'] and self.current_state['vessel_mask'] is not None:
            display_img = self.vessel_processor.create_overlay_image(
                self.current_state['original_img'].copy(),
                self.current_state['vessel_mask']
            )
            add_severity_label(display_img, 
                             self.current_state['current_severity'],
                             self.current_state['current_confidence'])
            
            vessel_color = (
                self.vessel_processor.settings['color_b'],
                self.vessel_processor.settings['color_g'], 
                self.vessel_processor.settings['color_r']
            )
            model_type = "UNet" if self.vessel_processor.settings['use_unet'] else "Traditional"
            cv2.putText(display_img, f"Vessel Density: {self.current_state['vessel_density']:.2f}% ({model_type})", 
                       (20, display_img.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, vessel_color, 2, cv2.LINE_AA)
            
            return display_img
        
        else:
            display_img = self.current_state['original_img'].copy()
            add_severity_label(display_img, 
                             self.current_state['current_severity'],
                             self.current_state['current_confidence'])
            
            if self.current_state['show_lesion_boxes']:
                display_img = self.draw_lesion_boxes(display_img)
            
            if self.current_state['show_macula_disc']:
                display_img = self.draw_macula_disc(display_img)
            
            if self.current_state['show_heatmap'] and self.current_state['heatmap_overlay'] is not None:
                display_img = cv2.addWeighted(display_img, 0.7, 
                                            self.current_state['heatmap_overlay'], 0.3, 0)
            
            return display_img
    
    def draw_lesion_boxes(self, img):
        for lesion in self.current_state['current_lesions']:
            x1, y1, x2, y2 = lesion["box"]
            color = (0, 255, 0)
            thickness_box = max(2, int(min(x2-x1, y2-y1) * 0.015))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness_box)
            
            label = f"{lesion['class']}: {lesion['confidence']:.2f}"
            font_scale = 0.6
            thickness_text = max(1, int(2))
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)[0]
            
            cv2.rectangle(img, 
                         (x1, y1 - label_size[1] - 8),
                         (x1 + label_size[0] + 8, y1),
                         color, -1)
            
            cv2.putText(img, label, (x1 + 4, y1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_text)
        
        return img
    
    def draw_macula_disc(self, img):
        for obj in self.current_state['macula_disc_boxes']:
            x1, y1, x2, y2 = obj["box"]
            color = (255, 0, 0) if obj["class"] == "macula" else (0, 255, 255)
            thickness_box = max(2, int(min(x2-x1, y2-y1) * 0.015))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness_box)
            
            label = f"{obj['class']}: {obj['confidence']:.2f}"
            font_scale = 0.6
            thickness_text = max(1, int(2))
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)[0]
            
            cv2.rectangle(img, 
                         (x1, y1 - label_size[1] - 8),
                         (x1 + label_size[0] + 8, y1),
                         color, -1)
            
            cv2.putText(img, label, (x1 + 4, y1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_text)
        
        return img
    
    def zoom_in(self):
        self.current_state['zoom_scale'] *= 1.2
        self.update_display()
    
    def zoom_out(self):
        self.current_state['zoom_scale'] /= 1.2
        if self.current_state['zoom_scale'] < 0.1:
            self.current_state['zoom_scale'] = 0.1
        self.update_display()
    
    def toggle_heatmap(self):
        self.current_state['show_heatmap'] = not self.current_state['show_heatmap']
        text = f"Heatmap: {'ON' if self.current_state['show_heatmap'] else 'OFF'}"
        self.buttons['heatmap'].config(text=text)
        self.update_display()
    
    def toggle_lesion_boxes(self):
        self.current_state['show_lesion_boxes'] = not self.current_state['show_lesion_boxes']
        text = f"Lesions: {'ON' if self.current_state['show_lesion_boxes'] else 'OFF'}"
        self.buttons['lesions'].config(text=text)
        self.update_display()
    
    def toggle_macula_disc(self):
        self.current_state['show_macula_disc'] = not self.current_state['show_macula_disc']
        text = f"Macula/Disc: {'ON' if self.current_state['show_macula_disc'] else 'OFF'}"
        self.buttons['macula'].config(text=text)
        self.update_display()
    
    def toggle_vessel_overlay(self):
        if not self.current_state['show_original_with_vessels'] and not self.current_state['show_vessels_only']:
            self.current_state['show_original_with_vessels'] = True
            self.buttons['vessels'].config(text="Vessels Overlay: ON", bg='#9b59b6')
        elif self.current_state['show_original_with_vessels'] and not self.current_state['show_vessels_only']:
            self.current_state['show_original_with_vessels'] = False
            self.current_state['show_vessels_only'] = True
            self.buttons['vessels'].config(text="Vessels Only: ON", bg='#e74c3c')
        else:
            self.current_state['show_vessels_only'] = False
            self.current_state['show_original_with_vessels'] = False
            self.buttons['vessels'].config(text="Vessels: OFF", bg='#95a5a6')
        
        self.update_display()
    
    def show_vessel_settings(self):
        dialog = VesselSettingsDialog(self.root, self.vessel_processor, self.update_display)
        dialog.show()
    
    def show_lesion_gallery(self):
        if not self.current_state['current_lesions']:
            messagebox.showinfo("No Lesions", "No lesions detected in the current image.")
            return
        
        LesionGalleryWindow(
            self.root,
            self.current_state['uploaded_img'],
            self.current_state['current_lesions'],
            self.current_state['macula_center'],
            self.current_state['optic_disc_diameter_pixels']
        )
    
    def update_status(self, message):
        self.status_label.set_status(message)
    
    def auto_send_analysis(self):
        if self.current_state['uploaded_img'] is None:
            return
        
        disc_info = f"\n  Optic disc diameter: {self.current_state['optic_disc_diameter_pixels']}px" if self.current_state['optic_disc_diameter_pixels'] > 0 else ""
        
        self.chat_display.add_user_message(
            f"Retina Scan Analysis\n"
            f"  Severity: {self.current_state['current_severity']}\n"
            f"  Confidence: {self.current_state['current_confidence']:.1%}\n"
            f"  Lesions detected: {len(self.current_state['current_lesions'])}\n"
            f"  Vessel density: {self.current_state['vessel_density']:.2f}%{disc_info}"
        )
        
        if self.api_client.is_available():
            self.chat_display.add_ai_message("Analyzing the scan...")
            
            analysis_data = {
                'severity': self.current_state['current_severity'],
                'confidence': self.current_state['current_confidence'],
                'lesion_count': len(self.current_state['current_lesions']),
                'vessel_density': self.current_state['vessel_density'],
                'vessel_method': "UNet" if self.vessel_processor.settings['use_unet'] else "Traditional",
                'optic_disc_diameter': self.current_state['optic_disc_diameter_pixels']
            }
            
            def on_ai_response(error, result):
                if error:
                    self.chat_display.add_ai_message(f"Error: {error}")
                else:
                    self.chat_display.add_ai_message(f"Clinical Assessment\n\n{result}")
            
            self.api_client.process_in_thread("analyze", on_ai_response, analysis_data=analysis_data)
    
    def send_message(self):
        message = self.chat_input.get().strip()
        if not message:
            return
        
        if not self.api_client.is_available():
            messagebox.showwarning("AI Not Available", 
                                   "OpenRouter API is not configured or failed to initialize.")
            return
        
        self.chat_display.add_user_message(message)
        self.chat_input.delete(0, tk.END)
        
        self.chat_display.add_ai_message("Processing your question...")
        
        lesion_types = {}
        for lesion in self.current_state['current_lesions']:
            lesion_type = lesion['class']
            lesion_types[lesion_type] = lesion_types.get(lesion_type, 0) + 1
        
        context_data = {
            'severity': self.current_state['current_severity'],
            'confidence': self.current_state['current_confidence'],
            'lesion_count': len(self.current_state['current_lesions']),
            'lesion_types': lesion_types,
            'vessel_density': self.current_state['vessel_density'],
            'vessel_method': "UNet" if self.vessel_processor.settings['use_unet'] else "Traditional",
            'optic_disc_diameter': self.current_state['optic_disc_diameter_pixels']
        }
        
        def on_ai_response(error, result):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("end-3l", "end-1l")  
            
            if error:
                self.chat_display.add_ai_message(f"Error: {error}")
            else:
                self.chat_display.add_ai_message(result)
        
        self.api_client.process_in_thread("question", on_ai_response, 
                                         question=message, context_data=context_data)