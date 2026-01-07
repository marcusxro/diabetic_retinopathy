import cv2
import numpy as np
import math
from config import SEVERITY_CLASSES, SEVERITY_COLORS, CLINICAL_NOTES
from utils.helpers import add_severity_label, calculate_distance

class ImageProcessor:
    def __init__(self, model_loader):
        self.models = model_loader.get_all_models()
        self.current_state = {
            'uploaded_img': None,
            'original_img': None,
            'display_img': None,
            'current_severity': "No_DR",
            'current_confidence': 0.0,
            'current_lesions': [],
            'heatmap_overlay': None,
            'vessel_mask': None,
            'vessel_density': 0.0,
            'macula_disc_boxes': [],
            'optic_disc_diameter_pixels': 0,
            'disc_center': None,
            'macula_center': None,
            'show_heatmap': False,
            'show_lesion_boxes': True,
            'show_vessels_only': False,
            'show_original_with_vessels': False,
            'show_macula_disc': True,
            'zoom_scale': 1.0,
            'vessel_settings': {
                'threshold': 0.3,
                'color_r': 255,
                'color_g': 0,
                'color_b': 0,
                'overlay_opacity': 0.35,
                'post_process': True,
                'use_unet': self.models['vessel_available'],
                'enhance_brightness': 1.2,
                'enhance_contrast': 1.5,
                'enhance_gamma': 1.0,
                'clahe_clip': 3.0,
                'green_boost': 1.3,
                'denoise_strength': 5,
                'invert_image': False,
                'equalize_hist': True,
            }
        }
    
    def set_image(self, img):
        self.current_state['uploaded_img'] = img
        self.current_state['original_img'] = img.copy()
        self.current_state['display_img'] = img.copy()
    
    def analyze_image(self):
        if self.current_state['uploaded_img'] is None:
            return "No image loaded"
        
        # Run all analysis steps
        self.classify_severity()
        self.detect_lesions()
        self.detect_macula_disc()
        self.generate_heatmap()
        
        return self.generate_analysis_report()
    
    def classify_severity(self):
        img = self.current_state['uploaded_img']
        model = self.models['severity']
        
        if not model:
            self.current_state['current_severity'] = "No_DR"
            self.current_state['current_confidence'] = 0.0
            return
        
        try:
            results = model(img, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs.data.cpu().numpy()
                    class_idx = np.argmax(probs)
                    confidence = float(probs[class_idx])
                    
                    if class_idx < len(SEVERITY_CLASSES):
                        self.current_state['current_severity'] = SEVERITY_CLASSES[class_idx]
                        self.current_state['current_confidence'] = confidence
                
                elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    max_conf_idx = np.argmax(boxes.conf.cpu().numpy())
                    class_idx = int(boxes.cls[max_conf_idx].item())
                    confidence = float(boxes.conf[max_conf_idx].item())
                    
                    if class_idx < len(SEVERITY_CLASSES):
                        self.current_state['current_severity'] = SEVERITY_CLASSES[class_idx]
                        self.current_state['current_confidence'] = confidence
            
        except Exception as e:
            print(f"Error classifying severity: {e}")
            self.current_state['current_severity'] = "No_DR"
            self.current_state['current_confidence'] = 0.0
    
    def detect_lesions(self):
        img = self.current_state['uploaded_img']
        model = self.models['lesion']
        boxes = []
        
        if not model:
            self.current_state['current_lesions'] = boxes
            return
        
        try:
            results = model(img, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
                        x1, y1, x2, y2 = xyxy
                        
                        cls_idx = int(box.cls.item())
                        cls_name = model.names.get(cls_idx, f"Lesion_{cls_idx}")
                        confidence = float(box.conf.item())
                        
                        boxes.append({
                            "box": [x1, y1, x2, y2], 
                            "class": cls_name,
                            "confidence": confidence
                        })
            
            self.current_state['current_lesions'] = boxes
        
        except Exception as e:
            print(f"Error detecting lesions: {e}")
            self.current_state['current_lesions'] = []
    
    def detect_macula_disc(self):
        img = self.current_state['uploaded_img']
        model = self.models['macula']
        
        self.current_state['macula_disc_boxes'] = []
        self.current_state['optic_disc_diameter_pixels'] = 0
        self.current_state['disc_center'] = None
        self.current_state['macula_center'] = None
        
        if not model:
            return
        
        try:
            results = model(img, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    macula_box = None
                    max_macula_conf = 0
                    disc_box = None
                    max_disc_conf = 0
                    
                    for box in result.boxes:
                        xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
                        x1, y1, x2, y2 = xyxy
                        
                        cls_idx = int(box.cls.item())
                        cls_name = model.names.get(cls_idx, f"Class_{cls_idx}")
                        
                        if cls_name == "blood":
                            continue
                        
                        confidence = float(box.conf.item())
                        
                        if cls_name == "macula":
                            if confidence > max_macula_conf:
                                max_macula_conf = confidence
                                macula_box = {
                                    "box": [x1, y1, x2, y2], 
                                    "class": cls_name,
                                    "confidence": confidence
                                }
                                self.current_state['macula_center'] = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        elif cls_name == "disc":
                            if confidence > max_disc_conf:
                                max_disc_conf = confidence
                                disc_box = {
                                    "box": [x1, y1, x2, y2], 
                                    "class": cls_name,
                                    "confidence": confidence
                                }
                            width = x2 - x1
                            height = y2 - y1
                            self.current_state['optic_disc_diameter_pixels'] = int(max(width, height) * 1.5)
                            self.current_state['disc_center'] = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    if macula_box:
                        self.current_state['macula_disc_boxes'].append(macula_box)
                    if disc_box:
                        self.current_state['macula_disc_boxes'].append(disc_box)
        
        except Exception as e:
            print(f"Error detecting macula/disc: {e}")
    
    def generate_heatmap(self):
        img = self.current_state['uploaded_img']
        h, w = img.shape[:2]
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        
        for lesion in self.current_state['current_lesions']:
            x1, y1, x2, y2 = lesion["box"]
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max((x2 - x1) // 2, (y2 - y1) // 2)
            
            cv2.circle(heatmap, (center_x, center_y), radius * 2, (0, 0, 255), -1)
        
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        self.current_state['heatmap_overlay'] = heatmap
    
    def generate_analysis_report(self):
        report_text = f"=== RETINA ANALYSIS REPORT ===\n\n"
        report_text += f"SEVERITY: {self.current_state['current_severity']}\n"
        report_text += f"Confidence: {self.current_state['current_confidence']:.1%}\n\n"
        
        if self.current_state['current_lesions']:
            report_text += "LESIONS DETECTED:\n"
            lesion_counts = {}
            for lesion in self.current_state['current_lesions']:
                lesion_type = lesion['class']
                lesion_counts[lesion_type] = lesion_counts.get(lesion_type, 0) + 1
            
            for lesion_type, count in lesion_counts.items():
                report_text += f"  {lesion_type}: {count}\n"
            
            report_text += f"\nTotal lesions: {len(self.current_state['current_lesions'])}\n"
            
            if (self.current_state['macula_center'] is not None and 
                self.current_state['optic_disc_diameter_pixels'] > 0):
                report_text += "\nLESION DISTANCES FROM MACULA:\n"
                radius_1dd = self.current_state['optic_disc_diameter_pixels'] / 2
                
                lesions_in_1dd = 0
                for lesion in self.current_state['current_lesions']:
                    x1, y1, x2, y2 = lesion["box"]
                    lesion_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    distance_pixels = calculate_distance(lesion_center, self.current_state['macula_center'])
                    distance_DD = distance_pixels / self.current_state['optic_disc_diameter_pixels']
                    
                    if distance_pixels <= radius_1dd:
                        lesions_in_1dd += 1
                        report_text += f"  {lesion['class']}: {distance_DD:.2f} DD (INSIDE 1DD CIRCLE)\n"
                
                report_text += f"\nLesions within 1 DD of macula: {lesions_in_1dd}\n"
        else:
            report_text += "No lesions detected.\n\n"
        
        if self.current_state['macula_disc_boxes']:
            report_text += "\nMACULA AND OPTIC DISC:\n"
            macula_count = 0
            disc_count = 0
            for obj in self.current_state['macula_disc_boxes']:
                if obj["class"] == "macula":
                    macula_count += 1
                elif obj["class"] == "disc":
                    disc_count += 1
            
            if macula_count > 0:
                report_text += f"  Macula detected: {macula_count}\n"
            if disc_count > 0:
                report_text += f"  Optic disc detected: {disc_count}\n"
                if self.current_state['optic_disc_diameter_pixels'] > 0:
                    report_text += f"  Optic disc diameter (DD): {self.current_state['optic_disc_diameter_pixels']} pixels\n"
                    report_text += f"  1 DD circle drawn around macula for reference\n"
        
        if self.current_state['vessel_density'] > 0:
            report_text += f"\nVESSEL ANALYSIS:\n"
            report_text += f"  Vessel density: {self.current_state['vessel_density']:.2f}%\n"
            method = "UNet (Trained Model)" if self.current_state['vessel_settings']['use_unet'] else "Traditional"
            report_text += f"  Segmentation method: {method}\n"
            report_text += f"  Detection threshold: {self.current_state['vessel_settings']['threshold']:.2f}\n"
            
            if self.current_state['vessel_density'] < 5:
                vessel_status = "Low density"
            elif self.current_state['vessel_density'] < 15:
                vessel_status = "Normal range"
            else:
                vessel_status = "High density - monitor closely"
            
            report_text += f"  Status: {vessel_status}\n"
        
        report_text += "\n=== CLINICAL SUMMARY ===\n"
        report_text += CLINICAL_NOTES.get(self.current_state['current_severity'], "Consult ophthalmologist.")
        
        return report_text
    
    def get_state(self):
        return self.current_state
    
    def update_state(self, key, value):
        if key in self.current_state:
            self.current_state[key] = value
    
    def get_lesions(self):
        return self.current_state['current_lesions']
    
    def get_severity(self):
        return self.current_state['current_severity'], self.current_state['current_confidence']
    
    def get_macula_info(self):
        return {
            'center': self.current_state['macula_center'],
            'disc_diameter': self.current_state['optic_disc_diameter_pixels'],
            'disc_center': self.current_state['disc_center']
        }
    
    def update_vessel_settings(self, settings):
        self.current_state['vessel_settings'].update(settings)