import cv2
import numpy as np
import math
from utils.helpers import (
    get_contour_measurements, 
    draw_measurement_lines, 
    create_lesion_thumbnail,
    calculate_distance
)

class LesionAnalyzer:
    def __init__(self):
        self.lesion_images = []
        self.lesion_measurements = []
    
    def analyze_lesions(self, uploaded_img, lesions, macula_center=None, optic_disc_diameter_pixels=0):
        self.lesion_images = []
        self.lesion_measurements = []
        
        if not lesions:
            return [], []
        
        if optic_disc_diameter_pixels > 0:
            pixels_per_micrometer = optic_disc_diameter_pixels / 1500.0  
        else:
            h, w = uploaded_img.shape[:2]
            pixels_per_micrometer = w / 15000.0  
            if pixels_per_micrometer <= 0:
                pixels_per_micrometer = 0.1
        
        for idx, lesion in enumerate(lesions):
            x1, y1, x2, y2 = lesion["box"]
            
            padding = 20
            h, w = uploaded_img.shape[:2]
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            lesion_roi = uploaded_img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            
            if lesion_roi.size == 0:
                continue
            
            lesion_type = lesion['class'].lower()
            vis_roi, contours = create_lesion_thumbnail(lesion_roi, lesion_type)
            
            measurement_text = ""
            if contours:
                try:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area_micrometers, width_micrometers, height_micrometers = get_contour_measurements(
                        largest_contour, pixels_per_micrometer
                    )
                    
                    rect = cv2.minAreaRect(largest_contour)
                    draw_measurement_lines(vis_roi, rect)
                    
                    mid_y = int(rect[0][1])
                    left_x = int(rect[0][0] - rect[1][0]/2)
                    cv2.putText(vis_roi, f"{width_micrometers:.1f}µm", 
                            (left_x + 5, mid_y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    mid_x = int(rect[0][0])
                    top_y = int(rect[0][1] - rect[1][1]/2)
                    bottom_y = int(rect[0][1] + rect[1][1]/2)
                    cv2.putText(vis_roi, f"{height_micrometers:.1f}µm", 
                            (mid_x + 15, (top_y + bottom_y)//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    measurement_text = f"Area: {area_micrometers:.1f}µm² | Size: {width_micrometers:.1f}×{height_micrometers:.1f}µm"
                except Exception as e:
                    measurement_text = f"Measurement error: {str(e)}"
            else:
                measurement_text = "No contour found for measurement"
            
            self.lesion_images.append(vis_roi)
            self.lesion_measurements.append(measurement_text)
        
        return self.lesion_images, self.lesion_measurements
    
    def get_lesion_summary(self, lesions, optic_disc_diameter_pixels, macula_center=None):
        if not lesions:
            return "No lesions detected"
        
        lesion_types = {}
        lesion_areas = []
        
        for lesion in lesions:
            lesion_type = lesion['class']
            lesion_types[lesion_type] = lesion_types.get(lesion_type, 0) + 1
        
        summary_text = "SUMMARY STATISTICS:\n"
        summary_text += f"Total Lesions: {len(lesions)}\n"
        
        for lesion_type, count in lesion_types.items():
            summary_text += f"{lesion_type}: {count} lesions\n"
        
        if macula_center is not None and optic_disc_diameter_pixels > 0:
            lesions_in_1dd = 0
            radius_1dd = optic_disc_diameter_pixels / 2
            
            for lesion in lesions:
                x1, y1, x2, y2 = lesion["box"]
                lesion_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                distance_pixels = calculate_distance(lesion_center, macula_center)
                
                if distance_pixels <= radius_1dd:
                    lesions_in_1dd += 1
            
            summary_text += f"\nLesions within 1 DD of macula: {lesions_in_1dd}\n"
        
        summary_text += f"\nConversion: 1 DD = {optic_disc_diameter_pixels}px ≈ 1500µm"
        
        return summary_text
    
    def calculate_lesion_distances(self, lesions, macula_center, optic_disc_diameter_pixels):
        distances = []
        
        if not macula_center or optic_disc_diameter_pixels <= 0:
            return distances
        
        for lesion in lesions:
            x1, y1, x2, y2 = lesion["box"]
            lesion_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance_pixels = calculate_distance(lesion_center, macula_center)
            distance_DD = distance_pixels / optic_disc_diameter_pixels
            
            distances.append({
                'lesion': lesion,
                'center': lesion_center,
                'distance_pixels': distance_pixels,
                'distance_DD': distance_DD,
                'within_1dd': distance_pixels <= (optic_disc_diameter_pixels / 2)
            })
        
        return distances
    
    def resize_for_gallery(self, img, max_size=250):
        h, w = img.shape[:2]
        scale = min(max_size / w, max_size / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        
        if new_w > 0 and new_h > 0:
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img