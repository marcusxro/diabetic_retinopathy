import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import DEFAULT_VESSEL_SETTINGS

class VesselProcessor:
    def __init__(self, vessel_model=None):
        self.vessel_model = vessel_model
        self.settings = DEFAULT_VESSEL_SETTINGS.copy()
        
        self.unet_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def enhance_for_unet(self, img):
        try:
            enhanced = img.copy()
            
            if len(enhanced.shape) == 2:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(
                clipLimit=self.settings['clahe_clip'], 
                tileGridSize=(8, 8)
            )
            l_enhanced = clahe.apply(l)
            
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            b, g, r = cv2.split(enhanced)
            g_boosted = cv2.convertScaleAbs(g, alpha=self.settings['green_boost'], beta=0)
            enhanced = cv2.merge([b, g_boosted, r])
            
            enhanced = cv2.convertScaleAbs(
                enhanced, 
                alpha=self.settings['enhance_contrast'], 
                beta=255 * (self.settings['enhance_brightness'] - 1)
            )
            
            inv_gamma = 1.0 / self.settings['enhance_gamma']
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
            
            if self.settings['denoise_strength'] > 0:
                h = self.settings['denoise_strength']
                enhanced = cv2.fastNlMeansDenoisingColored(
                    enhanced, 
                    None, 
                    h, h, 7, 21
                )
            
            if self.settings['equalize_hist']:
                ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)
                channels = cv2.split(ycrcb)
                channels[0] = cv2.equalizeHist(channels[0])
                ycrcb = cv2.merge(channels)
                enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            
            if self.settings['invert_image']:
                enhanced = cv2.bitwise_not(enhanced)
            
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
            
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return img
    
    def segment_with_unet(self, img):
        try:
            if self.vessel_model is None:
                return None, 0.0
            
            enhanced_img = self.enhance_for_unet(img)
            
            img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            augmented = self.unet_transform(image=img_rgb)
            img_tensor = augmented["image"].unsqueeze(0)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            img_tensor = img_tensor.to(device)
            
            with torch.no_grad():
                pred = torch.sigmoid(self.vessel_model(img_tensor))[0][0].cpu().numpy()
            
            pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            binary_mask = (pred > self.settings['threshold']).astype(np.uint8) * 255
            
            if self.settings['post_process']:
                binary_mask = self.post_process_mask(binary_mask)
            
            color = (self.settings['color_b'], self.settings['color_g'], self.settings['color_r'])
            overlay = np.zeros_like(img)
            overlay[binary_mask > 0] = color
            
            vessel_area = np.sum(binary_mask > 0)
            total_area = binary_mask.size
            vessel_density = (vessel_area / total_area) * 100
            
            return overlay, vessel_density
            
        except Exception as e:
            print(f"Error in UNet segmentation: {e}")
            return None, 0.0
    
    def segment_traditional(self, img):
        try:
            enhanced_img = self.enhance_for_unet(img)
            
            green = enhanced_img[:, :, 1]
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(green)
            
            enhanced = cv2.medianBlur(enhanced, 5)
            
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            color = (self.settings['color_b'], self.settings['color_g'], self.settings['color_r'])
            overlay = np.zeros_like(img)
            overlay[binary > 0] = color
            
            vessel_area = np.sum(binary > 0)
            total_area = binary.size
            vessel_density = (vessel_area / total_area) * 100
            
            return overlay, vessel_density
            
        except Exception as e:
            print(f"Error in traditional segmentation: {e}")
            return None, 0.0
    
    def post_process_mask(self, binary_mask):
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        
        return mask
    
    def segment_vessels(self, img):
        if self.settings['use_unet'] and self.vessel_model is not None:
            return self.segment_with_unet(img)
        else:
            return self.segment_traditional(img)
    
    def create_vessel_only_image(self, vessel_overlay, vessel_density):
        if vessel_overlay is None:
            return None
        
        vessel_color = (self.settings['color_b'], self.settings['color_g'], self.settings['color_r'])
        vessel_only = np.zeros_like(vessel_overlay)
        mask_indices = np.any(vessel_overlay > 0, axis=2)
        colored_vessels = vessel_overlay.copy()
        colored_vessels[mask_indices] = vessel_color
        vessel_only = colored_vessels
        model_type = "UNet" if self.settings['use_unet'] else "Traditional"
        cv2.putText(vessel_only, f"BLOOD VESSELS ({model_type})", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, vessel_color, 3)
        cv2.putText(vessel_only, f"Density: {vessel_density:.2f}%", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, vessel_color, 2)
        
        return vessel_only
    
    def create_overlay_image(self, original_img, vessel_overlay):
        if vessel_overlay is None:
            return original_img.copy()
        
        overlay = np.zeros_like(original_img)
        mask_indices = np.any(vessel_overlay > 0, axis=2)
        vessel_color = (self.settings['color_b'], self.settings['color_g'], self.settings['color_r'])
        overlay[mask_indices] = vessel_color
        
        opacity = self.settings['overlay_opacity']
        result = cv2.addWeighted(original_img, 1.0 - opacity, overlay, opacity, 0)
        
        return result
    
    def update_setting(self, setting, value):
        if setting.startswith('color_'):
            self.settings[setting] = max(0, min(255, int(value)))
        elif setting.startswith('enhance_') or setting in ['clahe_clip', 'green_boost', 'denoise_strength']:
            self.settings[setting] = float(value)
        elif setting == 'threshold' or setting == 'overlay_opacity':
            self.settings[setting] = float(value)
        else:
            self.settings[setting] = value
    
    def get_settings(self):
        return self.settings.copy()
    
    def reset_settings(self, vessel_model_available=True):
        self.settings = DEFAULT_VESSEL_SETTINGS.copy()
        self.settings['use_unet'] = vessel_model_available