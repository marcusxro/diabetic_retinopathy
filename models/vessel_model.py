import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

class VesselSegmentationModel:
    def __init__(self, model_path):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            self.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
            return True
        except Exception as e:
            print(f"Error loading vessel model: {e}")
            return False
    
    def predict(self, img, threshold=0.3):
        if self.model is None or self.transform is None:
            return None, 0.0
        
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=img_rgb)
            img_tensor = augmented["image"].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = torch.sigmoid(self.model(img_tensor))[0][0].cpu().numpy()
            
            pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            binary_mask = (pred > threshold).astype(np.uint8) * 255
            
            vessel_area = np.sum(binary_mask > 0)
            total_area = binary_mask.size
            vessel_density = (vessel_area / total_area) * 100
            
            return binary_mask, vessel_density
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, 0.0
    
    def post_process_mask(self, binary_mask):
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        
        return mask
    
    def create_vessel_overlay(self, img, binary_mask, color=(0, 0, 255), opacity=0.35):
        overlay = np.zeros_like(img)
        overlay[binary_mask > 0] = color
        result = cv2.addWeighted(img, 1.0 - opacity, overlay, opacity, 0)
        return result