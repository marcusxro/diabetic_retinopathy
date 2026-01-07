import os
import torch
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from config import SEVERITY_MODEL_PATH, LESION_MODEL_PATH, MACULA_MODEL_PATH, VESSEL_MODEL_PATH

class ModelLoader:
    def __init__(self):
        self.severity_model = None
        self.lesion_model = None
        self.macula_model = None
        self.vessel_model = None
        self.vessel_model_available = False
        
        self.load_models()
    
    def load_models(self):
        self.severity_model = self._load_yolo_model(SEVERITY_MODEL_PATH)
        self.lesion_model = self._load_yolo_model(LESION_MODEL_PATH)
        self.macula_model = self._load_yolo_model(MACULA_MODEL_PATH)
        self.vessel_model_available = self._load_vessel_model()
    
    def _load_yolo_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        try:
            model = YOLO(model_path)
            print(f"Successfully loaded model: {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None
    
    def _load_vessel_model(self):
        try:
            if os.path.exists(VESSEL_MODEL_PATH):
                print(f"Loading vessel model from: {VESSEL_MODEL_PATH}")
                vessel_unet = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights=None,
                    in_channels=3,
                    classes=1,
                    activation=None
                )
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                vessel_unet.load_state_dict(torch.load(VESSEL_MODEL_PATH, map_location=device))
                vessel_unet = vessel_unet.to(device)
                vessel_unet.eval()
                self.vessel_model = vessel_unet
                print("Vessel UNet model loaded successfully")
                return True
            else:
                print(f"Vessel model file not found: {VESSEL_MODEL_PATH}")
                return False
        except Exception as e:
            print(f"Error loading vessel model: {e}")
            return False
    
    def get_severity_model(self):
        return self.severity_model
    
    def get_lesion_model(self):
        return self.lesion_model
    
    def get_macula_model(self):
        return self.macula_model
    
    def get_vessel_model(self):
        return self.vessel_model
    
    def is_vessel_model_available(self):
        return self.vessel_model_available
    
    def get_all_models(self):
        return {
            'severity': self.severity_model,
            'lesion': self.lesion_model,
            'macula': self.macula_model,
            'vessel': self.vessel_model,
            'vessel_available': self.vessel_model_available
        }