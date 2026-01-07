import tkinter as tk
import warnings
warnings.filterwarnings('ignore')

from models.model_loader import ModelLoader
from processing.image_processor import ImageProcessor
from processing.vessel_processor import VesselProcessor
from processing.lesion_analyzer import LesionAnalyzer
from api.openrouter_api import OpenRouterAPI
from ui.app_ui import RetinaAnalyzerUI

def main():
    print("=" * 60)
    print("Initializing Retina AI Analyzer")
    print("=" * 60)
    
    print("\n1. Loading models...")
    model_loader = ModelLoader()
    
    print("\n2. Initializing processors...")
    image_processor = ImageProcessor(model_loader)
    
    vessel_model = model_loader.get_vessel_model()
    vessel_processor = VesselProcessor(vessel_model)
    
    lesion_analyzer = LesionAnalyzer()
    
    print("\n3. Initializing API client...")
    api_client = OpenRouterAPI()
    
    print("\n4. Creating application window...")
    
    root = tk.Tk()
    app = RetinaAnalyzerUI(root, image_processor, vessel_processor, api_client, lesion_analyzer)
    
    print("\n" + "=" * 60)
    print("INITIALIZATION COMPLETE")
    print("=" * 60)
    models = model_loader.get_all_models()
    print(f"✓ Severity Model: {'Loaded' if models['severity'] else 'Not available'}")
    print(f"✓ Lesion Model: {'Loaded' if models['lesion'] else 'Not available'}")
    print(f"✓ Macula Model: {'Loaded' if models['macula'] else 'Not available'}")
    print(f"✓ Vessel UNet: {'Loaded' if models['vessel_available'] else 'Not available'}")
    print(f"✓ OpenRouter API: {'Available' if api_client.is_available() else 'Not configured'}")
    print("=" * 60)
    print("\nApplication ready. Use 'Upload Retina Scan' to begin.")
    
    root.mainloop()

if __name__ == "__main__":
    main()