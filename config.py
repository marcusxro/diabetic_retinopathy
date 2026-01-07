import os

OPENROUTER_API_KEY = "ur api key here"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_AVAILABLE = True
SYSTEM_PROMPT = "You are RetinaExpert, an ophthalmology AI assistant specializing in diabetic retinopathy and retinal analysis."

MODELS_DIR = "models"
SEVERITY_MODEL_PATH = os.path.join(MODELS_DIR, "severity.pt")
LESION_MODEL_PATH = os.path.join(MODELS_DIR, "lesions.pt")
MACULA_MODEL_PATH = os.path.join(MODELS_DIR, "macula.pt")
VESSEL_MODEL_PATH = os.path.join(MODELS_DIR, "vessel_unet.pth")

SEVERITY_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative"]
SEVERITY_COLORS = {
    "No_DR": (0, 255, 0),
    "Mild": (0, 255, 255),
    "Moderate": (255, 255, 0),
    "Severe": (0, 165, 255),
    "Proliferative": (0, 0, 255)
}

CLINICAL_NOTES = {
    "No_DR": "No diabetic retinopathy detected. Annual screening recommended.",
    "Mild": "Mild NPDR. Microaneurysms present. Follow-up in 6-12 months.",
    "Moderate": "Moderate NPDR. Close monitoring needed. Follow-up in 3-6 months.",
    "Severe": "Severe NPDR. High risk progression. Specialist referral within 1 month.",
    "Proliferative": "PDR with neovascularization. Urgent specialist referral required."
}

DEFAULT_VESSEL_SETTINGS = {
    'threshold': 0.3,
    'color_r': 255,
    'color_g': 0,
    'color_b': 0,
    'overlay_opacity': 0.35,
    'post_process': True,
    'use_unet': True,
    'enhance_brightness': 1.2,
    'enhance_contrast': 1.5,
    'enhance_gamma': 1.0,
    'clahe_clip': 3.0,
    'green_boost': 1.3,
    'denoise_strength': 5,
    'invert_image': False,
    'equalize_hist': True,
}