import warnings
warnings.filterwarnings('ignore')

UI_COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_medium': '#16213e',
    'bg_light': '#0f3460',
    'accent_blue': '#3498db',
    'accent_green': '#2ecc71',
    'accent_red': '#e74c3c',
    'accent_orange': '#f39c12',
    'accent_purple': '#8e44ad',
    'accent_teal': '#16a085',
    'text_light': 'white',
    'text_dark': '#2c3e50',
    'chat_bg': '#ecf0f1',
    'chat_user_bg': '#3498db',
    'chat_ai_bg': '#dfe6e9'
}

DEFAULT_CANVAS_SIZE = (800, 600)
MAX_ZOOM_SCALE = 5.0
MIN_ZOOM_SCALE = 0.1
ZOOM_STEP = 1.2

DD_TO_MICROMETERS = 1500.0 
DEFAULT_PIXELS_PER_MICROMETER = 0.1

GALLERY_COLS = 4
MAX_GALLERY_IMAGE_SIZE = 250

VESSEL_DENSITY_LOW = 5.0
VESSEL_DENSITY_NORMAL = 15.0