import tkinter as tk
from tkinter import scrolledtext, ttk
from utils.constants import UI_COLORS

class ChatDisplay(scrolledtext.ScrolledText):
    def __init__(self, master=None, **kwargs):
        defaults = {
            'wrap': tk.WORD,
            'bg': UI_COLORS['chat_bg'],
            'fg': UI_COLORS['text_dark'],
            'font': ('Arial', 10),
            'state': tk.DISABLED,
            'relief': tk.FLAT,
            'borderwidth': 5
        }
        defaults.update(kwargs)
        super().__init__(master, **defaults)
        self.configure_tags()
    
    def configure_tags(self):
        self.tag_config("timestamp", foreground="#7f8c8d", font=('Arial', 8))
        self.tag_config("user_message", 
                       background=UI_COLORS['chat_user_bg'],
                       foreground="white",
                       relief=tk.RAISED,
                       borderwidth=2,
                       rmargin=50,
                       lmargin1=50,
                       lmargin2=50,
                       justify=tk.RIGHT)
        self.tag_config("ai_message", 
                       background=UI_COLORS['chat_ai_bg'],
                       foreground=UI_COLORS['text_dark'],
                       relief=tk.RAISED,
                       borderwidth=2,
                       rmargin=50,
                       lmargin1=10,
                       lmargin2=10,
                       justify=tk.LEFT)
    
    def add_user_message(self, message):
        self.config(state=tk.NORMAL)
        self.insert(tk.END, f"\nUSER: {message}\n\n", "user_message")
        self.config(state=tk.DISABLED)
        self.see(tk.END)
    
    def add_ai_message(self, message):
        self.config(state=tk.NORMAL)
        self.insert(tk.END, f"\nAI: {message}\n\n", "ai_message")
        self.config(state=tk.DISABLED)
        self.see(tk.END)
    
    def clear(self):
        self.config(state=tk.NORMAL)
        self.delete(1.0, tk.END)
        self.config(state=tk.DISABLED)

class AnalysisDisplay(scrolledtext.ScrolledText):
    def __init__(self, master=None, **kwargs):
        defaults = {
            'wrap': tk.WORD,
            'bg': UI_COLORS['bg_light'],
            'fg': 'white',
            'font': ('Courier', 10),
            'relief': tk.FLAT,
            'borderwidth': 5
        }
        defaults.update(kwargs)
        super().__init__(master, **defaults)
    
    def set_report(self, report_text):
        self.config(state=tk.NORMAL)
        self.delete(1.0, tk.END)
        self.insert(1.0, report_text)
        self.config(state=tk.DISABLED)
    
    def clear(self):
        self.set_report("Analysis Report will appear here...\n")

class ControlButton(tk.Button):
    def __init__(self, master, text, command, color=UI_COLORS['accent_blue'], **kwargs):
        defaults = {
            'bg': color,
            'fg': 'white',
            'font': ('Arial', 11),
            'padx': 15,
            'pady': 10,
            'relief': tk.RAISED,
            'borderwidth': 2,
            'cursor': 'hand2'
        }
        defaults.update(kwargs)
        super().__init__(master, text=text, command=command, **defaults)

class ImageCanvas(tk.Canvas):
    def __init__(self, master=None, **kwargs):
        defaults = {
            'bg': UI_COLORS['bg_light'],
            'highlightthickness': 0
        }
        defaults.update(kwargs)
        super().__init__(master, **defaults)
        
    def display_image(self, img_tk):
        self.delete("all")
        self.create_image(self.winfo_width() // 2, self.winfo_height() // 2, 
                         anchor=tk.CENTER, image=img_tk)
        self.image = img_tk  

class StatusLabel(tk.Label):
    def __init__(self, master=None, **kwargs):
        defaults = {
            'bg': UI_COLORS['bg_medium'],
            'fg': 'white',
            'font': ('Arial', 10),
            'anchor': 'w'
        }
        defaults.update(kwargs)
        super().__init__(master, **defaults)
    
    def set_status(self, message):
        self.config(text=f"Status: {message}")

class SettingsSlider(ttk.Frame):
    def __init__(self, master, label_text, from_, to, value, resolution=0.1, command=None, **kwargs):
        super().__init__(master, **kwargs)
        
        self.label = tk.Label(self, text=label_text, bg=UI_COLORS['bg_medium'], fg='white', font=('Arial', 10))
        self.label.pack(pady=(5, 0))
        
        self.slider = tk.Scale(
            self, 
            from_=from_, 
            to=to, 
            resolution=resolution,
            orient=tk.HORIZONTAL, 
            length=450,
            bg=UI_COLORS['bg_medium'],
            fg='white',
            highlightthickness=0,
            troughcolor=UI_COLORS['bg_light']
        )
        self.slider.set(value)
        if command:
            self.slider.config(command=command)
        self.slider.pack(pady=5)
    
    def get_value(self):
        return self.slider.get()
    
    def set_value(self, value):
        self.slider.set(value)

class ColorPreview(tk.Frame):
    def __init__(self, master, r, g, b, **kwargs):
        super().__init__(master, **kwargs)
        self.config(bg='#000000', height=30, width=100)
        self.set_color(r, g, b)
    
    def set_color(self, r, g, b):
        color_hex = f'#{r:02x}{g:02x}{b:02x}'
        self.config(bg=color_hex)