import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import json
import tempfile
import shutil
from pathlib import Path
import traceback
from omegaconf import OmegaConf, DictConfig
import csv
import time
import threading
import re

# Add SAM2 path to Python path - dynamically detect project root
def get_project_root():
    """Dynamically detect the project root directory (where sam2_ui.py lives)"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)

    # The directory containing sam2_ui.py is the Sam2UI root
    if os.path.basename(current_file) == 'sam2_ui.py':
        return current_dir

    # Fallback: search upwards for indicators
    indicators = ['sam2', 'setup.py']
    search_dir = current_dir
    while search_dir != os.path.dirname(search_dir):  # Not at filesystem root
        if all(os.path.exists(os.path.join(search_dir, indicator)) for indicator in indicators):
            return search_dir
        search_dir = os.path.dirname(search_dir)

    # Final fallback to current directory
    return current_dir

SAM2_PATH = get_project_root()

# Import torch for device detection
try:
    import torch
except ImportError:
    torch = None

# Check SAM3 availability
def _check_sam3_available():
    """Check if SAM3 is installed and usable"""
    try:
        sam3_path = os.path.join(SAM2_PATH, "sam_models", "sam3")
        if not os.path.exists(sam3_path):
            return False
        # Try importing SAM3
        from sam3.model_builder import build_sam3_video_predictor
        return True
    except ImportError:
        return False

SAM3_AVAILABLE = _check_sam3_available()

class SAM2VideoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM Video Segmentation Tool")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2b2b2b')
        
        # Dynamic paths - automatically detect project root
        self.sam2_base_path = SAM2_PATH  # Sam2UI root
        self.sam2_repo_path = os.path.join(SAM2_PATH, "sam_models", "sam2")  # SAM2 repository location
        self.checkpoint_dir = os.path.join(self.sam2_repo_path, "checkpoints")
        self.config_dir = os.path.join(self.sam2_repo_path, "sam2", "configs")  # Configs are in sam2/sam2/configs/
        
        # Variables
        self.video_path = None
        self.video_cap = None
        self.frames = []
        self.current_frame_idx = 0
        self.current_frame = None
        self.display_frame = None
        self.scale_factor = 1.0
        self.click_points = []  # Store click coordinates with object IDs
        self.masks = {}  # Store masks for each frame {frame_idx: {obj_id: mask}}
        self.playing = False
        self.inference_state = None
        self.current_object_id = 1  # Currently selected object ID
        self.max_object_id = 1  # Track highest object ID used
        self.max_total_objects = 100  # Maximum number of objects supported
        
        # Enhanced object management
        self.object_names = {}  # Maps obj_id to custom name
        self.object_colors = {}  # Dynamic color assignment
        self.point_removal_mode = False
        
        # Multi-frame annotation mode (always enabled)
        self.multi_frame_annotation_mode = True
        self.annotated_frames = set()  # Track which frames have been annotated

        # Mask flash animation state
        self.flash_in_progress = False
        self.flash_obj_id = None
        self.flash_white_on = False

        # Background task tracking
        self.active_exports = []
        self.active_segmentation = []

        # GPU selection
        self.available_gpus = self._detect_available_gpus()
        self.selected_gpu = tk.StringVar(value="auto")  # Default to auto selection
        self.gpu_device = None  # Will be set when model loads

        # Model selection
        self.available_models = self._detect_available_models()
        self.selected_model = tk.StringVar(value="auto")  # Auto-select best model
        self.current_model_info = None  # Store loaded model info

        # Model type (SAM2/SAM3)
        self.sam3_available = SAM3_AVAILABLE
        self.model_type_var = tk.StringVar(value="SAM2")  # Default to SAM2
        self.using_sam3 = False
        
        # Range-based processing for long videos
        self.limit_to_range_var = tk.BooleanVar(value=False)
        self.range_start_var = tk.IntVar(value=0)
        self.range_end_var = tk.IntVar(value=0)
        
        # Large video handling options
        self.downsample_frames_var = tk.BooleanVar(value=False)
        self.frame_skip_var = tk.IntVar(value=1)  # Skip every N frames
        self.scale_video_var = tk.BooleanVar(value=False)
        self.video_scale_factor = tk.DoubleVar(value=0.5)  # Scale factor for video resolution
        self.lazy_load_var = tk.BooleanVar(value=False)  # Load frames on demand
        self.video_cap_lazy = None  # Keep video capture open for lazy loading
        
        # Track original video dimensions for coordinate system consistency
        self.original_video_width = None
        self.original_video_height = None
        self.current_video_scale = 1.0  # Current scale applied to loaded frames

        # State for loading segmentation results
        self.loaded_from_results = False  # Flag if loaded from results
        self.original_video_path_for_resegment = None  # Path for re-segmentation
        self.segmented_video_displayed = False  # Track if displaying segmented vs original
        self.results_output_dir = None  # Store output directory

        # Initialize default colors and names
        self._initialize_objects()
        
        # SAM2 model
        self.sam2_model = None
        self.model_loaded = False

        
        # UI styling
        self.setup_styles()
        self.setup_ui()
        
    def _initialize_objects(self):
        """Initialize object colors and default names for up to max_total_objects"""
        # Generate distinct colors using HSV space
        for i in range(1, self.max_total_objects + 1):
            # Use HSV for better color distribution
            hue = (i * 137.5) % 360  # Golden angle approximation for good distribution
            saturation = 0.8 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.9 - (i % 2) * 0.2  # Vary brightness slightly
            
            # Convert HSV to RGB
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue/360, saturation, value)
            self.object_colors[i] = [int(r*255), int(g*255), int(b*255)]
            self.object_names[i] = f"Object_{i}"
        
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', background='#404040', foreground='white')
        style.map('TButton', 
                 background=[('active', '#505050'), ('pressed', '#303030')])
        
    def setup_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create paned window - use tk.PanedWindow instead of ttk.PanedWindow
        self.paned = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, 
                                    sashwidth=5, bg='#2b2b2b')
        self.paned.pack(fill=tk.BOTH, expand=True)
        
        # Create panels
        left_panel = ttk.Frame(self.paned)
        right_panel = ttk.Frame(self.paned)
        
        # Add panels
        self.paned.add(left_panel, width=280)  # Set initial width to 280px
        self.paned.add(right_panel)
        
        # Setup panels
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
        # Force sash position after window renders
        self.root.after(100, lambda: self.paned.sash_place(0, 280, 1))
        
    def setup_left_panel(self, parent):
        canvas = tk.Canvas(parent, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure canvas to expand scrollable_frame to canvas width
        def configure_scroll_frame(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Make the scrollable_frame match the canvas width
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        canvas.bind("<Configure>", configure_scroll_frame)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar FIRST, then canvas - this ensures scrollbar stays visible
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Mouse wheel scrolling - simplified working version
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        # Bind when mouse enters/leaves the parent frame
        parent.bind("<Enter>", lambda e: parent.bind_all("<MouseWheel>", _on_mousewheel))
        parent.bind("<Leave>", lambda e: parent.unbind_all("<MouseWheel>"))
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="SAM UI", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # File operations
        file_frame = ttk.LabelFrame(scrollable_frame, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="Load Video",
                  command=self.load_video, width=15).pack(fill=tk.X, pady=2)

        # Model Type Selection (SAM2/SAM3) - only show if SAM3 is available
        if self.sam3_available:
            ttk.Label(file_frame, text="Model Type:").pack(anchor=tk.W, pady=(10, 0))
            model_type_frame = ttk.Frame(file_frame)
            model_type_frame.pack(fill=tk.X, pady=(0, 5))

            ttk.Radiobutton(
                model_type_frame,
                text="SAM2",
                variable=self.model_type_var,
                value="SAM2",
                command=self.on_model_type_change
            ).pack(side=tk.LEFT, padx=(0, 10))

            ttk.Radiobutton(
                model_type_frame,
                text="SAM3",
                variable=self.model_type_var,
                value="SAM3",
                command=self.on_model_type_change
            ).pack(side=tk.LEFT)

            # Note about SAM3 capabilities
            note_label = ttk.Label(
                file_frame,
                text="Note: Text prompts coming soon",
                foreground="gray",
                font=("TkDefaultFont", 8, "italic")
            )
            note_label.pack(anchor=tk.W, pady=(0, 5))

        # Model Selection - Shows all available models in sam2/checkpoints/
        ttk.Label(file_frame, text="Model Variant:").pack(anchor=tk.W, pady=(5, 0))
        self.model_combo = ttk.Combobox(
            file_frame,
            textvariable=self.selected_model,
            values=self._format_model_list(),
            state="readonly",
            width=30
        )
        self.model_combo.pack(fill=tk.X, pady=(0, 5))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selection_change)

        ttk.Button(file_frame, text="Load SAM2 Model",
                  command=self.load_sam2_model, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Import Object List",
                  command=self.import_object_list, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export Object List",
                  command=self.export_object_list, width=15).pack(fill=tk.X, pady=2)

        # Model status
        self.model_status_label = ttk.Label(file_frame, text="Model Not Loaded",
                                           foreground='red')
        self.model_status_label.pack(pady=5)
        
        # GPU Selection
        gpu_frame = ttk.LabelFrame(scrollable_frame, text="GPU Selection", padding=10)
        gpu_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(gpu_frame, text="Device:").pack(anchor=tk.W)
        self.gpu_combo = ttk.Combobox(gpu_frame, textvariable=self.selected_gpu, 
                                     values=self.available_gpus, state="readonly", width=30)
        self.gpu_combo.pack(fill=tk.X, pady=(0, 5))
        self.gpu_combo.bind('<<ComboboxSelected>>', self.on_gpu_selection_change)
        
        # GPU info display
        self.gpu_info_label = ttk.Label(gpu_frame, text="", foreground='gray', font=('Arial', 8))
        self.gpu_info_label.pack(anchor=tk.W)
        
        # Update GPU info display
        self._update_gpu_info_display()
        
        # Enhanced Object Management
        obj_frame = ttk.LabelFrame(scrollable_frame, text="Object Management", padding=10)
        obj_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current object selection with more space
        current_obj_frame = ttk.Frame(obj_frame)
        current_obj_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(current_obj_frame, text="Current:").pack(side=tk.LEFT)
        
        self.object_var = tk.IntVar(value=1)
        self.object_spinbox = tk.Spinbox(current_obj_frame, from_=1, to=self.max_total_objects, 
                                        textvariable=self.object_var, width=5, 
                                        command=self.on_object_change,
                                        bg='#404040', fg='white', insertbackground='white')
        self.object_spinbox.pack(side=tk.LEFT, padx=(5, 5))
        
        # Object color indicator
        self.object_color_label = ttk.Label(current_obj_frame, text="", 
                                           foreground='cyan', font=('Arial', 16))
        self.object_color_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Object name entry
        name_frame = ttk.Frame(obj_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="Name:").pack(side=tk.LEFT)
        self.object_name_var = tk.StringVar(value="Object_1")
        self.object_name_entry = tk.Entry(name_frame, textvariable=self.object_name_var,
                                         bg='#404040', fg='white', insertbackground='white')
        self.object_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.object_name_entry.bind('<Return>', self.update_object_name)
        
        ttk.Button(name_frame, text="Save", command=self.update_object_name, width=5).pack(side=tk.RIGHT)
        
        # Object control buttons
        obj_buttons_frame = ttk.Frame(obj_frame)
        obj_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(obj_buttons_frame, text="Add New", 
                  command=self.add_new_object, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(obj_buttons_frame, text="Clear Obj", 
                  command=self.clear_current_object, width=10).pack(side=tk.LEFT)
        
        # Object list with scrollbar
        list_frame = ttk.Frame(obj_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create Treeview for object list
        self.object_tree = ttk.Treeview(list_frame, columns=("name", "points", "masks"), 
                                       show="tree headings", height=8)
        self.object_tree.heading("#0", text="ID")
        self.object_tree.heading("name", text="Name")
        self.object_tree.heading("points", text="Points")
        self.object_tree.heading("masks", text="Masks")
        
        self.object_tree.column("#0", width=40)
        self.object_tree.column("name", width=100)
        self.object_tree.column("points", width=60)
        self.object_tree.column("masks", width=60)
        
        # Scrollbar for object list
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.object_tree.yview)
        self.object_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.object_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.object_tree.bind('<ButtonRelease-1>', self.on_object_tree_select)
        
        # ADDED: Mark this widget as having its own scrollbar
        # When mouse is over this widget, it should handle its own scrolling
        self.object_tree.bind('<Enter>', lambda e: setattr(self, '_mouse_over_scrollable', True))
        self.object_tree.bind('<Leave>', lambda e: setattr(self, '_mouse_over_scrollable', False))
        
        # Segmentation controls
        seg_frame = ttk.LabelFrame(scrollable_frame, text="Segmentation", padding=10)
        seg_frame.pack(fill=tk.X, pady=(0, 10))

        # Annotation import/export first
        ttk.Button(seg_frame, text="Import Annotations",
                  command=self.import_annotations, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="Export Annotations",
                  command=self.export_annotations, width=15).pack(fill=tk.X, pady=2)

        # Import masks from processing output
        ttk.Button(seg_frame, text="Import Masks",
                  command=self.import_masks, width=15).pack(fill=tk.X, pady=2)

        # Point management buttons
        point_mgmt_frame = ttk.Frame(seg_frame)
        point_mgmt_frame.pack(fill=tk.X, pady=2)

        self.remove_point_button = tk.Button(point_mgmt_frame, text="Remove Point",
                  command=self.toggle_point_removal_mode, width=12,
                  bg='#404040', fg='white', activebackground='#505050')
        self.remove_point_button.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(point_mgmt_frame, text="Clear All",
                  command=self.clear_points, width=12).pack(side=tk.LEFT)

        ttk.Button(seg_frame, text="Show Frame Points",
                  command=self.show_frame_points, width=15).pack(fill=tk.X, pady=2)

        # Segmentation execution last
        ttk.Button(seg_frame, text="Segment Video",
                  command=self.segment_video, width=15).pack(fill=tk.X, pady=2)

        # Flash mask button
        ttk.Button(seg_frame, text="Flash Mask (F)",
                  command=self.flash_selected_object_mask, width=15).pack(fill=tk.X, pady=2)
        
        # Export controls
        export_frame = ttk.LabelFrame(scrollable_frame, text="Export", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(export_frame, text="Export Video",
                  command=self.export_video, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Export Masks",
                  command=self.export_masks, width=15).pack(fill=tk.X, pady=2)

        # Display options
        display_frame = ttk.LabelFrame(scrollable_frame, text="Display Options", padding=10)
        display_frame.pack(fill=tk.X, pady=(0, 10))

        # Show/Hide masks checkbox
        self.show_masks_var = tk.BooleanVar()
        ttk.Checkbutton(display_frame, text="Show Masks", 
                    variable=self.show_masks_var,
                    command=self.toggle_mask_display).pack(anchor=tk.W, pady=(0, 5))

        # Mask opacity slider
        opacity_frame = ttk.Frame(display_frame)
        opacity_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(opacity_frame, text="Mask Opacity:").pack(anchor=tk.W)

        self.mask_opacity_var = tk.DoubleVar(value=0.4)  # Default 40%
        opacity_slider = ttk.Scale(opacity_frame, from_=0.0, to=1.0, 
                                variable=self.mask_opacity_var, 
                                orient=tk.HORIZONTAL,
                                command=self.on_mask_opacity_change)
        opacity_slider.pack(fill=tk.X, padx=(0, 5))

        # Opacity percentage label
        self.opacity_label = ttk.Label(opacity_frame, text="40%", foreground='gray')
        self.opacity_label.pack(anchor=tk.W)
        
        # Status info
        status_frame = ttk.LabelFrame(scrollable_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="Ready", wraplength=250)
        self.status_label.pack(fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        
    def setup_right_panel(self, parent):
        """Setup the right video display panel"""
        # Video display area
        display_frame = ttk.Frame(parent)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas with scrollbars
        canvas_container = ttk.Frame(display_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, bg='#1a1a1a', highlightthickness=0)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Keyboard shortcuts
        self.root.bind('f', lambda e: self.flash_selected_object_mask())

        # Video controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X)
        
        # Playback controls
        playback_frame = ttk.Frame(controls_frame)
        playback_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.play_button = ttk.Button(playback_frame, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))

        # Prev/Next buttons with hold-to-scroll support
        self.prev_button = tk.Button(playback_frame, text="Prev", bg='#404040', fg='white')
        self.prev_button.pack(side=tk.LEFT, padx=(0, 5))
        self.prev_button.bind("<ButtonPress-1>", lambda e: self._start_continuous_nav("prev"))
        self.prev_button.bind("<ButtonRelease-1>", lambda e: self._stop_continuous_nav())

        self.next_button = tk.Button(playback_frame, text="Next", bg='#404040', fg='white')
        self.next_button.pack(side=tk.LEFT, padx=(0, 5))
        self.next_button.bind("<ButtonPress-1>", lambda e: self._start_continuous_nav("next"))
        self.next_button.bind("<ButtonRelease-1>", lambda e: self._stop_continuous_nav())

        ttk.Button(playback_frame, text="Reset", command=self.reset_video).pack(side=tk.LEFT, padx=(10, 0))
        
        # Jump to annotated frames buttons
        ttk.Button(playback_frame, text="◄ Ann", command=self.jump_to_prev_annotated_frame).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Button(playback_frame, text="Ann ►", command=self.jump_to_next_annotated_frame).pack(side=tk.LEFT, padx=(0, 5))

        # Frame slider
        slider_frame = ttk.Frame(controls_frame)
        slider_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT)
        
        self.frame_var = tk.IntVar()
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100, 
                                     orient=tk.HORIZONTAL, variable=self.frame_var,
                                     command=self.on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        
        self.frame_label = ttk.Label(slider_frame, text="0/0")
        self.frame_label.pack(side=tk.RIGHT)
        
        # Range selection for partial segmentation of long videos
        range_frame = ttk.Frame(controls_frame)
        range_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Checkbutton(range_frame, text="Limit to range", variable=self.limit_to_range_var).pack(side=tk.LEFT)
        ttk.Label(range_frame, text="Start:").pack(side=tk.LEFT, padx=(10, 2))
        self.range_start_spin = tk.Spinbox(range_frame, from_=0, to=0, textvariable=self.range_start_var, width=8,
                                           bg='#404040', fg='white', insertbackground='white')
        self.range_start_spin.pack(side=tk.LEFT)
        ttk.Label(range_frame, text="End:").pack(side=tk.LEFT, padx=(10, 2))
        self.range_end_spin = tk.Spinbox(range_frame, from_=0, to=0, textvariable=self.range_end_var, width=8,
                                         bg='#404040', fg='white', insertbackground='white')
        self.range_end_spin.pack(side=tk.LEFT)
        
        # Large video optimization options
        large_video_frame = ttk.LabelFrame(controls_frame, text="Large Video Options", padding=5)
        large_video_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Frame skipping option
        skip_frame = ttk.Frame(large_video_frame)
        skip_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Checkbutton(skip_frame, text="Skip frames:", variable=self.downsample_frames_var).pack(side=tk.LEFT)
        self.frame_skip_spin = tk.Spinbox(skip_frame, from_=1, to=10, textvariable=self.frame_skip_var, width=5,
                                         bg='#404040', fg='white', insertbackground='white')
        self.frame_skip_spin.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(skip_frame, text="(every N frames)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Video scaling option
        scale_frame = ttk.Frame(large_video_frame)
        scale_frame.pack(fill=tk.X, pady=(2, 0))
        ttk.Checkbutton(scale_frame, text="Scale video:", variable=self.scale_video_var).pack(side=tk.LEFT)
        self.scale_spin = tk.Spinbox(scale_frame, from_=0.1, to=1.0, increment=0.1, textvariable=self.video_scale_factor, 
                                   width=5, bg='#404040', fg='white', insertbackground='white')
        self.scale_spin.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(scale_frame, text="(reduces memory usage)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Lazy loading option
        lazy_frame = ttk.Frame(large_video_frame)
        lazy_frame.pack(fill=tk.X, pady=(2, 0))
        ttk.Checkbutton(lazy_frame, text="Lazy load frames", variable=self.lazy_load_var).pack(side=tk.LEFT)
        ttk.Label(lazy_frame, text="(load on demand - for very large videos)").pack(side=tk.LEFT, padx=(5, 0))

        # Info about when settings apply
        ttk.Label(large_video_frame, text="Note: any change made here applies to next video load",
                  foreground='gray', font=('Arial', 8, 'italic')).pack(pady=(5, 0))

        # Info panel
        info_frame = ttk.Frame(controls_frame)
        info_frame.pack(fill=tk.X)
        
        self.points_label = ttk.Label(info_frame, text="No points (Left: +, Right: -)")
        self.points_label.pack(fill=tk.X)
        
    def update_object_list(self):
        """Update the object list display"""
        # Clear current items
        for item in self.object_tree.get_children():
            self.object_tree.delete(item)
        
        # Add objects that have been used
        used_objects = set()
        
        # Find objects with points
        for _, _, _, obj_id, _ in self.click_points:
            used_objects.add(obj_id)
        
        # Find objects with masks
        for frame_masks in self.masks.values():
            used_objects.update(frame_masks.keys())
        
        # Always show current object
        used_objects.add(self.current_object_id)
        
        for obj_id in sorted(used_objects):
            # Count points for this object
            point_count = sum(1 for _, _, _, oid, _ in self.click_points if oid == obj_id)
            
            # Count masks for this object
            mask_count = sum(1 for frame_masks in self.masks.values() if obj_id in frame_masks)
            
            # Get color for display
            color_hex = self._rgb_to_hex(self.object_colors[obj_id])
            
            # Insert into tree
            item = self.object_tree.insert("", "end", text=str(obj_id),
                                          values=(self.object_names[obj_id], point_count, mask_count))
            
            # Highlight current object
            if obj_id == self.current_object_id:
                self.object_tree.selection_set(item)
                
    def on_object_tree_select(self, event):
        """Handle object tree selection"""
        selection = self.object_tree.selection()
        if selection:
            item = selection[0]
            obj_id = int(self.object_tree.item(item, "text"))
            self.current_object_id = obj_id
            self.object_var.set(obj_id)
            self.object_name_var.set(self.object_names[obj_id])
            self.update_object_color_display()
            if self.frames:
                self.display_current_frame()
                
    def import_object_list(self):
        """Import object names from CSV file"""
        file_path = filedialog.askopenfilename(
            title="Import Object List",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                imported_count = 0
                
                for row in reader:
                    if 'id' in row and 'name' in row:
                        try:
                            obj_id = int(row['id'])
                            if 1 <= obj_id <= self.max_total_objects:
                                self.object_names[obj_id] = row['name'].strip()
                                imported_count += 1
                        except ValueError:
                            continue
                            
                self.update_object_list()
                self.object_name_var.set(self.object_names[self.current_object_id])
                
                messagebox.showinfo("Import Complete", 
                                  f"Successfully imported {imported_count} object names.")
                                  
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import object list: {str(e)}")
            
    def export_object_list(self):
        """Export object names to CSV file"""
        file_path = filedialog.asksaveasfilename(
            title="Export Object List",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'name', 'color_r', 'color_g', 'color_b']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for obj_id in range(1, self.max_total_objects + 1):
                    if obj_id in self.object_colors:
                        color = self.object_colors[obj_id]
                        writer.writerow({
                            'id': obj_id,
                            'name': self.object_names.get(obj_id, f"Object_{obj_id}"),
                            'color_r': color[0],
                            'color_g': color[1],
                            'color_b': color[2]
                        })
                    
            messagebox.showinfo("Export Complete", f"Object list exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export object list: {str(e)}")
    
    def export_annotations(self):
        """Export click point annotations to JSON file"""
        if not self.click_points:
            messagebox.showwarning("No Annotations", "No annotations to export. Please add some click points first.")
            return
        
        try:
            # Get export file path
            file_path = self._get_export_file_path_with_creation(
                title="Export Annotations",
                default_name="sam2_annotations.json",
                file_types=[
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ],
                default_ext=".json"
            )
            
            if not file_path:
                return
            
            # Prepare annotation data
            annotation_data = {
                "video_path": self.video_path,
                "total_frames": len(self.frames),
                "total_annotations": len(self.click_points),
                "annotated_frames": sorted(list(self.annotated_frames)),
                "object_names": self.object_names,
                "object_colors": {str(k): v for k, v in self.object_colors.items()},
                
                # ADDED: Store original video dimensions and current scale
                "video_metadata": {
                    "original_width": self.original_video_width,
                    "original_height": self.original_video_height,
                    "current_scale": self.current_video_scale,
                    "frame_skip": self.frame_skip_var.get() if self.downsample_frames_var.get() else 1,
                    "lazy_load": self.lazy_load_var.get()
                },
                
                "annotations": []
            }
            
            # Convert click points to export format
            # NOTE: Points are already in ORIGINAL coordinate system
            for point in self.click_points:
                img_x, img_y, is_positive, obj_id, frame_idx = point
                annotation = {
                    "frame_index": frame_idx,
                    "x": float(img_x),  # These are in ORIGINAL video coordinates
                    "y": float(img_y),
                    "is_positive": is_positive,
                    "object_id": obj_id,
                    "object_name": self.object_names.get(obj_id, f"Object_{obj_id}")
                }
                annotation_data["annotations"].append(annotation)
            
            # Sort annotations by frame index
            annotation_data["annotations"].sort(key=lambda x: (x["frame_index"], x["object_id"]))
            
            # Add metadata
            annotation_data["export_info"] = {
                "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "app_version": "SAM Video UI v1.0",
                "coordinate_system": "original",  # ADDED: Indicate coordinate system
                "multi_frame_mode": self.multi_frame_annotation_mode
            }
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)
            
            # Show success message
            messagebox.showinfo("Export Complete", 
                              f"Annotations exported successfully!\n\n"
                              f"File: {file_path}\n"
                              f"Total annotations: {len(self.click_points)}\n"
                              f"Annotated frames: {len(self.annotated_frames)}\n"
                              f"Objects: {len(self.object_names)}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export annotations: {str(e)}")
    
    def import_annotations(self):
        """Import click point annotations from JSON file"""
        try:
            # Get import file path
            file_path = filedialog.askopenfilename(
                title="Import Annotations",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                return
            
            # Load annotation data
            with open(file_path, 'r') as f:
                annotation_data = json.load(f)
            
            # Validate the file format
            if "annotations" not in annotation_data:
                messagebox.showerror("Import Error", "Invalid annotation file format. Missing 'annotations' field.")
                return
            
            # Check if video is loaded
            if not self.frames:
                messagebox.showwarning("No Video", "Please load a video first before importing annotations.")
                return
            
            # ADDED: Check for coordinate system compatibility
            saved_metadata = annotation_data.get("video_metadata", {})
            saved_width = saved_metadata.get("original_width")
            saved_height = saved_metadata.get("original_height")
            saved_scale = saved_metadata.get("current_scale", 1.0)
            
            coordinate_system = annotation_data.get("export_info", {}).get("coordinate_system", "unknown")
            
            # Check if we need to warn about coordinate system
            needs_scaling_warning = False
            scale_correction_factor = 1.0
            
            if saved_width and saved_height and self.original_video_width:
                # Check if original video dimensions match
                if saved_width != self.original_video_width or saved_height != self.original_video_height:
                    needs_scaling_warning = True
                    messagebox.showwarning(
                        "Video Dimension Mismatch",
                        f"Warning: Annotations were created for a video with different dimensions!\n\n"
                        f"Saved annotations: {saved_width}x{saved_height}\n"
                        f"Current video: {self.original_video_width}x{self.original_video_height}\n\n"
                        f"Annotations may not appear in the correct locations.\n"
                        f"Please use the same source video."
                    )
            
            # Check for frame count mismatch
            saved_total_frames = annotation_data.get("total_frames", 0)
            current_total_frames = len(self.frames)
            
            if saved_total_frames != current_total_frames:
                result = messagebox.askyesnocancel(
                    "Frame Count Mismatch",
                    f"Warning: Annotation file has {saved_total_frames} frames, "
                    f"but current video has {current_total_frames} frames.\n\n"
                    f"This may happen if:\n"
                    f"- Video was loaded with different optimization settings\n"
                    f"- Video was loaded with frame skipping enabled\n"
                    f"- Different video file is loaded\n\n"
                    f"Coordinate system: {coordinate_system}\n\n"
                    f"Do you want to continue?\n\n"
                    f"Yes: Attempt to import (may skip invalid frame indices)\n"
                    f"No: Cancel import"
                )
                
                if not result:  # No or Cancel
                    return
            
            # Ask user if they want to clear existing annotations
            if self.click_points:
                result = messagebox.askyesnocancel(
                    "Existing Annotations",
                    f"You have {len(self.click_points)} existing annotations.\n\n"
                    f"What would you like to do?\n\n"
                    f"Yes: Clear existing and import new annotations\n"
                    f"No: Add to existing annotations\n"
                    f"Cancel: Abort import"
                )
                
                if result is None:  # Cancel
                    return
                elif result:  # Yes - clear existing
                    self.click_points.clear()
                    self.object_names.clear()
                    self.object_colors.clear()
                    self.annotated_frames.clear()
            
            # Import annotations with frame validation
            imported_count = 0
            skipped_count = 0
            
            for annotation in annotation_data["annotations"]:
                try:
                    frame_idx = annotation["frame_index"]
                    x = annotation["x"]  # These are in ORIGINAL coordinates
                    y = annotation["y"]
                    is_positive = annotation["is_positive"]
                    obj_id = annotation["object_id"]
                    obj_name = annotation.get("object_name", f"Object_{obj_id}")
                    
                    # Validate frame index is within current video bounds
                    if frame_idx >= len(self.frames):
                        skipped_count += 1
                        continue
                    
                    # ADDED: Coordinates are already in ORIGINAL system, 
                    # they will be scaled during display automatically
                    # No conversion needed here!
                    
                    # Add the annotation point (coordinates are in ORIGINAL scale)
                    self.click_points.append((x, y, is_positive, obj_id, frame_idx))
                    
                    # Update object names and colors
                    if obj_id not in self.object_names:
                        self.object_names[obj_id] = obj_name
                        # Assign a color if not already assigned
                        if obj_id not in self.object_colors:
                            # Try to load color from annotation data if available
                            if "object_colors" in annotation_data and str(obj_id) in annotation_data["object_colors"]:
                                self.object_colors[obj_id] = annotation_data["object_colors"][str(obj_id)]
                            else:
                                self.object_colors[obj_id] = self._get_next_color()
                    
                    # Track annotated frames
                    self.annotated_frames.add(frame_idx)
                    
                    imported_count += 1
                    
                except KeyError as e:
                    print(f"Warning: Skipping invalid annotation: {e}")
                    skipped_count += 1
                    continue
            
            # Update UI
            self.update_points_display()
            self.update_object_list()
            self.display_current_frame()
            
            # Show success message with warnings if applicable
            message = f"Annotations imported successfully!\n\n" \
                    f"File: {file_path}\n" \
                    f"Imported annotations: {imported_count}\n"
            
            if skipped_count > 0:
                message += f"Skipped annotations: {skipped_count} (invalid frame indices)\n"
            
            if coordinate_system == "original":
                message += f"\n✓ Using original coordinate system (compatible with video scaling)"
            
            message += f"\nTotal annotations: {len(self.click_points)}\n" \
                    f"Objects: {len(self.object_names)}"
            
            messagebox.showinfo("Import Complete", message)
            
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import annotations: {str(e)}")

    def import_masks(self):
        """Load segmentation results: segmented video, annotations, and mask directory"""
        try:
            # Select output directory
            output_dir = filedialog.askdirectory(title="Select Processing Output Directory")
            if not output_dir:
                return

            # Load metadata first
            metadata_path = os.path.join(output_dir, "processing_metadata.json")
            if not os.path.exists(metadata_path):
                messagebox.showerror("Error", "processing_metadata.json not found in directory")
                return

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Get file paths from metadata (with backward compatibility)
            file_paths = metadata.get("file_paths", {})
            segmented_video_filename = file_paths.get("segmented_video_filename", "segmented_video.mp4")

            # Store original video path for re-segmentation
            self.original_video_path_for_resegment = file_paths.get("original_video_path")

            # Locate segmented video
            segmented_video_path = os.path.join(output_dir, segmented_video_filename)
            if not os.path.exists(segmented_video_path):
                # Prompt user to locate the video
                messagebox.showwarning("Video Not Found",
                                     f"Segmented video not found at:\n{segmented_video_path}\n\n"
                                     f"Please locate the segmented video file.")
                segmented_video_path = filedialog.askopenfilename(
                    title="Select Segmented Video",
                    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
                )
                if not segmented_video_path:
                    return

            # Clear existing state
            self.frames = []
            self.masks = {}
            self.click_points = []
            self.annotated_frames = set()

            # Load segmented video
            print(f"Loading segmented video: {segmented_video_path}")
            self.video_path = segmented_video_path
            self.segmented_video_displayed = True
            self.results_output_dir = output_dir
            self.load_video_frames()

            # Set mask export directory for flash functionality
            masks_dir = os.path.join(output_dir, "masks")
            if not os.path.exists(masks_dir):
                messagebox.showerror("Error", "masks/ directory not found")
                return
            self.mask_export_dir = masks_dir

            # Load annotations from metadata
            self._load_annotations_from_metadata(metadata)

            # Validate masks exist on first frame (just check, don't load)
            first_frame_with_annotations = min(
                (pt[4] for pt in self.click_points),
                default=None
            )
            if first_frame_with_annotations is not None:
                # Try to find at least one mask file for the first frame
                found_mask = False
                for filename in os.listdir(masks_dir):
                    if filename.startswith(f"mask_f{first_frame_with_annotations:06d}_"):
                        found_mask = True
                        break

                if not found_mask:
                    print(f"WARNING: No mask files found for first annotated frame {first_frame_with_annotations}")

            # Update UI
            self.loaded_from_results = True
            self.update_object_list()
            self.display_current_frame()

            num_objects = len(self.object_names)
            num_annotations = len(self.click_points)
            messagebox.showinfo("Success",
                              f"Loaded segmentation results successfully!\n\n"
                              f"Video: {os.path.basename(segmented_video_path)}\n"
                              f"Frames: {len(self.frames)}\n"
                              f"Objects: {num_objects}\n"
                              f"Annotations: {num_annotations}")

        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import masks: {str(e)}")

    def _load_annotations_from_metadata(self, metadata):
        """Load annotations from processing metadata into UI state"""
        if "original_annotations" not in metadata:
            print("WARNING: No original_annotations found in metadata")
            return

        annotations = metadata["original_annotations"]

        # Load object names: {int: str}
        object_names_raw = annotations.get("object_names", {})
        self.object_names = {int(k): v for k, v in object_names_raw.items()}

        # Load object colors: {int: [R,G,B]}
        object_colors_raw = annotations.get("object_colors", {})
        self.object_colors = {int(k): v for k, v in object_colors_raw.items()}

        # Load click points: [(x, y, is_positive, obj_id, frame_idx), ...]
        annotations_list = annotations.get("annotations", [])
        for ann in annotations_list:
            self.click_points.append([
                ann["x"],
                ann["y"],
                ann["is_positive"],
                ann["object_id"],
                ann["frame_index"]
            ])

        # Update annotated frames set
        self.annotated_frames = set(ann["frame_index"] for ann in annotations_list)

        # Update max object ID
        if self.object_names:
            self.max_object_id = max(self.object_names.keys())
        else:
            self.max_object_id = 1

        print(f"Loaded annotations: {len(self.click_points)} points, "
              f"{len(self.object_names)} objects, "
              f"{len(self.annotated_frames)} annotated frames")

    def _get_original_video_for_resegmentation(self):
        """Get original video path for re-segmentation when working with loaded results"""
        # Check if we have stored path from metadata
        if self.original_video_path_for_resegment and os.path.exists(self.original_video_path_for_resegment):
            # Confirm with user
            result = messagebox.askyesno(
                "Use Stored Original Video?",
                f"Found original video path in metadata:\n\n"
                f"{self.original_video_path_for_resegment}\n\n"
                f"Use this video for re-segmentation?"
            )
            if result:
                return self.original_video_path_for_resegment

        # Prompt user to locate original video
        messagebox.showinfo(
            "Select Original Video",
            "Re-segmentation requires the original video (not the segmented version).\n\n"
            "Please select the original video file."
        )

        original_video_path = filedialog.askopenfilename(
            title="Select Original Video for Re-segmentation",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        return original_video_path if original_video_path else None

    def flash_selected_object_mask(self):
        """Flash the mask for the currently selected object with white color"""
        current_obj_id = self.current_object_id
        if current_obj_id == 0:
            messagebox.showinfo("Info", "Please select an object first")
            return

        # Try to load mask if not in memory (e.g., when loaded from results)
        if self.current_frame_idx not in self.masks or current_obj_id not in self.masks.get(self.current_frame_idx, {}):
            # Try loading from disk
            mask = self._load_mask(self.current_frame_idx, current_obj_id)
            if mask is None:
                messagebox.showinfo("Info", "No mask found for selected object on current frame")
                return

            # Temporarily add to self.masks for flash animation
            if self.current_frame_idx not in self.masks:
                self.masks[self.current_frame_idx] = {}
            self.masks[self.current_frame_idx][current_obj_id] = mask

        # Initialize flash state
        self.flash_in_progress = True
        self.flash_obj_id = current_obj_id
        self.flash_white_on = True

        # Flash 3 times: white for 0.3s, normal for 0.3s
        flash_count = [0]  # Use list to make it mutable in nested function
        max_flashes = 3

        def flash_step():
            if flash_count[0] >= max_flashes:
                # Restore original state
                self.flash_in_progress = False
                self.flash_obj_id = None
                self.flash_white_on = False
                self.display_current_frame()
                return

            # Toggle white on/off
            self.flash_white_on = not self.flash_white_on
            self.display_current_frame()

            # If we just turned white off, increment flash count
            if not self.flash_white_on:
                flash_count[0] += 1

            # Schedule next toggle after 300ms
            self.root.after(300, flash_step)

        # Start flashing
        flash_step()

    def _handle_annotation_frame_mismatch(self, annotation_data):
        """Handle frame count mismatch between saved annotations and current video"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Handle Frame Mismatch")
        dialog.geometry("500x350")
        dialog.configure(bg='#2b2b2b')
        dialog.transient(self.root)
        dialog.grab_set()

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        saved_frames = annotation_data.get("total_frames", 0)
        current_frames = len(self.frames)

        ttk.Label(main_frame, text="Frame Count Mismatch Options", 
                    font=('Arial', 14, 'bold')).pack(pady=(0, 15))

        info_text = f"Saved annotations: {saved_frames} frames\n" \
                    f"Current video: {current_frames} frames\n\n" \
                    f"Choose how to handle this mismatch:"

        ttk.Label(main_frame, text=info_text, justify=tk.LEFT).pack(pady=(0, 15))

        option_var = tk.StringVar(value="skip")

        ttk.Radiobutton(main_frame, text="Skip invalid frame indices (safest)", 
                        variable=option_var, value="skip").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(main_frame, text="Scale frame indices proportionally", 
                        variable=option_var, value="scale").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(main_frame, text="Cancel import", 
                        variable=option_var, value="cancel").pack(anchor=tk.W, pady=2)

        def apply_option():
            option = option_var.get()
            dialog.destroy()
            
            if option == "cancel":
                return
            elif option == "skip":
                self._import_annotations_skip_invalid(annotation_data)
            elif option == "scale":
                self._import_annotations_scale_indices(annotation_data)

        ttk.Button(main_frame, text="Apply", command=apply_option).pack(pady=(20, 0))

    def _import_annotations_skip_invalid(self, annotation_data):
        """Import annotations, skipping those with invalid frame indices"""
        imported_count = 0
        skipped_count = 0
        
        for annotation in annotation_data["annotations"]:
            try:
                frame_idx = annotation["frame_index"]
                
                # Skip if frame index is out of bounds
                if frame_idx >= len(self.frames):
                    skipped_count += 1
                    continue
                
                x = annotation["x"]
                y = annotation["y"]
                is_positive = annotation["is_positive"]
                obj_id = annotation["object_id"]
                obj_name = annotation.get("object_name", f"Object_{obj_id}")
                
                # Add the annotation point
                self.click_points.append((x, y, is_positive, obj_id, frame_idx))
                
                # Update object names and colors
                if obj_id not in self.object_names:
                    self.object_names[obj_id] = obj_name
                    if obj_id not in self.object_colors:
                        if "object_colors" in annotation_data and str(obj_id) in annotation_data["object_colors"]:
                            self.object_colors[obj_id] = annotation_data["object_colors"][str(obj_id)]
                        else:
                            self.object_colors[obj_id] = self._get_next_color()
                
                # Track annotated frames
                self.annotated_frames.add(frame_idx)
                imported_count += 1
                
            except KeyError as e:
                skipped_count += 1
                continue
        
        # Update UI
        self.update_points_display()
        self.update_object_list()
        self.display_current_frame()
        
        messagebox.showinfo("Import Complete", 
                        f"Imported {imported_count} annotations\n"
                        f"Skipped {skipped_count} annotations with invalid frame indices")

    def _import_annotations_scale_indices(self, annotation_data):
        """Import annotations, scaling frame indices proportionally"""
        saved_frames = annotation_data.get("total_frames", 1)
        current_frames = len(self.frames)
        scale_factor = current_frames / saved_frames
        
        imported_count = 0
        
        for annotation in annotation_data["annotations"]:
            try:
                original_frame_idx = annotation["frame_index"]
                # Scale the frame index
                scaled_frame_idx = int(original_frame_idx * scale_factor)
                
                # Clamp to valid range
                scaled_frame_idx = max(0, min(scaled_frame_idx, current_frames - 1))
                
                x = annotation["x"]
                y = annotation["y"]
                is_positive = annotation["is_positive"]
                obj_id = annotation["object_id"]
                obj_name = annotation.get("object_name", f"Object_{obj_id}")
                
                # Add the annotation point with scaled frame index
                self.click_points.append((x, y, is_positive, obj_id, scaled_frame_idx))
                
                # Update object names and colors
                if obj_id not in self.object_names:
                    self.object_names[obj_id] = obj_name
                    if obj_id not in self.object_colors:
                        if "object_colors" in annotation_data and str(obj_id) in annotation_data["object_colors"]:
                            self.object_colors[obj_id] = annotation_data["object_colors"][str(obj_id)]
                        else:
                            self.object_colors[obj_id] = self._get_next_color()
                
                # Track annotated frames
                self.annotated_frames.add(scaled_frame_idx)
                imported_count += 1
                
            except KeyError:
                continue
        
        # Update UI
        self.update_points_display()
        self.update_object_list()
        self.display_current_frame()
        
        messagebox.showinfo("Import Complete", 
                        f"Imported {imported_count} annotations with scaled frame indices\n"
                        f"Scale factor: {scale_factor:.3f}")

    def _get_next_color(self):
        """Get next available color for a new object"""
        # Find first unused object ID to get its color
        for obj_id in range(1, self.max_total_objects + 1):
            if obj_id not in self.object_colors:
                return self.object_colors.get(obj_id, [255, 255, 255])
        # Fallback to white if all colors used
        return [255, 255, 255]

    def toggle_multi_frame_annotation(self):
        """Multi-frame annotation mode is always enabled"""
        # Multi-frame annotation is always active, no need to toggle
        self.multi_frame_label.config(text="MULTI-FRAME ANNOTATION ACTIVE")
        self.status_label.config(text="Multi-frame mode: Navigate to different frames and add points, then segment")

    def toggle_point_removal_mode(self):
        """Toggle point removal mode for removing individual annotation points"""
        self.point_removal_mode = not self.point_removal_mode

        if self.point_removal_mode:
            # Change button color to indicate active state
            self.remove_point_button.config(bg='#DC143C', activebackground='#FF6347')  # Red
            self.status_label.config(text="Point removal mode: Click on annotation points to remove them")
        else:
            # Reset button color to normal
            self.remove_point_button.config(bg='#404040', activebackground='#505050')
            self.status_label.config(text="Point removal mode disabled")
    
    def remove_point_at_location(self, x, y, frame_idx):
        """Remove the closest annotation point to the given location"""
        if not self.click_points:
            return False
        
        # Find points on the current frame
        frame_points = [(i, point) for i, point in enumerate(self.click_points) 
                       if point[4] == frame_idx]  # point[4] is frame_idx
        
        if not frame_points:
            return False
        
        # Find the closest point to the click location
        min_distance = float('inf')
        closest_point_idx = None
        
        for point_idx, (px, py, is_pos, obj_id, f_idx) in frame_points:
            # CRITICAL FIX: Convert stored ORIGINAL coordinates to CURRENT (scaled) coordinates
            # for distance comparison with the click location
            if self.current_video_scale != 1.0 and self.original_video_width:
                scaled_px = px * self.current_video_scale
                scaled_py = py * self.current_video_scale
            else:
                scaled_px = px
                scaled_py = py
            
            # Calculate distance from click to point (both in current/scaled coordinates now)
            distance = ((x - scaled_px) ** 2 + (y - scaled_py) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_point_idx = point_idx
        
        # Only remove if click is close enough to a point (within 20 pixels)
        if closest_point_idx is not None and min_distance <= 20:
            removed_point = self.click_points.pop(closest_point_idx)
            px, py, is_pos, obj_id, f_idx = removed_point
            
            # Update annotated frames if no more points on this frame
            remaining_points_on_frame = [p for p in self.click_points if p[4] == frame_idx]
            if not remaining_points_on_frame:
                self.annotated_frames.discard(frame_idx)
            
            # Update object list and display
            self.update_object_list()
            self.display_current_frame()
            
            # Show confirmation (using ORIGINAL coordinates in the message)
            point_type = "positive" if is_pos else "negative"
            obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
            self.status_label.config(text=f"Removed {point_type} point for {obj_name} at ({px:.0f}, {py:.0f})")
            
            return True

        return False

    def update_object_name(self, event=None):
        """Update the name of the current object"""
        new_name = self.object_name_var.get().strip()
        if new_name:
            self.object_names[self.current_object_id] = new_name
            self.update_object_list()
   
    def on_object_change(self, event=None):
        obj_id = self.object_var.get()
        if 1 <= obj_id <= self.max_total_objects:
            self.current_object_id = obj_id
            self.object_name_var.set(self.object_names[obj_id])
            self.update_object_color_display()
            self.update_object_list()
            if self.frames:
                self.display_current_frame()

    def update_object_color_display(self):
        """Update the color indicator for the current object"""
        color = self.object_colors[self.current_object_id]
        # Display a colored square using unicode block character
        self.object_color_label.config(text="■", foreground=self._rgb_to_hex(color))
    def add_new_object(self):
        """Add a new object for segmentation"""
        if self.max_object_id < self.max_total_objects:
            self.max_object_id += 1
            self.current_object_id = self.max_object_id
            self.object_var.set(self.current_object_id)
            self.object_spinbox.config(to=min(self.max_object_id, self.max_total_objects))
            self.object_name_var.set(self.object_names[self.current_object_id])
            self.update_object_color_display()
            self.update_object_list()
            self.status_label.config(text=f"Added object {self.current_object_id}: {self.object_names[self.current_object_id]}")
        else:
            messagebox.showwarning("Limit Reached", f"Maximum {self.max_total_objects} objects supported.")
            
    def load_video(self):
        """Load video file and extract frames"""
        # Clean up temporary masks from previous video
        self._cleanup_temp_masks()

        # Clean up any existing lazy loading
        if hasattr(self, 'video_cap_lazy') and self.video_cap_lazy:
            self.video_cap_lazy.release()
            self.video_cap_lazy = None
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            self.video_path = file_path
            self.load_video_frames()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            
    def load_video_frames(self):
        """Extract frames from video with multiple backend fallbacks"""
        try:
            # Try different OpenCV backends in order of preference
            backends = [
                (cv2.CAP_FFMPEG, "FFMPEG"),
                (cv2.CAP_GSTREAMER, "GStreamer"),
                (cv2.CAP_ANY, "Auto")
            ]
            
            self.video_cap = None
            successful_backend = None
            
            for backend, name in backends:
                try:
                    self.video_cap = cv2.VideoCapture(self.video_path, backend)
                    if self.video_cap.isOpened():
                        successful_backend = name
                        break
                    else:
                        self.video_cap.release()
                except Exception as e:
                    print(f"Failed to open with {name} backend: {e}")
                    continue
            
            if not self.video_cap or not self.video_cap.isOpened():
                raise ValueError(
                    "Could not open video file with any backend.\n"
                    "Please install required codecs or convert video to MP4 format."
                )
            
            self.status_label.config(text=f"Video loaded using {successful_backend} backend")
            
            self.frames = []
            self.masks = {}
            self.click_points = []
            
            # Get video properties
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            original_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # ADDED: Store original dimensions for coordinate consistency
            self.original_video_width = original_width
            self.original_video_height = original_height
            
            # Calculate current scale factor
            skip_frames = self.frame_skip_var.get() if self.downsample_frames_var.get() else 1
            scale_factor = self.video_scale_factor.get() if self.scale_video_var.get() else 1.0
            
            # ADDED: Store current scale for coordinate conversions
            self.current_video_scale = scale_factor
            
            # Calculate memory usage estimate
            bytes_per_frame = original_width * original_height * 3  # RGB
            estimated_memory_mb = (total_frames * bytes_per_frame) / (1024 * 1024)
            
            # Check if we need optimization
            skip_frames = self.frame_skip_var.get() if self.downsample_frames_var.get() else 1
            scale_factor = self.video_scale_factor.get() if self.scale_video_var.get() else 1.0
            
            # Adjust memory estimate based on optimizations
            optimized_memory_mb = estimated_memory_mb / skip_frames * (scale_factor ** 2)
            
            # Show memory warning for large videos
            if estimated_memory_mb > 1000:  # > 1GB
                result = messagebox.askyesnocancel(
                    "Large Video Detected", 
                    f"Video size: {total_frames} frames ({original_width}x{original_height})\n"
                    f"Estimated memory: {estimated_memory_mb:.1f} MB\n"
                    f"Optimized memory: {optimized_memory_mb:.1f} MB\n\n"
                    f"Enable optimizations to reduce memory usage?\n"
                    f"- Frame skipping: {skip_frames}x\n"
                    f"- Video scaling: {scale_factor:.1f}x\n"
                    f"- Lazy loading: Load frames on demand\n\n"
                    f"Click Yes to proceed with optimizations,\n"
                    f"No to load without optimizations,\n"
                    f"Cancel to abort."
                )
                if result is None:  # Cancel
                    self.video_cap.release()
                    return
                elif result:  # Yes - enable optimizations
                    self.downsample_frames_var.set(True)
                    self.scale_video_var.set(True)
                    self.lazy_load_var.set(True)
                    skip_frames = self.frame_skip_var.get()
                    scale_factor = self.video_scale_factor.get()
            
            # Handle lazy loading
            if self.lazy_load_var.get():
                self._setup_lazy_loading(total_frames, original_width, original_height, fps, skip_frames, scale_factor)
                return
            
            self.status_label.config(text=f"Loading {total_frames} frames (optimized: {optimized_memory_mb:.1f} MB)...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.root.update()
            
            frame_count = 0
            loaded_count = 0
            
            while True:
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                
                # Skip frames if enabled
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                
                # Scale frame if enabled
                if scale_factor < 1.0:
                    new_width = int(original_width * scale_factor)
                    new_height = int(original_height * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame_rgb)
                loaded_count += 1
                frame_count += 1
                
                # Update progress
                progress = (frame_count / total_frames) * 100
                self.progress_var.set(progress)
                if loaded_count % 10 == 0:
                    self.root.update_idletasks()
            
            self.video_cap.release()
            self.progress_bar.pack_forget()
            
            if self.frames:
                self.current_frame_idx = 0
                self.frame_slider.config(to=len(self.frames)-1)
                self.display_current_frame()
                
                # Update status with optimization info
                status_text = f"Video loaded: {len(self.frames)} frames @ {fps:.1f} FPS"
                if skip_frames > 1:
                    status_text += f" (skipped {skip_frames-1} frames)"
                if scale_factor < 1.0:
                    status_text += f" (scaled {scale_factor:.1f}x)"
                self.status_label.config(text=status_text)
                
                self.update_object_list()
                # Initialize range spinboxes
                self.range_start_var.set(0)
                self.range_end_var.set(max(0, len(self.frames)-1))
                self.range_start_spin.config(to=max(0, len(self.frames)-1))
                self.range_end_spin.config(to=max(0, len(self.frames)-1))
            else:
                raise ValueError("No frames could be extracted from video")
                
        except Exception as e:
            self.progress_bar.pack_forget()
            raise e

    def _setup_lazy_loading(self, total_frames, original_width, original_height, fps, skip_frames, scale_factor):
        """Setup lazy loading for very large videos"""
        try:
            # Keep video capture open for lazy loading
            self.video_cap_lazy = cv2.VideoCapture(self.video_path)
            if not self.video_cap_lazy.isOpened():
                raise ValueError("Could not open video file for lazy loading")
            
            # ADDED: Store original dimensions
            self.original_video_width = original_width
            self.original_video_height = original_height
            self.current_video_scale = scale_factor
            
            # Store video properties for lazy loading
            self.video_props = {
                'total_frames': total_frames,
                'original_width': original_width,
                'original_height': original_height,
                'fps': fps,
                'skip_frames': skip_frames,
                'scale_factor': scale_factor
            }
            
            # Calculate how many frames we'll actually load
            frames_to_load = total_frames // skip_frames
            if total_frames % skip_frames != 0:
                frames_to_load += 1
            
            # Initialize frame cache (empty frames list)
            self.frames = [None] * frames_to_load
            self.frame_cache = {}  # Cache for loaded frames
            
            # Initialize UI
            self.current_frame_idx = 0
            self.frame_slider.config(to=frames_to_load-1)
            self.display_current_frame()
            
            status_text = f"Lazy loading: {frames_to_load} frames @ {fps:.1f} FPS"
            if skip_frames > 1:
                status_text += f" (skipped {skip_frames-1} frames)"
            if scale_factor < 1.0:
                status_text += f" (scaled {scale_factor:.1f}x)"
            self.status_label.config(text=status_text)
            
            self.update_object_list()
            # Initialize range spinboxes
            self.range_start_var.set(0)
            self.range_end_var.set(max(0, frames_to_load-1))
            self.range_start_spin.config(to=max(0, frames_to_load-1))
            self.range_end_spin.config(to=max(0, frames_to_load-1))
            
        except Exception as e:
            if self.video_cap_lazy:
                self.video_cap_lazy.release()
            raise e

    def _load_frame_lazy(self, frame_idx):
        """Load a specific frame on demand"""
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]
        
        if not self.video_cap_lazy or not hasattr(self, 'video_props'):
            return None
        
        try:
            # Calculate original frame index
            original_frame_idx = frame_idx * self.video_props['skip_frames']
            
            # Seek to the frame
            self.video_cap_lazy.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = self.video_cap_lazy.read()
            
            if not ret:
                return None
            
            # Scale frame if enabled
            if self.video_props['scale_factor'] < 1.0:
                new_width = int(self.video_props['original_width'] * self.video_props['scale_factor'])
                new_height = int(self.video_props['original_height'] * self.video_props['scale_factor'])
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Cache the frame (limit cache size to prevent memory issues)
            if len(self.frame_cache) > 50:  # Keep only last 50 frames in cache
                # Remove oldest frames
                oldest_keys = sorted(self.frame_cache.keys())[:10]
                for key in oldest_keys:
                    del self.frame_cache[key]
            
            self.frame_cache[frame_idx] = frame_rgb
            return frame_rgb
            
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            return None

    def display_current_frame(self):
        """Display current video frame with overlays"""
        if not self.frames:
            return
        
        # Handle lazy loading
        if self.lazy_load_var.get() and hasattr(self, 'video_props'):
            # Load frame on demand
            frame = self._load_frame_lazy(self.current_frame_idx)
            if frame is None:
                return
            self.current_frame = frame.copy()
        else:
            # Regular loading
            if self.current_frame_idx >= len(self.frames) or self.frames[self.current_frame_idx] is None:
                return
            self.current_frame = self.frames[self.current_frame_idx].copy()
        display_frame = self.current_frame.copy()

        # Add annotation indicator for multi-frame annotation mode
        if self.multi_frame_annotation_mode and self.current_frame_idx in self.annotated_frames:
            # Add blue border for annotated frames
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1]-1, display_frame.shape[0]-1), 
                         (0, 165, 255), 8)
        
        # Apply mask overlay if enabled and masks exist
        if self.show_masks_var.get() and self.current_frame_idx in self.masks:
            frame_masks = self.masks[self.current_frame_idx]

            # Debug: Print mask info for current frame
            print(f"\nDEBUG: Rendering frame {self.current_frame_idx}")
            print(f"  Frame shape: {display_frame.shape}")
            print(f"  Num masks: {len(frame_masks)}")

            # Collect all mask data first (to avoid cumulative blending bug)
            mask_data_list = []
            for obj_id in frame_masks.keys():
                # Load mask from disk (memory optimization)
                mask = self._load_mask(self.current_frame_idx, obj_id)
                if mask is None:
                    continue
                # If flash is in progress, only show the flashing object
                if self.flash_in_progress and obj_id != self.flash_obj_id:
                    continue

                if len(mask.shape) == 2:  # Single channel mask
                    # Debug: Print mask details
                    print(f"  Mask {obj_id}: shape={mask.shape}, dtype={mask.dtype}, "
                          f"nonzero={np.count_nonzero(mask)}, max={mask.max()}")

                    # Get object color
                    obj_color = self.object_colors.get(obj_id, [255, 255, 255])

                    # Check if mask needs resizing
                    if mask.shape != (display_frame.shape[0], display_frame.shape[1]):
                        print(f"  WARNING: Mask size {mask.shape} doesn't match frame size "
                              f"({display_frame.shape[0]}, {display_frame.shape[1]})")
                        print(f"  Resizing mask...")
                        from PIL import Image as PILImage
                        mask_pil = PILImage.fromarray(mask)
                        mask_pil = mask_pil.resize((display_frame.shape[1], display_frame.shape[0]), PILImage.NEAREST)
                        mask = np.array(mask_pil)
                        print(f"  Resized mask to: {mask.shape}")

                    # Convert mask to boolean
                    mask_bool = mask > 0

                    # If flashing, override color to white
                    if self.flash_in_progress and obj_id == self.flash_obj_id and self.flash_white_on:
                        obj_color = [255, 255, 255]

                    mask_data_list.append((mask_bool, obj_color, obj_id))

            # FIXED: Single-pass blending to avoid cumulative darkening
            if mask_data_list:
                height, width = display_frame.shape[:2]
                combined_overlay = np.zeros((height, width, 3), dtype=np.float32)
                overlap_count = np.zeros((height, width), dtype=np.int32)

                for mask_bool, obj_color, obj_id in mask_data_list:
                    # Add color to overlapping regions (accumulate for averaging)
                    combined_overlay[mask_bool] += obj_color
                    overlap_count[mask_bool] += 1

                # Average colors where masks overlap
                mask_pixels = overlap_count > 0
                combined_overlay[mask_pixels] /= overlap_count[mask_pixels, np.newaxis]
                combined_overlay = combined_overlay.astype(np.uint8)

                # Single blend operation - fixes cumulative darkening bug
                alpha = self.mask_opacity_var.get() if hasattr(self, 'mask_opacity_var') else 0.4
                print(f"  Blending {len(mask_data_list)} masks with alpha={alpha}")
                display_frame = cv2.addWeighted(display_frame, 1-alpha, combined_overlay, alpha, 0)
        
        # Draw click points only for the current frame
        for i, (x, y, is_positive, obj_id, frame_idx) in enumerate(self.click_points):
            if frame_idx != self.current_frame_idx:
                continue
            # Only draw points for current object or if showing all
            if obj_id == self.current_object_id or not hasattr(self, 'current_object_id'):
                # ADDED: Convert from ORIGINAL coordinates to CURRENT frame coordinates
                if self.current_video_scale != 1.0 and self.original_video_width:
                    scaled_x = x * self.current_video_scale
                    scaled_y = y * self.current_video_scale
                else:
                    scaled_x = x
                    scaled_y = y
                
                obj_color = self.object_colors.get(obj_id, [255, 255, 255])
                color = tuple(obj_color) if is_positive else (255, 0, 0)

                # Draw circle using SCALED coordinates
                cv2.circle(display_frame, (int(scaled_x), int(scaled_y)), 8, color, -1)
                cv2.circle(display_frame, (int(scaled_x), int(scaled_y)), 10, (255, 255, 255), 2)

                # Draw symbol using lines instead of text for perfect alignment
                line_length = 6
                line_thickness = 2
                white = (255, 255, 255)

                if is_positive:
                    # Draw + (vertical and horizontal lines)
                    # Vertical line
                    cv2.line(display_frame,
                            (int(scaled_x), int(scaled_y - line_length)),
                            (int(scaled_x), int(scaled_y + line_length)),
                            white, line_thickness)
                    # Horizontal line
                    cv2.line(display_frame,
                            (int(scaled_x - line_length), int(scaled_y)),
                            (int(scaled_x + line_length), int(scaled_y)),
                            white, line_thickness)
                else:
                    # Draw - (horizontal line only)
                    cv2.line(display_frame,
                            (int(scaled_x - line_length), int(scaled_y)),
                            (int(scaled_x + line_length), int(scaled_y)),
                            white, line_thickness)
                
                # Draw object name
                obj_name = self.object_names.get(obj_id, f"Obj{obj_id}")[:8]
                cv2.putText(display_frame, obj_name, (int(scaled_x)+15, int(scaled_y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        
        # Convert to PIL and display
        pil_image = Image.fromarray(display_frame)
        
        # Scale image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = pil_image.size
            
            # Calculate scale to fit canvas while maintaining aspect ratio
            scale_w = (canvas_width - 20) / img_width
            scale_h = (canvas_height - 20) / img_height
            self.scale_factor = min(scale_w, scale_h)  # Allow upscaling when window enlarged
            
            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and display
        self.display_frame = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        
        # Center the image on canvas
        canvas_center_x = canvas_width // 2
        canvas_center_y = canvas_height // 2
        self.canvas.create_image(canvas_center_x, canvas_center_y, image=self.display_frame)
        
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Update frame info
        self.frame_var.set(self.current_frame_idx)
        frame_text = f"{self.current_frame_idx + 1}/{len(self.frames)}"
        if self.multi_frame_annotation_mode and self.current_frame_idx in self.annotated_frames:
            frame_text += " (annotated)"
        self.frame_label.config(text=frame_text)

    def on_canvas_resize(self, event):
        """Handle canvas resize events with debouncing"""
        # Only respond to actual size changes, not other configure events
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Check if size actually changed
        if hasattr(self, '_last_canvas_size'):
            if (canvas_width, canvas_height) == self._last_canvas_size:
                return  # Size didn't change, ignore this event

        self._last_canvas_size = (canvas_width, canvas_height)

        # Cancel any pending resize callback
        if hasattr(self, '_resize_after_id'):
            self.root.after_cancel(self._resize_after_id)

        # Schedule display update after 100ms delay (debounce)
        # This prevents excessive updates during active resizing
        self._resize_after_id = self.root.after(100, self._handle_resize)

    def _handle_resize(self):
        """Actually handle the resize after debounce delay"""
        if self.frames:
            self.display_current_frame()

    def on_canvas_click(self, event):
        """Handle left mouse click (positive point or point removal)"""
        if self.point_removal_mode:
            self.handle_point_removal_click(event)
        else:
            self.add_click_point(event, is_positive=True)
        
    def on_canvas_right_click(self, event):
        """Handle right mouse click (negative point or point removal)"""
        if self.point_removal_mode:
            self.handle_point_removal_click(event)
        else:
            self.add_click_point(event, is_positive=False)
    
    def handle_point_removal_click(self, event):
        """Handle click in point removal mode"""
        if not self.frames or not self.current_frame_idx < len(self.frames):
            return
            
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates
        if self.scale_factor > 0:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Account for centering
            img_display_width = int(self.current_frame.shape[1] * self.scale_factor)
            img_display_height = int(self.current_frame.shape[0] * self.scale_factor)
            
            offset_x = (canvas_width - img_display_width) // 2
            offset_y = (canvas_height - img_display_height) // 2
            
            img_x = (canvas_x - offset_x) / self.scale_factor
            img_y = (canvas_y - offset_y) / self.scale_factor
            
            # Check if click is within image bounds
            if 0 <= img_x < self.current_frame.shape[1] and 0 <= img_y < self.current_frame.shape[0]:
                # Try to remove a point at this location
                if not self.remove_point_at_location(img_x, img_y, self.current_frame_idx):
                    self.status_label.config(text="No annotation point found near this location")
        else:
            self.status_label.config(text="Cannot remove points: invalid scale factor")
        
    def add_click_point(self, event, is_positive=True):
        """Add click point for segmentation"""
        if not self.frames or not self.current_frame_idx < len(self.frames):
            return
            
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates
        if self.scale_factor > 0:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Account for centering
            img_display_width = int(self.current_frame.shape[1] * self.scale_factor)
            img_display_height = int(self.current_frame.shape[0] * self.scale_factor)
            
            offset_x = (canvas_width - img_display_width) // 2
            offset_y = (canvas_height - img_display_height) // 2
            
            # Convert to current frame coordinates
            img_x = (canvas_x - offset_x) / self.scale_factor
            img_y = (canvas_y - offset_y) / self.scale_factor
            
            # Ensure coordinates are within current frame bounds
            img_height, img_width = self.current_frame.shape[:2]
            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                # ADDED: Convert to ORIGINAL video coordinates before storing
                # This ensures annotations work regardless of current scaling
                if self.current_video_scale != 1.0 and self.original_video_width:
                    original_x = img_x / self.current_video_scale
                    original_y = img_y / self.current_video_scale
                else:
                    original_x = img_x
                    original_y = img_y
                
                # Store frame-aware point with ORIGINAL coordinates
                self.click_points.append((original_x, original_y, is_positive, 
                                        self.current_object_id, self.current_frame_idx))
                
                # Track annotated frames in multi-frame annotation mode
                if self.multi_frame_annotation_mode:
                    self.annotated_frames.add(self.current_frame_idx)
                
                self.update_points_display()
                self.update_object_list()
                self.display_current_frame()
                
    def update_points_display(self):
        """Update the points display label"""
        if self.click_points:
            # Count points by object
            object_counts = {}
            for _, _, is_pos, obj_id, _ in self.click_points:
                if obj_id not in object_counts:
                    object_counts[obj_id] = {'pos': 0, 'neg': 0}
                if is_pos:
                    object_counts[obj_id]['pos'] += 1
                else:
                    object_counts[obj_id]['neg'] += 1
            
            # Create summary text
            total_points = len(self.click_points)
            current_obj_points = object_counts.get(self.current_object_id, {'pos': 0, 'neg': 0})
            obj_name = self.object_names.get(self.current_object_id, f"Obj{self.current_object_id}")
            points_text = f"Total: {total_points} | {obj_name}: +{current_obj_points['pos']}, -{current_obj_points['neg']}"
            
            # Add multi-frame annotation info
            if self.multi_frame_annotation_mode and self.annotated_frames:
                points_text += f" | Annotated frames: {len(self.annotated_frames)}"
        else:
            points_text = "No points (Left: +, Right: -)"
            if self.multi_frame_annotation_mode:
                points_text += " | Multi-frame mode active"
        
        self.points_label.config(text=points_text)
            
    def clear_points(self):
        """Clear all click points for the current object only"""
        # Filter out points for current object
        self.click_points = [p for p in self.click_points if p[3] != self.current_object_id]
        
        # Update annotated frames set - remove frames that no longer have any points
        remaining_frames = {p[4] for p in self.click_points}
        self.annotated_frames = self.annotated_frames.intersection(remaining_frames)
        
        self.update_points_display()
        self.update_object_list()
        if self.frames:
            self.display_current_frame()
    
    def clear_current_object(self):
        """Clear points and masks for current object only"""
        # Remove points for current object
        self.click_points = [p for p in self.click_points if p[3] != self.current_object_id]
        
        # Remove masks for current object
        for frame_idx in self.masks:
            if self.current_object_id in self.masks[frame_idx]:
                del self.masks[frame_idx][self.current_object_id]
        
        self.update_points_display()
        self.update_object_list()
        if self.frames:
            self.display_current_frame()
            
    def toggle_mask_display(self):
        """Toggle mask overlay display"""
        if self.frames:
            self.display_current_frame()

    def on_mask_opacity_change(self, value=None):
        """Handle mask opacity slider change"""
        opacity = self.mask_opacity_var.get()
        # Update label to show percentage
        self.opacity_label.config(text=f"{int(opacity * 100)}%")
        
        # Redraw frame if masks are visible
        if self.show_masks_var.get() and self.frames:
            self.display_current_frame()
            
    def prev_frame(self):
        """Go to previous frame"""
        if self.frames and self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.display_current_frame()
            
    def next_frame(self):
        """Go to next frame"""
        if self.frames and self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            self.display_current_frame()

    def _start_continuous_nav(self, direction):
        """Start continuous frame navigation on button hold"""
        # Cancel any existing navigation
        self._stop_continuous_nav()

        # Record start time for threshold detection
        self._nav_start_time = time.time()
        self._nav_direction = direction

        # Initial single frame move (immediate response)
        if direction == "prev":
            self.prev_frame()
        else:
            self.next_frame()

        # Schedule check for continuous navigation after threshold
        self._nav_check_id = self.root.after(300, self._check_continuous_nav)

    def _check_continuous_nav(self):
        """Check if button is still held and start continuous navigation"""
        # If we get here, button has been held for 300ms - start continuous mode
        self._continuous_nav()

    def _continuous_nav(self):
        """Continuously navigate frames while button is held"""
        if not hasattr(self, '_nav_direction'):
            return

        # Move to next/prev frame
        if self._nav_direction == "prev":
            if self.frames and self.current_frame_idx > 0:
                self.current_frame_idx -= 1
                self.display_current_frame()
        else:  # next
            if self.frames and self.current_frame_idx < len(self.frames) - 1:
                self.current_frame_idx += 1
                self.display_current_frame()

        # Schedule next move (faster rate during continuous nav)
        self._nav_repeat_id = self.root.after(50, self._continuous_nav)

    def _stop_continuous_nav(self):
        """Stop continuous frame navigation"""
        # Cancel pending navigation callbacks
        if hasattr(self, '_nav_check_id'):
            self.root.after_cancel(self._nav_check_id)
            delattr(self, '_nav_check_id')

        if hasattr(self, '_nav_repeat_id'):
            self.root.after_cancel(self._nav_repeat_id)
            delattr(self, '_nav_repeat_id')

        if hasattr(self, '_nav_direction'):
            delattr(self, '_nav_direction')

        if hasattr(self, '_nav_start_time'):
            delattr(self, '_nav_start_time')

    def reset_video(self):
        """Reset to first frame"""
        if self.frames:
            self.current_frame_idx = 0
            self.display_current_frame()

    def jump_to_prev_annotated_frame(self):
        """Jump to the previous frame with annotations for the current object"""
        # Get frames with annotations for current object only
        current_obj_frames = set()
        for _, _, _, obj_id, frame_idx in self.click_points:
            if obj_id == self.current_object_id:
                current_obj_frames.add(frame_idx)
        
        if not current_obj_frames:
            obj_name = self.object_names.get(self.current_object_id, f"Object_{self.current_object_id}")
            messagebox.showinfo("No Annotations", f"No annotated frames found for {obj_name}.")
            return
        
        sorted_annotated = sorted(list(current_obj_frames))
        prev_frames = [f for f in sorted_annotated if f < self.current_frame_idx]
        
        if prev_frames:
            self.current_frame_idx = prev_frames[-1]
            self.display_current_frame()
        else:
            self.current_frame_idx = sorted_annotated[-1]
            self.display_current_frame()
            obj_name = self.object_names.get(self.current_object_id, f"Object_{self.current_object_id}")
            messagebox.showinfo("Jump to Annotation", f"Wrapped to last annotated frame for {obj_name}")
    
    def jump_to_next_annotated_frame(self):
        """Jump to the next frame with annotations for the current object"""
        # Get frames with annotations for current object only
        current_obj_frames = set()
        for _, _, _, obj_id, frame_idx in self.click_points:
            if obj_id == self.current_object_id:
                current_obj_frames.add(frame_idx)
        
        if not current_obj_frames:
            obj_name = self.object_names.get(self.current_object_id, f"Object_{self.current_object_id}")
            messagebox.showinfo("No Annotations", f"No annotated frames found for {obj_name}.")
            return
        
        sorted_annotated = sorted(list(current_obj_frames))
        next_frames = [f for f in sorted_annotated if f > self.current_frame_idx]
        
        if next_frames:
            self.current_frame_idx = next_frames[0]
            self.display_current_frame()
        else:
            self.current_frame_idx = sorted_annotated[0]
            self.display_current_frame()
            obj_name = self.object_names.get(self.current_object_id, f"Object_{self.current_object_id}")
            messagebox.showinfo("Jump to Annotation", f"Wrapped to first annotated frame for {obj_name}")
            
    def on_slider_change(self, value):
        """Handle frame slider change"""
        if self.frames and not self.playing:
            self.current_frame_idx = int(float(value))
            self.display_current_frame()
            
    def toggle_play(self):
        """Toggle video playback"""
        if not self.frames:
            return
            
        self.playing = not self.playing
        if self.playing:
            self.play_button.config(text="Pause")
            threading.Thread(target=self.play_video, daemon=True).start()
        else:
            self.play_button.config(text="Play")
            
    def play_video(self):
        """Play video in separate thread"""
        while self.playing and self.frames:
            if self.current_frame_idx < len(self.frames) - 1:
                self.current_frame_idx += 1
                self.root.after(0, self.display_current_frame)
                threading.Event().wait(0.033)  # ~30 FPS
            else:
                self.playing = False
                self.root.after(0, lambda: self.play_button.config(text="Play"))
                break
    
    def _detect_available_gpus(self):
        """Detect available GPUs and return list of options"""
        gpu_options = []
        
        try:
            if torch and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_options.append("auto")  # Let PyTorch choose
                gpu_options.append("cpu")   # Force CPU
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                    gpu_options.append(f"cuda:{i} ({gpu_name} - {gpu_memory:.1f}GB)")
            else:
                gpu_options.append("cpu")
                gpu_options.append("auto")
        except Exception as e:
            print(f"Error detecting GPUs: {e}")
            gpu_options = ["cpu", "auto"]
        
        return gpu_options
    
    def _get_selected_device(self):
        """Get the selected device string and convert to PyTorch device"""
        selected = self.selected_gpu.get()

        if selected == "auto":
            if torch and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        elif selected == "cpu":
            return "cpu"
        elif "cuda:" in selected:
            # Extract cuda device from display string like "cuda:0 (GPU Name - 24.0GB)"
            # This handles both plain "cuda:0" and formatted "cuda:0 (...)" strings
            device_part = selected.split("cuda:")[1].split(" ")[0]
            return f"cuda:{device_part}"
        else:
            return "cpu"
    
    def on_gpu_selection_change(self, event=None):
        """Handle GPU selection change"""
        device = self._get_selected_device()
        self.gpu_device = device
        self._update_gpu_info_display()
        
        # Update status
        self.status_label.config(text=f"GPU selection changed to: {device}")
        
        # If model is already loaded, warn user that they need to reload
        if hasattr(self, 'sam2_model') and self.sam2_model is not None:
            messagebox.showwarning(
                "Model Reload Required",
                f"GPU selection changed to: {device}\n\n"
                f"The SAM2 model is currently loaded on a different device.\n"
                f"Please reload the model to use the new GPU selection."
            )
    
    def _update_gpu_info_display(self):
        """Update the GPU info display"""
        try:
            device = self._get_selected_device()
            
            if device == "cpu":
                info_text = "Using CPU (slower but works on any system)"
            elif device == "cuda":
                if torch and torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    info_text = f"Auto-selected: {gpu_name} ({gpu_memory:.1f}GB)"
                else:
                    info_text = "Auto-selected: CPU (CUDA not available)"
            elif device.startswith("cuda:"):
                gpu_id = int(device.split(":")[1])
                if torch and torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                    info_text = f"Selected: {gpu_name} ({gpu_memory:.1f}GB)"
                else:
                    info_text = f"Selected: GPU {gpu_id} (not available)"
            else:
                info_text = f"Selected: {device}"
                
            self.gpu_info_label.config(text=info_text)
            
        except Exception as e:
            self.gpu_info_label.config(text=f"Error: {str(e)}")

    def _detect_available_models(self):
        """Detect available SAM2 models from checkpoints"""
        from pathlib import Path

        checkpoint_dir = Path(self.checkpoint_dir)
        if not checkpoint_dir.exists():
            return ["auto"]

        models = []

        # Model mapping: checkpoint filename -> (display name, config path)
        model_mapping = {
            # SAM2.1 models (preferred)
            "sam2.1_hiera_tiny.pt": ("SAM2.1 Tiny (fastest)", "sam2.1/sam2.1_hiera_t.yaml"),
            "sam2.1_hiera_small.pt": ("SAM2.1 Small", "sam2.1/sam2.1_hiera_s.yaml"),
            "sam2.1_hiera_base_plus.pt": ("SAM2.1 Base+", "sam2.1/sam2.1_hiera_b+.yaml"),
            "sam2.1_hiera_large.pt": ("SAM2.1 Large (best)", "sam2.1/sam2.1_hiera_l.yaml"),

            # SAM2 legacy models
            "sam2_hiera_tiny.pt": ("SAM2 Tiny", "sam2/sam2_hiera_t.yaml"),
            "sam2_hiera_small.pt": ("SAM2 Small", "sam2/sam2_hiera_s.yaml"),
            "sam2_hiera_base_plus.pt": ("SAM2 Base+", "sam2/sam2_hiera_b+.yaml"),
            "sam2_hiera_large.pt": ("SAM2 Large", "sam2/sam2_hiera_l.yaml"),
        }

        models.append("auto")  # Auto-selection option

        for checkpoint_file, (display_name, config_path) in model_mapping.items():
            checkpoint_path = checkpoint_dir / checkpoint_file
            full_config_path = Path(self.config_dir) / config_path

            if checkpoint_path.exists() and full_config_path.exists():
                models.append(f"{display_name}|{checkpoint_file}|{config_path}")

        return models if len(models) > 1 else ["auto"]

    def _auto_select_best_model(self):
        """Auto-select best available model based on GPU memory"""
        models = self.available_models[1:]  # Skip "auto" option

        if not models:
            return None

        # Preference: SAM2.1 models first, then larger models for better quality
        # Adjust based on available GPU memory
        preference_order = [
            # SAM2.1 models (preferred)
            "sam2.1_hiera_large.pt",       # Best quality
            "sam2.1_hiera_base_plus.pt",   # High quality
            "sam2.1_hiera_small.pt",       # Good balance
            "sam2.1_hiera_tiny.pt",        # Fastest
            # SAM2 legacy models (fallback)
            "sam2_hiera_large.pt",
            "sam2_hiera_base_plus.pt",
            "sam2_hiera_small.pt",
            "sam2_hiera_tiny.pt",
        ]

        # Check GPU memory and adjust preference
        try:
            if torch and torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                # Adjust based on GPU memory
                if gpu_mem_gb >= 8:
                    # Prefer large models (best quality)
                    pass  # Keep original order
                elif gpu_mem_gb >= 4:
                    # Prefer base+ and small models
                    preference_order = [p for p in preference_order if 'large' not in p]
                else:
                    # Low memory: prefer small and tiny models
                    preference_order = [p for p in preference_order if 'large' not in p and 'base' not in p]
        except:
            # Default to base+ if can't detect GPU
            preference_order = [p for p in preference_order if 'large' not in p]

        # Find first available model in preference order
        for preferred in preference_order:
            for model in models:
                if preferred in model:
                    return model

        # Fallback to first available
        return models[0]

    def _format_model_list(self):
        """Format model list for display in combobox"""
        formatted = ["Auto-select best model"]
        for model in self.available_models[1:]:  # Skip first "auto"
            display_name = model.split('|')[0]
            formatted.append(display_name)
        return formatted

    def on_model_type_change(self):
        """Handle model type change between SAM2 and SAM3"""
        new_type = self.model_type_var.get()

        # Warn if model is already loaded
        if self.model_loaded:
            result = messagebox.askyesno(
                "Change Model Type",
                f"Switching to {new_type} will clear the current model and all annotations.\n\n"
                "Do you want to continue?",
                icon='warning'
            )
            if not result:
                # Revert to previous selection
                old_type = "SAM3" if new_type == "SAM2" else "SAM2"
                self.model_type_var.set(old_type)
                return

            # Clear current model
            self.sam2_model = None
            self.model_loaded = False
            self.using_sam3 = False
            self.inference_state = None
            self.current_model_info = None
            self.model_status_label.config(text="No model loaded", foreground='red')

        # Update model dropdown to show appropriate models
        # For SAM3, we'll keep the same model detection since both use similar checkpoint structure
        # The difference is in which builder we use during loading
        status_msg = f"{new_type} selected. Load a model to begin."
        self.status_label.config(text=status_msg)

    def on_model_selection_change(self, event=None):
        """Handle model selection change"""
        # Get selected display name
        selected_display = self.model_combo.get()

        if selected_display == "Auto-select best model":
            self.selected_model.set("auto")
        else:
            # Find corresponding model info
            for model in self.available_models[1:]:
                if model.startswith(selected_display + "|"):
                    self.selected_model.set(model)
                    break

        # Warn user if model already loaded
        if hasattr(self, 'sam2_model') and self.sam2_model is not None:
            response = messagebox.askyesno(
                "Model Change",
                "Changing the model requires reloading.\n\n"
                "This will clear current segmentation state.\n\n"
                "Continue?",
                icon='warning'
            )
            if response:
                self.load_sam2_model()
            else:
                # Revert selection
                if self.current_model_info:
                    display_name = self.current_model_info.split('|')[0]
                    self.model_combo.set(display_name)

    def _ensure_model_dtype_consistency(self):
        """Ensure model is using consistent dtypes to avoid BFloat16/Float mismatches"""
        try:
            if hasattr(self, 'sam2_model') and self.sam2_model is not None:
                # Set model to float32 precision
                if hasattr(self.sam2_model, 'model'):
                    self.sam2_model.model = self.sam2_model.model.to(dtype=torch.float32)
                    self.sam2_model.model.eval()
                
                # Force all model parameters to float32
                if hasattr(self.sam2_model, 'model'):
                    for param in self.sam2_model.model.parameters():
                        param.data = param.data.to(dtype=torch.float32)
                
                # Disable autocast to prevent mixed precision issues
                if hasattr(torch, 'autocast'):
                    # This will be handled in the inference calls
                    pass
                    
        except Exception as e:
            print(f"Warning: Could not ensure dtype consistency: {e}")
    
    def _disable_autocast_for_inference(self):
        """Disable autocast during inference to prevent dtype mismatches"""
        try:
            # Get the device type for autocast
            device = self._get_selected_device()
            device_type = 'cuda' if 'cuda' in device else 'cpu'
            
            # Create a context manager that disables autocast
            if hasattr(torch, 'autocast'):
                # In PyTorch 1.10+, use torch.autocast with device_type
                return torch.autocast(device_type=device_type, enabled=False)
            elif hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                # Fallback for older PyTorch versions with CUDA autocast
                return torch.cuda.amp.autocast(enabled=False)
            else:
                # Fallback to no_grad
                return torch.no_grad()
        except Exception as e:
            print(f"Warning: Could not create autocast context: {e}")
            # Return a simple no_grad context as fallback
            return torch.no_grad()
    
    def _force_model_float32(self):
        """Force all model components to float32 to prevent dtype mismatches"""
        try:
            if hasattr(self, 'sam2_model') and self.sam2_model is not None:
                # Force all model parameters to float32
                if hasattr(self.sam2_model, 'model'):
                    for name, param in self.sam2_model.model.named_parameters():
                        if param.dtype != torch.float32:
                            param.data = param.data.to(dtype=torch.float32)
                    
                    # Also force buffers to float32
                    for name, buffer in self.sam2_model.model.named_buffers():
                        if buffer.dtype != torch.float32:
                            buffer.data = buffer.data.to(dtype=torch.float32)
                
                # Force any other model components
                for attr_name in dir(self.sam2_model):
                    if not attr_name.startswith('_'):
                        attr = getattr(self.sam2_model, attr_name)
                        if hasattr(attr, 'to') and hasattr(attr, 'dtype'):
                            try:
                                if attr.dtype != torch.float32:
                                    setattr(self.sam2_model, attr_name, attr.to(dtype=torch.float32))
                            except:
                                pass  # Skip if not convertible
                
                # Force the entire model to float32 recursively
                self._recursive_float32_conversion(self.sam2_model)
                                
        except Exception as e:
            print(f"Warning: Could not force model to float32: {e}")
    
    def _recursive_float32_conversion(self, obj):
        """Recursively convert all tensors in an object to float32"""
        try:
            if hasattr(obj, 'to') and hasattr(obj, 'dtype'):
                if obj.dtype != torch.float32:
                    obj.data = obj.data.to(dtype=torch.float32)
            elif hasattr(obj, '__dict__'):
                for attr_name, attr_value in obj.__dict__.items():
                    if not attr_name.startswith('_'):
                        self._recursive_float32_conversion(attr_value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    self._recursive_float32_conversion(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    self._recursive_float32_conversion(value)
        except Exception as e:
            pass  # Skip if conversion fails
    
    def _patch_model_for_float32(self):
        """Patch the model to force float32 operations"""
        try:
            if hasattr(self, 'sam2_model') and self.sam2_model is not None:
                # Patch the model's forward method to force float32
                if hasattr(self.sam2_model, 'model'):
                    original_forward = self.sam2_model.model.forward
                    
                    def float32_forward(*args, **kwargs):
                        # Convert all tensor inputs to float32
                        new_args = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor):
                                if arg.dtype != torch.float32:
                                    new_args.append(arg.to(dtype=torch.float32))
                                else:
                                    new_args.append(arg)
                            else:
                                new_args.append(arg)
                        
                        # Convert tensor kwargs to float32
                        new_kwargs = {}
                        for key, value in kwargs.items():
                            if isinstance(value, torch.Tensor):
                                if value.dtype != torch.float32:
                                    new_kwargs[key] = value.to(dtype=torch.float32)
                                else:
                                    new_kwargs[key] = value
                            else:
                                new_kwargs[key] = value
                        
                        # Call original forward with float32 tensors
                        return original_forward(*new_args, **new_kwargs)
                    
                    # Replace the forward method
                    self.sam2_model.model.forward = float32_forward
                    
                    print("Model forward method patched for float32")
                    
        except Exception as e:
            print(f"Warning: Could not patch model for float32: {e}")
    
    def _disable_mixed_precision_globally(self):
        """Disable mixed precision globally to prevent dtype issues"""
        try:
            # Set environment variables to disable mixed precision
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Disable autocast globally
            if hasattr(torch, 'autocast'):
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
                
            print("Mixed precision disabled globally")
            
        except Exception as e:
            print(f"Warning: Could not disable mixed precision globally: {e}")
    
    def _force_float32_context(self):
        """Context manager that forces all operations to float32"""
        class Float32Context:
            def __init__(self, parent):
                self.parent = parent
                self.original_autocast = None
                
            def __enter__(self):
                # Disable autocast completely
                if hasattr(torch, 'autocast'):
                    self.original_autocast = torch.autocast(enabled=False)
                    self.original_autocast.__enter__()
                
                # Force model to float32
                self.parent._force_model_float32()
                
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Restore autocast
                if self.original_autocast:
                    self.original_autocast.__exit__(exc_type, exc_val, exc_tb)
        
        return Float32Context(self)
    
    def show_frame_points(self):
        """Show a list of annotation points on the current frame"""
        if not self.click_points:
            messagebox.showinfo("No Points", "No annotation points found on any frame.")
            return
        
        # Get points on current frame
        current_frame_points = [point for point in self.click_points 
                              if point[4] == self.current_frame_idx]
        
        if not current_frame_points:
            messagebox.showinfo("No Points", f"No annotation points found on frame {self.current_frame_idx + 1}.")
            return
        
        # Create a simple dialog showing the points
        points_info = f"Annotation points on frame {self.current_frame_idx + 1}:\n\n"
        
        for i, (x, y, is_pos, obj_id, frame_idx) in enumerate(current_frame_points):
            point_type = "Positive" if is_pos else "Negative"
            obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
            points_info += f"{i+1}. {point_type} point for {obj_name} at ({x:.0f}, {y:.0f})\n"
        
        points_info += f"\nTotal: {len(current_frame_points)} points on this frame"
        points_info += f"\nTotal: {len(self.click_points)} points on all frames"
        
        messagebox.showinfo("Frame Points", points_info)
    
    def _prepare_tensors_for_inference(self, points, labels):
        """Prepare points and labels tensors with consistent dtype and device"""
        try:
            device = self._get_selected_device()
            
            # Convert to tensors with consistent dtype
            if isinstance(points, (list, np.ndarray)):
                points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
            else:
                points_tensor = points.to(dtype=torch.float32, device=device)
                
            if isinstance(labels, (list, np.ndarray)):
                labels_tensor = torch.tensor(labels, dtype=torch.int64, device=device)
            else:
                labels_tensor = labels.to(dtype=torch.int64, device=device)
            
            # Ensure tensors are contiguous and have correct shape
            points_tensor = points_tensor.contiguous()
            labels_tensor = labels_tensor.contiguous()
            
            # Ensure points are 2D (N, 2) and labels are 1D (N,)
            if len(points_tensor.shape) == 1:
                points_tensor = points_tensor.unsqueeze(0)
            if len(labels_tensor.shape) == 0:
                labels_tensor = labels_tensor.unsqueeze(0)
            
            return points_tensor, labels_tensor
            
        except Exception as e:
            print(f"Error preparing tensors: {e}")
            # Fallback to original values
            return points, labels
    
    def _debug_dtype_mismatch(self, points, labels, frame_idx, obj_id):
        """Debug method to identify dtype mismatches"""
        try:
            print(f"Debug: Processing frame {frame_idx}, object {obj_id}")
            print(f"  Points type: {type(points)}, shape: {getattr(points, 'shape', 'N/A')}")
            print(f"  Labels type: {type(labels)}, shape: {getattr(labels, 'shape', 'N/A')}")
            
            if hasattr(points, 'dtype'):
                print(f"  Points dtype: {points.dtype}")
            if hasattr(labels, 'dtype'):
                print(f"  Labels dtype: {labels.dtype}")
                
            # Check model dtype
            if hasattr(self, 'sam2_model') and hasattr(self.sam2_model, 'model'):
                for name, param in self.sam2_model.model.named_parameters():
                    if param.dtype != torch.float32:
                        print(f"  WARNING: Model parameter {name} has dtype {param.dtype}, not float32")
                        break
                        
        except Exception as e:
            print(f"Debug error: {e}")

    def segment_video(self):
        """Segment video using SAM2 model"""
        if not self.frames:
            messagebox.showwarning("Warning", "Please load a video first")
            return
            
        if not self.model_loaded or not self.sam2_model:
            messagebox.showwarning("Warning", "Please load SAM2 model first")
            return
            
        if not self.click_points:
            messagebox.showwarning("Warning", "Please add some click points first")
            return
        
        # Validate annotations
        is_valid, error_msg = self._validate_annotations_before_segmentation()
        if not is_valid:
            messagebox.showerror("Invalid Annotations", 
                            f"{error_msg}\n\n"
                            f"This usually happens when:\n"
                            f"1. Annotations were saved with different video settings\n"
                            f"2. Video was reloaded with frame skipping enabled\n\n"
                            f"Please reload the video with the same settings used when creating annotations, "
                            f"or create new annotations for the current video.")
            return
        
        # Ask for output directory for auto-export
        export_dir_choice = filedialog.askdirectory(
            title="Select Output Directory for Segmentation Results (Cancel to abort)",
            initialdir=os.path.expanduser("~")
        )

        if not export_dir_choice:
            # User cancelled - abort segmentation
            return

        # Store export directory for auto-export after segmentation
        self.auto_export_after_segmentation = True
        self.auto_export_directory = export_dir_choice
            
        try:
            self.status_label.config(text="Preparing frames for SAM2...")

            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(0)
            self.root.update()
            
            # Create temporary directory for frames
            temp_dir = tempfile.mkdtemp(prefix='sam2_frames_')

            # Create temporary directory for mask storage (always enabled for memory efficiency)
            export_dir = tempfile.mkdtemp(prefix="sam2_masks_")
            print(f"Temporary mask storage: {export_dir}")

            # Check if we need to use original video for re-segmentation
            segmentation_video_path = None
            if self.segmented_video_displayed:
                segmentation_video_path = self._get_original_video_for_resegmentation()
                if not segmentation_video_path:
                    return  # User cancelled

            try:
                # Determine which frames to save based on processing range
                if self.limit_to_range_var.get():
                    start_idx = max(0, min(self.range_start_var.get(), len(self.frames)-1))
                    end_idx = max(0, min(self.range_end_var.get(), len(self.frames)-1))
                    if end_idx < start_idx:
                        start_idx, end_idx = end_idx, start_idx
                    frames_to_save = list(range(start_idx, end_idx + 1))
                    self.status_label.config(text=f"Saving frames {start_idx+1}-{end_idx+1} for processing...")
                    print(f"Processing limited range: frames {start_idx+1} to {end_idx+1} (total: {len(frames_to_save)} frames)")
                else:
                    frames_to_save = list(range(len(self.frames)))
                    print(f"Processing full video: {len(frames_to_save)} frames")
                
                # Save only the frames we need to process
                if segmentation_video_path:
                    # Re-segmentation: Extract frames from original video
                    print(f"Re-segmentation mode: Extracting frames from original video")
                    cap = cv2.VideoCapture(segmentation_video_path)

                    for save_idx, frame_idx in enumerate(frames_to_save):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame_bgr = cap.read()
                        if not ret:
                            print(f"Warning: Could not read frame {frame_idx} from original video, skipping...")
                            continue

                        frame_path = os.path.join(temp_dir, f"{save_idx:05d}.jpg")
                        cv2.imwrite(frame_path, frame_bgr)

                        progress = (save_idx / len(frames_to_save)) * 30
                        self.progress_var.set(progress)
                        if save_idx % 10 == 0:
                            self.root.update_idletasks()

                    cap.release()
                else:
                    # Normal segmentation: Use loaded frames
                    for save_idx, frame_idx in enumerate(frames_to_save):
                        # Handle lazy loading - load frame on demand if needed
                        if self.lazy_load_var.get() and hasattr(self, 'video_props'):
                            frame = self._load_frame_lazy(frame_idx)
                            if frame is None:
                                # Skip this frame if it can't be loaded
                                print(f"Warning: Could not load frame {frame_idx}, skipping...")
                                continue
                        else:
                            frame = self.frames[frame_idx]
                            if frame is None:
                                print(f"Warning: Frame {frame_idx} is None, skipping...")
                                continue

                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame_path = os.path.join(temp_dir, f"{save_idx:05d}.jpg")
                        cv2.imwrite(frame_path, frame_bgr)
                    
                    progress = (save_idx / len(frames_to_save)) * 30
                    self.progress_var.set(progress)
                    if save_idx % 10 == 0:
                        self.root.update_idletasks()
                
                self.status_label.config(text="Initializing SAM2 inference...")
                self.progress_var.set(35)
                self.root.update()
                
                # Always reset inference state to clear dimension cache
                # This ensures SAM2 uses current frame dimensions, not cached ones
                if hasattr(self, 'inference_state') and self.inference_state is not None:
                    # Reset existing state
                    try:
                        self.sam2_model.reset_state(self.inference_state)
                    except:
                        pass  # If reset_state doesn't exist, just clear the reference
                    self.inference_state = None

                # Initialize fresh state with current frames
                # Enable memory optimizations (always on)
                self.inference_state = self.sam2_model.init_state(
                    video_path=temp_dir,
                    offload_video_to_cpu=True,  # Reduce GPU memory usage
                    offload_state_to_cpu=True   # Offload model state to CPU when not in use
                )
                print("Memory optimizations enabled: CPU offloading")

                # Calculate coordinate scaling if re-segmenting
                coord_scale_x = 1.0
                coord_scale_y = 1.0
                if segmentation_video_path:
                    cap_temp = cv2.VideoCapture(segmentation_video_path)
                    orig_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
                    orig_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap_temp.release()

                    seg_height, seg_width = self.frames[0].shape[:2]
                    coord_scale_x = orig_width / seg_width
                    coord_scale_y = orig_height / seg_height

                    print(f"Re-segmentation coordinate scaling: {coord_scale_x:.3f}x (width), {coord_scale_y:.3f}x (height)")
                    print(f"  Segmented video: {seg_width}x{seg_height}")
                    print(f"  Original video: {orig_width}x{orig_height}")

                # Group click points by frame and by object ID
                points_by_frame_and_object = {}
                for x, y, is_pos, obj_id, frame_idx in self.click_points:
                    # Map original frame index to limited video index
                    if self.limit_to_range_var.get():
                        if frame_idx not in frames_to_save:
                            continue  # Skip points outside the processing range
                        limited_frame_idx = frames_to_save.index(frame_idx)
                    else:
                        limited_frame_idx = frame_idx
                    
                    # Handle coordinate scaling for re-segmentation
                    if coord_scale_x != 1.0 or coord_scale_y != 1.0:
                        # Re-segmentation: Scale from segmented video dimensions to original video dimensions
                        scaled_x = x * coord_scale_x
                        scaled_y = y * coord_scale_y
                    # ADDED: Convert from ORIGINAL coordinates to CURRENT frame coordinates
                    # for SAM2 processing
                    elif self.current_video_scale != 1.0 and self.original_video_width:
                        scaled_x = x * self.current_video_scale
                        scaled_y = y * self.current_video_scale
                    else:
                        scaled_x = x
                        scaled_y = y
                    
                    if limited_frame_idx not in points_by_frame_and_object:
                        points_by_frame_and_object[limited_frame_idx] = {}
                    if obj_id not in points_by_frame_and_object[limited_frame_idx]:
                        points_by_frame_and_object[limited_frame_idx][obj_id] = {'points': [], 'labels': []}
                    
                    # Use SCALED coordinates for SAM2 (matching current frame size)
                    points_by_frame_and_object[limited_frame_idx][obj_id]['points'].append([scaled_x, scaled_y])
                    points_by_frame_and_object[limited_frame_idx][obj_id]['labels'].append(1 if is_pos else 0)
                    
                    
                
                self.status_label.config(text="Adding prompts to SAM2...")
                self.progress_var.set(40)
                self.root.update()

                # Initialize metadata list instead of pre-allocating masks (memory optimization)
                mask_metadata = []
                
                # Determine which frames to annotate based on where points exist
                annotation_frames = sorted(points_by_frame_and_object.keys())
                # Use for metadata
                ann_frame_idx = annotation_frames[0] if annotation_frames else self.current_frame_idx
                self.ann_frame_idx = ann_frame_idx
                
                # Process each annotation frame and its objects
                for ann_frame in annotation_frames:
                    obj_dict = points_by_frame_and_object[ann_frame]
                    for obj_id, point_data in obj_dict.items():
                        points = np.array(point_data['points'], dtype=np.float32)
                        labels = np.array(point_data['labels'], dtype=np.int32)
                        
                        obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                        self.status_label.config(text=f"Processing {obj_name} on frame {ann_frame+1}...")
                        self.root.update()
                        
                        try:
                            # Get device
                            device = self._get_selected_device()
                            
                            # Convert to tensors with Float32 (autocast will handle BFloat16 conversion)
                            points_tensor = torch.from_numpy(points).to(dtype=torch.float32, device=device)
                            labels_tensor = torch.from_numpy(labels).to(dtype=torch.int64, device=device)
                            
                            # Make sure tensors are contiguous
                            points_tensor = points_tensor.contiguous()
                            labels_tensor = labels_tensor.contiguous()
                            
                            # Add points (autocast context handles dtype conversion automatically)
                            _ = self.sam2_model.add_new_points(
                                inference_state=self.inference_state,
                                frame_idx=ann_frame,
                                obj_id=obj_id,
                                points=points_tensor,
                                labels=labels_tensor,
                            )
                                    
                            print(f"Successfully added {len(points)} points for {obj_name} on frame {ann_frame}")
                            
                        except Exception as e:
                            error_msg = f"Error adding points for {obj_name} on frame {ann_frame}: {e}"
                            print(error_msg)
                            traceback.print_exc()
                            messagebox.showerror("Segmentation Error", error_msg)
                            return
                
                # Determine processing range
                if self.limit_to_range_var.get():
                    start_idx = max(0, min(self.range_start_var.get(), len(self.frames)-1))
                    end_idx = max(0, min(self.range_end_var.get(), len(self.frames)-1))
                    if end_idx < start_idx:
                        start_idx, end_idx = end_idx, start_idx
                    frames_to_process = list(range(start_idx, end_idx + 1))
                    self.status_label.config(text=f"Propagating through selected range {start_idx+1}-{end_idx+1}...")
                else:
                    self.status_label.config(text="Propagating through entire video...")
                    frames_to_process = list(range(len(self.frames)))
                
                # Store the processing range for later use
                self.processing_range = frames_to_process
                
                self.progress_var.set(45)
                self.root.update()
                
                processed_frames = 0
                
                try:
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(self.inference_state):
                        # Map limited video frame index back to original frame index
                        if self.limit_to_range_var.get():
                            if out_frame_idx >= len(frames_to_save):
                                continue
                            original_frame_idx = frames_to_save[out_frame_idx]
                        else:
                            original_frame_idx = out_frame_idx

                        # Only process frames that are in our target range
                        if original_frame_idx not in frames_to_process:
                            continue
                            
                        # Process each object mask
                        for i, out_obj_id in enumerate(out_obj_ids):
                            # Only keep masks for objects that were annotated anywhere
                            if any(out_obj_id in obj_dict for obj_dict in points_by_frame_and_object.values()):
                                mask_logits = out_mask_logits[i]
                                
                                # Convert from BFloat16 to Float32 only when moving to CPU for numpy
                                if hasattr(mask_logits, 'cpu'):
                                    # Convert to float32 for CPU operations (numpy doesn't support bfloat16)
                                    mask_logits = mask_logits.float().cpu()
                                if hasattr(mask_logits, 'numpy'):
                                    mask_logits = mask_logits.numpy()
                                
                                # Convert to binary mask
                                mask = (mask_logits > 0.0)

                                # Ensure mask is 2D
                                if len(mask.shape) > 2:
                                    mask = mask.squeeze()

                                # Always export to disk (streaming) for memory efficiency
                                mask_uint8 = (mask * 255).astype(np.uint8)
                                self._export_mask_to_disk(original_frame_idx, out_obj_id, mask_uint8, export_dir, mask_metadata)

                                # Explicitly delete tensors
                                del mask_logits
                                del mask

                        # Clean up old frames from cache to prevent memory growth
                        self._cleanup_frame_cache(out_frame_idx, self.inference_state)

                        # Monitor memory usage
                        self._monitor_memory(out_frame_idx, len(frame_indices))

                        processed_frames += 1
                        progress = 45 + (processed_frames / max(1, len(frames_to_process))) * 55
                        self.progress_var.set(min(progress, 100))
                        
                        if processed_frames % 10 == 0:
                            self.status_label.config(text=f"Processing frame {processed_frames}/{len(frames_to_process)}")
                            self.root.update_idletasks()
                            
                except Exception as e:
                    print(f"Error during forward propagation: {e}")
                    traceback.print_exc()
                    messagebox.showwarning("Propagation Warning",
                                        f"Encountered issue during forward propagation: {str(e)}")

                # BACKWARD PROPAGATION - if earliest annotation is not at frame 0
                earliest_frame = min(annotation_frames) if annotation_frames else 0

                if earliest_frame > 0:
                    self.status_label.config(text=f"Propagating backward from frame {earliest_frame+1}...")
                    self.root.update()

                    try:
                        backward_processed = 0
                        frames_to_backward = earliest_frame  # Number of frames before first annotation

                        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(
                            self.inference_state, reverse=True
                        ):
                            # Map limited video frame index back to original frame index
                            if self.limit_to_range_var.get():
                                if out_frame_idx >= len(frames_to_save):
                                    continue
                                original_frame_idx = frames_to_save[out_frame_idx]
                            else:
                                original_frame_idx = out_frame_idx

                            # Only process frames before the earliest annotation
                            if original_frame_idx >= earliest_frame:
                                continue

                            # Process each object mask
                            for i, out_obj_id in enumerate(out_obj_ids):
                                if any(out_obj_id in obj_dict for obj_dict in points_by_frame_and_object.values()):
                                    mask_logits = out_mask_logits[i]

                                    # Convert from BFloat16 to Float32 only when moving to CPU for numpy
                                    if hasattr(mask_logits, 'cpu'):
                                        mask_logits = mask_logits.float().cpu()
                                    if hasattr(mask_logits, 'numpy'):
                                        mask_logits = mask_logits.numpy()

                                    # Convert to binary mask
                                    mask = (mask_logits > 0.0)

                                    # Ensure mask is 2D
                                    if len(mask.shape) > 2:
                                        mask = mask.squeeze()

                                    # Always export to disk (streaming) for memory efficiency
                                    mask_uint8 = (mask * 255).astype(np.uint8)
                                    self._export_mask_to_disk(original_frame_idx, out_obj_id, mask_uint8, export_dir, mask_metadata)

                                    # Explicitly delete tensors
                                    del mask_logits
                                    del mask

                            # Clean up old frames from cache to prevent memory growth
                            self._cleanup_frame_cache(out_frame_idx, self.inference_state)

                            # Monitor memory usage
                            self._monitor_memory(out_frame_idx, earliest_frame)

                            backward_processed += 1
                            # Update progress (backward propagation gets remaining progress from 50% to 100%)
                            if frames_to_backward > 0:
                                backward_progress = 50 + (backward_processed / frames_to_backward) * 50
                                self.progress_var.set(min(backward_progress, 100))

                            if backward_processed % 10 == 0:
                                self.status_label.config(text=f"Backward: Processing frame {backward_processed}/{frames_to_backward}")
                                self.root.update_idletasks()

                            print(f"Backward propagation complete: processed {backward_processed} frames")

                    except Exception as e:
                        print(f"Error during backward propagation: {e}")
                        traceback.print_exc()
                        messagebox.showwarning("Backward Propagation Warning",
                                            f"Encountered issue during backward propagation: {str(e)}")

                # Save metadata and temporary directory
                metadata_path = os.path.join(export_dir, "segmentation_metadata.json")
                with open(metadata_path, 'w') as f:
                    import json
                    json.dump({
                        'video_path': self.video_path,
                        'total_frames': len(self.frames),
                        'objects': {obj_id: self.object_names.get(obj_id, f"Object {obj_id}")
                                   for obj_id in self.object_ids},
                        'masks': mask_metadata
                    }, f, indent=2)

                # Store export directory for later use (video export, cleanup)
                self.mask_export_dir = export_dir
                print(f"\nMasks saved to temporary directory: {export_dir}")
                print(f"Metadata saved to: {metadata_path}")

                # For backward compatibility, populate self.masks with empty dicts
                # (actual masks will be loaded from disk on demand)
                if not hasattr(self, 'masks'):
                    self.masks = {}
                for item in mask_metadata:
                    frame_idx = item['frame_idx']
                    obj_id = item['obj_id']
                    if frame_idx not in self.masks:
                        self.masks[frame_idx] = {}
                    # Store a placeholder - actual mask will be loaded from disk when needed
                    self.masks[frame_idx][obj_id] = None

                self.progress_bar.pack_forget()

                # Count results
                total_masks = sum(len(frame_masks) for frame_masks in self.masks.values())
                unique_objects = set()
                for frame_masks in self.masks.values():
                    unique_objects.update(frame_masks.keys())
                
                if total_masks > 0:
                    self.status_label.config(text=f"Segmentation complete! Generated {total_masks} masks for {len(unique_objects)} objects")
                    # Multi-frame annotation mode stays active for continued annotation

                    self.show_masks_var.set(True)
                    self.update_object_list()
                    self.display_current_frame()
                    
                    # Auto-export if requested
                    if getattr(self, 'auto_export_after_segmentation', False):
                        self._perform_auto_export_after_segmentation()
                    
                    messagebox.showinfo("Success", 
                                      f"Segmentation completed!\n"
                                      f"Objects: {len(unique_objects)}\n"
                                      f"Total masks: {total_masks}\n"
                                      f"Frames processed: {len(frames_to_process)}")
                else:
                    self.status_label.config(text="No masks generated")
                    messagebox.showwarning("Warning", "No masks were generated. Try different points.")
                
            finally:
                # Clean up
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Could not clean up temp directory: {e}")
                
        except Exception as e:
            self.progress_bar.pack_forget()
            self.status_label.config(text="Segmentation failed")
            traceback.print_exc()
            messagebox.showerror("Segmentation Error", f"Segmentation failed: {str(e)}")

    def _validate_annotations_before_segmentation(self):
        """Validate that annotations have valid frame indices for current video"""
        if not self.click_points:
            return False, "No annotation points found"

        invalid_points = []
        for i, (x, y, is_pos, obj_id, frame_idx) in enumerate(self.click_points):
            if frame_idx >= len(self.frames):
                invalid_points.append((i, frame_idx))

        if invalid_points:
            return False, f"Found {len(invalid_points)} annotations with invalid frame indices"

        return True, "All annotations valid"

    def _perform_auto_export_after_segmentation(self):
        """Auto-export masks and video to pre-selected directory after segmentation"""
        if not hasattr(self, 'auto_export_directory') or not self.auto_export_directory:
            print("WARNING: Auto-export requested but no directory specified")
            return

        export_dir = self.auto_export_directory

        # Create timestamped subdirectory to avoid overwriting
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(self.video_path))[0] if self.video_path else "video"
        export_subdir = os.path.join(export_dir, f"{video_name}_{timestamp}")
        os.makedirs(export_subdir, exist_ok=True)

        print(f"Auto-exporting results to: {export_subdir}")

        try:
            # Export masks
            masks_dir = os.path.join(export_subdir, "masks")
            os.makedirs(masks_dir, exist_ok=True)

            metadata_list = []
            total_exported = 0

            for frame_idx, frame_masks in self.masks.items():
                for obj_id in frame_masks.keys():
                    # Export mask using existing method
                    self._export_mask_to_disk(frame_idx, obj_id, frame_masks[obj_id], masks_dir, metadata_list)
                    total_exported += 1

            # Save metadata JSON
            metadata_path = os.path.join(export_subdir, "segmentation_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    "video_path": self.video_path,
                    "total_frames": len(self.frames),
                    "total_masks_exported": total_exported,
                    "objects": {obj_id: self.object_names.get(obj_id, f"Object_{obj_id}")
                               for obj_id in self.object_names.keys()},
                    "masks": metadata_list
                }, f, indent=2)

            # Export annotations JSON
            annotations_path = os.path.join(export_subdir, "annotations.json")
            annotations_data = {
                "video_path": self.video_path,
                "object_names": self.object_names,
                "object_colors": self.object_colors,
                "annotations": [
                    {
                        "frame_index": frame_idx,
                        "object_id": obj_id,
                        "x": x,
                        "y": y,
                        "is_positive": is_pos
                    }
                    for x, y, is_pos, obj_id, frame_idx in self.click_points
                ]
            }
            with open(annotations_path, 'w') as f:
                json.dump(annotations_data, f, indent=2)

            print(f"✅ Auto-export complete: {total_exported} masks exported to {export_subdir}")
            self.status_label.config(text=f"Auto-export complete: {total_exported} masks")

        except Exception as e:
            print(f"❌ Auto-export failed: {e}")
            traceback.print_exc()
            messagebox.showerror("Auto-Export Error", f"Failed to auto-export results: {str(e)}")

    def export_masks(self):
        """Export masks with enhanced metadata including object names"""
        if not self.masks:
            messagebox.showwarning("Warning", "No masks to export. Please segment the video first.")
            return
        
        try:
            self._start_foreground_mask_export()
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export masks: {str(e)}")
            print(f"Error starting foreground mask export: {e}")

    def _start_foreground_mask_export(self):
        """Export masks in foreground (blocking)"""
        if not self.masks:
            messagebox.showwarning("Warning", "No masks to export.")
            return
        
        # Get export directory
        export_dir = filedialog.askdirectory(title="Select Export Directory for Masks")
        if not export_dir:
            return

        # Check if masks already exist in the directory
        existing_masks = [f for f in os.listdir(export_dir) if f.startswith('mask_f') and f.endswith('.png')]
        if existing_masks:
            response = messagebox.askyesnocancel(
                "Masks Exist",
                f"Directory already contains {len(existing_masks)} mask files.\n\n"
                "Yes: Delete existing masks and export new ones\n"
                "No: Keep existing masks and add new ones\n"
                "Cancel: Abort export"
            )

            if response is None:  # Cancel
                return
            elif response:  # Yes - delete existing
                import shutil
                for mask_file in existing_masks:
                    try:
                        os.remove(os.path.join(export_dir, mask_file))
                    except Exception as e:
                        print(f"Warning: Could not delete {mask_file}: {e}")
                # Also delete metadata files if they exist
                for meta_file in ["segmentation_metadata.json", "object_mapping.csv"]:
                    meta_path = os.path.join(export_dir, meta_file)
                    if os.path.exists(meta_path):
                        try:
                            os.remove(meta_path)
                        except Exception as e:
                            print(f"Warning: Could not delete {meta_file}: {e}")
            # else: No - keep existing, just add new ones

        try:
            self.status_label.config(text="Exporting masks...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            
            # Export each frame's masks
            total_frames = len(self.masks)
            for idx, (frame_idx, frame_masks) in enumerate(self.masks.items()):
                for obj_id in frame_masks.keys():
                    # Load mask from disk (memory optimization)
                    mask = self._load_mask(frame_idx, obj_id)
                    if mask is None:
                        continue
                    obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                    # Use consolidated filename format: mask_f{frame_idx:06d}_{obj_name}_id{obj_id}.png
                    mask_filename = f"mask_f{frame_idx:06d}_{obj_name}_id{obj_id}.png"
                    mask_path = os.path.join(export_dir, mask_filename)
                    
                    progress = (exported_count / total_masks) * 90
                    self.progress_var.set(progress)
                    
                    if exported_count % 50 == 0:
                        self.root.update_idletasks()
            
            # Enhanced metadata export
            metadata = {
                "video_path": self.video_path,
                "total_frames": len(self.frames),
                "objects": {},
                "click_points_by_object": {},
                "prompt_frame": getattr(self, 'ann_frame_idx', self.current_frame_idx),
                "export_timestamp": str(__import__('datetime').datetime.now()),
                "sam2_info": {
                    "base_path": self.sam2_base_path,
                    "checkpoint_dir": self.checkpoint_dir,
                    "config_dir": self.config_dir
                }
            }
            
            # Object information with names and colors
            for obj_id in range(1, self.max_total_objects + 1):
                if any(obj_id in frame_masks for frame_masks in self.masks.values()):
                    mask_count = sum(1 for frame_masks in self.masks.values() if obj_id in frame_masks)
                    point_count = sum(1 for _, _, _, oid in self.click_points if oid == obj_id)
                    
                    metadata["objects"][obj_id] = {
                        "name": self.object_names.get(obj_id, f"Object_{obj_id}"),
                        "mask_count": mask_count,
                        "point_count": point_count,
                        "color": self.object_colors.get(obj_id, [255, 255, 255])
                    }
            
            # Click points grouped by object
            for x, y, is_pos, obj_id, frame_idx in self.click_points:
                if obj_id not in metadata["click_points_by_object"]:
                    metadata["click_points_by_object"][obj_id] = []
                metadata["click_points_by_object"][obj_id].append({
                    "x": float(x), 
                    "y": float(y), 
                    "positive": bool(is_pos),
                    "object_name": self.object_names.get(obj_id, f"Object_{obj_id}"),
                    "frame": int(frame_idx)
                })
            
            metadata_path = os.path.join(folder_path, "segmentation_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Also export object mapping CSV
            csv_path = os.path.join(folder_path, "object_mapping.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'name', 'mask_count', 'point_count', 'color_hex']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for obj_id, obj_info in metadata["objects"].items():
                    color_hex = self._rgb_to_hex(obj_info["color"])
                    writer.writerow({
                        'id': obj_id,
                        'name': obj_info['name'],
                        'mask_count': obj_info['mask_count'],
                        'point_count': obj_info['point_count'],
                        'color_hex': color_hex
                    })
            
            self.progress_var.set(100)
            self.progress_bar.pack_forget()
            self.status_label.config(text=f"Export complete: {exported_count} masks + metadata")
            
            # Show brief success message without dialog
            messagebox.showinfo("Export Complete", 
                              f"Successfully exported:\n"
                              f"- {exported_count} mask images (PNG)\n"
                              f"- 1 metadata file (JSON)\n"
                              f"- 1 object mapping (CSV)\n\n"
                              f"Location: {folder_path}")
                
        except Exception as e:
            self.progress_bar.pack_forget()
            self.status_label.config(text="Export failed")
            messagebox.showerror("Export Error", f"Failed to export masks: {str(e)}")
    
    def _export_masks_background(self, export_task):
        """Background mask export implementation"""
        try:
            folder_path = export_task['folder_path']
            masks = export_task['masks']
            object_names = export_task['object_names']
            object_colors = export_task['object_colors']
            click_points = export_task['click_points']
            video_path = export_task['video_path']
            frames_count = export_task['frames']
            ann_frame_idx = export_task['ann_frame_idx']
            sam2_info = export_task['sam2_info']
            
            exported_count = 0
            total_masks = sum(len(fm) for fm in masks.values())
            
            # Export masks with object names in filename
            for frame_idx, frame_masks in masks.items():
                for obj_id in frame_masks.keys():
                    # Load mask from disk (memory optimization)
                    mask = self._load_mask(frame_idx, obj_id)
                    if mask is None:
                        continue
                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                    # Use consolidated filename format: mask_f{frame_idx:06d}_{obj_name}_id{obj_id}.png
                    mask_path = os.path.join(folder_path, f"mask_f{frame_idx:06d}_{obj_name}_id{obj_id}.png")
                    cv2.imwrite(mask_path, mask)
                    exported_count += 1
            
            # Enhanced metadata export
            metadata = {
                "video_path": video_path,
                "total_frames": frames_count,
                "objects": {},
                "click_points_by_object": {},
                "prompt_frame": ann_frame_idx,
                "export_timestamp": str(__import__('datetime').datetime.now()),
                "sam2_info": sam2_info,
                "export_mode": "background"
            }
            
            # Object information with names and colors
            # Note: max_total_objects should be available in the export task context
            # For background export, we'll iterate through actual objects used
            used_obj_ids = set()
            for frame_masks in masks.values():
                used_obj_ids.update(frame_masks.keys())
            for point in click_points:
                if len(point) >= 4:
                    used_obj_ids.add(point[3])  # obj_id is at index 3
            
            for obj_id in sorted(used_obj_ids):
                if any(obj_id in frame_masks for frame_masks in masks.values()):
                    mask_count = sum(1 for frame_masks in masks.values() if obj_id in frame_masks)
                    point_count = sum(1 for point in click_points if len(point) >= 4 and point[3] == obj_id)
                    
                    metadata["objects"][obj_id] = {
                        "name": object_names.get(obj_id, f"Object_{obj_id}"),
                        "mask_count": mask_count,
                        "point_count": point_count,
                        "color": object_colors.get(obj_id, [255, 255, 255])
                    }
            
            # Click points grouped by object
            for x, y, is_pos, obj_id, frame_idx in click_points:
                if obj_id not in metadata["click_points_by_object"]:
                    metadata["click_points_by_object"][obj_id] = []
                metadata["click_points_by_object"][obj_id].append({
                    "x": float(x), 
                    "y": float(y), 
                    "positive": bool(is_pos),
                    "object_name": object_names.get(obj_id, f"Object_{obj_id}"),
                    "frame": int(frame_idx)
                })
            
            metadata_path = os.path.join(folder_path, "segmentation_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Also export object mapping CSV
            csv_path = os.path.join(folder_path, "object_mapping.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'name', 'mask_count', 'point_count', 'color_hex']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for obj_id, obj_info in metadata["objects"].items():
                    color_hex = self._rgb_to_hex(obj_info["color"])
                    writer.writerow({
                        'id': obj_id,
                        'name': obj_info['name'],
                        'mask_count': obj_info['mask_count'],
                        'point_count': obj_info['point_count'],
                        'color_hex': color_hex
                    })
            
            return {
                'success': True,
                'exported_count': exported_count,
                'folder_path': folder_path,
                'export_type': 'masks'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'export_type': 'masks'
            }

    def export_video(self):
        """Export video with mask overlays and enhanced object visualization"""
        if not self.frames:
            messagebox.showwarning("Warning", "No video loaded")
            return
            
        if not self.masks:
            messagebox.showwarning("Warning", "No masks to export. Please segment the video first.")
            return
        
        # Use checkbox setting to determine export mode
        if self.background_export_var.get():
            self._start_background_video_export()
        else:
            self._start_foreground_video_export()
    
    def _start_background_video_export(self):
        """Start background video export"""
        # Get export settings from user first
        export_dialog = tk.Toplevel(self.root)
        export_dialog.title("Background Video Export Settings")
        export_dialog.geometry("400x500")
        export_dialog.configure(bg='#2b2b2b')
        export_dialog.transient(self.root)
        export_dialog.grab_set()
        
        # Variables for export settings
        export_format = tk.StringVar(value="mp4")
        overlay_opacity = tk.DoubleVar(value=0.4)
        show_object_names = tk.BooleanVar(value=True)
        show_object_ids = tk.BooleanVar(value=False)
        show_boundaries = tk.BooleanVar(value=True)
        fps_var = tk.DoubleVar(value=30.0)
        quality_var = tk.StringVar(value="medium")
        export_mode = tk.StringVar(value="overlay")
        
        # UI Elements (simplified for background export)
        main_frame = ttk.Frame(export_dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(main_frame, text="Background Video Export", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 15))
        
        # Export mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Export Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(mode_frame, text="Original + Mask Overlay", 
                        variable=export_mode, value="overlay").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Masks Only (Black Background)", 
                        variable=export_mode, value="masks_only").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Side by Side", 
                        variable=export_mode, value="side_by_side").pack(anchor=tk.W)
        
        # Video settings
        video_frame = ttk.LabelFrame(main_frame, text="Video Settings", padding=10)
        video_frame.pack(fill=tk.X, pady=(0, 10))
        
        # FPS setting
        fps_frame = ttk.Frame(video_frame)
        fps_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        fps_spinbox = tk.Spinbox(fps_frame, from_=1, to=60, 
                                textvariable=fps_var, width=10,
                                bg='#404040', fg='white', insertbackground='white')
        fps_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Quality setting
        quality_frame = ttk.Frame(video_frame)
        quality_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(quality_frame, text="Quality:").pack(side=tk.LEFT)
        quality_combo = ttk.Combobox(quality_frame, textvariable=quality_var,
                                    values=["low", "medium", "high"], state="readonly", width=10)
        quality_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Format setting
        format_frame = ttk.Frame(video_frame)
        format_frame.pack(fill=tk.X)
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT)
        format_combo = ttk.Combobox(format_frame, textvariable=export_format,
                                   values=["mp4", "avi", "mov"], state="readonly", width=10)
        format_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def start_background_export():
            export_dialog.destroy()
            self._perform_background_video_export(export_format.get(), overlay_opacity.get(), 
                                                show_object_names.get(), show_object_ids.get(), 
                                                show_boundaries.get(), fps_var.get(), 
                                                quality_var.get(), export_mode.get())
            
        def cancel_export():
            export_dialog.destroy()
            
        ttk.Button(button_frame, text="Start Background Export", command=start_background_export).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Cancel", command=cancel_export).pack(side=tk.RIGHT)
        
        # Wait for dialog to close
        export_dialog.wait_window()
    
    def _start_foreground_video_export(self):
        """Start foreground video export (original behavior)"""
            
        # Get export settings from user
        export_dialog = tk.Toplevel(self.root)
        export_dialog.title("Export Video Settings")
        export_dialog.geometry("400x500")
        export_dialog.configure(bg='#2b2b2b')
        export_dialog.transient(self.root)
        export_dialog.grab_set()
        
        # Variables for export settings
        export_format = tk.StringVar(value="mp4")
        overlay_opacity = tk.DoubleVar(value=0.4)
        show_object_names = tk.BooleanVar(value=True)
        show_object_ids = tk.BooleanVar(value=False)
        show_boundaries = tk.BooleanVar(value=True)
        fps_var = tk.DoubleVar(value=30.0)
        quality_var = tk.StringVar(value="medium")
        
        # Export modes
        export_mode = tk.StringVar(value="overlay")
        
        # UI Elements
        main_frame = ttk.Frame(export_dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(main_frame, text="Video Export Settings", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 15))
        
        # Export mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Export Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(mode_frame, text="Original + Mask Overlay", 
                        variable=export_mode, value="overlay").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Masks Only (Black Background)", 
                        variable=export_mode, value="masks_only").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Side by Side", 
                        variable=export_mode, value="side_by_side").pack(anchor=tk.W)
        
        # Overlay settings
        overlay_frame = ttk.LabelFrame(main_frame, text="Overlay Settings", padding=10)
        overlay_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(overlay_frame, text="Mask Opacity:").pack(anchor=tk.W)
        opacity_scale = ttk.Scale(overlay_frame, from_=0.1, to=1.0, 
                                 variable=overlay_opacity, orient=tk.HORIZONTAL)
        opacity_scale.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Checkbutton(overlay_frame, text="Show Object Names", 
                        variable=show_object_names).pack(anchor=tk.W)
        ttk.Checkbutton(overlay_frame, text="Show Object IDs", 
                        variable=show_object_ids).pack(anchor=tk.W)
        ttk.Checkbutton(overlay_frame, text="Show Mask Boundaries", 
                        variable=show_boundaries).pack(anchor=tk.W)
        
        # Video settings
        video_frame = ttk.LabelFrame(main_frame, text="Video Settings", padding=10)
        video_frame.pack(fill=tk.X, pady=(0, 10))
        
        # FPS setting
        fps_frame = ttk.Frame(video_frame)
        fps_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        fps_spinbox = tk.Spinbox(fps_frame, from_=1, to=60, 
                                textvariable=fps_var, width=10,
                                bg='#404040', fg='white', insertbackground='white')
        fps_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Quality setting
        quality_frame = ttk.Frame(video_frame)
        quality_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(quality_frame, text="Quality:").pack(side=tk.LEFT)
        quality_combo = ttk.Combobox(quality_frame, textvariable=quality_var,
                                    values=["low", "medium", "high"], state="readonly", width=10)
        quality_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Format setting
        format_frame = ttk.Frame(video_frame)
        format_frame.pack(fill=tk.X)
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT)
        format_combo = ttk.Combobox(format_frame, textvariable=export_format,
                                   values=["mp4", "avi", "mov"], state="readonly", width=10)
        format_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def start_export():
            export_dialog.destroy()
            self._perform_video_export(export_format.get(), overlay_opacity.get(), 
                                      show_object_names.get(), show_object_ids.get(), 
                                      show_boundaries.get(), fps_var.get(), 
                                      quality_var.get(), export_mode.get())
            
        def cancel_export():
            export_dialog.destroy()
            
        ttk.Button(button_frame, text="Export", command=start_export).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Cancel", command=cancel_export).pack(side=tk.RIGHT)
        
        # Wait for dialog to close
        export_dialog.wait_window()

    def _perform_video_export(self, export_format, overlay_opacity, show_object_names, 
                             show_object_ids, show_boundaries, fps, quality, export_mode):
        """Perform the actual video export based on settings"""
        
        # Get output file path with folder creation
        file_path = self._get_export_file_path_with_creation(
            title="Save Video As",
            default_name=f"sam2_video.{export_format}",
            file_types=[
                (f"{export_format.upper()} files", f"*.{export_format}"),
                ("All files", "*.*")
            ],
            default_ext=f".{export_format}"
        )
        
        if not file_path:
            return
            
        try:
            self.status_label.config(text="Exporting video...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(0)
            self.root.update()
            
            # Setup video writer
            height, width = self.frames[0].shape[:2]
            
            # Adjust dimensions based on export mode
            if export_mode == "side_by_side":
                output_width = width * 2
                output_height = height
            else:
                output_width = width
                output_height = height
            
            # Video codec settings
            if export_format == "mp4":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif export_format == "avi":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            elif export_format == "mov":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            out = cv2.VideoWriter(file_path, fourcc, fps, 
                                (output_width, output_height))
            
            if not out.isOpened():
                raise ValueError("Could not open video writer")
            
            # Only export frames that were processed during segmentation
            if hasattr(self, 'processing_range') and self.processing_range:
                frames_to_export = self.processing_range
                total_frames = len(frames_to_export)
                self.status_label.config(text=f"Exporting {total_frames} processed frames...")
            else:
                frames_to_export = list(range(len(self.frames)))
                total_frames = len(self.frames)
            
            for export_idx, frame_idx in enumerate(frames_to_export):
                frame = self.frames[frame_idx]
                # Create output frame based on mode
                if export_mode == "overlay":
                    output_frame = self._create_overlay_frame(
                        frame, frame_idx, overlay_opacity,
                        show_object_names, show_object_ids,
                        show_boundaries
                    )
                elif export_mode == "masks_only":
                    output_frame = self._create_masks_only_frame(
                        frame, frame_idx, show_object_names,
                        show_object_ids, show_boundaries
                    )
                elif export_mode == "side_by_side":
                    output_frame = self._create_side_by_side_frame(
                        frame, frame_idx, overlay_opacity,
                        show_object_names, show_object_ids,
                        show_boundaries
                    )
                
                # Convert RGB to BGR for OpenCV
                if len(output_frame.shape) == 3:
                    output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                else:
                    output_frame_bgr = output_frame
                
                out.write(output_frame_bgr)
                
                # Update progress
                self.progress_var.set((idx + 1) / total_frames * 100)
                self.root.update()
            
            self.progress_bar.pack_forget()
            messagebox.showinfo("Export Complete", f"Masks exported to {export_dir}")
            self.status_label.config(text="Mask export complete")
            
            # Determine if this was a limited export
            if hasattr(self, 'processing_range') and self.processing_range and len(self.processing_range) < len(self.frames):
                range_info = f" (frames {min(self.processing_range)+1}-{max(self.processing_range)+1} of {len(self.frames)})"
            else:
                range_info = ""
            
            # Show brief success message without asking
            messagebox.showinfo("Export Complete", 
                              f"Video exported successfully!\n"
                              f"Location: {file_path}\n"
                              f"Frames: {total_frames}{range_info}\n"
                              f"FPS: {fps}\n"
                              f"Format: {export_format.upper()}")
                              
        except Exception as e:
            self.progress_bar.pack_forget()
            raise e
        
    def _get_export_file_path_with_creation(self, title, default_name, file_types, default_ext):
        """Get file path for export with directory creation support"""
        file_path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=default_ext,
            filetypes=file_types,
            initialfile=default_name
        )
        
        if file_path:
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(file_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
        
        return file_path

    def _rgb_to_hex(self, rgb):
        """Convert RGB color list to hex string"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def export_video(self):
        """Export segmented video with various options"""
        if not self.masks:
            messagebox.showwarning("Warning", "No masks to export. Please segment the video first.")
            return
            
        if not self.frames:
            messagebox.showwarning("Warning", "No video frames available.")
            return
        
        # Create export options dialog
        export_dialog = tk.Toplevel(self.root)
        export_dialog.title("Export Video Options")
        export_dialog.geometry("450x500")
        export_dialog.configure(bg='#2b2b2b')
        export_dialog.transient(self.root)
        export_dialog.grab_set()
        
        # Center the dialog
        export_dialog.update_idletasks()
        x = (export_dialog.winfo_screenwidth() // 2) - (export_dialog.winfo_width() // 2)
        y = (export_dialog.winfo_screenheight() // 2) - (export_dialog.winfo_height() // 2)
        export_dialog.geometry(f"+{x}+{y}")
        
        # Configure dialog style
        dialog_style = ttk.Style()
        dialog_style.theme_use('clam')
        dialog_style.configure('Dialog.TFrame', background='#2b2b2b')
        dialog_style.configure('Dialog.TLabel', background='#2b2b2b', foreground='white')
        dialog_style.configure('Dialog.TButton', background='#404040', foreground='white')
        
        main_frame = ttk.Frame(export_dialog, style='Dialog.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(main_frame, text="Video Export Options", 
                  font=('Arial', 14, 'bold'), style='Dialog.TLabel').pack(pady=(0, 20))
        
        # Export type selection
        export_type_var = tk.StringVar(value="overlay")
        
        ttk.Label(main_frame, text="Export Type:", font=('Arial', 10, 'bold'), 
                 style='Dialog.TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        type_frame = ttk.Frame(main_frame, style='Dialog.TFrame')
        type_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Radiobutton(type_frame, text="Original with mask overlay", 
                       variable=export_type_var, value="overlay").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(type_frame, text="Mask only (black background)", 
                       variable=export_type_var, value="mask_only").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(type_frame, text="Segmented object only", 
                       variable=export_type_var, value="object_only").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(type_frame, text="Side-by-side comparison", 
                       variable=export_type_var, value="side_by_side").pack(anchor=tk.W, pady=2)
        
        # FPS setting
        ttk.Label(main_frame, text="Video Quality:", font=('Arial', 10, 'bold'), 
                 style='Dialog.TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        quality_frame = ttk.Frame(main_frame, style='Dialog.TFrame')
        quality_frame.pack(fill=tk.X, pady=(0, 15))
        
        fps_var = tk.DoubleVar(value=30.0)
        ttk.Label(quality_frame, text="FPS:", style='Dialog.TLabel').pack(side=tk.LEFT)
        tk.Spinbox(quality_frame, from_=1, to=60, textvariable=fps_var, 
                  width=8, bg='#404040', fg='white', insertbackground='white').pack(side=tk.LEFT, padx=(5, 0))
        
        # Overlay transparency
        ttk.Label(main_frame, text="Overlay Settings:", font=('Arial', 10, 'bold'), 
                 style='Dialog.TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        overlay_frame = ttk.Frame(main_frame, style='Dialog.TFrame')
        overlay_frame.pack(fill=tk.X, pady=(0, 15))
        
        overlay_alpha_var = tk.DoubleVar(value=0.4)
        ttk.Label(overlay_frame, text="Transparency:", style='Dialog.TLabel').pack(anchor=tk.W)
        ttk.Scale(overlay_frame, from_=0.1, to=0.8, variable=overlay_alpha_var, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Object selection
        ttk.Label(main_frame, text="Objects to Export:", font=('Arial', 10, 'bold'), 
                 style='Dialog.TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        objects_container = ttk.Frame(main_frame, style='Dialog.TFrame')
        objects_container.pack(fill=tk.X, pady=(0, 20))
        
        # Get unique objects from masks
        unique_objects = set()
        for frame_masks in self.masks.values():
            unique_objects.update(frame_masks.keys())
        unique_objects = sorted(list(unique_objects))
        
        export_objects_vars = {}
        if unique_objects:
            for obj_id in unique_objects:
                export_objects_vars[obj_id] = tk.BooleanVar(value=True)
                color_hex = self._rgb_to_hex(self.object_colors.get(obj_id, [255, 255, 255]))
                
                obj_frame = ttk.Frame(objects_container, style='Dialog.TFrame')
                obj_frame.pack(anchor=tk.W, pady=1)
                
                ttk.Checkbutton(obj_frame, text=f"Object {obj_id}", 
                              variable=export_objects_vars[obj_id]).pack(side=tk.LEFT)
                
                ttk.Label(obj_frame, text="●", foreground=color_hex, 
                         font=('Arial', 12), style='Dialog.TLabel').pack(side=tk.LEFT, padx=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame, style='Dialog.TFrame')
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def start_export():
            selected_objects = [obj_id for obj_id, var in export_objects_vars.items() if var.get()]
            if not selected_objects and export_objects_vars:
                messagebox.showwarning("Warning", "Please select at least one object to export.")
                return
            
            export_dialog.destroy()
            self._export_video_with_options(
                export_type_var.get(),
                fps_var.get(),
                overlay_alpha_var.get(),
                selected_objects if export_objects_vars else []
            )
        
        ttk.Button(button_frame, text="Cancel", command=export_dialog.destroy, 
                  width=12).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Export Video", command=start_export, 
                  width=15).pack(side=tk.RIGHT, padx=(5, 5))
    
    def _export_video_with_options(self, export_type, fps, overlay_alpha, selected_objects):
        """Export video with specified options"""
        output_path = filedialog.asksaveasfilename(
            title="Save Video As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("All files", "*.*")
            ]
        )
        
        if not output_path:
            return

        # Check if video file already exists
        if os.path.exists(output_path):
            response = messagebox.askyesno(
                "Video Exists",
                f"Video file already exists:\n{os.path.basename(output_path)}\n\nOverwrite?",
                default=messagebox.NO
            )
            if not response:
                return

        try:
            self.status_label.config(text="Preparing video export...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(0)
            self.root.update()
            
            # Get dimensions
            height, width = self.frames[0].shape[:2]
            
            if export_type == "side_by_side":
                output_width = width * 2
                output_height = height
            else:
                output_width = width
                output_height = height
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
            
            if not out.isOpened():
                raise ValueError("Could not open video writer. Try a different format.")
            
            total_frames = len(self.frames)
            
            for frame_idx, frame in enumerate(self.frames):
                # Get masks for selected objects
                masks_dict = {}
                if frame_idx in self.masks:
                    for obj_id in self.masks[frame_idx].keys():
                        if obj_id in selected_objects:
                            # Load mask from disk (memory optimization)
                            mask = self._load_mask(frame_idx, obj_id)
                            if mask is not None:
                                masks_dict[obj_id] = mask
                
                # Create output frame based on type
                if export_type == "overlay":
                    output_frame = frame.copy()
                    for obj_id, mask in masks_dict.items():
                        color = self.object_colors.get(obj_id, [255, 255, 255])
                        colored_mask = np.zeros_like(frame)
                        colored_mask[mask > 0] = color
                        output_frame = cv2.addWeighted(output_frame, 1, colored_mask, overlay_alpha, 0)
                        
                elif export_type == "mask_only":
                    output_frame = np.zeros_like(frame)
                    for obj_id, mask in masks_dict.items():
                        color = self.object_colors.get(obj_id, [255, 255, 255])
                        output_frame[mask > 0] = color
                        
                elif export_type == "object_only":
                    output_frame = np.zeros_like(frame)
                    for obj_id, mask in masks_dict.items():
                        output_frame[mask > 0] = frame[mask > 0]
                        
                elif export_type == "side_by_side":
                    left = frame.copy()
                    for obj_id, mask in masks_dict.items():
                        color = self.object_colors.get(obj_id, [255, 255, 255])
                        colored_mask = np.zeros_like(frame)
                        colored_mask[mask > 0] = color
                        left = cv2.addWeighted(left, 1, colored_mask, overlay_alpha, 0)
                    output_frame = np.hstack([left, frame])
                else:
                    output_frame = frame.copy()
                
                # Convert RGB to BGR for OpenCV
                output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                out.write(output_frame_bgr)
                
                # Update progress
                progress = ((frame_idx + 1) / total_frames) * 100
                self.progress_var.set(progress)
                self.status_label.config(text=f"Exporting frame {frame_idx + 1}/{total_frames}")
                self.root.update()
            
            out.release()
            self.progress_bar.pack_forget()
            self.status_label.config(text="Video export complete!")
            
            messagebox.showinfo("Export Complete", f"Video exported to:\n{output_path}")
            
        except Exception as e:
            self.progress_bar.pack_forget()
            self.status_label.config(text="Export failed")
            messagebox.showerror("Export Error", f"Failed to export video: {str(e)}")
    def load_sam2_model(self):
        """Load SAM2 or SAM3 model with correct video predictor initialization"""
        try:
            # Determine which model type to load
            model_type = self.model_type_var.get() if self.sam3_available else "SAM2"
            self.status_label.config(text=f"Loading {model_type} model...")
            self.model_status_label.config(text="Loading...", foreground='orange')
            self.root.update()

            # Select and validate device FIRST (works for both SAM2 and SAM3)
            device = self._get_selected_device()

            # Validate device selection and provide feedback
            if device.startswith("cuda:"):
                gpu_id = int(device.split(":")[1])
                if not torch or not torch.cuda.is_available():
                    device = "cpu"
                    self.status_label.config(text="CUDA not available, using CPU...")
                elif gpu_id >= torch.cuda.device_count():
                    device = "cpu"
                    self.status_label.config(text=f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPU(s) found), using CPU...")
                else:
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    self.status_label.config(text=f"Using GPU {gpu_id} ({gpu_name}) for inference...")
            elif device == "cuda":
                if torch and torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    self.status_label.config(text=f"Using CUDA GPU ({gpu_name}) for inference...")
                else:
                    device = "cpu"
                    self.status_label.config(text="CUDA not available, using CPU...")
            else:  # cpu
                self.status_label.config(text="Using CPU for inference (slower)...")

            self.root.update()

            # Load model based on type
            if model_type == "SAM3":
                # SAM3 loading with HuggingFace model and SAM2-compatible API
                try:
                    from sam3.model_builder import build_sam3_video_model
                except ImportError:
                    raise ImportError(
                        "SAM3 not found. Please install SAM3:\n"
                        "1. Run setup.py and choose to install SAM3\n"
                        "2. Follow HuggingFace authentication steps\n"
                        "3. Download checkpoints from https://huggingface.co/facebook/sam3"
                    )

                # Build SAM3 model with SAM2-compatible API
                self.status_label.config(text="Building SAM3 model (may download from HuggingFace)...")
                self.root.update()

                sam3_model = build_sam3_video_model(device=device)

                # Extract the predictor using SAM2-compatible interface
                self.sam2_model = sam3_model.tracker
                self.sam2_model.backbone = sam3_model.detector.backbone

                self.using_sam3 = True
                display_name = "SAM3 (HuggingFace)"

            else:
                # SAM2 loading logic
                # Determine model to load
                model_selection = self.selected_model.get()

                if model_selection == "auto":
                    model_info = self._auto_select_best_model()
                    if not model_info:
                        raise ValueError("No models available. Please run setup.py first to download models.")
                else:
                    model_info = model_selection

                # Parse model info: "Display Name|checkpoint_file|config_path"
                parts = model_info.split('|')
                if len(parts) != 3:
                    raise ValueError(f"Invalid model selection: {model_info}")

                display_name, checkpoint_file, config_path = parts

                # Build full paths using new structure
                sam2_checkpoint = os.path.join(self.checkpoint_dir, checkpoint_file)
                model_cfg = os.path.join(self.config_dir, config_path)

                # Hydra on Linux strips leading '/', so prepend extra '/' for absolute paths
                if model_cfg.startswith('/'):
                    model_cfg = '/' + model_cfg

                # Check files exist
                if not os.path.exists(sam2_checkpoint):
                    raise FileNotFoundError(f"Checkpoint not found: {sam2_checkpoint}\n\nPlease run setup.py to download models.")
                if not os.path.exists(model_cfg):
                    raise FileNotFoundError(f"Config not found: {model_cfg}\n\nPlease ensure SAM2 is properly installed.")

                # Import the correct builder for VIDEO segmentation
                from sam2.build_sam import build_sam2_video_predictor
                self.using_sam3 = False

                # Build the VIDEO predictor for SAM2
                self.sam2_model = build_sam2_video_predictor(
                    config_file=model_cfg,
                    ckpt_path=sam2_checkpoint,
                    device=device
                )

            # Setup autocast context for BFloat16 (following SAM2 notebook pattern)
            if device != "cpu" and torch.cuda.is_available():
                # Enable autocast for BFloat16 as per SAM2 official notebook
                self.autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
                self.autocast_context.__enter__()

                # Enable TF32 for Ampere GPUs (compute capability >= 8.0)
                try:
                    gpu_id = 0 if device == "cuda" else int(device.split(":")[1])
                    if torch.cuda.get_device_properties(gpu_id).major >= 8:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        print(f"Enabled TF32 for Ampere GPU (compute capability {torch.cuda.get_device_properties(gpu_id).major}.x)")
                except Exception as e:
                    print(f"Could not enable TF32: {e}")

                print(f"Model loaded on {device} with BFloat16 autocast enabled")
            else:
                self.autocast_context = None
                print(f"Model loaded on {device} (CPU mode, no autocast)")

            self.model_loaded = True
            # Store model info (for SAM2 only; SAM3 doesn't use this)
            if not self.using_sam3:
                self.current_model_info = model_info
            else:
                self.current_model_info = "SAM3|sam3_hiera_l.pt|sam3_hiera_l.yaml"

            # Update status display
            dtype_str = "BF16+TF32" if self.autocast_context else "FP32"
            model_type_display = model_type if self.sam3_available else "SAM2"
            self.model_status_label.config(
                text=f"{display_name} ({device.upper()}/{dtype_str})",
                foreground='green'
            )
            self.status_label.config(text=f"{model_type_display} loaded: {display_name} on {device.upper()}")

            # Test that the model has the required methods
            if not hasattr(self.sam2_model, 'init_state'):
                raise AttributeError("Model does not have 'init_state' method. Check SAM2 installation.")
            if not hasattr(self.sam2_model, 'add_new_points'):
                raise AttributeError("Model does not have 'add_new_points' method. Check SAM2 installation.")

        except Exception as e:
            self.model_loaded = False
            self.model_status_label.config(text="Load Failed", foreground='red')
            traceback.print_exc()

            # Determine which model type failed
            model_type = self.model_type_var.get() if self.sam3_available else "SAM2"

            error_msg = f"Failed to load {model_type} model:\n\n{str(e)}\n\nPossible solutions:\n"
            if model_type == "SAM3":
                error_msg += "1. Run setup.py and install SAM3\n"
                error_msg += "2. Authenticate with HuggingFace: huggingface-cli login\n"
                error_msg += "3. Request access at https://huggingface.co/facebook/sam3\n"
                error_msg += "4. Check Python ≥3.12, PyTorch ≥2.7, CUDA ≥12.6\n"
                error_msg += "5. Try switching to SAM2 instead"
            else:
                error_msg += "1. Run setup.py to download models and install SAM2\n"
                error_msg += "2. Check that sam2/ directory exists with checkpoints\n"
                error_msg += "3. Verify SAM2 package is installed (pip list | grep sam2)\n"
                error_msg += "4. Try a different model from the dropdown"

            messagebox.showerror("Model Load Error", error_msg)

    def _monitor_memory(self, frame_idx, total_frames):
        """Monitor and log GPU/RAM memory usage (every 50 frames)"""
        # Only monitor every 50 frames to reduce overhead
        if frame_idx % 50 != 0:
            return

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3

            # Log to console
            print(f"  GPU Memory - Frame {frame_idx}/{total_frames}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {peak:.2f}GB peak")

            # Reset peak stats every 50 frames to track incremental growth
            torch.cuda.reset_peak_memory_stats()

        # Monitor RAM
        try:
            import psutil
            ram_used = psutil.virtual_memory().used / 1024**3
            ram_total = psutil.virtual_memory().total / 1024**3
            ram_percent = psutil.virtual_memory().percent
            print(f"  RAM: {ram_used:.2f}/{ram_total:.2f} GB ({ram_percent:.1f}%)")
        except ImportError:
            pass  # psutil not available, skip RAM monitoring

    def _export_mask_to_disk(self, frame_idx, obj_id, mask, output_dir, metadata_list):
        """
        Export mask to disk immediately and store only metadata

        Args:
            frame_idx: Frame index
            obj_id: Object ID
            mask: Binary mask (H, W) numpy array
            output_dir: Directory to save masks
            metadata_list: List to append metadata to
        """
        # Create masks subdirectory
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        # Generate filename
        mask_filename = f"frame_{frame_idx:05d}_obj_{obj_id}.png"
        mask_path = os.path.join(masks_dir, mask_filename)

        # Save mask to disk
        cv2.imwrite(mask_path, mask)

        # Get object metadata
        obj_name = self.object_names.get(obj_id, f"Object {obj_id}")
        obj_color = self.object_colors.get(obj_id, (255, 0, 0))

        # Store only metadata (filename, name, color) - NOT the full mask array
        metadata = {
            'frame_idx': frame_idx,
            'obj_id': obj_id,
            'mask_file': mask_filename,
            'mask_path': mask_path,
            'object_name': obj_name,
            'color': obj_color,
            'mask_shape': mask.shape
        }
        metadata_list.append(metadata)

        # Explicitly delete mask array to free memory immediately
        del mask

    def _cleanup_frame_cache(self, current_frame_idx, inference_state):
        """
        Clean up old frames from inference state to prevent memory growth

        SAM2 only needs last 6-16 frames for temporal memory attention.
        Keeping more than necessary wastes GPU memory.

        Args:
            current_frame_idx: Current frame being processed
            inference_state: SAM2 inference state object
        """
        frames_to_keep = 20  # Fixed value, always keep last 20 frames

        # Get non-conditioning frames from inference state
        # This is where SAM2 stores frame features and outputs
        if hasattr(inference_state, 'non_cond_frame_outputs'):
            non_cond = inference_state.non_cond_frame_outputs

            # Find frames older than the cache window
            old_frames = [f for f in non_cond.keys() if f < current_frame_idx - frames_to_keep]

            # Delete old frame outputs
            for old_frame in old_frames:
                del non_cond[old_frame]

            if old_frames and current_frame_idx % 50 == 0:
                print(f"  Cleaned up {len(old_frames)} old frames from cache (keeping last {frames_to_keep})")

        # Periodic GPU memory cleanup
        if current_frame_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_mask(self, frame_idx, obj_id):
        """
        Load mask from disk with pattern matching fallback

        Args:
            frame_idx: Frame index
            obj_id: Object ID

        Returns:
            mask: Binary mask array (H, W) or None if not found
        """
        if not hasattr(self, 'mask_export_dir') or self.mask_export_dir is None:
            print(f"WARNING: No mask export directory found for frame {frame_idx}, obj {obj_id}")
            return None

        # Try 1: Exact name match with new pattern
        obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
        mask_filename = f"mask_f{frame_idx:06d}_{obj_name}_id{obj_id}.png"
        mask_path = os.path.join(self.mask_export_dir, mask_filename)

        if os.path.exists(mask_path):
            return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Try 2: Legacy pattern (for backward compatibility)
        mask_filename_legacy = f"frame_{frame_idx:05d}_obj_{obj_id}.png"
        mask_path_legacy = os.path.join(self.mask_export_dir, "masks", mask_filename_legacy)

        if os.path.exists(mask_path_legacy):
            print(f"WARNING: Using legacy mask pattern for frame {frame_idx}, obj {obj_id}")
            return cv2.imread(mask_path_legacy, cv2.IMREAD_GRAYSCALE)

        # Try 3: Pattern matching by frame and ID (name mismatch fallback)
        for filename in os.listdir(self.mask_export_dir):
            match = re.match(rf'mask_f{frame_idx:06d}_(.+)_id{obj_id}\.png', filename)
            if match:
                found_name = match.group(1)
                print(f"WARNING: Mask name mismatch for obj {obj_id}. "
                      f"Expected '{obj_name}', found '{found_name}'. Using found mask.")
                mask_path_fallback = os.path.join(self.mask_export_dir, filename)
                return cv2.imread(mask_path_fallback, cv2.IMREAD_GRAYSCALE)

        print(f"WARNING: Mask file not found for frame {frame_idx}, obj {obj_id}")
        return None

    def _cleanup_temp_masks(self):
        """Clean up temporary mask directory"""
        if hasattr(self, 'mask_export_dir') and self.mask_export_dir and os.path.exists(self.mask_export_dir):
            try:
                import shutil
                shutil.rmtree(self.mask_export_dir)
                print(f"Cleaned up temporary mask directory: {self.mask_export_dir}")
            except Exception as e:
                print(f"WARNING: Failed to clean up temporary masks: {e}")
            self.mask_export_dir = None


def main():
    """Main application entry point"""
    root = tk.Tk()
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    app = SAM2VideoUI(root)
    
    def on_closing():
        # Check for active exports and segmentation
        active_exports = len(app.active_exports) if app.active_exports else 0
        active_segmentation = len(app.active_segmentation) if app.active_segmentation else 0
        total_active = active_exports + active_segmentation
        
        if total_active > 0:
            result = messagebox.askyesnocancel(
                "Active Background Tasks Detected",
                f"You have {total_active} background task(s) running:\n"
                f"- {active_exports} export(s)\n"
                f"- {active_segmentation} segmentation(s)\n\n"
                f"What would you like to do?\n\n"
                f"Yes: Save task status and quit (tasks will continue)\n"
                f"No: Quit without saving (tasks will be lost)\n"
                f"Cancel: Stay in application"
            )
            
            if result is None:  # Cancel
                return
            elif result:  # Yes - save task status and let tasks continue
                if hasattr(app, '_handle_background_tasks_save_on_exit'):
                    app._handle_background_tasks_save_on_exit()
                # Don't stop workers - let them continue
            # else: No - user wants to quit without saving, just exit

        # Clean up video capture if lazy loading
        if hasattr(app, 'video_cap_lazy') and app.video_cap_lazy:
            app.video_cap_lazy.release()

        # Clean up temporary masks
        app._cleanup_temp_masks()

        # CRITICAL: Always destroy root at the end
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()
