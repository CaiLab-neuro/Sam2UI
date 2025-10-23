import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import json
import sys
import tempfile
import shutil
from pathlib import Path
import traceback
from omegaconf import OmegaConf, DictConfig
import csv
import queue
import time

# Add SAM2 path to Python path - dynamically detect project root
def get_project_root():
    """Dynamically detect the project root directory"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    # Look for project root indicators
    indicators = ['sam2', 'checkpoints', 'configs', 'pyproject.toml', 'setup.py']
    
    # Start from current file and go up directories
    search_dir = current_dir
    while search_dir != os.path.dirname(search_dir):  # Not at filesystem root
        if all(os.path.exists(os.path.join(search_dir, indicator)) for indicator in indicators[:3]):
            return search_dir
        search_dir = os.path.dirname(search_dir)
    
    # Fallback to current directory if not found
    return current_dir

SAM2_PATH = get_project_root()
if SAM2_PATH not in sys.path:
    sys.path.append(SAM2_PATH)

# Import torch for device detection
try:
    import torch
except ImportError:
    torch = None

class SAM2VideoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM2 Video Segmentation Tool - Enhanced")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2b2b2b')
        
        # Dynamic paths - automatically detect project root
        self.sam2_base_path = SAM2_PATH
        self.checkpoint_dir = os.path.join(SAM2_PATH, "checkpoints")
        self.config_dir = os.path.join(SAM2_PATH, "configs")
        
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
        self.max_object_id = 1  # Track highest object ID used (now up to 30)
        
        # Enhanced object management
        self.object_names = {}  # Maps obj_id to custom name
        self.object_colors = {}  # Dynamic color assignment
        self.refinement_mode = False
        self.selected_frames_for_refinement = set()
        self.point_removal_mode = False
        
        # Multi-frame annotation mode (always enabled)
        self.multi_frame_annotation_mode = True
        self.annotated_frames = set()  # Track which frames have been annotated
        
        # GPU selection
        self.available_gpus = self._detect_available_gpus()
        self.selected_gpu = tk.StringVar(value="auto")  # Default to auto selection
        self.gpu_device = None  # Will be set when model loads
        
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
        
        # Initialize default colors and names
        self._initialize_objects()
        
        # SAM2 model
        self.sam2_model = None
        self.model_loaded = False
        
        # Background export management
        self.export_queue = queue.Queue()
        self.active_exports = {}  # Track active export threads
        self.export_results = {}  # Store export results
        self.export_thread = None
        self.export_running = False
        
        # Background segmentation management
        self.segmentation_queue = queue.Queue()
        self.active_segmentation = {}  # Track active segmentation threads
        self.segmentation_results = {}  # Store segmentation results
        self.segmentation_thread = None
        self.segmentation_running = False
        
        # UI styling
        self.setup_styles()
        self.setup_ui()
        
    def _initialize_objects(self):
        """Initialize object colors and default names for up to 30 objects"""
        # Generate distinct colors for 30 objects using HSV space
        for i in range(1, 31):
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
        # Main container with better layout
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create paned window for resizable layout
        paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)
        
        # Right panel for video display
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=3)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, parent):
        """Setup the left control panel with enhanced object management"""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="SAM2 Enhanced", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # File operations
        file_frame = ttk.LabelFrame(scrollable_frame, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Video", 
                  command=self.load_video, width=15).pack(fill=tk.X, pady=2)
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
        self.object_spinbox = tk.Spinbox(current_obj_frame, from_=1, to=30, 
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
        
        # Segmentation controls
        seg_frame = ttk.LabelFrame(scrollable_frame, text="Segmentation", padding=10)
        seg_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(seg_frame, text="Segment Video", 
                  command=self.segment_video, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="Refine Segment", 
                  command=self.toggle_refinement_mode, width=15).pack(fill=tk.X, pady=2)
        
        # Point management buttons
        point_mgmt_frame = ttk.Frame(seg_frame)
        point_mgmt_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(point_mgmt_frame, text="Remove Point", 
                  command=self.toggle_point_removal_mode, width=12).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(point_mgmt_frame, text="Clear All", 
                  command=self.clear_points, width=12).pack(side=tk.LEFT)
        
        ttk.Button(seg_frame, text="Show Frame Points", 
                  command=self.show_frame_points, width=15).pack(fill=tk.X, pady=2)
        
        ttk.Button(seg_frame, text="Export Annotations", 
                  command=self.export_annotations, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="Import Annotations", 
                  command=self.import_annotations, width=15).pack(fill=tk.X, pady=2)
        
        # Multi-frame annotation mode indicator (always active)
        self.multi_frame_label = ttk.Label(seg_frame, text="MULTI-FRAME ANNOTATION ACTIVE", foreground='blue')
        self.multi_frame_label.pack(pady=2)
        
        # Refinement mode indicator
        self.refinement_label = ttk.Label(seg_frame, text="", foreground='orange')
        self.refinement_label.pack(pady=2)
        
        # Point removal mode indicator
        self.point_removal_label = ttk.Label(seg_frame, text="", foreground='red')
        self.point_removal_label.pack(pady=2)
        
        # Export controls
        export_frame = ttk.LabelFrame(scrollable_frame, text="Export", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Export mode selection
        self.background_export_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(export_frame, text="Background Export Mode", 
                       variable=self.background_export_var,
                       command=self.on_export_mode_change).pack(anchor=tk.W, pady=(0, 5))
        
        # Export mode indicator
        self.export_mode_label = ttk.Label(export_frame, text="Foreground Mode", 
                                          foreground='blue', font=('Arial', 8))
        self.export_mode_label.pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Button(export_frame, text="Export Masks", 
                  command=self.export_masks, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Export Video", 
                  command=self.export_video, width=15).pack(fill=tk.X, pady=2)
        
        # Display options
        display_frame = ttk.LabelFrame(scrollable_frame, text="Display Options", padding=10)
        display_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.show_masks_var = tk.BooleanVar()
        ttk.Checkbutton(display_frame, text="Show Masks", 
                       variable=self.show_masks_var,
                       command=self.toggle_mask_display).pack(anchor=tk.W)
        
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
        
        # Video controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X)
        
        # Playback controls
        playback_frame = ttk.Frame(controls_frame)
        playback_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.play_button = ttk.Button(playback_frame, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(playback_frame, text="Prev", command=self.prev_frame).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(playback_frame, text="Next", command=self.next_frame).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(playback_frame, text="Reset", command=self.reset_video).pack(side=tk.LEFT, padx=(10, 0))
        
        # Frame selection for refinement (always present; enabled when active)
        self.select_frame_button = ttk.Button(playback_frame, text="Select Frame",
                      command=self.toggle_frame_selection)
        self.select_frame_button.state(["disabled"])
        self.select_frame_button.pack(side=tk.LEFT, padx=(10, 0))
        
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
                            if 1 <= obj_id <= 30:
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
                
                for obj_id in range(1, 31):
                    color = self.object_colors[obj_id]
                    writer.writerow({
                        'id': obj_id,
                        'name': self.object_names[obj_id],
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
                "annotations": []
            }
            
            # Convert click points to export format
            for point in self.click_points:
                img_x, img_y, is_positive, obj_id, frame_idx = point
                annotation = {
                    "frame_index": frame_idx,
                    "x": int(img_x),
                    "y": int(img_y),
                    "is_positive": is_positive,
                    "object_id": obj_id,
                    "object_name": self.object_names.get(obj_id, f"Object_{obj_id}"),
                    "timestamp": frame_idx / 30.0 if len(self.frames) > 0 else 0  # Assuming 30 FPS
                }
                annotation_data["annotations"].append(annotation)
            
            # Sort annotations by frame index
            annotation_data["annotations"].sort(key=lambda x: (x["frame_index"], x["object_id"]))
            
            # Add metadata
            annotation_data["export_info"] = {
                "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "app_version": "SAM2 Video UI Enhanced",
                "multi_frame_mode": self.multi_frame_annotation_mode,
                "refinement_mode": self.refinement_mode
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
            
            # Import annotations
            imported_count = 0
            for annotation in annotation_data["annotations"]:
                try:
                    frame_idx = annotation["frame_index"]
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
                        # Assign a color if not already assigned
                        if obj_id not in self.object_colors:
                            self.object_colors[obj_id] = self._get_next_color()
                    
                    # Track annotated frames
                    self.annotated_frames.add(frame_idx)
                    
                    imported_count += 1
                    
                except KeyError as e:
                    print(f"Warning: Skipping invalid annotation: {e}")
                    continue
            
            # Update UI
            self.update_points_display()
            self.update_object_list()
            self.display_current_frame()
            
            # Show success message
            messagebox.showinfo("Import Complete", 
                              f"Annotations imported successfully!\n\n"
                              f"File: {file_path}\n"
                              f"Imported annotations: {imported_count}\n"
                              f"Total annotations: {len(self.click_points)}\n"
                              f"Objects: {len(self.object_names)}")
            
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import annotations: {str(e)}")
    
    def toggle_multi_frame_annotation(self):
        """Multi-frame annotation mode is always enabled"""
        # Multi-frame annotation is always active, no need to toggle
        self.multi_frame_label.config(text="MULTI-FRAME ANNOTATION ACTIVE")
        self.status_label.config(text="Multi-frame mode: Navigate to different frames and add points, then segment")
    
    def toggle_refinement_mode(self):
        """Toggle refinement mode for improving segmentation"""
        self.refinement_mode = not self.refinement_mode
        
        if self.refinement_mode:
            self.refinement_label.config(text="REFINEMENT MODE ACTIVE")
            self.status_label.config(text="Refinement mode: Select frames to improve, add points, then re-segment")
            if hasattr(self, 'select_frame_button'):
                self.select_frame_button.state(["!disabled"])  # enable
            # Multi-frame annotation mode stays active
        else:
            self.refinement_label.config(text="")
            self.selected_frames_for_refinement.clear()
            self.status_label.config(text="Refinement mode disabled")
            if hasattr(self, 'select_frame_button'):
                self.select_frame_button.state(["disabled"])  # disable
    
    def toggle_point_removal_mode(self):
        """Toggle point removal mode for removing individual annotation points"""
        self.point_removal_mode = not self.point_removal_mode
        
        if self.point_removal_mode:
            self.point_removal_label.config(text="POINT REMOVAL MODE ACTIVE")
            self.status_label.config(text="Point removal mode: Click on annotation points to remove them")
            # Disable other modes
            self.refinement_mode = False
            self.refinement_label.config(text="")
        else:
            self.point_removal_label.config(text="")
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
            # Calculate distance from click to point
            distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            
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
            
            # Show confirmation
            point_type = "positive" if is_pos else "negative"
            obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
            self.status_label.config(text=f"Removed {point_type} point for {obj_name} at ({px:.0f}, {py:.0f})")
            
            return True
        
        return False
            
    def toggle_frame_selection(self):
        """Toggle current frame selection for refinement"""
        if not self.refinement_mode:
            return
            
        if self.current_frame_idx in self.selected_frames_for_refinement:
            self.selected_frames_for_refinement.remove(self.current_frame_idx)
        else:
            self.selected_frames_for_refinement.add(self.current_frame_idx)
            
        self.display_current_frame()
        
    def update_object_name(self, event=None):
        """Update the name of the current object"""
        new_name = self.object_name_var.get().strip()
        if new_name:
            self.object_names[self.current_object_id] = new_name
            self.update_object_list()
            
    def add_new_object(self):
        """Add a new object for segmentation"""
        if self.max_object_id < 30:
            self.max_object_id += 1
            self.current_object_id = self.max_object_id
            self.object_var.set(self.current_object_id)
            self.object_spinbox.config(to=self.max_object_id)
            self.object_name_var.set(self.object_names[self.current_object_id])
            self.update_object_color_display()
            self.update_object_list()
            self.status_label.config(text=f"Added object {self.current_object_id}: {self.object_names[self.current_object_id]}")
        else:
            messagebox.showwarning("Limit Reached", "Maximum 30 objects supported.")
            
    def load_video(self):
        """Load video file and extract frames"""
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
        """Extract frames from video with memory optimization options"""
        try:
            self.video_cap = cv2.VideoCapture(self.video_path)
            if not self.video_cap.isOpened():
                raise ValueError("Could not open video file")
            
            self.frames = []
            self.masks = {}
            self.click_points = []
            
            # Get video properties
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            original_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
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
        
        # Add frame selection indicator for refinement mode
        if self.refinement_mode and self.current_frame_idx in self.selected_frames_for_refinement:
            # Add orange border for selected frames
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1]-1, display_frame.shape[0]-1), 
                         (255, 165, 0), 10)
        
        # Add annotation indicator for multi-frame annotation mode
        if self.multi_frame_annotation_mode and self.current_frame_idx in self.annotated_frames:
            # Add blue border for annotated frames
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1]-1, display_frame.shape[0]-1), 
                         (0, 165, 255), 8)
        
        # Apply mask overlay if enabled and masks exist
        if self.show_masks_var.get() and self.current_frame_idx in self.masks:
            frame_masks = self.masks[self.current_frame_idx]
            
            # Apply each object's mask with its color
            for obj_id, mask in frame_masks.items():
                if len(mask.shape) == 2:  # Single channel mask
                    # Get object color
                    obj_color = self.object_colors.get(obj_id, [255, 255, 255])
                    
                    # Create colored overlay
                    overlay = np.zeros_like(display_frame)
                    overlay[mask > 0] = obj_color
                    
                    # Blend with current frame
                    alpha = 0.4
                    display_frame = cv2.addWeighted(display_frame, 1-alpha, overlay, alpha, 0)
        
        # Draw click points only for the current frame
        for i, (x, y, is_positive, obj_id, frame_idx) in enumerate(self.click_points):
            if frame_idx != self.current_frame_idx:
                continue
            # Only draw points for current object or if showing all
            if obj_id == self.current_object_id or not hasattr(self, 'current_object_id'):
                obj_color = self.object_colors.get(obj_id, [255, 255, 255])
                color = tuple(obj_color) if is_positive else (255, 0, 0)  # Object color for positive, red for negative
                symbol = "+" if is_positive else "-"
                
                # Draw circle
                cv2.circle(display_frame, (int(x), int(y)), 8, color, -1)
                cv2.circle(display_frame, (int(x), int(y)), 10, (255, 255, 255), 2)
                
                # Draw symbol
                cv2.putText(display_frame, symbol, (int(x)-5, int(y)+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw object name instead of just ID
                obj_name = self.object_names.get(obj_id, f"Obj{obj_id}")[:8]  # Truncate long names
                cv2.putText(display_frame, obj_name, (int(x)+15, int(y)-10), 
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
            self.scale_factor = min(scale_w, scale_h, 1.0)
            
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
            
            img_x = (canvas_x - offset_x) / self.scale_factor
            img_y = (canvas_y - offset_y) / self.scale_factor
            
            # Ensure coordinates are within image bounds
            img_height, img_width = self.current_frame.shape[:2]
            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                # Store frame-aware point
                self.click_points.append((img_x, img_y, is_positive, self.current_object_id, self.current_frame_idx))
                
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
        """Clear all click points"""
        self.click_points = []
        self.annotated_frames.clear()
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
            
    def reset_video(self):
        """Reset to first frame"""
        if self.frames:
            self.current_frame_idx = 0
            self.display_current_frame()
            
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
    
    def check_background_tasks(self):
        """Periodically check for completed background tasks"""
        # Check for completed background segmentation results
        self.check_background_segmentation_results()
        
        # Schedule next check
        self.root.after(2000, self.check_background_tasks)  # Check every 2 seconds
    
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
        elif selected.startswith("cuda:"):
            return selected
        else:
            # Extract cuda device from display string
            if "cuda:" in selected:
                device_part = selected.split("cuda:")[1].split(" ")[0]
                return f"cuda:{device_part}"
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
            # Create a context manager that disables autocast
            if hasattr(torch, 'autocast'):
                return torch.autocast(enabled=False, device_type='cuda' if torch.cuda.is_available() else 'cpu')
            else:
                return torch.no_grad()
        except Exception as e:
            print(f"Warning: Could not disable autocast: {e}")
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
        """Enhanced segmentation with refinement support"""
        if not self.frames:
            messagebox.showwarning("Warning", "Please load a video first")
            return
            
        if not self.model_loaded or not self.sam2_model:
            messagebox.showwarning("Warning", "Please load SAM2 model first")
            return
            
        if not self.click_points:
            messagebox.showwarning("Warning", "Please add some click points first")
            return
        
        # Ask about processing mode before segmentation
        processing_choice = messagebox.askyesnocancel(
            "Segmentation Processing Options",
            "How would you like to process the segmentation?\n\n"
            "Yes: Background processing (segment + export, can close app)\n"
            "No: Foreground processing (wait for completion)\n"
            "Cancel: Abort segmentation"
        )
        
        if processing_choice is None:  # Cancel
            return
        elif processing_choice:  # Yes - background processing
            self.background_segmentation = True
            self.auto_export_after_segmentation = True
        else:  # No - foreground processing
            self.background_segmentation = False
            # Ask about export after segmentation
            export_choice = messagebox.askyesnocancel(
                "Export After Segmentation",
                "Would you like to automatically export after segmentation?\n\n"
                "Yes: Export masks and video after segmentation\n"
                "No: Just segment (no automatic export)\n"
                "Cancel: Abort segmentation"
            )
            
            if export_choice is None:  # Cancel
                return
            elif export_choice:  # Yes - will export after segmentation
                self.auto_export_after_segmentation = True
            else:  # No - just segment
                self.auto_export_after_segmentation = False
        
        # Start background segmentation if requested
        if getattr(self, 'background_segmentation', False):
            self._start_background_segmentation()
            return
            
        try:
            # Determine if this is refinement or initial segmentation
            is_refinement = self.refinement_mode and self.selected_frames_for_refinement
            
            if is_refinement:
                self.status_label.config(text="Refining segmentation for selected frames...")
            else:
                self.status_label.config(text="Preparing frames for SAM2...")
                
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(0)
            self.root.update()
            
            # Create temporary directory for frames
            temp_dir = tempfile.mkdtemp(prefix='sam2_frames_')
            
            try:
                # Determine which frames to save based on processing range
                if self.limit_to_range_var.get() and not is_refinement:
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
                for save_idx, frame_idx in enumerate(frames_to_save):
                    frame = self.frames[frame_idx]
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
                
                # Initialize or reuse inference state
                if not hasattr(self, 'inference_state') or self.inference_state is None:
                    self.inference_state = self.sam2_model.init_state(video_path=temp_dir)
                
                # Group click points by frame and by object ID
                points_by_frame_and_object = {}
                for x, y, is_pos, obj_id, frame_idx in self.click_points:
                    # Map original frame index to limited video index
                    if self.limit_to_range_var.get() and not is_refinement:
                        if frame_idx not in frames_to_save:
                            continue  # Skip points outside the processing range
                        limited_frame_idx = frames_to_save.index(frame_idx)
                    else:
                        limited_frame_idx = frame_idx
                    
                    if limited_frame_idx not in points_by_frame_and_object:
                        points_by_frame_and_object[limited_frame_idx] = {}
                    if obj_id not in points_by_frame_and_object[limited_frame_idx]:
                        points_by_frame_and_object[limited_frame_idx][obj_id] = {'points': [], 'labels': []}
                    points_by_frame_and_object[limited_frame_idx][obj_id]['points'].append([x, y])
                    points_by_frame_and_object[limited_frame_idx][obj_id]['labels'].append(1 if is_pos else 0)
                
                self.status_label.config(text="Adding prompts to SAM2...")
                self.progress_var.set(40)
                self.root.update()
                
                # Initialize masks dictionary if not exists
                if not hasattr(self, 'masks'):
                    self.masks = {}
                    
                for frame_idx in range(len(self.frames)):
                    if frame_idx not in self.masks:
                        self.masks[frame_idx] = {}
                
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
                            # Debug dtype information
                            self._debug_dtype_mismatch(points, labels, ann_frame, obj_id)
                            
                            # Ensure points and labels are in correct dtype and device
                            points_tensor, labels_tensor = self._prepare_tensors_for_inference(points, labels)
                            
                            # Disable autocast to prevent dtype mismatches
                            with self._disable_autocast_for_inference():
                                # Force model to float32 before inference
                                self._force_model_float32()
                                
                                # Add points for this object at this frame with dtype error handling
                                try:
                                    # Use custom context manager to force float32
                                    with self._force_float32_context():
                                        _ = self.sam2_model.add_new_points(
                                            inference_state=self.inference_state,
                                            frame_idx=ann_frame,
                                            obj_id=obj_id,
                                            points=points_tensor,
                                            labels=labels_tensor,
                                        )
                                except RuntimeError as e:
                                    if "dtype" in str(e).lower() or "bfloat16" in str(e).lower():
                                        print(f"Dtype error detected, attempting recovery...")
                                        # Try with explicit float32 conversion and model patching
                                        self._force_model_float32()
                                        points_tensor = points_tensor.float()
                                        labels_tensor = labels_tensor.long()
                                        with self._force_float32_context():
                                            _ = self.sam2_model.add_new_points(
                                                inference_state=self.inference_state,
                                                frame_idx=ann_frame,
                                                obj_id=obj_id,
                                                points=points_tensor,
                                                labels=labels_tensor,
                                            )
                                    else:
                                        # If all else fails, try on CPU
                                        print(f"Attempting CPU fallback for dtype error...")
                                        try:
                                            points_cpu = points_tensor.cpu().float()
                                            labels_cpu = labels_tensor.cpu().long()
                                            # Temporarily move model to CPU
                                            original_device = next(self.sam2_model.model.parameters()).device
                                            self.sam2_model.model = self.sam2_model.model.cpu()
                                            _ = self.sam2_model.add_new_points(
                                                inference_state=self.inference_state,
                                                frame_idx=ann_frame,
                                                obj_id=obj_id,
                                                points=points_cpu,
                                                labels=labels_cpu,
                                            )
                                            # Move model back to original device
                                            self.sam2_model.model = self.sam2_model.model.to(original_device)
                                        except Exception as cpu_e:
                                            print(f"CPU fallback also failed: {cpu_e}")
                                            raise e
                        except Exception as e:
                            print(f"Error adding points for {obj_name} on frame {ann_frame}: {e}")
                            continue
                
                # Propagate through video (or just selected frames in refinement)
                if is_refinement:
                    self.status_label.config(text="Refining selected frames...")
                    frames_to_process = sorted(self.selected_frames_for_refinement)
                else:
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
                        if self.limit_to_range_var.get() and not is_refinement:
                            if out_frame_idx >= len(frames_to_save):
                                continue
                            original_frame_idx = frames_to_save[out_frame_idx]
                        else:
                            original_frame_idx = out_frame_idx
                        
                        # Skip frames not in processing list during refinement
                        if is_refinement and original_frame_idx not in frames_to_process:
                            continue
                        # Skip frames not in selected range (when limiting)
                        if not is_refinement and self.limit_to_range_var.get() and original_frame_idx not in frames_to_process:
                            continue
                        
                        # Only process frames that are in our target range
                        if original_frame_idx not in frames_to_process:
                            continue
                            
                        # Process each object mask
                        for i, out_obj_id in enumerate(out_obj_ids):
                            # Only keep masks for objects that were annotated anywhere
                            if any(out_obj_id in obj_dict for obj_dict in points_by_frame_and_object.values()):
                                mask_logits = out_mask_logits[i]
                                
                                # Handle torch tensors
                                if hasattr(mask_logits, 'cpu'):
                                    mask_logits = mask_logits.cpu()
                                if hasattr(mask_logits, 'numpy'):
                                    mask_logits = mask_logits.numpy()
                                
                                # Convert to binary mask
                                mask = (mask_logits > 0.0)
                                
                                # Ensure mask is 2D
                                if len(mask.shape) > 2:
                                    mask = mask.squeeze()
                                
                                # Store mask with original frame index
                                self.masks[original_frame_idx][out_obj_id] = (mask * 255).astype(np.uint8)
                        
                        processed_frames += 1
                        progress = 45 + (processed_frames / max(1, len(frames_to_process))) * 55
                        self.progress_var.set(min(progress, 100))
                        
                        if processed_frames % 10 == 0:
                            self.status_label.config(text=f"Processing frame {processed_frames}/{len(frames_to_process)}")
                            self.root.update_idletasks()
                            
                except Exception as e:
                    print(f"Error during propagation: {e}")
                    messagebox.showwarning("Propagation Warning", 
                                         f"Encountered issue: {str(e)}")
                
                self.progress_bar.pack_forget()
                
                # Count results
                total_masks = sum(len(frame_masks) for frame_masks in self.masks.values())
                unique_objects = set()
                for frame_masks in self.masks.values():
                    unique_objects.update(frame_masks.keys())
                
                if total_masks > 0:
                    if is_refinement:
                        self.status_label.config(text=f"Refinement complete! Updated masks for {len(unique_objects)} objects")
                        self.refinement_mode = False
                        self.refinement_label.config(text="")
                        self.selected_frames_for_refinement.clear()
                    else:
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

    def export_masks(self):
        """Export masks with enhanced metadata including object names"""
        if not self.masks:
            messagebox.showwarning("Warning", "No masks to export. Please segment the video first.")
            return
            
        # Use checkbox setting to determine export mode
        if self.background_export_var.get():
            self._start_background_mask_export()
        else:
            self._start_foreground_mask_export()
    
    def _start_background_mask_export(self):
        """Start background mask export"""
        folder_path = self._get_export_folder_with_creation(
            title="Select folder to save masks", 
            default_name="sam2_masks"
        )
        if not folder_path:
            return
        
        # Generate unique export ID
        export_id = f"masks_{int(time.time())}"
        
        # Create export task
        export_task = {
            'id': export_id,
            'type': 'masks',
            'folder_path': folder_path,
            'masks': self.masks.copy(),
            'object_names': self.object_names.copy(),
            'object_colors': self.object_colors.copy(),
            'click_points': self.click_points.copy(),
            'video_path': self.video_path,
            'frames': len(self.frames),
            'ann_frame_idx': getattr(self, 'ann_frame_idx', self.current_frame_idx),
            'sam2_info': {
                'base_path': self.sam2_base_path,
                'checkpoint_dir': self.checkpoint_dir,
                'config_dir': self.config_dir
            }
        }
        
        # Start export worker if not running
        self._start_export_worker()
        
        # Add task to queue
        self.export_queue.put(export_task)
        
        # Show status
        self.status_label.config(text=f"Background mask export started (ID: {export_id})")
        messagebox.showinfo("Export Started", 
                          f"Mask export started in background.\n"
                          f"Export ID: {export_id}\n"
                          f"Destination: {folder_path}\n\n"
                          f"You can now close the application if needed.\n"
                          f"The export will continue in the background.")
    
    def _start_foreground_mask_export(self):
        """Start foreground mask export (original behavior)"""
        folder_path = self._get_export_folder_with_creation(
            title="Select folder to save masks", 
            default_name="sam2_masks"
        )
        if not folder_path:
            return
            
        try:
            self.status_label.config(text="Exporting masks...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.root.update()
            
            exported_count = 0
            total_masks = sum(len(fm) for fm in self.masks.values())
            
            # Export masks with object names in filename
            for frame_idx, frame_masks in self.masks.items():
                for obj_id, mask in frame_masks.items():
                    obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                    # Clean filename
                    clean_name = "".join(c for c in obj_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    mask_path = os.path.join(folder_path, f"mask_f{frame_idx:05d}_{clean_name}_id{obj_id}.png")
                    cv2.imwrite(mask_path, mask)
                    exported_count += 1
                    
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
                },
                "refinement_info": {
                    "refinement_used": hasattr(self, 'selected_frames_for_refinement') and len(self.selected_frames_for_refinement) > 0,
                    "refined_frames": list(getattr(self, 'selected_frames_for_refinement', set()))
                }
            }
            
            # Object information with names and colors
            for obj_id in range(1, 31):
                if any(obj_id in frame_masks for frame_masks in self.masks.values()):
                    mask_count = sum(1 for frame_masks in self.masks.values() if obj_id in frame_masks)
                    point_count = sum(1 for _, _, _, oid in self.click_points if oid == obj_id)
                    
                    metadata["objects"][obj_id] = {
                        "name": self.object_names[obj_id],
                        "mask_count": mask_count,
                        "point_count": point_count,
                        "color": self.object_colors[obj_id]
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
            
            # Ask if user wants to open the export folder
            result = messagebox.askyesno("Export Complete", 
                              f"Successfully exported:\n"
                              f"- {exported_count} mask images (PNG)\n"
                              f"- 1 metadata file (JSON)\n"
                              f"- 1 object mapping (CSV)\n\n"
                              f"Location: {folder_path}\n\n"
                              f"Would you like to open the export folder?")
            
            if result:
                self._open_folder(folder_path)
                
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
                for obj_id, mask in frame_masks.items():
                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                    # Clean filename
                    clean_name = "".join(c for c in obj_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    mask_path = os.path.join(folder_path, f"mask_f{frame_idx:05d}_{clean_name}_id{obj_id}.png")
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
            for obj_id in range(1, 31):
                if any(obj_id in frame_masks for frame_masks in masks.values()):
                    mask_count = sum(1 for frame_masks in masks.values() if obj_id in frame_masks)
                    point_count = sum(1 for _, _, _, oid, _ in click_points if oid == obj_id)
                    
                    metadata["objects"][obj_id] = {
                        "name": object_names[obj_id],
                        "mask_count": mask_count,
                        "point_count": point_count,
                        "color": object_colors[obj_id]
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
                progress = ((export_idx + 1) / total_frames) * 100
                self.progress_var.set(progress)
                
                if export_idx % 10 == 0:
                    self.status_label.config(text=f"Exporting frame {export_idx + 1}/{total_frames} (original frame {frame_idx + 1})")
                    self.root.update_idletasks()
            
            out.release()
            self.progress_bar.pack_forget()
            self.status_label.config(text=f"Video exported successfully: {file_path}")
            
            # Determine if this was a limited export
            if hasattr(self, 'processing_range') and self.processing_range and len(self.processing_range) < len(self.frames):
                range_info = f" (frames {min(self.processing_range)+1}-{max(self.processing_range)+1} of {len(self.frames)})"
            else:
                range_info = ""
            
            # Ask if user wants to open the export folder
            result = messagebox.askyesno("Export Complete", 
                              f"Video exported successfully!\n"
                              f"Location: {file_path}\n"
                              f"Frames: {total_frames}{range_info}\n"
                              f"FPS: {fps}\n"
                              f"Format: {export_format.upper()}\n\n"
                              f"Would you like to open the export folder?")
            
            if result:
                export_folder = os.path.dirname(file_path)
                self._open_folder(export_folder)
                              
        except Exception as e:
            self.progress_bar.pack_forget()
            self.status_label.config(text="Video export failed")
            messagebox.showerror("Export Error", f"Failed to export video: {str(e)}")
    
    def _perform_background_video_export(self, export_format, overlay_opacity, show_object_names, 
                                       show_object_ids, show_boundaries, fps, quality, export_mode):
        """Start background video export"""
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
        
        # Generate unique export ID
        export_id = f"video_{int(time.time())}"
        
        # Create export task
        export_task = {
            'id': export_id,
            'type': 'video',
            'file_path': file_path,
            'export_format': export_format,
            'overlay_opacity': overlay_opacity,
            'show_object_names': show_object_names,
            'show_object_ids': show_object_ids,
            'show_boundaries': show_boundaries,
            'fps': fps,
            'quality': quality,
            'export_mode': export_mode,
            'frames': self.frames.copy(),
            'masks': self.masks.copy(),
            'object_names': self.object_names.copy(),
            'object_colors': self.object_colors.copy(),
            'processing_range': getattr(self, 'processing_range', list(range(len(self.frames))))
        }
        
        # Start export worker if not running
        self._start_export_worker()
        
        # Add task to queue
        self.export_queue.put(export_task)
        
        # Show status
        self.status_label.config(text=f"Background video export started (ID: {export_id})")
        messagebox.showinfo("Export Started", 
                          f"Video export started in background.\n"
                          f"Export ID: {export_id}\n"
                          f"Destination: {file_path}\n\n"
                          f"You can now close the application if needed.\n"
                          f"The export will continue in the background.")
    
    def _export_video_background(self, export_task):
        """Background video export implementation"""
        try:
            file_path = export_task['file_path']
            export_format = export_task['export_format']
            overlay_opacity = export_task['overlay_opacity']
            show_object_names = export_task['show_object_names']
            show_object_ids = export_task['show_object_ids']
            show_boundaries = export_task['show_boundaries']
            fps = export_task['fps']
            quality = export_task['quality']
            export_mode = export_task['export_mode']
            frames = export_task['frames']
            masks = export_task['masks']
            object_names = export_task['object_names']
            object_colors = export_task['object_colors']
            processing_range = export_task['processing_range']
            
            # Setup video writer
            height, width = frames[0].shape[:2]
            
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
            
            # Export frames
            frames_to_export = processing_range
            total_frames = len(frames_to_export)
            
            for export_idx, frame_idx in enumerate(frames_to_export):
                frame = frames[frame_idx]
                # Create output frame based on mode
                if export_mode == "overlay":
                    output_frame = self._create_overlay_frame_background(
                        frame, frame_idx, overlay_opacity,
                        show_object_names, show_object_ids,
                        show_boundaries, masks, object_names, object_colors
                    )
                elif export_mode == "masks_only":
                    output_frame = self._create_masks_only_frame_background(
                        frame, frame_idx, show_object_names,
                        show_object_ids, show_boundaries, masks, object_names, object_colors
                    )
                elif export_mode == "side_by_side":
                    output_frame = self._create_side_by_side_frame_background(
                        frame, frame_idx, overlay_opacity,
                        show_object_names, show_object_ids,
                        show_boundaries, masks, object_names, object_colors
                    )
                
                # Convert RGB to BGR for OpenCV
                if len(output_frame.shape) == 3:
                    output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                else:
                    output_frame_bgr = output_frame
                
                out.write(output_frame_bgr)
            
            out.release()
            
            return {
                'success': True,
                'file_path': file_path,
                'total_frames': total_frames,
                'export_type': 'video'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'export_type': 'video'
            }

    def _create_overlay_frame(self, frame, frame_idx, opacity, show_names, show_ids, show_boundaries):
        """Create frame with mask overlay"""
        output_frame = frame.copy()
        
        if frame_idx in self.masks:
            frame_masks = self.masks[frame_idx]
            
            for obj_id, mask in frame_masks.items():
                if len(mask.shape) == 2:  # Single channel mask
                    # Get object color
                    obj_color = self.object_colors.get(obj_id, [255, 255, 255])
                    
                    # Create colored overlay
                    overlay = np.zeros_like(output_frame)
                    overlay[mask > 0] = obj_color
                    
                    # Blend with current frame
                    output_frame = cv2.addWeighted(output_frame, 1-opacity, overlay, opacity, 0)
                    
                    # Add boundaries if requested
                    if show_boundaries:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(output_frame, contours, -1, tuple(obj_color), 2)
                    
                    # Add labels if requested
                    if show_names or show_ids:
                        # Find centroid of mask for label placement
                        moments = cv2.moments(mask)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            
                            label_parts = []
                            if show_names:
                                label_parts.append(self.object_names.get(obj_id, f"Obj{obj_id}"))
                            if show_ids:
                                label_parts.append(f"ID:{obj_id}")
                            
                            label = " - ".join(label_parts)
                            
                            # Add text background
                            (text_width, text_height), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            cv2.rectangle(output_frame, 
                                        (cx - text_width//2 - 5, cy - text_height - 5),
                                        (cx + text_width//2 + 5, cy + 5),
                                        (0, 0, 0), -1)
                            
                            # Add text
                            cv2.putText(output_frame, label, 
                                      (cx - text_width//2, cy), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                      tuple(obj_color), 2)
        
        return output_frame

    def _create_masks_only_frame(self, frame, frame_idx, show_names, show_ids, show_boundaries):
        """Create frame showing only masks on black background"""
        height, width = frame.shape[:2]
        output_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if frame_idx in self.masks:
            frame_masks = self.masks[frame_idx]
            
            for obj_id, mask in frame_masks.items():
                if len(mask.shape) == 2:  # Single channel mask
                    # Get object color
                    obj_color = self.object_colors.get(obj_id, [255, 255, 255])
                    
                    # Apply mask color
                    output_frame[mask > 0] = obj_color
                    
                    # Add boundaries if requested
                    if show_boundaries:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(output_frame, contours, -1, (255, 255, 255), 2)
                    
                    # Add labels if requested
                    if show_names or show_ids:
                        # Find centroid of mask for label placement
                        moments = cv2.moments(mask)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            
                            label_parts = []
                            if show_names:
                                label_parts.append(self.object_names.get(obj_id, f"Obj{obj_id}"))
                            if show_ids:
                                label_parts.append(f"ID:{obj_id}")
                            
                            label = " - ".join(label_parts)
                            
                            # Add text
                            cv2.putText(output_frame, label, 
                                      (cx - 50, cy), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                      (255, 255, 255), 2)
        
        return output_frame

    def _create_side_by_side_frame(self, frame, frame_idx, opacity, show_names, show_ids, show_boundaries):
        """Create side-by-side frame with original and overlay"""
        overlay_frame = self._create_overlay_frame(frame, frame_idx, opacity, 
                                                  show_names, show_ids, show_boundaries)
        
        # Concatenate horizontally
        output_frame = np.hstack((frame, overlay_frame))
        return output_frame
    
    def _create_overlay_frame_background(self, frame, frame_idx, opacity, show_names, show_ids, show_boundaries, masks, object_names, object_colors):
        """Create frame with mask overlay for background export"""
        output_frame = frame.copy()
        
        if frame_idx in masks:
            frame_masks = masks[frame_idx]
            
            for obj_id, mask in frame_masks.items():
                if len(mask.shape) == 2:  # Single channel mask
                    # Get object color
                    obj_color = object_colors.get(obj_id, [255, 255, 255])
                    
                    # Create colored overlay
                    overlay = np.zeros_like(output_frame)
                    overlay[mask > 0] = obj_color
                    
                    # Blend with current frame
                    output_frame = cv2.addWeighted(output_frame, 1-opacity, overlay, opacity, 0)
                    
                    # Add boundaries if requested
                    if show_boundaries:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(output_frame, contours, -1, tuple(obj_color), 2)
                    
                    # Add labels if requested
                    if show_names or show_ids:
                        # Find centroid of mask for label placement
                        moments = cv2.moments(mask)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            
                            label_parts = []
                            if show_names:
                                label_parts.append(object_names.get(obj_id, f"Obj{obj_id}"))
                            if show_ids:
                                label_parts.append(f"ID:{obj_id}")
                            
                            label = " - ".join(label_parts)
                            
                            # Add text background
                            (text_width, text_height), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            cv2.rectangle(output_frame, 
                                        (cx - text_width//2 - 5, cy - text_height - 5),
                                        (cx + text_width//2 + 5, cy + 5),
                                        (0, 0, 0), -1)
                            
                            # Add text
                            cv2.putText(output_frame, label, 
                                      (cx - text_width//2, cy), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                      tuple(obj_color), 2)
        
        return output_frame

    def _create_masks_only_frame_background(self, frame, frame_idx, show_names, show_ids, show_boundaries, masks, object_names, object_colors):
        """Create frame showing only masks on black background for background export"""
        height, width = frame.shape[:2]
        output_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if frame_idx in masks:
            frame_masks = masks[frame_idx]
            
            for obj_id, mask in frame_masks.items():
                if len(mask.shape) == 2:  # Single channel mask
                    # Get object color
                    obj_color = object_colors.get(obj_id, [255, 255, 255])
                    
                    # Apply mask color
                    output_frame[mask > 0] = obj_color
                    
                    # Add boundaries if requested
                    if show_boundaries:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(output_frame, contours, -1, (255, 255, 255), 2)
                    
                    # Add labels if requested
                    if show_names or show_ids:
                        # Find centroid of mask for label placement
                        moments = cv2.moments(mask)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            
                            label_parts = []
                            if show_names:
                                label_parts.append(object_names.get(obj_id, f"Obj{obj_id}"))
                            if show_ids:
                                label_parts.append(f"ID:{obj_id}")
                            
                            label = " - ".join(label_parts)
                            
                            # Add text
                            cv2.putText(output_frame, label, 
                                      (cx - 50, cy), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                      (255, 255, 255), 2)
        
        return output_frame

    def _create_side_by_side_frame_background(self, frame, frame_idx, opacity, show_names, show_ids, show_boundaries, masks, object_names, object_colors):
        """Create side-by-side frame with original and overlay for background export"""
        overlay_frame = self._create_overlay_frame_background(frame, frame_idx, opacity, 
                                                           show_names, show_ids, show_boundaries, 
                                                           masks, object_names, object_colors)
        
        # Concatenate horizontally
        output_frame = np.hstack((frame, overlay_frame))
        return output_frame

    def _rgb_to_hex(self, rgb):
        """Convert RGB color to hex"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def _open_folder(self, folder_path):
        """Open folder in system's default file manager"""
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Windows":
                subprocess.run(["explorer", folder_path], check=True)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", folder_path], check=True)
            elif system == "Linux":
                subprocess.run(["xdg-open", folder_path], check=True)
            else:
                # Fallback for other systems
                subprocess.run(["open", folder_path], check=True)
                
        except Exception as e:
            # If opening fails, show a message with the path
            messagebox.showinfo("Export Location", 
                              f"Could not open folder automatically.\n"
                              f"Export location: {folder_path}")
    
    def _get_export_folder_with_creation(self, title="Select Export Folder", default_name="sam2_export"):
        """Get export folder path with option to create new folder"""
        # First, ask user to select a base directory
        base_dir = filedialog.askdirectory(title=f"{title} - Select Base Directory")
        if not base_dir:
            return None
        
        # Ask if user wants to create a new subfolder
        create_subfolder = messagebox.askyesno(
            "Create Export Folder",
            f"Would you like to create a new subfolder for this export?\n\n"
            f"Base directory: {base_dir}\n\n"
            f"Yes: Create new subfolder\n"
            f"No: Use base directory directly"
        )
        
        if create_subfolder:
            # Ask for folder name
            folder_name = tk.simpledialog.askstring(
                "Export Folder Name",
                "Enter name for the export folder:",
                initialvalue=default_name
            )
            
            if not folder_name:
                return None
            
            # Create the full path
            export_folder = os.path.join(base_dir, folder_name)
            
            # Create the folder if it doesn't exist
            try:
                os.makedirs(export_folder, exist_ok=True)
                return export_folder
            except Exception as e:
                messagebox.showerror("Folder Creation Error", f"Could not create folder: {str(e)}")
                return None
        else:
            return base_dir
    
    def _get_export_file_path_with_creation(self, title="Save Export File", default_name="sam2_export", 
                                          file_types=[("All files", "*.*")], default_ext=""):
        """Get export file path with option to create parent folder"""
        # Get the file path
        file_path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=default_ext,
            filetypes=file_types,
            initialfile=default_name
        )
        
        if not file_path:
            return None
        
        # Check if parent directory exists, if not ask to create it
        parent_dir = os.path.dirname(file_path)
        if not os.path.exists(parent_dir):
            create_dir = messagebox.askyesno(
                "Create Directory",
                f"Directory does not exist:\n{parent_dir}\n\n"
                f"Would you like to create it?"
            )
            
            if create_dir:
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                except Exception as e:
                    messagebox.showerror("Directory Creation Error", f"Could not create directory: {str(e)}")
                    return None
            else:
                return None
        
        return file_path
    
    def _start_export_worker(self):
        """Start the background export worker thread"""
        if not self.export_running:
            self.export_running = True
            # Make thread non-daemon so it can continue after UI closes
            self.export_thread = threading.Thread(target=self._export_worker, daemon=False)
            self.export_thread.start()
    
    def _export_worker(self):
        """Background worker thread for handling exports"""
        while self.export_running:
            try:
                # Get export task from queue (with timeout)
                export_task = self.export_queue.get(timeout=1.0)
                
                if export_task is None:  # Shutdown signal
                    break
                
                export_type = export_task.get('type')
                export_id = export_task.get('id')
                
                # Mark export as active
                self.active_exports[export_id] = {
                    'type': export_type,
                    'status': 'running',
                    'start_time': time.time(),
                    'progress': 0
                }
                
                try:
                    if export_type == 'masks':
                        result = self._export_masks_background(export_task)
                    elif export_type == 'video':
                        result = self._export_video_background(export_task)
                    else:
                        result = {'success': False, 'error': f'Unknown export type: {export_type}'}
                    
                    # Store result
                    self.export_results[export_id] = result
                    self.active_exports[export_id]['status'] = 'completed'
                    
                except Exception as e:
                    self.export_results[export_id] = {
                        'success': False, 
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                    self.active_exports[export_id]['status'] = 'failed'
                
                finally:
                    self.export_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Export worker error: {e}")
                continue
    
    def _stop_export_worker(self):
        """Stop the background export worker thread"""
        self.export_running = False
        if self.export_thread and self.export_thread.is_alive():
            # Send shutdown signal
            self.export_queue.put(None)
            self.export_thread.join(timeout=5.0)
    
    def _handle_background_tasks_save_on_exit(self):
        """Handle saving background task information when exiting with active tasks"""
        try:
            # Create a comprehensive task status file
            task_status = {
                'active_exports': self.active_exports,
                'export_results': self.export_results,
                'active_segmentation': self.active_segmentation,
                'segmentation_results': self.segmentation_results,
                'timestamp': time.time(),
                'app_version': 'SAM2 Video UI Enhanced'
            }
            
            # Save to a temporary file
            status_file = os.path.join(tempfile.gettempdir(), 'sam2_background_tasks_status.json')
            with open(status_file, 'w') as f:
                json.dump(task_status, f, indent=2)
            
            # Show user where to find task status
            messagebox.showinfo(
                "Background Tasks Status Saved",
                f"Background tasks status saved to:\n{status_file}\n\n"
                f"Active tasks will continue in the background.\n"
                f"You can check this file to monitor progress."
            )
            
        except Exception as e:
            print(f"Error saving background tasks status: {e}")
            messagebox.showwarning(
                "Background Tasks Status Warning",
                f"Could not save task status: {str(e)}\n\n"
                f"Tasks will continue in background but status won't be saved."
            )
    
    def on_object_change(self):
        """Handle object selection change"""
        self.current_object_id = self.object_var.get()
        self.object_name_var.set(self.object_names[self.current_object_id])
        self.update_object_color_display()
        self.update_object_list()
        if self.frames:
            self.display_current_frame()
    
    def update_object_color_display(self):
        """Update the object color indicator"""
        color = self.object_colors[self.current_object_id]
        color_hex = self._rgb_to_hex(color)
        self.object_color_label.config(foreground=color_hex)
    
    def on_export_mode_change(self):
        """Handle export mode checkbox change"""
        if self.background_export_var.get():
            self.export_mode_label.config(text="Background Mode (can close app)", foreground='green')
        else:
            self.export_mode_label.config(text="Foreground Mode (wait for completion)", foreground='blue')
    
    def _perform_auto_export_after_segmentation(self):
        """Automatically export masks and video after successful segmentation"""
        try:
            # Ask user for export preferences
            export_dialog = tk.Toplevel(self.root)
            export_dialog.title("Auto-Export After Segmentation")
            export_dialog.geometry("400x300")
            export_dialog.configure(bg='#2b2b2b')
            export_dialog.transient(self.root)
            export_dialog.grab_set()
            
            # Variables for export settings
            export_masks = tk.BooleanVar(value=True)
            export_video = tk.BooleanVar(value=True)
            export_format = tk.StringVar(value="mp4")
            overlay_opacity = tk.DoubleVar(value=0.4)
            show_object_names = tk.BooleanVar(value=True)
            show_boundaries = tk.BooleanVar(value=True)
            fps_var = tk.DoubleVar(value=30.0)
            
            # UI Elements
            main_frame = ttk.Frame(export_dialog)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            ttk.Label(main_frame, text="Auto-Export Settings", 
                     font=('Arial', 14, 'bold')).pack(pady=(0, 15))
            
            # Export type selection
            type_frame = ttk.LabelFrame(main_frame, text="Export Types", padding=10)
            type_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Checkbutton(type_frame, text="Export Masks", 
                           variable=export_masks).pack(anchor=tk.W)
            ttk.Checkbutton(type_frame, text="Export Video", 
                           variable=export_video).pack(anchor=tk.W)
            
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
            
            def start_auto_export():
                export_dialog.destroy()
                self._execute_auto_export(export_masks.get(), export_video.get(), 
                                        export_format.get(), overlay_opacity.get(),
                                        show_object_names.get(), show_boundaries.get(), 
                                        fps_var.get())
                
            def skip_export():
                export_dialog.destroy()
                
            ttk.Button(button_frame, text="Start Auto-Export", command=start_auto_export).pack(side=tk.RIGHT, padx=(10, 0))
            ttk.Button(button_frame, text="Skip Export", command=skip_export).pack(side=tk.RIGHT)
            
            # Wait for dialog to close
            export_dialog.wait_window()
            
        except Exception as e:
            print(f"Error in auto-export dialog: {e}")
            messagebox.showerror("Auto-Export Error", f"Failed to start auto-export: {str(e)}")
    
    def _execute_auto_export(self, export_masks, export_video, export_format, 
                           overlay_opacity, show_object_names, show_boundaries, fps):
        """Execute the auto-export based on user preferences"""
        try:
            if export_masks:
                # Start background mask export
                self._start_background_mask_export()
            
            if export_video:
                # Start background video export with settings
                self._start_background_video_export_with_settings(
                    export_format, overlay_opacity, show_object_names, 
                    False, show_boundaries, fps, "overlay"
                )
            
            if export_masks or export_video:
                self.status_label.config(text="Auto-export started in background")
                messagebox.showinfo("Auto-Export Started", 
                                  f"Auto-export started in background.\n"
                                  f"Masks: {'Yes' if export_masks else 'No'}\n"
                                  f"Video: {'Yes' if export_video else 'No'}\n\n"
                                  f"You can now close the application if needed.")
            
        except Exception as e:
            messagebox.showerror("Auto-Export Error", f"Failed to start auto-export: {str(e)}")
    
    def _start_background_video_export_with_settings(self, export_format, overlay_opacity, 
                                                   show_object_names, show_object_ids, 
                                                   show_boundaries, fps, export_mode):
        """Start background video export with predefined settings"""
        # Get output file path with folder creation
        file_path = self._get_export_file_path_with_creation(
            title="Save Auto-Export Video As",
            default_name=f"sam2_auto_video.{export_format}",
            file_types=[
                (f"{export_format.upper()} files", f"*.{export_format}"),
                ("All files", "*.*")
            ],
            default_ext=f".{export_format}"
        )
        
        if not file_path:
            return
        
        # Generate unique export ID
        export_id = f"auto_video_{int(time.time())}"
        
        # Create export task
        export_task = {
            'id': export_id,
            'type': 'video',
            'file_path': file_path,
            'export_format': export_format,
            'overlay_opacity': overlay_opacity,
            'show_object_names': show_object_names,
            'show_object_ids': show_object_ids,
            'show_boundaries': show_boundaries,
            'fps': fps,
            'quality': 'medium',
            'export_mode': export_mode,
            'frames': self.frames.copy(),
            'masks': self.masks.copy(),
            'object_names': self.object_names.copy(),
            'object_colors': self.object_colors.copy(),
            'processing_range': getattr(self, 'processing_range', list(range(len(self.frames))))
        }
        
        # Start export worker if not running
        self._start_export_worker()
        
        # Add task to queue
        self.export_queue.put(export_task)
        
        # Show status
        self.status_label.config(text=f"Auto video export started (ID: {export_id})")
    
    def _start_background_segmentation(self):
        """Start background segmentation with auto-export"""
        try:
            # Get export preferences first
            export_dialog = tk.Toplevel(self.root)
            export_dialog.title("Background Segmentation + Export")
            export_dialog.geometry("450x400")
            export_dialog.configure(bg='#2b2b2b')
            export_dialog.transient(self.root)
            export_dialog.grab_set()
            
            # Variables for export settings
            export_masks = tk.BooleanVar(value=True)
            export_video = tk.BooleanVar(value=True)
            export_format = tk.StringVar(value="mp4")
            overlay_opacity = tk.DoubleVar(value=0.4)
            show_object_names = tk.BooleanVar(value=True)
            show_boundaries = tk.BooleanVar(value=True)
            fps_var = tk.DoubleVar(value=30.0)
            
            # UI Elements
            main_frame = ttk.Frame(export_dialog)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            ttk.Label(main_frame, text="Background Processing Settings", 
                     font=('Arial', 14, 'bold')).pack(pady=(0, 15))
            
            # Info text
            info_text = "Segmentation and export will run in background.\nYou can close the application during processing."
            ttk.Label(main_frame, text=info_text, 
                     foreground='green', font=('Arial', 10)).pack(pady=(0, 15))
            
            # Export type selection
            type_frame = ttk.LabelFrame(main_frame, text="Export Types", padding=10)
            type_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Checkbutton(type_frame, text="Export Masks", 
                           variable=export_masks).pack(anchor=tk.W)
            ttk.Checkbutton(type_frame, text="Export Video", 
                           variable=export_video).pack(anchor=tk.W)
            
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
            
            def start_background_processing():
                export_dialog.destroy()
                # Get export locations before starting
                self._get_background_export_locations_and_start(
                    export_masks.get(), export_video.get(), 
                    export_format.get(), overlay_opacity.get(),
                    show_object_names.get(), show_boundaries.get(), 
                    fps_var.get()
                )
                
            def cancel_processing():
                export_dialog.destroy()
                
            ttk.Button(button_frame, text="Start Background Processing", command=start_background_processing).pack(side=tk.RIGHT, padx=(10, 0))
            ttk.Button(button_frame, text="Cancel", command=cancel_processing).pack(side=tk.RIGHT)
            
            # Wait for dialog to close
            export_dialog.wait_window()
            
        except Exception as e:
            print(f"Error in background segmentation dialog: {e}")
            messagebox.showerror("Background Processing Error", f"Failed to start background processing: {str(e)}")
    
    def _get_background_export_locations_and_start(self, export_masks, export_video, export_format, 
                                                 overlay_opacity, show_object_names, show_boundaries, fps):
        """Get export locations and start background segmentation"""
        try:
            masks_folder = None
            video_file = None
            
            # Get mask export location if requested
            if export_masks:
                masks_folder = self._get_export_folder_with_creation(
                    title="Select folder to save masks", 
                    default_name="sam2_masks"
                )
                if not masks_folder:
                    return
            
            # Get video export location if requested
            if export_video:
                video_file = self._get_export_file_path_with_creation(
                    title="Save video file",
                    default_name=f"sam2_video.{export_format}",
                    file_types=[
                        (f"{export_format.upper()} files", f"*.{export_format}"),
                        ("All files", "*.*")
                    ],
                    default_ext=f".{export_format}"
                )
                if not video_file:
                    return
            
            # Start background segmentation with export locations
            self._execute_background_segmentation_with_locations(
                export_masks, export_video, export_format, overlay_opacity,
                show_object_names, show_boundaries, fps, masks_folder, video_file
            )
            
        except Exception as e:
            messagebox.showerror("Export Location Error", f"Failed to get export locations: {str(e)}")
    
    def _execute_background_segmentation_with_locations(self, export_masks, export_video, export_format, 
                                                       overlay_opacity, show_object_names, show_boundaries, 
                                                       fps, masks_folder, video_file):
        """Execute background segmentation with predefined export locations"""
        try:
            # Generate unique segmentation ID
            segmentation_id = f"seg_{int(time.time())}"
            
            # Create segmentation task with export locations
            segmentation_task = {
                'id': segmentation_id,
                'type': 'segmentation',
                'export_masks': export_masks,
                'export_video': export_video,
                'export_format': export_format,
                'overlay_opacity': overlay_opacity,
                'show_object_names': show_object_names,
                'show_boundaries': show_boundaries,
                'fps': fps,
                'masks_folder': masks_folder,
                'video_file': video_file,
                'frames': self.frames.copy(),
                'click_points': self.click_points.copy(),
                'object_names': self.object_names.copy(),
                'object_colors': self.object_colors.copy(),
                'video_path': self.video_path,
                'sam2_model': self.sam2_model,
                'inference_state': getattr(self, 'inference_state', None),
                'refinement_mode': self.refinement_mode,
                'selected_frames_for_refinement': self.selected_frames_for_refinement.copy(),
                'limit_to_range_var': self.limit_to_range_var.get(),
                'range_start_var': self.range_start_var.get(),
                'range_end_var': self.range_end_var.get(),
                'multi_frame_annotation_mode': self.multi_frame_annotation_mode,
                'annotated_frames': self.annotated_frames.copy()
            }
            
            # Start segmentation worker if not running
            self._start_segmentation_worker()
            
            # Add task to queue
            self.segmentation_queue.put(segmentation_task)
            
            # Show status
            self.status_label.config(text=f"Background segmentation started (ID: {segmentation_id})")
            messagebox.showinfo("Background Processing Started", 
                              f"Background segmentation and export started.\n"
                              f"Segmentation ID: {segmentation_id}\n"
                              f"Masks: {'Yes' if export_masks else 'No'}\n"
                              f"Video: {'Yes' if export_video else 'No'}\n\n"
                              f"You can now close the application.\n"
                              f"Processing will continue in the background.")
            
        except Exception as e:
            messagebox.showerror("Background Processing Error", f"Failed to start background processing: {str(e)}")
    
    def _execute_background_segmentation(self, export_masks, export_video, export_format, 
                                       overlay_opacity, show_object_names, show_boundaries, fps):
        """Execute background segmentation with auto-export"""
        try:
            # Generate unique segmentation ID
            segmentation_id = f"seg_{int(time.time())}"
            
            # Create segmentation task
            segmentation_task = {
                'id': segmentation_id,
                'type': 'segmentation',
                'export_masks': export_masks,
                'export_video': export_video,
                'export_format': export_format,
                'overlay_opacity': overlay_opacity,
                'show_object_names': show_object_names,
                'show_boundaries': show_boundaries,
                'fps': fps,
                'frames': self.frames.copy(),
                'click_points': self.click_points.copy(),
                'object_names': self.object_names.copy(),
                'object_colors': self.object_colors.copy(),
                'video_path': self.video_path,
                'sam2_model': self.sam2_model,
                'inference_state': getattr(self, 'inference_state', None),
                'refinement_mode': self.refinement_mode,
                'selected_frames_for_refinement': self.selected_frames_for_refinement.copy(),
                'limit_to_range_var': self.limit_to_range_var.get(),
                'range_start_var': self.range_start_var.get(),
                'range_end_var': self.range_end_var.get(),
                'multi_frame_annotation_mode': self.multi_frame_annotation_mode,
                'annotated_frames': self.annotated_frames.copy()
            }
            
            # Start segmentation worker if not running
            self._start_segmentation_worker()
            
            # Add task to queue
            self.segmentation_queue.put(segmentation_task)
            
            # Show status
            self.status_label.config(text=f"Background segmentation started (ID: {segmentation_id})")
            messagebox.showinfo("Background Processing Started", 
                              f"Background segmentation and export started.\n"
                              f"Segmentation ID: {segmentation_id}\n"
                              f"Masks: {'Yes' if export_masks else 'No'}\n"
                              f"Video: {'Yes' if export_video else 'No'}\n\n"
                              f"You can now close the application.\n"
                              f"Processing will continue in the background.")
            
        except Exception as e:
            messagebox.showerror("Background Processing Error", f"Failed to start background processing: {str(e)}")
    
    def _start_segmentation_worker(self):
        """Start the background segmentation worker thread"""
        if not self.segmentation_running:
            self.segmentation_running = True
            # Make thread non-daemon so it can continue after UI closes
            self.segmentation_thread = threading.Thread(target=self._segmentation_worker, daemon=False)
            self.segmentation_thread.start()
    
    def _segmentation_worker(self):
        """Background worker thread for handling segmentation"""
        while self.segmentation_running:
            try:
                # Get segmentation task from queue (with timeout)
                segmentation_task = self.segmentation_queue.get(timeout=1.0)
                
                if segmentation_task is None:  # Shutdown signal
                    break
                
                segmentation_id = segmentation_task.get('id')
                
                # Mark segmentation as active
                self.active_segmentation[segmentation_id] = {
                    'type': 'segmentation',
                    'status': 'running',
                    'start_time': time.time(),
                    'progress': 0
                }
                
                try:
                    result = self._perform_background_segmentation(segmentation_task)
                    
                    # Store result
                    self.segmentation_results[segmentation_id] = result
                    self.active_segmentation[segmentation_id]['status'] = 'completed'
                    
                except Exception as e:
                    self.segmentation_results[segmentation_id] = {
                        'success': False, 
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                    self.active_segmentation[segmentation_id]['status'] = 'failed'
                
                finally:
                    self.segmentation_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Segmentation worker error: {e}")
                continue
    
    def _stop_segmentation_worker(self):
        """Stop the background segmentation worker thread"""
        self.segmentation_running = False
        if self.segmentation_thread and self.segmentation_thread.is_alive():
            # Send shutdown signal
            self.segmentation_queue.put(None)
    
    def check_background_segmentation_results(self):
        """Check for completed background segmentation results and offer refinement"""
        if not self.segmentation_results:
            return
        
        # Find completed segmentation results
        completed_results = []
        for seg_id, result in self.segmentation_results.items():
            if result.get('success', False) and seg_id in self.active_segmentation:
                if self.active_segmentation[seg_id]['status'] == 'completed':
                    completed_results.append((seg_id, result))
        
        if not completed_results:
            return
        
        # Process the most recent completed result
        seg_id, result = completed_results[-1]  # Get the latest result
        
        # Load the segmentation results into the UI
        if 'masks' in result:
            self._load_background_segmentation_results(result)
            
            # Remove from active segmentation
            if seg_id in self.active_segmentation:
                del self.active_segmentation[seg_id]
            
            # Offer refinement options
            self._offer_refinement_after_background_segmentation(result)
    
    def _load_background_segmentation_results(self, result):
        """Load background segmentation results into the UI"""
        try:
            # Load masks into the UI
            if 'masks' in result:
                self.masks = result['masks']
            
            # Load object information if available
            if 'object_names' in result:
                self.object_names.update(result['object_names'])
            if 'object_colors' in result:
                self.object_colors.update(result['object_colors'])
            
            # Update UI elements
            self.show_masks_var.set(True)
            self.update_object_list()
            self.display_current_frame()
            
            # Update status
            total_masks = sum(len(frame_masks) for frame_masks in self.masks.values())
            unique_objects = set()
            for frame_masks in self.masks.values():
                unique_objects.update(frame_masks.keys())
            
            self.status_label.config(text=f"Background segmentation loaded! {total_masks} masks for {len(unique_objects)} objects")
            
        except Exception as e:
            print(f"Error loading background segmentation results: {e}")
            messagebox.showerror("Load Error", f"Failed to load background segmentation results: {str(e)}")
    
    def _offer_refinement_after_background_segmentation(self, result):
        """Offer refinement options after background segmentation completes"""
        total_masks = sum(len(frame_masks) for frame_masks in result.get('masks', {}).values())
        unique_objects = set()
        for frame_masks in result.get('masks', {}).values():
            unique_objects.update(frame_masks.keys())
        
        # Show completion message with refinement option
        refinement_choice = messagebox.askyesnocancel(
            "Background Segmentation Complete",
            f"Background segmentation completed successfully!\n\n"
            f"Results:\n"
            f"- Total masks: {total_masks}\n"
            f"- Objects: {len(unique_objects)}\n"
            f"- Processed frames: {result.get('processed_frames', 'Unknown')}\n\n"
            f"Would you like to refine the segmentation?\n\n"
            f"Yes: Enter refinement mode to improve masks\n"
            f"No: Keep current results as-is\n"
            f"Cancel: View results without refinement"
        )
        
        if refinement_choice is None:  # Cancel - just view results
            self.status_label.config(text="Background segmentation results loaded. Use controls to navigate and view masks.")
        elif refinement_choice:  # Yes - enter refinement mode
            self._enter_refinement_mode_after_background()
        else:  # No - keep results as-is
            self.status_label.config(text="Background segmentation results loaded. Ready for export or further annotation.")
    
    def _enter_refinement_mode_after_background(self):
        """Enter refinement mode after background segmentation"""
        # Enable refinement mode
        self.refinement_mode = True
        self.refinement_label.config(text="REFINEMENT MODE ACTIVE")
        
        # Update status
        self.status_label.config(text="Refinement mode: Select frames to improve, add points, then re-segment")
        
        # Enable frame selection button if it exists
        if hasattr(self, 'select_frame_button'):
            self.select_frame_button.state(["!disabled"])
        
        # Show refinement instructions
        messagebox.showinfo(
            "Refinement Mode Active",
            "Refinement mode is now active!\n\n"
            "To refine the segmentation:\n"
            "1. Navigate to frames with poor segmentation\n"
            "2. Click 'Select Frame' to mark frames for refinement\n"
            "3. Add positive/negative points to improve masks\n"
            "4. Click 'Segment Video' to re-segment selected frames\n\n"
            "The refined masks will replace the original ones."
        )
    
    def _perform_background_segmentation(self, segmentation_task):
        """Perform segmentation in background with auto-export"""
        try:
            # Extract task parameters
            segmentation_id = segmentation_task['id']
            export_masks = segmentation_task['export_masks']
            export_video = segmentation_task['export_video']
            export_format = segmentation_task['export_format']
            overlay_opacity = segmentation_task['overlay_opacity']
            show_object_names = segmentation_task['show_object_names']
            show_boundaries = segmentation_task['show_boundaries']
            fps = segmentation_task['fps']
            frames = segmentation_task['frames']
            click_points = segmentation_task['click_points']
            object_names = segmentation_task['object_names']
            object_colors = segmentation_task['object_colors']
            video_path = segmentation_task['video_path']
            sam2_model = segmentation_task['sam2_model']
            inference_state = segmentation_task['inference_state']
            refinement_mode = segmentation_task['refinement_mode']
            selected_frames_for_refinement = segmentation_task['selected_frames_for_refinement']
            limit_to_range_var = segmentation_task['limit_to_range_var']
            range_start_var = segmentation_task['range_start_var']
            range_end_var = segmentation_task['range_end_var']
            multi_frame_annotation_mode = segmentation_task['multi_frame_annotation_mode']
            annotated_frames = segmentation_task['annotated_frames']
            
            # Create temporary directory for frames
            temp_dir = tempfile.mkdtemp(prefix='sam2_bg_frames_')
            
            try:
                # Determine which frames to save based on processing range
                if limit_to_range_var:
                    start_idx = max(0, min(range_start_var, len(frames)-1))
                    end_idx = max(0, min(range_end_var, len(frames)-1))
                    if end_idx < start_idx:
                        start_idx, end_idx = end_idx, start_idx
                    frames_to_save = list(range(start_idx, end_idx + 1))
                else:
                    frames_to_save = list(range(len(frames)))
                
                # Save frames for processing
                for save_idx, frame_idx in enumerate(frames_to_save):
                    frame = frames[frame_idx]
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_path = os.path.join(temp_dir, f"{save_idx:05d}.jpg")
                    cv2.imwrite(frame_path, frame_bgr)
                
                # Initialize or reuse inference state
                if not inference_state:
                    inference_state = sam2_model.init_state(video_path=temp_dir)
                
                # Group click points by frame and by object ID
                points_by_frame_and_object = {}
                for x, y, is_pos, obj_id, frame_idx in click_points:
                    # Map original frame index to limited video index
                    if limit_to_range_var:
                        if frame_idx not in frames_to_save:
                            continue
                        limited_frame_idx = frames_to_save.index(frame_idx)
                    else:
                        limited_frame_idx = frame_idx
                    
                    if limited_frame_idx not in points_by_frame_and_object:
                        points_by_frame_and_object[limited_frame_idx] = {}
                    if obj_id not in points_by_frame_and_object[limited_frame_idx]:
                        points_by_frame_and_object[limited_frame_idx][obj_id] = {'points': [], 'labels': []}
                    points_by_frame_and_object[limited_frame_idx][obj_id]['points'].append([x, y])
                    points_by_frame_and_object[limited_frame_idx][obj_id]['labels'].append(1 if is_pos else 0)
                
                # Initialize masks dictionary
                masks = {}
                for frame_idx in range(len(frames)):
                    masks[frame_idx] = {}
                
                # Process each annotation frame and its objects
                annotation_frames = sorted(points_by_frame_and_object.keys())
                ann_frame_idx = annotation_frames[0] if annotation_frames else 0
                
                for ann_frame in annotation_frames:
                    obj_dict = points_by_frame_and_object[ann_frame]
                    for obj_id, point_data in obj_dict.items():
                        points = np.array(point_data['points'], dtype=np.float32)
                        labels = np.array(point_data['labels'], dtype=np.int32)
                        
                        try:
                            # Ensure points and labels are in correct dtype and device
                            points_tensor, labels_tensor = self._prepare_tensors_for_inference(points, labels)
                            
                            # Disable autocast to prevent dtype mismatches
                            with self._disable_autocast_for_inference():
                                _ = sam2_model.add_new_points(
                                    inference_state=inference_state,
                                    frame_idx=ann_frame,
                                    obj_id=obj_id,
                                    points=points_tensor,
                                    labels=labels_tensor,
                                )
                        except Exception as e:
                            print(f"Error adding points for object {obj_id} on frame {ann_frame}: {e}")
                            continue
                
                # Determine which frames to process
                if refinement_mode and selected_frames_for_refinement:
                    frames_to_process = sorted(selected_frames_for_refinement)
                else:
                    if limit_to_range_var:
                        start_idx = max(0, min(range_start_var, len(frames)-1))
                        end_idx = max(0, min(range_end_var, len(frames)-1))
                        if end_idx < start_idx:
                            start_idx, end_idx = end_idx, start_idx
                        frames_to_process = list(range(start_idx, end_idx + 1))
                    else:
                        frames_to_process = list(range(len(frames)))
                
                # Propagate through video
                processed_frames = 0
                
                for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(inference_state):
                    # Map limited video frame index back to original frame index
                    if limit_to_range_var:
                        if out_frame_idx >= len(frames_to_save):
                            continue
                        original_frame_idx = frames_to_save[out_frame_idx]
                    else:
                        original_frame_idx = out_frame_idx
                    
                    # Skip frames not in processing list
                    if original_frame_idx not in frames_to_process:
                        continue
                        
                    # Process each object mask
                    for i, out_obj_id in enumerate(out_obj_ids):
                        if any(out_obj_id in obj_dict for obj_dict in points_by_frame_and_object.values()):
                            mask_logits = out_mask_logits[i]
                            
                            # Handle torch tensors
                            if hasattr(mask_logits, 'cpu'):
                                mask_logits = mask_logits.cpu()
                            if hasattr(mask_logits, 'numpy'):
                                mask_logits = mask_logits.numpy()
                            
                            # Convert to binary mask
                            mask = (mask_logits > 0.0)
                            
                            # Ensure mask is 2D
                            if len(mask.shape) > 2:
                                mask = mask.squeeze()
                            
                            # Store mask with original frame index
                            masks[original_frame_idx][out_obj_id] = (mask * 255).astype(np.uint8)
                    
                    processed_frames += 1
                
                # Count results
                total_masks = sum(len(frame_masks) for frame_masks in masks.values())
                unique_objects = set()
                for frame_masks in masks.values():
                    unique_objects.update(frame_masks.keys())
                
                if total_masks > 0:
                    # Start auto-export if requested using predefined locations
                    if export_masks or export_video:
                        masks_folder = segmentation_task.get('masks_folder')
                        video_file = segmentation_task.get('video_file')
                        self._start_background_export_after_segmentation_with_locations(
                            masks, object_names, object_colors, click_points, 
                            video_path, len(frames), ann_frame_idx,
                            export_masks, export_video, export_format,
                            overlay_opacity, show_object_names, show_boundaries, fps,
                            frames, frames_to_process, masks_folder, video_file
                        )
                    
                    return {
                        'success': True,
                        'total_masks': total_masks,
                        'unique_objects': len(unique_objects),
                        'processed_frames': processed_frames,
                        'masks': masks,
                        'export_started': export_masks or export_video
                    }
                else:
                    return {
                        'success': False,
                        'error': 'No masks were generated',
                        'total_masks': 0
                    }
                
            finally:
                # Clean up temp directory
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Could not clean up temp directory: {e}")
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _start_background_export_after_segmentation(self, masks, object_names, object_colors, 
                                                  click_points, video_path, frames_count, 
                                                  ann_frame_idx, export_masks, export_video, 
                                                  export_format, overlay_opacity, show_object_names, 
                                                  show_boundaries, fps, frames, processing_range):
        """Start background export after successful segmentation"""
        try:
            if export_masks:
                # Get folder path for masks with user selection
                masks_folder = self._get_export_folder_with_creation(
                    title="Select folder to save masks", 
                    default_name=f"sam2_masks_{int(time.time())}"
                )
                if not masks_folder:
                    return
                
                # Create mask export task
                mask_export_task = {
                    'id': f"auto_masks_{int(time.time())}",
                    'type': 'masks',
                    'folder_path': masks_folder,
                    'masks': masks,
                    'object_names': object_names,
                    'object_colors': object_colors,
                    'click_points': click_points,
                    'video_path': video_path,
                    'frames': frames_count,
                    'ann_frame_idx': ann_frame_idx,
                    'sam2_info': {
                        'base_path': self.sam2_base_path,
                        'checkpoint_dir': self.checkpoint_dir,
                        'config_dir': self.config_dir
                    }
                }
                
                # Start export worker if not running
                self._start_export_worker()
                self.export_queue.put(mask_export_task)
            
            if export_video:
                # Get file path for video with user selection
                video_file = self._get_export_file_path_with_creation(
                    title="Save video file",
                    default_name=f"sam2_video_{int(time.time())}.{export_format}",
                    file_types=[
                        (f"{export_format.upper()} files", f"*.{export_format}"),
                        ("All files", "*.*")
                    ],
                    default_ext=f".{export_format}"
                )
                if not video_file:
                    return
                
                # Create video export task
                video_export_task = {
                    'id': f"auto_video_{int(time.time())}",
                    'type': 'video',
                    'file_path': video_file,
                    'export_format': export_format,
                    'overlay_opacity': overlay_opacity,
                    'show_object_names': show_object_names,
                    'show_object_ids': False,
                    'show_boundaries': show_boundaries,
                    'fps': fps,
                    'quality': 'medium',
                    'export_mode': 'overlay',
                    'frames': frames,
                    'masks': masks,
                    'object_names': object_names,
                    'object_colors': object_colors,
                    'processing_range': processing_range
                }
                
                # Start export worker if not running
                self._start_export_worker()
                self.export_queue.put(video_export_task)
                
        except Exception as e:
            print(f"Error starting background export after segmentation: {e}")
    
    def _start_background_export_after_segmentation_with_locations(self, masks, object_names, object_colors, 
                                                                 click_points, video_path, frames_count, 
                                                                 ann_frame_idx, export_masks, export_video, 
                                                                 export_format, overlay_opacity, show_object_names, 
                                                                 show_boundaries, fps, frames, processing_range,
                                                                 masks_folder, video_file):
        """Start background export after successful segmentation using predefined locations"""
        try:
            if export_masks and masks_folder:
                # Create mask export task with predefined folder
                mask_export_task = {
                    'id': f"auto_masks_{int(time.time())}",
                    'type': 'masks',
                    'folder_path': masks_folder,
                    'masks': masks,
                    'object_names': object_names,
                    'object_colors': object_colors,
                    'click_points': click_points,
                    'video_path': video_path,
                    'frames': frames_count,
                    'ann_frame_idx': ann_frame_idx,
                    'sam2_info': {
                        'base_path': self.sam2_base_path,
                        'checkpoint_dir': self.checkpoint_dir,
                        'config_dir': self.config_dir
                    }
                }
                
                # Start export worker if not running
                self._start_export_worker()
                self.export_queue.put(mask_export_task)
            
            if export_video and video_file:
                # Create video export task with predefined file
                video_export_task = {
                    'id': f"auto_video_{int(time.time())}",
                    'type': 'video',
                    'file_path': video_file,
                    'export_format': export_format,
                    'overlay_opacity': overlay_opacity,
                    'show_object_names': show_object_names,
                    'show_object_ids': False,
                    'show_boundaries': show_boundaries,
                    'fps': fps,
                    'quality': 'medium',
                    'export_mode': 'overlay',
                    'frames': frames,
                    'masks': masks,
                    'object_names': object_names,
                    'object_colors': object_colors,
                    'processing_range': processing_range
                }
                
                # Start export worker if not running
                self._start_export_worker()
                self.export_queue.put(video_export_task)
                
        except Exception as e:
            print(f"Error starting background export after segmentation with locations: {e}")
    

    def load_sam2_model(self):
        """Load SAM2 model with correct video predictor initialization"""
        try:
            self.status_label.config(text="Loading SAM2 model...")
            self.model_status_label.config(text="Loading...", foreground='orange')
            self.root.update()

            # Use the correct checkpoint and config for video segmentation
            sam2_checkpoint = os.path.join(self.checkpoint_dir, "sam2_hiera_small.pt")
            model_cfg = os.path.join(self.sam2_base_path, "configs", "sam2_hiera_s.yaml")

            # Check files exist
            if not os.path.exists(sam2_checkpoint):
                raise FileNotFoundError(f"Checkpoint not found: {sam2_checkpoint}")
            if not os.path.exists(model_cfg):
                raise FileNotFoundError(f"Config not found: {model_cfg}")

            # Import the correct builder for VIDEO segmentation
            from sam2.build_sam import build_sam2_video_predictor
        
            # Select device based on user selection
            device = self._get_selected_device()
            
            # Validate device selection
            if device.startswith("cuda:"):
                gpu_id = int(device.split(":")[1])
                if not torch or not torch.cuda.is_available():
                    device = "cpu"
                    self.status_label.config(text="CUDA not available, using CPU...")
                elif gpu_id >= torch.cuda.device_count():
                    device = "cpu"
                    self.status_label.config(text=f"GPU {gpu_id} not available, using CPU...")
                else:
                    self.status_label.config(text=f"Using GPU {gpu_id} for inference...")
            elif device == "cuda":
                if torch and torch.cuda.is_available():
                    self.status_label.config(text="Using CUDA GPU for inference...")
                else:
                    device = "cpu"
                    self.status_label.config(text="CUDA not available, using CPU...")
            else:  # cpu
                self.status_label.config(text="Using CPU for inference (slower)...")

            # Build the VIDEO predictor (not the base model)
            with self._disable_autocast_for_inference():
                self.sam2_model = build_sam2_video_predictor(
                    config_file=model_cfg,
                    ckpt_path=sam2_checkpoint,
                    device=device
                )
            
            # Ensure consistent dtype handling
            if hasattr(self.sam2_model, 'model') and hasattr(self.sam2_model.model, 'to'):
                # Convert model to float32 to avoid dtype mismatches
                self.sam2_model.model = self.sam2_model.model.to(dtype=torch.float32)
            
            # Set model to evaluation mode
            if hasattr(self.sam2_model, 'model'):
                self.sam2_model.model.eval()
            
            # Ensure dtype consistency to prevent BFloat16/Float mismatches
            self._ensure_model_dtype_consistency()
            
            # Force all model components to float32
            self._force_model_float32()
            
            # Patch model to force float32 operations
            self._patch_model_for_float32()
            
            # Disable mixed precision globally
            self._disable_mixed_precision_globally()

            self.model_loaded = True

            model_name = os.path.basename(sam2_checkpoint).replace('.pt', '')
            self.model_status_label.config(text=f"{model_name} ({device.upper()})", foreground='green')
            self.status_label.config(text=f"SAM2 video predictor loaded successfully on {device.upper()}")
        
            # Test that the model has the required methods
            if not hasattr(self.sam2_model, 'init_state'):
                raise AttributeError("Model does not have 'init_state' method. Check SAM2 installation.")
            if not hasattr(self.sam2_model, 'add_new_points'):
                raise AttributeError("Model does not have 'add_new_points' method. Check SAM2 installation.")
            
        except Exception as e:
            self.model_status_label.config(text="Load Failed", foreground='red')
            traceback.print_exc()
            error_msg = f"Failed to load SAM2 model: {str(e)}\n\nPossible solutions:\n"
            error_msg += "1. Ensure SAM2 is properly installed\n"
            error_msg += "2. Check checkpoint and config file paths\n"
            error_msg += "3. Verify you have the video segmentation version\n"
            error_msg += "4. Try reinstalling SAM2 from the official repository"
            messagebox.showerror("Model Load Error", error_msg)


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
                app._handle_background_tasks_save_on_exit()
                # Don't stop workers - let them continue
                # Just destroy the UI and let the process continue
                root.destroy()
                return
            # No - just quit (tasks will be lost)
            else:
                # Stop workers and quit normally
                app._stop_export_worker()
                app._stop_segmentation_worker()
        else:
            # No active tasks, just quit normally
            app._stop_export_worker()
            app._stop_segmentation_worker()
        
        # Clean up video capture if lazy loading
        if hasattr(app, 'video_cap_lazy') and app.video_cap_lazy:
            app.video_cap_lazy.release()
        
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start background task checking
    app.check_background_tasks()
    
    root.mainloop()

if __name__ == "__main__":
    main()
