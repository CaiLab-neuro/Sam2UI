import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
        
        # Hardcoded paths
        self.sam2_base_path = None # r"C:\Users\kevin\segment-anything-2"
        self.checkpoint_dir = None # r"C:\Users\kevin\segment-anything-2\checkpoints"
        self.config_dir = None # r"C:\Users\kevin\segment-anything-2\configs"
        
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
        
        # Initialize default colors and names
        self._initialize_objects()
        
        # SAM2 model
        self.sam2_model = None
        self.model_loaded = False
        
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
        
    def set_sam2_path(self):
        """Allow user to select SAM2 installation directory"""
        folder_path = filedialog.askdirectory(
            title="Select SAM2 Installation Directory",
            initialdir=os.path.expanduser("~")
        )
        
        if not folder_path:
            return
            
        # Validate the selected path
        expected_files = [
            os.path.join(folder_path, "checkpoints"),
            os.path.join(folder_path, "sam2")
        ]
        
        
        missing_items = [item for item in expected_files if not os.path.exists(item)]
        
        if missing_items:
            messagebox.showerror("Invalid SAM2 Path", 
                            f"The selected directory doesn't appear to be a valid SAM2 installation.\n\n"
                            f"Missing items:\n" + "\n".join(f"• {os.path.basename(item)}" for item in missing_items))
            return
        
        # Set the paths
        self.sam2_base_path = folder_path
        self.checkpoint_dir = os.path.join(folder_path, "checkpoints")
        self.config_dir = os.path.join(folder_path, "sam2", "configs")
        
        # Add to Python path
        if folder_path not in sys.path:
            sys.path.append(folder_path)
        
        self.status_label.config(text=f"SAM2 path set: {folder_path}")
        messagebox.showinfo("Path Set", f"SAM2 path configured successfully!\n\nPath: {folder_path}")

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
        # Title
        title_label = ttk.Label(parent, text="SAM2 Enhanced", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # File operations
        file_frame = ttk.LabelFrame(parent, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="📁 Set SAM2 Path", 
                   command=self.set_sam2_path, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="📁 Load Video",
                   command=self.load_video, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="🤖 Load SAM2 Model",
                   command=self.load_sam2_model, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="📤 Import Object List", 
                   command=self.import_object_list, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="📥 Export Object List",
                   command=self.export_object_list, width=15).pack(fill=tk.X, pady=2)
        
        # Model status
        self.model_status_label = ttk.Label(file_frame, text="❌ Model Not Loaded", 
                                           foreground='red')
        self.model_status_label.pack(pady=5)
        
        # Enhanced Object Management
        obj_frame = ttk.LabelFrame(parent, text="Object Management", padding=10)
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
        self.object_color_label = ttk.Label(current_obj_frame, text="●", 
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
        
        ttk.Button(name_frame, text="✓", command=self.update_object_name, width=3).pack(side=tk.RIGHT)
        
        # Object control buttons
        obj_buttons_frame = ttk.Frame(obj_frame)
        obj_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(obj_buttons_frame, text="➕ Add New", 
                  command=self.add_new_object, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(obj_buttons_frame, text="🗑️ Clear Obj", 
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
        seg_frame = ttk.LabelFrame(parent, text="Segmentation", padding=10)
        seg_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(seg_frame, text="🎯 Segment Video", 
                  command=self.segment_video, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="🔧 Refine Segment", 
                  command=self.toggle_refinement_mode, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="🗑️ Clear All Points", 
                  command=self.clear_points, width=15).pack(fill=tk.X, pady=2)
        
        # Refinement mode indicator
        self.refinement_label = ttk.Label(seg_frame, text="", foreground='orange')
        self.refinement_label.pack(pady=2)
        
        # Export controls
        export_frame = ttk.LabelFrame(parent, text="Export", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(export_frame, text="💾 Export Masks", 
                  command=self.export_masks, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="🎬 Export Video", 
                  command=self.export_video, width=15).pack(fill=tk.X, pady=2)
        
        # Display options
        display_frame = ttk.LabelFrame(parent, text="Display Options", padding=10)
        display_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.show_masks_var = tk.BooleanVar()
        ttk.Checkbutton(display_frame, text="Show Masks", 
                       variable=self.show_masks_var,
                       command=self.toggle_mask_display).pack(anchor=tk.W)
        
        # Status info
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
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
        
        self.play_button = ttk.Button(playback_frame, text="▶️ Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(playback_frame, text="⏮️ Prev", command=self.prev_frame).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(playback_frame, text="⏭️ Next", command=self.next_frame).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(playback_frame, text="⏹️ Reset", command=self.reset_video).pack(side=tk.LEFT, padx=(10, 0))
        
        # Frame selection for refinement
        if self.refinement_mode:
            ttk.Button(playback_frame, text="📌 Select Frame", 
                      command=self.toggle_frame_selection).pack(side=tk.LEFT, padx=(10, 0))
        
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
        for _, _, _, obj_id in self.click_points:
            used_objects.add(obj_id)
        
        # Find objects with masks
        for frame_masks in self.masks.values():
            used_objects.update(frame_masks.keys())
        
        # Always show current object
        used_objects.add(self.current_object_id)
        
        for obj_id in sorted(used_objects):
            # Count points for this object
            point_count = sum(1 for _, _, _, oid in self.click_points if oid == obj_id)
            
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
    
    def toggle_refinement_mode(self):
        """Toggle refinement mode for improving segmentation"""
        self.refinement_mode = not self.refinement_mode
        
        if self.refinement_mode:
            self.refinement_label.config(text="🔧 REFINEMENT MODE ACTIVE")
            self.status_label.config(text="Refinement mode: Select frames to improve, add points, then re-segment")
        else:
            self.refinement_label.config(text="")
            self.selected_frames_for_refinement.clear()
            self.status_label.config(text="Refinement mode disabled")
            
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
        """Extract all frames from video"""
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
            
            self.status_label.config(text=f"Loading {total_frames} frames...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.root.update()
            
            frame_count = 0
            while True:
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame_rgb)
                frame_count += 1
                
                # Update progress
                progress = (frame_count / total_frames) * 100
                self.progress_var.set(progress)
                self.root.update_idletasks()
            
            self.video_cap.release()
            self.progress_bar.pack_forget()
            
            if self.frames:
                self.current_frame_idx = 0
                self.frame_slider.config(to=len(self.frames)-1)
                self.display_current_frame()
                self.status_label.config(text=f"Video loaded: {len(self.frames)} frames @ {fps:.1f} FPS")
                self.update_object_list()
            else:
                raise ValueError("No frames could be extracted from video")
                
        except Exception as e:
            self.progress_bar.pack_forget()
            raise e

    def display_current_frame(self):
        """Display current video frame with overlays"""
        if not self.frames:
            return
            
        self.current_frame = self.frames[self.current_frame_idx].copy()
        display_frame = self.current_frame.copy()
        
        # Add frame selection indicator for refinement mode
        if self.refinement_mode and self.current_frame_idx in self.selected_frames_for_refinement:
            # Add orange border for selected frames
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1]-1, display_frame.shape[0]-1), 
                         (255, 165, 0), 10)
        
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
        
        # Draw click points
        for i, (x, y, is_positive, obj_id) in enumerate(self.click_points):
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
        self.frame_label.config(text=f"{self.current_frame_idx + 1}/{len(self.frames)}")
        
    def on_canvas_click(self, event):
        """Handle left mouse click (positive point)"""
        self.add_click_point(event, is_positive=True)
        
    def on_canvas_right_click(self, event):
        """Handle right mouse click (negative point)"""
        self.add_click_point(event, is_positive=False)
        
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
                self.click_points.append((img_x, img_y, is_positive, self.current_object_id))
                self.update_points_display()
                self.update_object_list()
                self.display_current_frame()
                
    def update_points_display(self):
        """Update the points display label"""
        if self.click_points:
            # Count points by object
            object_counts = {}
            for _, _, is_pos, obj_id in self.click_points:
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
        else:
            points_text = "No points (Left: +, Right: -)"
        
        self.points_label.config(text=points_text)
            
    def clear_points(self):
        """Clear all click points"""
        self.click_points = []
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
            self.play_button.config(text="⏸️ Pause")
            threading.Thread(target=self.play_video, daemon=True).start()
        else:
            self.play_button.config(text="▶️ Play")
            
    def play_video(self):
        """Play video in separate thread"""
        while self.playing and self.frames:
            if self.current_frame_idx < len(self.frames) - 1:
                self.current_frame_idx += 1
                self.root.after(0, self.display_current_frame)
                threading.Event().wait(0.033)  # ~30 FPS
            else:
                self.playing = False
                self.root.after(0, lambda: self.play_button.config(text="▶️ Play"))
                break

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
                # Save frames
                for frame_idx, frame in enumerate(self.frames):
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_path = os.path.join(temp_dir, f"{frame_idx:05d}.jpg")
                    cv2.imwrite(frame_path, frame_bgr)
                    
                    progress = (frame_idx / len(self.frames)) * 30
                    self.progress_var.set(progress)
                    if frame_idx % 10 == 0:
                        self.root.update_idletasks()
                
                self.status_label.config(text="Initializing SAM2 inference...")
                self.progress_var.set(35)
                self.root.update()
                
                # Initialize or reuse inference state
                if not hasattr(self, 'inference_state') or self.inference_state is None:
                    self.inference_state = self.sam2_model.init_state(video_path=temp_dir)
                
                # Group click points by object ID
                points_by_object = {}
                for x, y, is_pos, obj_id in self.click_points:
                    if obj_id not in points_by_object:
                        points_by_object[obj_id] = {'points': [], 'labels': []}
                    points_by_object[obj_id]['points'].append([x, y])
                    points_by_object[obj_id]['labels'].append(1 if is_pos else 0)
                
                self.status_label.config(text="Adding prompts to SAM2...")
                self.progress_var.set(40)
                self.root.update()
                
                # Initialize masks dictionary if not exists
                if not hasattr(self, 'masks'):
                    self.masks = {}
                    
                for frame_idx in range(len(self.frames)):
                    if frame_idx not in self.masks:
                        self.masks[frame_idx] = {}
                
                # Use current frame as annotation frame (or first selected frame in refinement)
                if is_refinement and self.selected_frames_for_refinement:
                    ann_frame_idx = min(self.selected_frames_for_refinement)
                else:
                    ann_frame_idx = self.current_frame_idx
                
                # Process each object
                for obj_id, point_data in points_by_object.items():
                    points = np.array(point_data['points'], dtype=np.float32)
                    labels = np.array(point_data['labels'], dtype=np.int32)
                    
                    obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                    self.status_label.config(text=f"Processing {obj_name}...")
                    self.root.update()
                    
                    try:
                        # Add points for this object
                        result = self.sam2_model.add_new_points(
                            inference_state=self.inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=obj_id,
                            points=points,
                            labels=labels,
                        )
                    except Exception as e:
                        print(f"Error adding points for {obj_name}: {e}")
                        continue
                
                # Propagate through video (or just selected frames in refinement)
                if is_refinement:
                    self.status_label.config(text="Refining selected frames...")
                    frames_to_process = sorted(self.selected_frames_for_refinement)
                else:
                    self.status_label.config(text="Propagating through entire video...")
                    frames_to_process = list(range(len(self.frames)))
                
                self.progress_var.set(45)
                self.root.update()
                
                processed_frames = 0
                
                try:
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(self.inference_state):
                        # Skip frames not in processing list during refinement
                        if is_refinement and out_frame_idx not in frames_to_process:
                            continue
                            
                        # Process each object mask
                        for i, out_obj_id in enumerate(out_obj_ids):
                            if out_obj_id in points_by_object:
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
                                
                                # Store mask
                                self.masks[out_frame_idx][out_obj_id] = (mask * 255).astype(np.uint8)
                        
                        processed_frames += 1
                        progress = 45 + (processed_frames / len(frames_to_process)) * 55
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
                    
                    self.show_masks_var.set(True)
                    self.update_object_list()
                    self.display_current_frame()
                    
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
            
        folder_path = filedialog.askdirectory(title="Select folder to save masks")
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
            for x, y, is_pos, obj_id in self.click_points:
                if obj_id not in metadata["click_points_by_object"]:
                    metadata["click_points_by_object"][obj_id] = []
                metadata["click_points_by_object"][obj_id].append({
                    "x": float(x), 
                    "y": float(y), 
                    "positive": bool(is_pos),
                    "object_name": self.object_names.get(obj_id, f"Object_{obj_id}")
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
            
            messagebox.showinfo("Export Complete", 
                              f"Successfully exported:\n"
                              f"• {exported_count} mask images (PNG)\n"
                              f"• 1 metadata file (JSON)\n"
                              f"• 1 object mapping (CSV)\n\n"
                              f"Location: {folder_path}")
                
        except Exception as e:
            self.progress_bar.pack_forget()
            self.status_label.config(text="Export failed")
            messagebox.showerror("Export Error", f"Failed to export masks: {str(e)}")

    def export_video(self):
        """Export video with mask overlays and enhanced object visualization"""
        if not self.frames:
            messagebox.showwarning("Warning", "No video loaded")
            return
            
        if not self.masks:
            messagebox.showwarning("Warning", "No masks to export. Please segment the video first.")
            return
            
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
        
        # Get output file path
        file_path = filedialog.asksaveasfilename(
            title="Save Video As",
            defaultextension=f".{export_format}",
            filetypes=[
                (f"{export_format.upper()} files", f"*.{export_format}"),
                ("All files", "*.*")
            ]
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
            
            total_frames = len(self.frames)
            
            for frame_idx, frame in enumerate(self.frames):
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
                progress = ((frame_idx + 1) / total_frames) * 100
                self.progress_var.set(progress)
                
                if frame_idx % 10 == 0:
                    self.status_label.config(text=f"Exporting frame {frame_idx + 1}/{total_frames}")
                    self.root.update_idletasks()
            
            out.release()
            self.progress_bar.pack_forget()
            self.status_label.config(text=f"Video exported successfully: {file_path}")
            
            messagebox.showinfo("Export Complete", 
                              f"Video exported successfully!\n"
                              f"Location: {file_path}\n"
                              f"Frames: {total_frames}\n"
                              f"FPS: {fps}\n"
                              f"Format: {export_format.upper()}")
                              
        except Exception as e:
            self.progress_bar.pack_forget()
            self.status_label.config(text="Video export failed")
            messagebox.showerror("Export Error", f"Failed to export video: {str(e)}")

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

    def _rgb_to_hex(self, rgb):
        """Convert RGB color to hex"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
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

    def load_sam2_model(self):
        """Load SAM2 model with explicit top-level config"""
        if not self.sam2_base_path:
            messagebox.showwarning("SAM2 Path Not Set", 
                                "Please set the SAM2 installation path first by clicking 'Set SAM2 Path'.")
            return
            
        try:
            self.status_label.config(text="Loading SAM2 model...")
            self.model_status_label.config(text="⏳ Loading...", foreground='orange')
            self.root.update()

            # Explicit paths (ignore configs/sam2/)
            sam2_checkpoint = os.path.join(self.checkpoint_dir, "sam2.1_hiera_small.pt")
            model_cfg = os.path.join(self.config_dir, "sam2.1", "sam2.1_hiera_s.yaml")

            # Check files exist
            if not os.path.exists(sam2_checkpoint):
                raise FileNotFoundError(f"Checkpoint not found: {sam2_checkpoint}")
            if not os.path.exists(model_cfg):
                raise FileNotFoundError(f"Config not found: {model_cfg}")
            if model_cfg.startswith('/'):
                model_cfg = '/' + model_cfg 
                # for some reason, the hydra package called by the sam2 builder will ignore the first slash.

            # Import the correct builder for VIDEO segmentation
            from sam2.build_sam import build_sam2_video_predictor
        
            # Select device
            if torch and torch.cuda.is_available():
                device = "cuda"
                self.status_label.config(text="Using CUDA GPU for inference...")
            else:
                device = "cpu"
                self.status_label.config(text="Using CPU for inference (slower)...")

            # Build the VIDEO predictor (not the base model)
            self.sam2_model = build_sam2_video_predictor(
                config_file=model_cfg,
                ckpt_path=sam2_checkpoint,
                device=device
            )

            self.model_loaded = True

            model_name = os.path.basename(sam2_checkpoint).replace('.pt', '')
            self.model_status_label.config(text=f"✅ {model_name} ({device.upper()})", foreground='green')
            self.status_label.config(text=f"SAM2 video predictor loaded successfully on {device.upper()}")
        
            # Test that the model has the required methods
            if not hasattr(self.sam2_model, 'init_state'):
                raise AttributeError("Model does not have 'init_state' method. Check SAM2 installation.")
            if not hasattr(self.sam2_model, 'add_new_points'):
                raise AttributeError("Model does not have 'add_new_points' method. Check SAM2 installation.")
            
        except Exception as e:
            self.model_status_label.config(text="❌ Load Failed", foreground='red')
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
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()