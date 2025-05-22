from typing import List, Tuple, Set, Any, Dict, Protocol, Optional
from dataclasses import dataclass, field
import numpy as np

from ur_project.data_processing.arc_types import ARCGrid, ARCPixel

@dataclass
class ARCObject:
    """Represents a detected object within an ARC grid."""
    id: int # Unique ID for this object within its grid
    color: ARCPixel
    pixels: Set[Tuple[int, int]] # Set of (row, col) coordinates
    bounding_box: Tuple[int, int, int, int] # (min_row, min_col, max_row, max_col)
    pixel_count: int
    centroid: Tuple[float, float] # (avg_row, avg_col)
    raw_object_grid: Optional[ARCGrid] = None # The object isolated on its own minimal grid
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GridFeatures:
    """Represents the extracted features from a single ARC grid."""
    grid_height: int
    grid_width: int
    objects: List[ARCObject]
    background_color: ARCPixel # Most common color, or a defined background
    unique_colors: Set[ARCPixel]
    object_colors: Set[ARCPixel] # Colors that form objects (excluding background if distinct)
    color_counts: Dict[ARCPixel, int] # Count of each pixel color
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- Neural Network Components for Feature Extraction ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """(Convolution => [BN] => ReLU) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels) # Adjusted for concat

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    A simple U-Net for semantic segmentation of ARC grids.
    Input: (B, NumInputChannels, H, W) - NumInputChannels typically 10 (for one-hot ARC colors 0-9)
    Output: (B, NumClasses, H, W) - NumClasses typically 10 (for ARC colors 0-9)
    """
    def __init__(self, n_input_channels=10, n_output_classes=10, bilinear=True):
        super(UNet, self).__init__()
        self.n_input_channels = n_input_channels
        self.n_output_classes = n_output_classes
        self.bilinear = bilinear

        # Adjust channel sizes for potentially small inputs
        # These are kept small for ARC grids
        self.inc = ConvBlock(n_input_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        # Removed further downsampling for small ARC grids (e.g. 30x30)
        # factor = 2 if bilinear else 1
        # self.down3 = Down(64, 128 // factor) 
        
        # self.up1 = Up(128, 64 // factor, bilinear)
        self.up1 = Up(64, 32, bilinear) # From 64 (bottleneck) to 32 (to concat with down1's 32)
        self.up2 = Up(32, 16, bilinear) # From 32 (up1 output) to 16 (to concat with inc's 16)
        self.outc = OutConv(16, n_output_classes)

        # Bottleneck layer (between down2 and up1)
        self.bottleneck = ConvBlock(64, 64)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3) # If more layers were used
        
        # Bottleneck
        bn = self.bottleneck(x3)

        # x = self.up1(bn, x3) # x3 is from down2
        # x = self.up2(x, x2)   # x2 is from down1
        # x = self.up3(x, x1)   # x1 is from inc
        # Logits
        # logits = self.outc(x)
        
        # Corrected upsampling path for the shallower U-Net
        x_up1 = self.up1(bn, x2) # bn (64) + x2 (32 from down1) -> up1 (32 out)
        x_up2 = self.up2(x_up1, x1) # x_up1 (32) + x1 (16 from inc) -> up2 (16 out)
        logits = self.outc(x_up2)
        return logits

# --- End of Neural Network Components ---


class BaseFeatureExtractor(Protocol):
    """Protocol for an ARC grid feature extractor."""

    def extract_features(self, grid: ARCGrid, background_color_heuristic: str = 'most_frequent') -> Optional[GridFeatures]:
        """
        Extracts features from the given ARC grid.

        Args:
            grid (ARCGrid): The input ARC grid.
            background_color_heuristic (str): Method to determine background color 
                                             ('most_frequent', 'zero', or specific int).
        Returns:
            Optional[GridFeatures]: Extracted features, or None if extraction fails.
        """
        ...

class BasicARCFeatureExtractor(BaseFeatureExtractor):
    """Extracts basic features from an ARC grid, including objects and their properties."""

    def _determine_background_color(self, grid_np: np.ndarray, heuristic: str = 'most_frequent') -> ARCPixel:
        """Determines the background color of the grid based on a heuristic."""
        if not grid_np.size > 0: # Empty grid
            return ARCPixel(0) # Default for empty

        if heuristic.isdigit():
            return ARCPixel(int(heuristic))
        elif heuristic == 'zero':
            return ARCPixel(0)
        elif heuristic == 'most_frequent':
            colors, counts = np.unique(grid_np, return_counts=True)
            return ARCPixel(colors[np.argmax(counts)])
        else: # Default to most frequent if heuristic is unknown
            print(f"Warning: Unknown background_color_heuristic '{heuristic}'. Defaulting to 'most_frequent'.")
            colors, counts = np.unique(grid_np, return_counts=True)
            return ARCPixel(colors[np.argmax(counts)])

    def _find_objects_connected_components(self, grid_np: np.ndarray, background_color: ARCPixel) -> List[ARCObject]:
        """Identifies objects using a basic connected components algorithm (4-connectivity)."""
        if not grid_np.size > 0:
            return []

        rows, cols = grid_np.shape
        visited = np.zeros_like(grid_np, dtype=bool)
        objects: List[ARCObject] = []
        object_id_counter = 0

        for r_start in range(rows):
            for c_start in range(cols):
                if visited[r_start,c_start] or grid_np[r_start,c_start] == background_color:
                    continue

                # Start of a new object
                object_id_counter += 1
                current_object_color = ARCPixel(grid_np[r_start,c_start])
                object_pixels: Set[Tuple[int, int]] = set()
                q: List[Tuple[int, int]] = [(r_start,c_start)]
                visited[r_start,c_start] = True
                
                min_r, max_r = r_start, r_start
                min_c, max_c = c_start, c_start
                pixel_sum_r, pixel_sum_c = 0.0, 0.0 # Use float for sums for centroid calculation

                while q:
                    curr_r, curr_c = q.pop(0)
                    object_pixels.add((curr_r, curr_c))

                    min_r = min(min_r, curr_r)
                    max_r = max(max_r, curr_r)
                    min_c = min(min_c, curr_c)
                    max_c = max(max_c, curr_c)
                    pixel_sum_r += curr_r
                    pixel_sum_c += curr_c

                    # Check neighbors (4-connectivity)
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and \
                           not visited[nr,nc] and grid_np[nr,nc] == current_object_color:
                            visited[nr,nc] = True
                            q.append((nr,nc))
                
                if object_pixels:
                    pixel_count = len(object_pixels)
                    centroid_r = pixel_sum_r / pixel_count
                    centroid_c = pixel_sum_c / pixel_count
                    
                    # Extract raw object grid
                    obj_height = max_r - min_r + 1
                    obj_width = max_c - min_c + 1
                    # Initialize with background color, ensuring correct dtype for ARCPixel mapping
                    raw_obj_grid_np = np.full((obj_height, obj_width), int(background_color), dtype=int)
                    for obj_r, obj_c in object_pixels:
                        raw_obj_grid_np[obj_r - min_r, obj_c - min_c] = int(current_object_color)
                    
                    raw_obj_grid: ARCGrid = [[ARCPixel(cell) for cell in row] for row in raw_obj_grid_np.tolist()]

                    objects.append(ARCObject(
                        id=object_id_counter,
                        color=current_object_color,
                        pixels=object_pixels,
                        bounding_box=(min_r, min_c, max_r, max_c),
                        pixel_count=pixel_count,
                        centroid=(centroid_r, centroid_c),
                        raw_object_grid=raw_obj_grid
                    ))
        return objects

    def extract_features(self, grid: ARCGrid, background_color_heuristic: str = 'most_frequent') -> Optional[GridFeatures]:
        if not grid or not (grid and grid[0]):
            return GridFeatures(
                grid_height=len(grid) if grid else 0, 
                grid_width=len(grid[0]) if grid and grid[0] else 0, 
                objects=[], 
                background_color=ARCPixel(0),
                unique_colors=set(), 
                object_colors=set(), 
                color_counts={}
            )

        grid_np = np.array([[int(p) for p in row] for row in grid], dtype=int)
        height, width = grid_np.shape

        bg_color = self._determine_background_color(grid_np, heuristic=background_color_heuristic)
        
        arc_objects = self._find_objects_connected_components(grid_np, bg_color)

        unique_colors_on_grid = set(ARCPixel(c) for c in np.unique(grid_np)) if grid_np.size > 0 else set()
        object_colors_on_grid = set(obj.color for obj in arc_objects)
        
        color_counts_on_grid = {ARCPixel(color): count for color, count in zip(*np.unique(grid_np, return_counts=True))} if grid_np.size > 0 else {}
        
        # --- Future UNet Integration Point ---
        # if self.use_unet_preprocessing and self.unet_model and self.device:
        #     # 1. Preprocess grid_np for U-Net (e.g., one-hot, normalize, to tensor)
        #     #    Input shape: (1, num_colors, H, W)
        #     #    Max ARC colors = 10 (0-9)
        #     one_hot_grid = F.one_hot(torch.from_numpy(grid_np).long(), num_classes=10).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        #     
        #     with torch.no_grad():
        #         processed_logits = self.unet_model(one_hot_grid) # (1, 10, H, W)
        #     
        #     # 2. Post-process U-Net output (e.g., argmax to get color predictions)
        #     #    This "cleaned_grid_np" would then be used for object detection.
        #     cleaned_grid_np = torch.argmax(processed_logits, dim=1).squeeze(0).cpu().numpy().astype(int)
        #     
        #     # Potentially re-determine background color if U-Net changes it significantly
        #     # bg_color = self._determine_background_color(cleaned_grid_np, heuristic=background_color_heuristic)
        #     # arc_objects = self._find_objects_connected_components(cleaned_grid_np, bg_color)
        #     # unique_colors_on_grid = set(ARCPixel(c) for c in np.unique(cleaned_grid_np))
        #     # object_colors_on_grid = set(obj.color for obj in arc_objects)
        #     # color_counts_on_grid = {ARCPixel(color): count for color, count in zip(*np.unique(cleaned_grid_np, return_counts=True))}
        #
        #     # For now, we are not using the U-Net output to overwrite grid_np, 
        #     # but the infrastructure is here.
        #     # The U-Net features could be added to metadata.
        #     pass
        # --- End of UNet Integration Point ---

        return GridFeatures(
            grid_height=height,
            grid_width=width,
            objects=arc_objects,
            background_color=bg_color,
            unique_colors=unique_colors_on_grid,
            object_colors=object_colors_on_grid,
            color_counts=color_counts_on_grid
        )

# Example Usage:
# if __name__ == '__main__':
#     extractor = BasicARCFeatureExtractor()

#     # Test Case 1: Simple grid
#     grid1: ARCGrid = [
#         [ARCPixel(0), ARCPixel(1), ARCPixel(1)],
#         [ARCPixel(0), ARCPixel(0), ARCPixel(1)],
#         [ARCPixel(2), ARCPixel(2), ARCPixel(0)]
#     ]
#     print("--- Grid 1 ---")
#     features1 = extractor.extract_features(grid1)
#     if features1:
#         print(f"Background: {features1.background_color}")
#         print(f"Unique Colors: {features1.unique_colors}")
#         print(f"Object Colors: {features1.object_colors}")
#         print(f"Color Counts: {features1.color_counts}")
#         print(f"Num Objects: {len(features1.objects)}")
#         for obj in features1.objects:
#             print(f"  Obj ID: {obj.id}, Color: {obj.color}, Pixels: {obj.pixel_count}, Centroid: {obj.centroid}, BBox: {obj.bounding_box}")
#             # print(f"    Raw Grid: {obj.raw_object_grid}")
    
#     # Test Case 2: Grid with only one color (should be background, no objects)
#     grid2: ARCGrid = [
#         [ARCPixel(5), ARCPixel(5)],
#         [ARCPixel(5), ARCPixel(5)]
#     ]
#     print("\n--- Grid 2 (Single Color) ---")
#     features2 = extractor.extract_features(grid2)
#     if features2:
#         print(f"Background: {features2.background_color}")
#         print(f"Num Objects: {len(features2.objects)}") # Expected 0

#     # Test Case 3: Empty Grid
#     grid3: ARCGrid = []
#     print("\n--- Grid 3 (Empty) ---")
#     features3 = extractor.extract_features(grid3)
#     if features3:
#         print(f"Grid Height: {features3.grid_height}, Width: {features3.grid_width}")
#         print(f"Num Objects: {len(features3.objects)}")
    
#     # Test Case 4: Grid with one empty row
#     grid4: ARCGrid = [[]]
#     print("\n--- Grid 4 (One empty row) ---")
#     features4 = extractor.extract_features(grid4)
#     if features4:
#         print(f"Grid Height: {features4.grid_height}, Width: {features4.grid_width}")
#         print(f"Num Objects: {len(features4.objects)}")

#     # Test Case 5: More complex objects
#     grid5: ARCGrid = [
#         [ARCPixel(1), ARCPixel(1), ARCPixel(0), ARCPixel(2)],
#         [ARCPixel(1), ARCPixel(0), ARCPixel(0), ARCPixel(2)],
#         [ARCPixel(0), ARCPixel(0), ARCPixel(3), ARCPixel(3)],
#         [ARCPixel(4), ARCPixel(4), ARCPixel(3), ARCPixel(0)],
#     ]
#     print("\n--- Grid 5 (Complex) ---")
#     features5 = extractor.extract_features(grid5, background_color_heuristic='zero')
#     if features5:
#         print(f"Background: {features5.background_color}")
#         print(f"Num Objects: {len(features5.objects)}")
#         for obj in features5.objects:
#             print(f"  Obj ID: {obj.id}, Color: {obj.color}, Pixels: {obj.pixel_count}, BBox: {obj.bounding_box}")
#             print(f"    Raw Grid: {obj.raw_object_grid}") 


class UNetARCFeatureExtractor(BasicARCFeatureExtractor):
    """
    Feature extractor that can optionally use a U-Net for initial grid processing
    before applying basic feature extraction.
    """
    def __init__(self, use_unet_preprocessing: bool = False, model_path: Optional[str] = None, num_input_channels: int = 10, num_classes: int = 10):
        super().__init__()
        self.use_unet_preprocessing = use_unet_preprocessing
        self.unet_model: Optional[UNet] = None
        self.device: Optional[torch.device] = None

        if self.use_unet_preprocessing:
            if not torch.cuda.is_available():
                print("Warning: UNetARCFeatureExtractor configured to use U-Net, but CUDA is not available. U-Net will run on CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda")
            
            self.unet_model = UNet(n_input_channels=num_input_channels, n_output_classes=num_classes)
            
            if model_path:
                try:
                    self.unet_model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"U-Net model loaded from {model_path}")
                except FileNotFoundError:
                    print(f"Warning: U-Net model path {model_path} not found. Using randomly initialized U-Net.")
                except Exception as e:
                    print(f"Warning: Error loading U-Net model from {model_path}: {e}. Using randomly initialized U-Net.")
            else:
                print("Warning: No U-Net model path provided. Using randomly initialized U-Net.")
            
            self.unet_model.to(self.device)
            self.unet_model.eval() # Set to evaluation mode

    def extract_features(self, grid: ARCGrid, background_color_heuristic: str = 'most_frequent') -> Optional[GridFeatures]:
        if not grid or not (grid and grid[0]):
            return GridFeatures(grid_height=len(grid) if grid else 0, 
                                grid_width=len(grid[0]) if grid and grid[0] else 0, 
                                objects=[], background_color=ARCPixel(0),
                                unique_colors=set(), object_colors=set(), color_counts={})

        grid_np_original = np.array([[int(p) for p in row] for row in grid], dtype=int)
        
        processed_grid_np = grid_np_original.copy() # Start with original

        unet_output_features = None # To store raw U-Net output if needed

        if self.use_unet_preprocessing and self.unet_model and self.device:
            # One-hot encode the grid: (H, W) -> (H, W, num_colors) -> (num_colors, H, W)
            # Max ARC colors = 10 (0-9 assumed for num_classes in one_hot)
            num_classes_for_onehot = self.unet_model.n_input_channels
            
            # Ensure grid values are within [0, num_classes_for_onehot - 1]
            # This is important if num_classes_for_onehot is less than max possible pixel value (e.g. 9)
            clamped_grid_np = np.clip(grid_np_original, 0, num_classes_for_onehot - 1)

            grid_tensor = torch.from_numpy(clamped_grid_np).long().to(self.device)
            one_hot_grid = F.one_hot(grid_tensor, num_classes=num_classes_for_onehot)
            one_hot_grid = one_hot_grid.permute(2, 0, 1).unsqueeze(0).float() # (1, num_colors, H, W)
            
            with torch.no_grad():
                logits = self.unet_model(one_hot_grid) # (1, num_output_classes, H, W)
            
            unet_output_features = logits.cpu().numpy() # Store raw logits if needed later
            
            # Convert logits to a "cleaned" grid by taking argmax over color channels
            # This results in a grid where each pixel has the color with the highest logit
            processed_grid_np = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(int)
            
            # Note: The background color and objects will now be determined from this
            # U-Net processed grid. This is a significant change in how features are derived.
            # print("U-Net preprocessing applied. Features will be extracted from U-Net's output grid.")


        # Proceed with feature extraction using the (potentially U-Net processed) grid
        height, width = processed_grid_np.shape
        bg_color = self._determine_background_color(processed_grid_np, heuristic=background_color_heuristic)
        arc_objects = self._find_objects_connected_components(processed_grid_np, bg_color)
        
        unique_colors_on_grid = set(ARCPixel(c) for c in np.unique(processed_grid_np)) if processed_grid_np.size > 0 else set()
        object_colors_on_grid = set(obj.color for obj in arc_objects)
        color_counts_on_grid = {ARCPixel(color): count for color, count in zip(*np.unique(processed_grid_np, return_counts=True))} if processed_grid_np.size > 0 else {}

        grid_features = GridFeatures(
            grid_height=height,
            grid_width=width,
            objects=arc_objects,
            background_color=bg_color,
            unique_colors=unique_colors_on_grid,
            object_colors=object_colors_on_grid,
            color_counts=color_counts_on_grid,
            metadata={} # Initialize empty metadata
        )

        if unet_output_features is not None:
            grid_features.metadata['unet_raw_output'] = unet_output_features
            # Add information that U-Net was used for this GridFeatures instance
            grid_features.metadata['unet_preprocessing_applied'] = True


        # TODO: Implement topological, symmetry, repetition, and enhanced counting features
        # These will operate on the `arc_objects` derived from either the original or U-Net processed grid.
        # For example:
        # grid_features.metadata['adjacency_relations'] = self._extract_adjacency(arc_objects)
        # grid_features.metadata['symmetry_info'] = self._analyze_symmetry(processed_grid_np, arc_objects) 
        
        # Enhance features after initial objects are found
        self._extract_topological_features(grid_features, processed_grid_np)
        self._extract_symmetry_features(grid_features)
        self._extract_repetition_features(grid_features)
        self._perform_enhanced_counting(grid_features)

        return grid_features

    def _np_from_arc_grid(self, arc_grid: ARCGrid) -> np.ndarray:
        """Converts an ARCGrid (list of lists of ARCPixel) to a NumPy array."""
        if not arc_grid or not arc_grid[0]:
            return np.array([[]], dtype=int)
        return np.array([[int(p) for p in row] for row in arc_grid], dtype=int)

    def _extract_topological_features(self, grid_features: GridFeatures, grid_np: np.ndarray):
        """
        Extracts adjacency, containment, centroid distances, and relative angles.
        Updates ARCObject.metadata and GridFeatures.metadata.
        """
        objects = grid_features.objects
        object_map = {obj.id: obj for obj in objects}
        
        # Create a grid representation with object IDs for quick lookups
        # Initialize with 0 (or a special ID for background/unassigned)
        object_id_grid = np.zeros(grid_np.shape, dtype=int)
        for obj in objects:
            for r, c in obj.pixels:
                object_id_grid[r, c] = obj.id

        for i in range(len(objects)):
            obj1 = objects[i]
            obj1.metadata['adjacent_object_ids'] = set()
            obj1.metadata['contained_object_ids'] = [] # Objects obj1 contains
            obj1.metadata['is_contained_by_ids'] = [] # Objects that contain obj1
            obj1.metadata['relative_positions'] = []

            # Adjacency
            for r, c in obj1.pixels:
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]: # 4-connectivity for adjacency
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_np.shape[0] and 0 <= nc < grid_np.shape[1]:
                        neighbor_obj_id = object_id_grid[nr, nc]
                        if neighbor_obj_id != 0 and neighbor_obj_id != obj1.id:
                            obj1.metadata['adjacent_object_ids'].add(neighbor_obj_id)
            
            # Convert set to list for JSON compatibility if needed later
            obj1.metadata['adjacent_object_ids'] = sorted(list(obj1.metadata['adjacent_object_ids']))


            for j in range(len(objects)):
                if i == j:
                    continue
                obj2 = objects[j]

                # Centroid distance and angle
                dist = np.linalg.norm(np.array(obj1.centroid) - np.array(obj2.centroid))
                angle = np.arctan2(obj2.centroid[0] - obj1.centroid[0], obj2.centroid[1] - obj1.centroid[1])
                obj1.metadata['relative_positions'].append({
                    'object_id': obj2.id,
                    'distance': dist,
                    'angle': angle
                })

                # Containment (obj1 contains obj2)
                # Bounding box check: obj2's bbox must be strictly inside obj1's bbox
                b1_min_r, b1_min_c, b1_max_r, b1_max_c = obj1.bounding_box
                b2_min_r, b2_min_c, b2_max_r, b2_max_c = obj2.bounding_box

                if b1_min_r < b2_min_r and b1_min_c < b2_min_c and \
                   b1_max_r > b2_max_r and b1_max_c > b2_max_c:
                    # Pixel check: all pixels of obj2 must be within the area defined by obj1's pixels
                    # This is a simplified check: are all obj2 pixels also obj1 pixels of the *same color*?
                    # Or, are all obj2 pixels on *any* part of obj1 if obj1 is a container?
                    # A more robust check: check if obj2 is surrounded by obj1's pixels or grid boundary within obj1's bbox.
                    # For now, let's assume if bbox2 is inside bbox1, and all pixels of obj2 are not outside obj1.
                    
                    # Simplified: if obj2's pixels are a subset of obj1's pixels (implies same color)
                    # This is more like sub-object, not general containment.
                    # if obj2.pixels.issubset(obj1.pixels):
                    #    obj1.metadata['contained_object_ids'].append(obj2.id)
                    #    obj2.metadata['is_contained_by_ids'].append(obj1.id)

                    # More general containment: obj2 is inside obj1, obj1 pixels form a boundary
                    # This check assumes obj2 is "floating" inside obj1, not necessarily same color.
                    # It checks that no pixel of obj2 is on the border of obj1's bounding box.
                    is_contained = True
                    for r_obj2, c_obj2 in obj2.pixels:
                        # Check if the pixel from obj2 is also a pixel of obj1
                        if (r_obj2, c_obj2) not in obj1.pixels:
                            # If obj2 has pixels not part of obj1, it's not simple containment of same-color parts
                            # This could be fine if obj1 is a "hollow" container of a different color.
                            # For now, let's be strict for initial implementation: obj2 pixel must be on obj1 pixel.
                            # This will mostly find sub-parts of the same object if not careful.
                            pass # This condition depends on how "containment" is defined.

                        # Check if obj2 touches the inner border of obj1's bounding box.
                        # If it does, it's not "fully" contained in a floating sense.
                        # This check is problematic if obj1 is not rectangular.
                        # if r_obj2 <= b1_min_r or r_obj2 >= b1_max_r or \
                        #    c_obj2 <= b1_min_c or c_obj2 >= b1_max_c:
                        #     is_contained = False; break
                    
                    # Let's use a definition: obj1 contains obj2 if obj2's bbox is inside obj1's bbox,
                    # AND all pixels of obj2 are "covered" by obj1 (i.e., for each (r,c) in obj2.pixels,
                    # (r,c) is also in obj1.pixels OR obj1.color is the background of obj2's raw grid).
                    # This is still too complex for now.
                    # Using a simpler definition: obj2 bbox inside obj1 bbox, and all pixels of obj2
                    # are within the set of pixels defined by obj1. This is suitable for objects of the same color.
                    if obj2.pixels.issubset(obj1.pixels):
                         obj1.metadata['contained_object_ids'].append(obj2.id)
                         obj2.metadata['is_contained_by_ids'].append(obj1.id)


    def _extract_symmetry_features(self, grid_features: GridFeatures):
        """Extracts symmetry features for each object. Updates ARCObject.metadata."""
        for obj in grid_features.objects:
            if not obj.raw_object_grid:
                obj.metadata['symmetries'] = {} # Should not happen if object extraction is correct
                continue

            obj_np = self._np_from_arc_grid(obj.raw_object_grid)
            if obj_np.size == 0:
                obj.metadata['symmetries'] = {}
                continue

            symmetries = {}
            # Horizontal symmetry
            symmetries['horizontal'] = np.array_equal(obj_np, np.fliplr(obj_np))
            # Vertical symmetry
            symmetries['vertical'] = np.array_equal(obj_np, np.flipud(obj_np))
            
            # Rotational (need square array for easy np.rot90)
            # If not square, it can't have 90/270 deg rotational symmetry in the simple sense.
            # 180 deg rotational symmetry is (i,j) == (H-1-i, W-1-j)
            symmetries['rotational_180'] = np.array_equal(obj_np, np.rot90(np.rot90(obj_np))) # Flip H and W
            if obj_np.shape[0] == obj_np.shape[1]:
                symmetries['rotational_90'] = np.array_equal(obj_np, np.rot90(obj_np, k=1))
                symmetries['rotational_270'] = np.array_equal(obj_np, np.rot90(obj_np, k=3))
            else:
                symmetries['rotational_90'] = False
                symmetries['rotational_270'] = False

            # Diagonal (main: top-left to bottom-right)
            if obj_np.shape[0] == obj_np.shape[1]:
                symmetries['diagonal_main'] = np.array_equal(obj_np, obj_np.T)
            else:
                symmetries['diagonal_main'] = False # Non-square cannot have this simple transpose symmetry

            # Diagonal (anti: top-right to bottom-left)
            # Check (i,j) against (W-1-j, H-1-i) after flipping.
            if obj_np.shape[0] == obj_np.shape[1]:
                 symmetries['diagonal_anti'] = np.array_equal(obj_np, np.fliplr(np.flipud(obj_np).T))
            else:
                 symmetries['diagonal_anti'] = False # Non-square

            obj.metadata['symmetries'] = symmetries
    
    def _get_object_shape_signature(self, obj_np: np.ndarray, obj_color: ARCPixel) -> str:
        """Creates a canonical string representation of an object's shape, ignoring color."""
        # Replace actual color with a placeholder (e.g., 1) for shape comparison
        shape_np = np.where(obj_np == int(obj_color), 1, 0)
        return str(shape_np.tolist())


    def _extract_repetition_features(self, grid_features: GridFeatures):
        """Detects repeated objects based on shape and color. Updates GridFeatures.metadata."""
        objects = grid_features.objects
        object_signatures = {} # signature -> list of object_ids

        for obj in objects:
            if obj.raw_object_grid:
                obj_np = self._np_from_arc_grid(obj.raw_object_grid)
                # Signature combines color and shape string
                # Shape signature is relative to its own bounding box and uses a fixed color.
                shape_sig = self._get_object_shape_signature(obj_np, obj.color)
                full_signature = f"c{int(obj.color)}_s{shape_sig}"
                
                if full_signature not in object_signatures:
                    object_signatures[full_signature] = []
                object_signatures[full_signature].append(obj.id)
        
        # Filter out non-repeated objects
        repeated_groups = [ids for ids in object_signatures.values() if len(ids) > 1]
        grid_features.metadata['object_repetitions'] = repeated_groups

    def _perform_enhanced_counting(self, grid_features: GridFeatures):
        """Performs enhanced counting and stores in GridFeatures.metadata."""
        objects = grid_features.objects
        grid_features.metadata['object_counts_by_color'] = {}
        grid_features.metadata['object_counts_by_shape'] = {} # shape signature -> count
        grid_features.metadata['object_counts_by_size'] = {} # pixel_count -> count

        shape_signatures_map = {} # obj_id -> shape_sig for quick lookup

        for obj in objects:
            # Count by color
            color = obj.color
            grid_features.metadata['object_counts_by_color'][color] = \
                grid_features.metadata['object_counts_by_color'].get(color, 0) + 1

            # Count by size (pixel_count)
            size = obj.pixel_count
            grid_features.metadata['object_counts_by_size'][size] = \
                grid_features.metadata['object_counts_by_size'].get(size, 0) + 1

            # Count by shape
            if obj.raw_object_grid:
                obj_np = self._np_from_arc_grid(obj.raw_object_grid)
                shape_sig = self._get_object_shape_signature(obj_np, obj.color)
                obj.metadata['shape_signature'] = shape_sig # Store on object
                shape_signatures_map[obj.id] = shape_sig # Store for later use if needed
                grid_features.metadata['object_counts_by_shape'][shape_sig] = \
                    grid_features.metadata['object_counts_by_shape'].get(shape_sig, 0) + 1
        
        # Store all shape signatures for reference, if desired
        # grid_features.metadata['object_shape_signatures'] = shape_signatures_map

    def _calculate_pixel_jaccard_index(self, pixels1: Set[Tuple[int, int]], pixels2: Set[Tuple[int, int]]) -> float:
        """Calculates Jaccard index between two sets of pixels."""
        if not pixels1 and not pixels2:
            return 1.0 # Both empty, considered identical
        if not pixels1 or not pixels2:
            return 0.0 # One empty, other not
        intersection_size = len(pixels1.intersection(pixels2))
        union_size = len(pixels1.union(pixels2))
        return intersection_size / union_size if union_size > 0 else 0.0

    def _are_shapes_identical(self, obj1: ARCObject, obj2: ARCObject) -> bool:
        """Checks if two objects have identical raw_object_grid shapes and content."""
        if obj1.raw_object_grid is None and obj2.raw_object_grid is None:
            return True # Both None, considered same for this purpose
        if obj1.raw_object_grid is None or obj2.raw_object_grid is None:
            return False
        
        # Convert to numpy for easier comparison if needed, but list comparison is fine
        grid1_np = self._np_from_arc_grid(obj1.raw_object_grid)
        grid2_np = self._np_from_arc_grid(obj2.raw_object_grid)

        if grid1_np.shape != grid2_np.shape:
            return False
        
        # Compare content. Note: raw_object_grid stores actual colors.
        # For shape comparison, we might want to ignore color, but here we check exact match.
        return np.array_equal(grid1_np, grid2_np)


    def analyze_object_transformations(
        self, 
        input_features: GridFeatures, 
        output_features: GridFeatures,
        max_centroid_distance_for_identity: float = 5.0, # Max distance for an object to be "the same but moved slightly"
        min_jaccard_for_general_match: float = 0.3 # Min pixel overlap for general matching
    ) -> Dict[str, Any]:
        """
        Analyzes transformations of objects between an input and output grid.
        Focuses on Object Permanence/Continuity.
        """
        input_objects = {obj.id: obj for obj in input_features.objects}
        output_objects = {obj.id: obj for obj in output_features.objects}

        matched_input_ids: Set[int] = set()
        matched_output_ids: Set[int] = set()
        
        persistent_objects = [] # Stores info about matched objects and their transformations

        # --- Pass 1: Strong Matches (Identical objects, possibly moved slightly) ---
        for in_id, in_obj in input_objects.items():
            if in_id in matched_input_ids:
                continue
            
            best_match_out_id = -1
            min_dist = float('inf')

            for out_id, out_obj in output_objects.items():
                if out_id in matched_output_ids:
                    continue

                if in_obj.color == out_obj.color and \
                   in_obj.pixel_count == out_obj.pixel_count and \
                   self._are_shapes_identical(in_obj, out_obj):
                    
                    dist = np.linalg.norm(np.array(in_obj.centroid) - np.array(out_obj.centroid))
                    if dist < max_centroid_distance_for_identity and dist < min_dist:
                        min_dist = dist
                        best_match_out_id = out_id
            
            if best_match_out_id != -1:
                in_obj_matched = input_objects[in_id]
                out_obj_matched = output_objects[best_match_out_id]
                
                transformation_details = {
                    'input_object_id': in_id,
                    'output_object_id': best_match_out_id,
                    'input_properties': {'color': in_obj_matched.color, 'centroid': in_obj_matched.centroid, 'pixel_count': in_obj_matched.pixel_count},
                    'output_properties': {'color': out_obj_matched.color, 'centroid': out_obj_matched.centroid, 'pixel_count': out_obj_matched.pixel_count},
                    'type': 'identity_persist',
                    'delta_centroid': tuple(np.array(out_obj_matched.centroid) - np.array(in_obj_matched.centroid)),
                    'color_changed': False,
                    'shape_changed': False, # Based on raw_object_grid exact match
                    'size_changed': False,
                    'match_score': 1.0 # Perfect match on properties
                }
                persistent_objects.append(transformation_details)
                matched_input_ids.add(in_id)
                matched_output_ids.add(best_match_out_id)

        # --- Pass 2: Color Change Matches (Same shape and size, different color) ---
        for in_id, in_obj in input_objects.items():
            if in_id in matched_input_ids:
                continue
            
            best_match_out_id = -1
            min_dist = float('inf')

            for out_id, out_obj in output_objects.items():
                if out_id in matched_output_ids:
                    continue
                
                if in_obj.pixel_count == out_obj.pixel_count and \
                   self._are_shapes_identical(in_obj, out_obj) and \
                   in_obj.color != out_obj.color: # Key difference: color changed
                    
                    dist = np.linalg.norm(np.array(in_obj.centroid) - np.array(out_obj.centroid))
                    if dist < max_centroid_distance_for_identity and dist < min_dist: # Still expect it to be roughly in place
                        min_dist = dist
                        best_match_out_id = out_id
            
            if best_match_out_id != -1:
                in_obj_matched = input_objects[in_id]
                out_obj_matched = output_objects[best_match_out_id]
                transformation_details = {
                    'input_object_id': in_id,
                    'output_object_id': best_match_out_id,
                    'input_properties': {'color': in_obj_matched.color, 'centroid': in_obj_matched.centroid, 'pixel_count': in_obj_matched.pixel_count},
                    'output_properties': {'color': out_obj_matched.color, 'centroid': out_obj_matched.centroid, 'pixel_count': out_obj_matched.pixel_count},
                    'type': 'color_change',
                    'delta_centroid': tuple(np.array(out_obj_matched.centroid) - np.array(in_obj_matched.centroid)),
                    'color_changed': True,
                    'color_from': in_obj_matched.color,
                    'color_to': out_obj_matched.color,
                    'shape_changed': False,
                    'size_changed': False,
                    'match_score': 0.9 # High confidence, but color changed
                }
                persistent_objects.append(transformation_details)
                matched_input_ids.add(in_id)
                matched_output_ids.add(best_match_out_id)
        
        # --- Pass 3: General Overlap-Based Matching (Fallback for more complex transformations) ---
        # This pass tries to match remaining objects based on pixel overlap (Jaccard index).
        # This can find objects that moved significantly, changed shape, or partially changed.
        # Using a simple greedy approach here. A bipartite matching algorithm would be more robust for optimal pairings.
        
        # Create a list of Jaccard indices for all remaining input/output object pairs
        potential_matches = []
        for in_id, in_obj in input_objects.items():
            if in_id in matched_input_ids:
                continue
            for out_id, out_obj in output_objects.items():
                if out_id in matched_output_ids:
                    continue
                
                jaccard = self._calculate_pixel_jaccard_index(in_obj.pixels, out_obj.pixels)
                if jaccard >= min_jaccard_for_general_match:
                    potential_matches.append({'in_id': in_id, 'out_id': out_id, 'jaccard': jaccard, 
                                              'in_obj': in_obj, 'out_obj': out_obj})
        
        # Sort by Jaccard index (descending) to prioritize stronger overlaps
        potential_matches.sort(key=lambda x: x['jaccard'], reverse=True)
        
        for match_candidate in potential_matches:
            in_id = match_candidate['in_id']
            out_id = match_candidate['out_id']
            
            if in_id in matched_input_ids or out_id in matched_output_ids:
                continue # Already matched by a previous stronger overlap or different pass
            
            in_obj_matched = match_candidate['in_obj']
            out_obj_matched = match_candidate['out_obj']

            color_changed = in_obj_matched.color != out_obj_matched.color
            shape_changed = not self._are_shapes_identical(in_obj_matched, out_obj_matched)
            size_changed = in_obj_matched.pixel_count != out_obj_matched.pixel_count
            
            match_type = "overlap_transform"
            if color_changed and not shape_changed and not size_changed: match_type = "color_change_overlap"
            elif not color_changed and shape_changed and not size_changed: match_type = "shape_change_overlap"
            elif not color_changed and not shape_changed and size_changed: match_type = "size_change_overlap"
            
            transformation_details = {
                'input_object_id': in_id,
                'output_object_id': out_id,
                'input_properties': {'color': in_obj_matched.color, 'centroid': in_obj_matched.centroid, 'pixel_count': in_obj_matched.pixel_count},
                'output_properties': {'color': out_obj_matched.color, 'centroid': out_obj_matched.centroid, 'pixel_count': out_obj_matched.pixel_count},
                'type': match_type,
                'delta_centroid': tuple(np.array(out_obj_matched.centroid) - np.array(in_obj_matched.centroid)),
                'color_changed': color_changed,
                'shape_changed': shape_changed,
                'size_changed': size_changed,
                'jaccard_index': match_candidate['jaccard'],
                'match_score': match_candidate['jaccard'] # Use Jaccard as score
            }
            if color_changed:
                transformation_details['color_from'] = in_obj_matched.color
                transformation_details['color_to'] = out_obj_matched.color

            persistent_objects.append(transformation_details)
            matched_input_ids.add(in_id)
            matched_output_ids.add(out_id)

        # --- Identify Vanished and Appeared Objects ---
        vanished_objects = []
        for in_id, in_obj in input_objects.items():
            if in_id not in matched_input_ids:
                vanished_objects.append({
                    'object_id': in_id,
                    'color': in_obj.color,
                    'centroid': in_obj.centroid,
                    'pixel_count': in_obj.pixel_count
                })

        appeared_objects = []
        for out_id, out_obj in output_objects.items():
            if out_id not in matched_output_ids:
                appeared_objects.append({
                    'object_id': out_id,
                    'color': out_obj.color,
                    'centroid': out_obj.centroid,
                    'pixel_count': out_obj.pixel_count
                })
        
        # TODO: More sophisticated analysis for splits/merges
        # e.g., one input object overlaps with multiple output objects, or vice versa.
        # This would involve checking unmatched objects against already matched ones.

        return {
            "persistent_objects": persistent_objects,
            "vanished_objects": vanished_objects,
            "appeared_objects": appeared_objects,
            # "split_merges": [] # Placeholder
        }