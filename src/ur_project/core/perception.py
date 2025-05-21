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
        if not grid or not (grid and grid[0]): # Handle empty or malformed grid (e.g. [[]])
            # print("Warning: Attempted to extract features from an empty or malformed grid.")
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