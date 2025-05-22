import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Set, Optional

# Assuming perception.py is in the same directory (core)
# and arc_types.py is in data_processing, one level up from core
from .perception import GridFeatures, ARCObject 
from ..data_processing.arc_types import ARCPixel

class PredicateNet(nn.Module):
    """A simple feed-forward network for a binary predicate."""
    def __init__(self, input_size: int, hidden_size: int = 32, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # Output probability
        return x

class SymbolGroundingModel(nn.Module):
    """
    A model to ground symbolic predicates from features extracted from ARC grids.
    Each predicate is represented by a small neural network.
    """
    MAX_ARC_COLORS = 10 # Standard 0-9 colors

    # Define feature sizes
    # Object features: color_one_hot (10) + centroid_norm (2) + size_norm (1) + bbox_norm (2) + symmetries (3: H, V, R180)
    OBJECT_FEATURE_SIZE = MAX_ARC_COLORS + 2 + 1 + 2 + 3 # = 18
    
    # Pairwise features: obj1_feat (18) + obj2_feat (18) + rel_dist (1) + rel_angle_sincos (2)
    PAIRWISE_FEATURE_SIZE = OBJECT_FEATURE_SIZE * 2 + 1 + 2 # = 36 + 3 = 39

    def __init__(self, hidden_size: int = 32, dropout_rate: float = 0.1):
        super().__init__()

        # --- Predicate Networks ---
        # Pairwise predicates
        self.is_touching_net = PredicateNet(self.PAIRWISE_FEATURE_SIZE, hidden_size, dropout_rate)
        self.is_inside_net = PredicateNet(self.PAIRWISE_FEATURE_SIZE, hidden_size, dropout_rate) # objA is inside objB
        self.is_same_color_net = PredicateNet(self.PAIRWISE_FEATURE_SIZE, hidden_size, dropout_rate)
        self.is_same_shape_net = PredicateNet(self.PAIRWISE_FEATURE_SIZE, hidden_size, dropout_rate)
        
        # Unary predicates (operating on single objects)
        self.has_horizontal_symmetry_net = PredicateNet(self.OBJECT_FEATURE_SIZE, hidden_size, dropout_rate)
        self.has_vertical_symmetry_net = PredicateNet(self.OBJECT_FEATURE_SIZE, hidden_size, dropout_rate)
        
        # self.is_repeated_object_net = PredicateNet(self.OBJECT_FEATURE_SIZE, hidden_size) # Requires context of other objects

        # Device handling (useful if this model is moved to GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def _normalize_val(self, val, max_val):
        return val / max_val if max_val > 0 else 0.0

    def _prepare_object_features(self, obj: ARCObject, grid_h: int, grid_w: int) -> torch.Tensor:
        """Converts an ARCObject's properties into a flat feature tensor."""
        features = []

        # Color (one-hot)
        color_one_hot = F.one_hot(torch.tensor(int(obj.color)), num_classes=self.MAX_ARC_COLORS).float()
        features.extend(color_one_hot.tolist())

        # Centroid (normalized)
        features.append(self._normalize_val(obj.centroid[0], grid_h))
        features.append(self._normalize_val(obj.centroid[1], grid_w))

        # Pixel count (normalized by grid area)
        features.append(self._normalize_val(obj.pixel_count, grid_h * grid_w))

        # Bounding box (height, width, normalized)
        bb_h = obj.bounding_box[2] - obj.bounding_box[0] + 1
        bb_w = obj.bounding_box[3] - obj.bounding_box[1] + 1
        features.append(self._normalize_val(bb_h, grid_h))
        features.append(self._normalize_val(bb_w, grid_w))

        # Symmetries (from metadata, ensure they exist or default to 0)
        sym = obj.metadata.get('symmetries', {})
        features.append(float(sym.get('horizontal', False)))
        features.append(float(sym.get('vertical', False)))
        features.append(float(sym.get('rotational_180', False)))
        
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def _prepare_pairwise_features(
        self, 
        obj1_feat_tensor: torch.Tensor, 
        obj2_feat_tensor: torch.Tensor, 
        obj1: ARCObject, obj2: ARCObject, 
        grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """Prepares combined features for object pairs."""
        
        # Relative distance (normalized by grid diagonal)
        grid_diag = np.sqrt(grid_h**2 + grid_w**2)
        dist = np.linalg.norm(np.array(obj1.centroid) - np.array(obj2.centroid))
        rel_dist = self._normalize_val(dist, grid_diag)

        # Relative angle (decomposed into sin/cos)
        angle = np.arctan2(obj2.centroid[0] - obj1.centroid[0], obj2.centroid[1] - obj1.centroid[1])
        rel_angle_sin = np.sin(angle)
        rel_angle_cos = np.cos(angle)

        pairwise_specific_feats = torch.tensor([rel_dist, rel_angle_sin, rel_angle_cos], dtype=torch.float32).to(self.device)
        
        return torch.cat([obj1_feat_tensor, obj2_feat_tensor, pairwise_specific_feats])

    def forward(self, grid_features: GridFeatures) -> Dict[str, torch.Tensor]:
        """
        A conceptual forward pass that could compute all predicates for a grid.
        For practical use, get_grounded_symbols is more structured.
        This method's output structure would need careful design if used directly for training.
        """
        # This is a placeholder. Actual multi-task training would require more complex batching
        # and handling of different input types (single objects vs pairs).
        # For now, it's not the primary way to get predicate values.
        print("Warning: SymbolGroundingModel.forward() is conceptual. Use get_grounded_symbols().")
        return {}


    def get_grounded_symbols(self, grid_features: GridFeatures) -> Dict[str, Any]:
        """
        Computes probabilities for all defined symbolic predicates based on the input GridFeatures.
        """
        if not grid_features or not grid_features.objects:
            return { # Return empty lists for all predicates if no objects
                "is_touching": [], "is_inside": [], "is_same_color": [],
                "is_same_shape": [], "has_horizontal_symmetry": [], "has_vertical_symmetry": []
            }

        grounded_symbols: Dict[str, Any] = {
            "is_touching": [],
            "is_inside": [],
            "is_same_color": [],
            "is_same_shape": [],
            "has_horizontal_symmetry": [],
            "has_vertical_symmetry": []
        }
        
        # Pre-compute all object features
        obj_feature_tensors: Dict[int, torch.Tensor] = {}
        for obj in grid_features.objects:
            obj_feature_tensors[obj.id] = self._prepare_object_features(obj, grid_features.grid_height, grid_features.grid_width)

        # Unary predicates
        for obj_id, obj_feat_tensor in obj_feature_tensors.items():
            obj = next(o for o in grid_features.objects if o.id == obj_id) # Get the ARCObject instance

            # Horizontal Symmetry
            prob_h_sym = self.has_horizontal_symmetry_net(obj_feat_tensor).item()
            grounded_symbols["has_horizontal_symmetry"].append((obj_id, prob_h_sym))
            # For training: Use obj.metadata.get('symmetries', {}).get('horizontal', False) as label

            # Vertical Symmetry
            prob_v_sym = self.has_vertical_symmetry_net(obj_feat_tensor).item()
            grounded_symbols["has_vertical_symmetry"].append((obj_id, prob_v_sym))
            # For training: Use obj.metadata.get('symmetries', {}).get('vertical', False) as label

        # Pairwise predicates
        object_list = grid_features.objects
        for i in range(len(object_list)):
            for j in range(i + 1, len(object_list)): # Ensure (obj1, obj2) and not (obj2, obj1) for symmetric, avoid self-pairs
                obj1 = object_list[i]
                obj2 = object_list[j]
                
                obj1_id = obj1.id
                obj2_id = obj2.id
                
                obj1_feat_tensor = obj_feature_tensors[obj1_id]
                obj2_feat_tensor = obj_feature_tensors[obj2_id]

                pairwise_feat_tensor = self._prepare_pairwise_features(
                    obj1_feat_tensor, obj2_feat_tensor, 
                    obj1, obj2, 
                    grid_features.grid_height, grid_features.grid_width
                )

                # is_touching(obj1, obj2)
                prob_touching = self.is_touching_net(pairwise_feat_tensor).item()
                grounded_symbols["is_touching"].append(((obj1_id, obj2_id), prob_touching))
                # For training: Use obj1_id in obj2.metadata.get('adjacent_object_ids', []) as label

                # is_same_color(obj1, obj2)
                prob_same_color = self.is_same_color_net(pairwise_feat_tensor).item()
                grounded_symbols["is_same_color"].append(((obj1_id, obj2_id), prob_same_color))
                # For training: Use (obj1.color == obj2.color) as label

                # is_same_shape(obj1, obj2)
                prob_same_shape = self.is_same_shape_net(pairwise_feat_tensor).item()
                grounded_symbols["is_same_shape"].append(((obj1_id, obj2_id), prob_same_shape))
                # For training: Use (obj1.metadata.get('shape_signature') == obj2.metadata.get('shape_signature')) as label
                                
                # is_inside(obj1, obj2) -> obj1 is inside obj2
                # Need to check both (obj1 in obj2) and (obj2 in obj1) as separate predicates if relation is asymmetric
                # For (obj1 in obj2):
                prob_inside_o1_o2 = self.is_inside_net(pairwise_feat_tensor).item() 
                grounded_symbols["is_inside"].append(((obj1_id, obj2_id), prob_inside_o1_o2))
                # For training: Use obj1_id in obj2.metadata.get('contained_object_ids', []) as label
                
                # For (obj2 in obj1):
                # We need to prepare features with obj2 as the first element for the network if it's not symmetric by design
                # Or, ensure the training data covers both orderings if the network should learn symmetry.
                # For now, let's assume the network input order matters.
                pairwise_feat_tensor_rev = self._prepare_pairwise_features(
                    obj2_feat_tensor, obj1_feat_tensor,
                    obj2, obj1,
                    grid_features.grid_height, grid_features.grid_width
                )
                prob_inside_o2_o1 = self.is_inside_net(pairwise_feat_tensor_rev).item()
                grounded_symbols["is_inside"].append(((obj2_id, obj1_id), prob_inside_o2_o1))
                # For training: Use obj2_id in obj1.metadata.get('contained_object_ids', []) as label

        return grounded_symbols

# Example Usage (conceptual, assuming features are extracted elsewhere):
# if __name__ == '__main__':
#     # 1. Assume `grid_features` is an instance of GridFeatures, populated by UNetARCFeatureExtractor
#     # For testing, we'd need to mock GridFeatures and ARCObject instances.
#     
#     # Mock ARCObject
#     mock_obj1 = ARCObject(id=1, color=ARCPixel(1), pixels={(0,0)}, bounding_box=(0,0,0,0), pixel_count=1, centroid=(0.0,0.0), metadata={'symmetries': {'horizontal': True, 'vertical': False, 'rotational_180': False}, 'shape_signature': "s1"})
#     mock_obj2 = ARCObject(id=2, color=ARCPixel(2), pixels={(1,1)}, bounding_box=(1,1,1,1), pixel_count=1, centroid=(1.0,1.0), metadata={'symmetries': {'horizontal': False, 'vertical': True, 'rotational_180': True}, 'shape_signature': "s2", 'adjacent_object_ids': [1]})
#     mock_obj3 = ARCObject(id=3, color=ARCPixel(1), pixels={(2,2)}, bounding_box=(2,2,2,2), pixel_count=1, centroid=(2.0,2.0), metadata={'symmetries': {}, 'shape_signature': "s1"})
      # mock_obj1.metadata['adjacent_object_ids'] = [2] # Make obj1 adjacent to obj2 for is_touching label
# 
#     mock_grid_features = GridFeatures(
#         grid_height=10, grid_width=10, 
#         objects=[mock_obj1, mock_obj2, mock_obj3],
#         background_color=ARCPixel(0), 
#         unique_colors={ARCPixel(0), ARCPixel(1), ARCPixel(2)},
#         object_colors={ARCPixel(1), ARCPixel(2)},
#         color_counts={ARCPixel(0):97, ARCPixel(1):2, ARCPixel(2):1}
#     )
# 
#     # 2. Instantiate the grounding model
#     symbol_grounder = SymbolGroundingModel(hidden_size=64)
# 
#     # 3. Get grounded symbols (probabilities, since model is not trained)
#     # Make sure the model is in eval mode if dropout/batchnorm were used differently for train/eval
#     symbol_grounder.eval() 
#     with torch.no_grad(): # Important if not training
#         grounded_output = symbol_grounder.get_grounded_symbols(mock_grid_features)
# 
#     print("Grounded Symbol Probabilities (Untrained Model):")
#     for predicate_name, results in grounded_output.items():
#         print(f"  {predicate_name}:")
#         for result_item in results:
#             print(f"    {result_item}")
#
#     # Example of how one might get training labels (conceptual)
#     # For is_touching(obj1, obj2):
#     # label = 1.0 if mock_obj1.id in mock_obj2.metadata.get('adjacent_object_ids', []) else 0.0
#     # print(f"Training label for is_touching(1,2) would be: {label}")
#     # label_h_sym_obj1 = 1.0 if mock_obj1.metadata.get('symmetries', {}).get('horizontal', False) else 0.0
#     # print(f"Training label for has_horizontal_symmetry(1) would be: {label_h_sym_obj1}")
#     # label_same_color_o1_o3 = 1.0 if mock_obj1.color == mock_obj3.color else 0.0
#     # print(f"Training label for is_same_color(1,3) would be: {label_same_color_o1_o3}")
#     # label_same_shape_o1_o3 = 1.0 if mock_obj1.metadata.get('shape_signature') == mock_obj3.metadata.get('shape_signature') else 0.0
#     # print(f"Training label for is_same_shape(1,3) would be: {label_same_shape_o1_o3}")

print("SymbolGroundingModel structure defined in symbol_grounding.py")
