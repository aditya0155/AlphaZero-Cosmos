# src/ur_project/core/knowledge_base.py
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

from ur_project.core.perception import ARCObject, GridFeatures # Reusing existing types
from ur_project.data_processing.arc_types import ARCPixel, ARCGrid # Reusing existing types

# --- Basic Symbolic Primitives ---

@dataclass
class SymbolicValue:
    """Represents a generic symbolic value, which could be a number, color, string, etc."""
    value_type: str  # e.g., "color", "integer", "string", "boolean"
    value: Any

    def __hash__(self):
        return hash((self.value_type, self.value))

    def __eq__(self, other):
        if not isinstance(other, SymbolicValue):
            return NotImplemented
        return self.value_type == other.value_type and self.value == other.value

@dataclass
class SymbolicProperty:
    """Represents a property of an entity, e.g., (object_A, color, red_symbolic_value)."""
    name: str  # e.g., "color", "size", "shape_type"
    value: SymbolicValue

    def __hash__(self):
        return hash((self.name, self.value))

    def __eq__(self, other):
        if not isinstance(other, SymbolicProperty):
            return NotImplemented
        return self.name == other.name and self.value == other.value

@dataclass
class SymbolicEntity:
    """Represents an entity in the ARC world, typically an ARCObject or the grid itself."""
    id: str  # Unique identifier for this entity (e.g., "object_1", "grid_input")
    entity_type: str  # e.g., "arc_object", "arc_grid"
    properties: List[SymbolicProperty] = field(default_factory=list)
    # For ARC objects, we might store a direct reference or its ID
    source_arc_object_id: Optional[int] = None 
    source_grid_features_id: Optional[str] = None # If representing features of a whole grid

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, SymbolicEntity):
            return NotImplemented
        return self.id == other.id
    
    def add_property(self, name: str, value: Any, value_type: Optional[str] = None):
        if value_type is None:
            if isinstance(value, ARCPixel): value_type = "color"
            elif isinstance(value, int): value_type = "integer"
            elif isinstance(value, str): value_type = "string"
            elif isinstance(value, bool): value_type = "boolean"
            else: value_type = "unknown"
        
        sym_val = SymbolicValue(value_type=value_type, value=value)
        self.properties.append(SymbolicProperty(name=name, value=sym_val))

# --- Relationships ---

@dataclass
class SymbolicRelationship:
    """Represents a relationship between two or more entities."""
    type: str  # e.g., "spatial_relation", "transformational_rule_component"
    name: str  # e.g., "is_left_of", "is_larger_than", "is_transformed_into"
    subject: SymbolicEntity 
    object: Optional[SymbolicEntity] = None # Can be unary (property of subject) or binary
    # For more complex relationships, 'object' could be a list of entities or another relationship
    attributes: Dict[str, SymbolicValue] = field(default_factory=dict) # e.g. distance for "is_left_of"

    def __hash__(self):
        return hash((self.type, self.name, self.subject, self.object, tuple(sorted(self.attributes.items()))))

    def __eq__(self, other):
        if not isinstance(other, SymbolicRelationship):
            return NotImplemented
        return (self.type == other.type and
                self.name == other.name and
                self.subject == other.subject and
                self.object == other.object and
                self.attributes == other.attributes)

# --- Rules and Transformations (Very Basic Initial Schema) ---

@dataclass
class SymbolicTransformationRule:
    """
    Represents a very basic abstract rule, e.g., 
    "IF an object has property P1, THEN its corresponding output object has property P2"
    or "IF grid has N objects of color C, THEN output grid has M objects of color D".
    This is a placeholder and will need significant expansion.
    """
    id: str
    description: str
    # Conditions could be lists of required properties/relationships on input entities
    conditions: List[Union[SymbolicProperty, SymbolicRelationship]] = field(default_factory=list) 
    # Actions could describe changes or properties of output entities
    actions: List[Union[SymbolicProperty, SymbolicRelationship]] = field(default_factory=list)
    # Meta-data about the rule
    source: Optional[str] = None # e.g., "human_defined", "induced_from_task_XYZ"
    confidence: float = 1.0

# --- Knowledge Base ---

class ARCKnowledgeBase:
    """
    A simple in-memory knowledge base for storing symbolic facts and rules about ARC tasks.
    This will start as a collection of dictionaries and lists.
    Future: Could be backed by a graph database.
    """
    def __init__(self):
        self.entities: Dict[str, SymbolicEntity] = {} # entity_id -> SymbolicEntity
        self.relationships: List[SymbolicRelationship] = []
        self.rules: Dict[str, SymbolicTransformationRule] = {} # rule_id -> Rule
        self._temp_object_map: Dict[int, str] = {} # For mapping ARCObject.id to SymbolicEntity.id

    def clear_task_context(self):
        """Clears entities and relationships that are specific to a single task instance."""
        # This is a simplification. A more robust KB would tag entities/relations by task.
        self.entities.clear()
        self.relationships.clear()
        self._temp_object_map.clear()

    def _get_symbolic_entity_id_for_arc_object(self, arc_object: ARCObject, grid_context: str = "input") -> str:
        """Creates a unique ID for a symbolic entity based on an ARCObject."""
        # Simple ID generation for now
        # Ensure a mapping from arc_object.id (int) to a unique string ID for SymbolicEntity
        if arc_object.id not in self._temp_object_map:
            self._temp_object_map[arc_object.id] = f"{grid_context}_obj_{arc_object.id}"
        return self._temp_object_map[arc_object.id]

    def add_arc_object_as_entity(self, arc_object: ARCObject, grid_context: str = "input"):
        """Converts an ARCObject into a SymbolicEntity and adds it to the KB."""
        entity_id = self._get_symbolic_entity_id_for_arc_object(arc_object, grid_context)
        
        if entity_id in self.entities:
            # Potentially update existing entity, for now we assume new context = new entities
            # This depends on how we manage entities across input/output/multiple examples.
            # For now, let's allow re-adding if context changes (though clear_task_context helps)
            pass

        entity = SymbolicEntity(id=entity_id, entity_type="arc_object", source_arc_object_id=arc_object.id)
        
        # Add basic properties from ARCObject
        entity.add_property(name="color", value=arc_object.color, value_type="color")
        entity.add_property(name="pixel_count", value=arc_object.pixel_count, value_type="integer")
        # Bounding box properties
        entity.add_property(name="bbox_min_row", value=arc_object.bounding_box[0], value_type="integer")
        entity.add_property(name="bbox_min_col", value=arc_object.bounding_box[1], value_type="integer")
        entity.add_property(name="bbox_max_row", value=arc_object.bounding_box[2], value_type="integer")
        entity.add_property(name="bbox_max_col", value=arc_object.bounding_box[3], value_type="integer")
        entity.add_property(name="bbox_height", value=arc_object.bounding_box[2] - arc_object.bounding_box[0] + 1, value_type="integer")
        entity.add_property(name="bbox_width", value=arc_object.bounding_box[3] - arc_object.bounding_box[1] + 1, value_type="integer")
        # Centroid (might need a "coordinate" value_type or store as two floats)
        entity.add_property(name="centroid_row", value=arc_object.centroid[0], value_type="float")
        entity.add_property(name="centroid_col", value=arc_object.centroid[1], value_type="float")

        # Placeholder for shape (requires shape recognition logic from perception)
        # entity.add_property(name="shape_type", value="unknown", value_type="string") 

        self.entities[entity.id] = entity
        return entity

    def add_grid_features_as_entity(self, grid_features: GridFeatures, grid_id: str = "input_grid"):
        """Converts GridFeatures into a SymbolicEntity and adds it to the KB."""
        if grid_id in self.entities:
            # As with objects, decide update strategy or rely on clearing context
            pass
            
        entity = SymbolicEntity(id=grid_id, entity_type="arc_grid", source_grid_features_id=grid_id)
        
        entity.add_property(name="height", value=grid_features.grid_height, value_type="integer")
        entity.add_property(name="width", value=grid_features.grid_width, value_type="integer")
        entity.add_property(name="background_color", value=grid_features.background_color, value_type="color")
        entity.add_property(name="num_objects", value=len(grid_features.objects), value_type="integer")
        # Could add more global features like total_pixels_of_color_X etc.
        
        self.entities[entity.id] = entity
        return entity

    def add_relationship(self, subject_entity_id: str, object_entity_id: str, rel_type: str, rel_name: str, attributes: Optional[Dict[str, Any]] = None):
        subj = self.entities.get(subject_entity_id)
        obj = self.entities.get(object_entity_id)
        if not subj or not obj:
            print(f"Warning: Could not add relationship '{rel_name}' - subject or object entity not found.")
            return

        attr_sym_values = {}
        if attributes:
            for k, v_tuple in attributes.items(): # Expect v_tuple as (value, value_type_str)
                attr_sym_values[k] = SymbolicValue(value_type=v_tuple[1], value=v_tuple[0])
        
        rel = SymbolicRelationship(
            type=rel_type, 
            name=rel_name, 
            subject=subj, 
            object=obj,
            attributes=attr_sym_values
        )
        if rel not in self.relationships: # Avoid duplicates if added multiple times
            self.relationships.append(rel)
        return rel

    def add_rule(self, rule: SymbolicTransformationRule):
        if rule.id not in self.rules:
            self.rules[rule.id] = rule
        else:
            # Decide on update strategy for existing rules
            print(f"Warning: Rule with ID {rule.id} already exists. Overwriting.")
            self.rules[rule.id] = rule
            
    def get_entity(self, entity_id: str) -> Optional[SymbolicEntity]:
        return self.entities.get(entity_id)

    def query_relationships(self, subject_id: Optional[str] = None, rel_name: Optional[str] = None, object_id: Optional[str] = None) -> List[SymbolicRelationship]:
        """Basic querying for relationships."""
        results = []
        for rel in self.relationships:
            match = True
            if subject_id and rel.subject.id != subject_id:
                match = False
            if rel_name and rel.name != rel_name:
                match = False
            if object_id and (rel.object is None or rel.object.id != object_id):
                match = False
            if match:
                results.append(rel)
        return results

    # TODO: More sophisticated methods for populating KB from ARCPuzzle (input/output pairs)
    # TODO: Methods for inferring new relationships or rules (Phase 3+)

    def infer_spatial_relationships(self, relation_name: str = "is_left_of"):
        """
        Performs basic inference for transitive spatial relationships.
        Example: If A is_left_of B, and B is_left_of C, then infer A is_left_of C.
        This is a very basic example and would need to be generalized.
        """
        newly_inferred_relationships: List[SymbolicRelationship] = []
        # Limit iterations to prevent infinite loops in more complex scenarios, though not strictly needed for simple transitivity
        MAX_INFERENCE_ITERATIONS = 5 

        for _ in range(MAX_INFERENCE_ITERATIONS):
            inferred_in_this_iteration = 0
            current_rels_of_type = [r for r in self.relationships if r.name == relation_name and r.object is not None]

            for rel1 in current_rels_of_type:
                # rel1 is A -> B (A.subject relation_name B.object)
                A = rel1.subject
                B = rel1.object
                if B is None: continue # Should not happen due to filter but defensive

                # Find all relations where B is the subject: B -> C
                for rel2 in current_rels_of_type:
                    if rel2.subject.id == B.id:
                        C = rel2.object
                        if C is None: continue

                        # We found A -> B and B -> C. Infer A -> C.
                        # Check if A -> C already exists
                        existing_A_C_rel = False
                        for existing_rel in current_rels_of_type + newly_inferred_relationships:
                            if existing_rel.subject.id == A.id and \
                               existing_rel.object and existing_rel.object.id == C.id:
                                existing_A_C_rel = True
                                break
                        
                        if not existing_A_C_rel and A.id != C.id: # Avoid self-loops
                            # For now, attributes are not combined/inferred, could be complex (e.g., sum distances)
                            inferred_rel = SymbolicRelationship(
                                type=rel1.type, # Assume same type
                                name=relation_name,
                                subject=A,
                                object=C,
                                attributes={}
                            )
                            if inferred_rel not in newly_inferred_relationships and inferred_rel not in self.relationships:
                                newly_inferred_relationships.append(inferred_rel)
                                inferred_in_this_iteration += 1
            
            if inferred_in_this_iteration == 0:
                break # No new inferences in this pass
        
        # Add all unique new relationships to the main list
        for new_rel in newly_inferred_relationships:
            if new_rel not in self.relationships:
                self.relationships.append(new_rel)
        
        if newly_inferred_relationships:
            print(f"Inferred {len(newly_inferred_relationships)} new '{relation_name}' relationships.")

    def get_all_object_properties_for_prompt(self) -> List[str]:
        """
        Formats properties of all 'arc_object' entities for inclusion in a prompt.
        """
        object_property_strings = []
        for entity_id, entity in self.entities.items():
            if entity.entity_type == "arc_object":
                props_summary = []
                for prop in entity.properties:
                    # Format value more nicely for prompt
                    val_str = str(prop.value.value)
                    if prop.value.value_type == "color":
                        # Assuming ARCPixel has a simple representation
                        val_str = f"Color({prop.value.value.value})" if hasattr(prop.value.value, 'value') else str(prop.value.value)
                    
                    if "bbox_" in prop.name and ("min_row" in prop.name or "min_col" in prop.name): 
                        # Consolidate bbox info later if desired, for now list important ones
                        props_summary.append(f"{prop.name}: {val_str}")
                    elif "bbox_height" == prop.name or "bbox_width" == prop.name:
                        props_summary.append(f"{prop.name}: {val_str}")
                    elif prop.name in ["color", "pixel_count", "centroid_row", "centroid_col"]:
                         props_summary.append(f"{prop.name}: {val_str}")
                
                # Construct a summary like: "Object 'obj_1' (color: C, pixel_count: N, bbox_height: H, bbox_width: W)"
                # More sophisticated summary could be done here.
                details = ", ".join(props_summary)
                object_property_strings.append(f"Object '{entity.id}': {details}")
        return object_property_strings

    def get_all_relationships_for_prompt(self) -> List[str]:
        """
        Formats all relationships for inclusion in a prompt.
        """
        relationship_strings = []
        for rel in self.relationships:
            subject_id = rel.subject.id
            object_id = rel.object.id if rel.object else "None"
            attrs_summary = []
            if rel.attributes:
                for attr_name, attr_sym_val in rel.attributes.items():
                    attrs_summary.append(f"{attr_name}: {attr_sym_val.value}")
            
            attr_str = ""
            if attrs_summary:
                attr_str = f" ({', '.join(attrs_summary)})"
            
            relationship_strings.append(f"Relationship: {subject_id} {rel.name} {object_id}{attr_str}")
        return relationship_strings


if __name__ == '__main__':
    kb = ARCKnowledgeBase()

    # Example: Populate from a hypothetical ARCObject (normally from Perception module)
    obj_a_data = ARCObject(id=1, color=ARCPixel(1), pixels={(0,0),(0,1)}, bounding_box=(0,0,0,1), pixel_count=2, centroid=(0,0.5))
    obj_b_data = ARCObject(id=2, color=ARCPixel(2), pixels={(1,1)}, bounding_box=(1,1,1,1), pixel_count=1, centroid=(1,1))

    entity_a = kb.add_arc_object_as_entity(obj_a_data, grid_context="input1")
    entity_b = kb.add_arc_object_as_entity(obj_b_data, grid_context="input1")

    print(f"Entity A: {entity_a.id}, Properties: {[(p.name, p.value.value) for p in entity_a.properties]}")
    print(f"Entity B: {entity_b.id}, Properties: {[(p.name, p.value.value) for p in entity_b.properties]}")

    # Add a spatial relationship (hypothetical, needs actual calculation logic)
    # Assuming obj_a is "left_of" obj_b for this example
    kb.add_relationship(entity_a.id, entity_b.id, 
                        rel_type="spatial", rel_name="is_left_of", 
                        attributes={"distance_pixels": (1, "integer")})

    left_of_rels = kb.query_relationships(rel_name="is_left_of")
    for rel in left_of_rels:
        print(f"Relationship: {rel.subject.id} {rel.name} {rel.object.id if rel.object else ''} (Attr: {rel.attributes})")

    # Add another for transitivity: B is_left_of D
    obj_d_data = ARCObject(id=3, color=ARCPixel(3), pixels={(1,2)}, bounding_box=(1,2,1,2), pixel_count=1, centroid=(1,2))
    entity_d = kb.add_arc_object_as_entity(obj_d_data, grid_context="input1")
    kb.add_relationship(entity_b.id, entity_d.id,
                        rel_type="spatial", rel_name="is_left_of",
                        attributes={"distance_pixels": (1, "integer")})

    print("\n--- Before Inference ---")
    all_left_of_rels_before = kb.query_relationships(rel_name="is_left_of")
    for rel in all_left_of_rels_before:
        print(f"Relationship: {rel.subject.id} {rel.name} {rel.object.id if rel.object else ''} (Attr: {rel.attributes})")

    kb.infer_spatial_relationships(relation_name="is_left_of")

    print("\n--- After Inference ---")
    all_left_of_rels_after = kb.query_relationships(rel_name="is_left_of")
    for rel in all_left_of_rels_after:
        print(f"Relationship: {rel.subject.id} {rel.name} {rel.object.id if rel.object else ''} (Attr: {rel.attributes})")

    # Example Rule (very abstract)
    rule1_cond_prop = SymbolicProperty(name="color", value=SymbolicValue(value_type="color", value=ARCPixel(1)))
    rule1_action_prop = SymbolicProperty(name="color", value=SymbolicValue(value_type="color", value=ARCPixel(5))) # Change color to 5
    
    rule1 = SymbolicTransformationRule(
        id="rule_color_change_1_to_5",
        description="If an input object is color 1, its output counterpart is color 5.",
        conditions=[rule1_cond_prop],
        actions=[rule1_action_prop],
        source="hypothetical"
    )
    kb.add_rule(rule1)
    print(f"Rule '{rule1.id}': Conditions: {rule1.conditions}, Actions: {rule1.actions}")

    kb.clear_task_context()
    print(f"KB entities after clear: {len(kb.entities)}")

    # Example with GridFeatures
    example_grid_for_features: ARCGrid = [[ARCPixel(1), ARCPixel(0)], [ARCPixel(0), ARCPixel(1)]]
    # Assume BasicARCFeatureExtractor produces these (simplified for example)
    grid_feats = GridFeatures(
        grid_height=2, grid_width=2, objects=[obj_a_data], 
        background_color=ARCPixel(0), unique_colors={ARCPixel(0), ARCPixel(1)},
        object_colors={ARCPixel(1)}, color_counts={ARCPixel(0):2, ARCPixel(1):2}
    )
    grid_entity = kb.add_grid_features_as_entity(grid_feats, grid_id="my_input_grid")
    print(f"Grid Entity: {grid_entity.id}, Properties: {[(p.name, p.value.value) for p in grid_entity.properties]}") 