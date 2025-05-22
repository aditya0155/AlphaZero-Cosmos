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
    context_tags: List[str] = field(default_factory=list)

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
    subject: Union[SymbolicEntity, 'EntityPlaceholder'] 
    object: Optional[Union[SymbolicEntity, 'EntityPlaceholder']] = None
    # For more complex relationships, 'object' could be a list of entities or another relationship
    attributes: Dict[str, SymbolicValue] = field(default_factory=dict) # e.g. distance for "is_left_of"
    context_tags: List[str] = field(default_factory=list)

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
    # Conditions and actions now use more structured types for clarity with variables
    conditions: List['RuleCondition'] = field(default_factory=list) 
    actions: List['RuleAction'] = field(default_factory=list)
    # Meta-data about the rule
    source: Optional[str] = None # e.g., "human_defined", "induced_from_task_XYZ"
    confidence: float = 1.0
    context_tags: List[str] = field(default_factory=list)


@dataclass
class EntityPlaceholder:
    """Represents a variable that stands for an entity in a rule."""
    variable_name: str  # e.g., "?X", "?objectA"

    def __hash__(self):
        # Ensure hashability if used as dict keys or in sets directly
        return hash(self.variable_name)

    def __eq__(self, other):
        if not isinstance(other, EntityPlaceholder):
            return NotImplemented
        return self.variable_name == other.variable_name
    
    def __str__(self): # For easier debugging
        return self.variable_name

@dataclass
class RuleCondition:
    """
    Describes a condition within a rule.
    A condition can be:
    1. A property that an entity variable must have (property_pattern on entity_var).
       The property_pattern.value can itself be a variable (e.g., "?color").
    2. A relationship that must exist (relationship_pattern).
       The subject/object in relationship_pattern can be EntityPlaceholders.
    """
    # For property conditions:
    entity_var: Optional[EntityPlaceholder] = None
    property_pattern: Optional[SymbolicProperty] = None # e.g. SymbolicProperty(name="color", value=SymbolicValue("color", "?C"))
    
    # For relationship conditions:
    relationship_pattern: Optional[SymbolicRelationship] = None # e.g. SymbolicRelationship(subject=EntityPlaceholder("?X"), name="is_left_of", object=EntityPlaceholder("?Y"))

    # For direct value comparison if needed (less common, usually properties of entities)
    # value_comparison: Optional[Tuple[SymbolicValue, str, SymbolicValue]] = None # e.g. ("?var1", "equals", "?var2")

    def __post_init__(self):
        if self.entity_var and self.property_pattern:
            if self.relationship_pattern: # or self.value_comparison
                raise ValueError("RuleCondition cannot have both property and relationship patterns.")
        elif self.relationship_pattern:
            if self.entity_var or self.property_pattern: # or self.value_comparison
                raise ValueError("RuleCondition with relationship_pattern should not have other fields set.")
        # Add similar checks if value_comparison is implemented
        elif not (self.entity_var and self.property_pattern) and not self.relationship_pattern: # and not self.value_comparison
             raise ValueError("RuleCondition must specify either a property pattern or a relationship pattern.")


@dataclass
class RuleAction:
    """
    Describes an action to be taken if a rule's conditions are met.
    An action can be:
    1. Modifying/asserting a property of an entity variable (property_assert on entity_var).
    2. Adding a relationship (relationship_add).
    3. Creating a new entity (create_entity_var, with type and optional properties).
    4. Deleting an entity (delete_entity_var).
    (Create and Delete are more advanced and might be deferred for full implementation)
    """
    # For property changes/assertions:
    entity_var: Optional[EntityPlaceholder] = None
    property_assert: Optional[SymbolicProperty] = None # e.g. SymbolicProperty(name="color", value=SymbolicValue("color", "blue"))
                                                       # or SymbolicProperty(name="color", value=SymbolicValue("color", "?new_color_var"))

    # For adding relationships:
    relationship_add: Optional[SymbolicRelationship] = None

    # --- Future extensions for more complex actions ---
    # For removing relationships: (Pattern to match relationships to remove)
    # relationship_remove_pattern: Optional[SymbolicRelationship] = None 
    
    # For creating new entities:
    # create_entity_placeholder: Optional[EntityPlaceholder] = None # e.g., "?new_object"
    # new_entity_type: Optional[str] = None # e.g., "arc_object"
    # new_entity_properties: List[SymbolicProperty] = field(default_factory=list) # Properties for the new entity

    # For deleting entities:
    # delete_entity_placeholder: Optional[EntityPlaceholder] = None 

    def __post_init__(self):
        # Basic validation that at least one action type is specified
        action_fields = [
            (self.entity_var and self.property_assert), 
            self.relationship_add,
            # self.relationship_remove_pattern, 
            # self.create_entity_placeholder, 
            # self.delete_entity_placeholder
        ]
        if sum(bool(field_is_set) for field_is_set in action_fields) != 1:
            raise ValueError("RuleAction must specify exactly one action type (e.g., property_assert, relationship_add).")
        
        if self.entity_var and not self.property_assert:
             raise ValueError("If entity_var is specified for property assertion, property_assert must also be specified.")
        if self.property_assert and not self.entity_var:
             raise ValueError("If property_assert is specified, entity_var must also be specified.")


# --- Knowledge Base ---

@dataclass
class StoredTaskSolution:
    """Represents a stored solution attempt for an ARC task or task pair."""
    solution_id: str  # Unique ID, e.g., "taskXYZ_train0_sol1"
    task_id: str      # Original ARC task ID
    example_id: Optional[str] = None # e.g., "train_0", "test_0" if for a specific pair

    # Sequence of applied transformations (high-level or DSL-level).
    # Each item is a dict, e.g., 
    # {'type': 'rule_application', 'rule_id': 'rule123', 'bindings': {'?X': 'obj1', '?C': 'red'}}
    # {'type': 'dsl_command', 'command': 'draw_line(...)'}
    applied_rules_or_transformations: List[Dict[str, Any]] = field(default_factory=list)
    
    output_grid_achieved: Optional[ARCGrid] = None # The actual output grid produced
    is_successful: bool = False # Whether this solution solved the pair/task
    confidence: float = 1.0
    reasoning_trace: Optional[str] = None # Textual description or log
    
    context_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.solution_id)

    def __eq__(self, other):
        if not isinstance(other, StoredTaskSolution):
            return NotImplemented
        return self.solution_id == other.solution_id

@dataclass
class ARCPattern:
    """Represents a common visual or transformational pattern observed in ARC tasks."""
    pattern_id: str # Unique ID, e.g., "pattern_reflection_H_color_swap"
    description: str
    pattern_type: str # e.g., "visual_motif", "transformational_sequence", "object_property_cluster"
    
    # Flexible representation of the pattern itself.
    # Examples:
    # For visual_motif: {"type": "grid_patch", "patch": [[1,0],[0,1]], "ignore_color": True}
    # For transformational_sequence: {"type": "rule_ids_sequence", "rules": ["rule_move_down", "rule_change_color_blue"]}
    # For object_property_cluster: {"type": "property_set", "properties": [{"name": "color", "value": "blue"}, {"name": "shape", "value": "square"}]}
    representation: Dict[str, Any] = field(default_factory=dict)
    
    related_rule_ids: List[str] = field(default_factory=list) # Links to SymbolicTransformationRules
    # References to specific examples where this pattern is observed
    example_references: List[Dict[str,str]] = field(default_factory=list) # e.g., [{'task_id': 'abc', 'example_id': 'train_0', 'entity_id': 'obj1'}]
    
    context_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.pattern_id)

    def __eq__(self, other):
        if not isinstance(other, ARCPattern):
            return NotImplemented
        return self.pattern_id == other.pattern_id


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
        self.solutions: Dict[str, StoredTaskSolution] = {} # solution_id -> StoredTaskSolution
        self.patterns: Dict[str, ARCPattern] = {} # pattern_id -> ARCPattern
        # _temp_object_map is removed as entity IDs should be unique and context-aware from the start
        # For example, entity_id could be "task1_example0_input_obj_1"

    def _matches_context(self, item_tags: List[str], query_tags: Optional[List[str]], match_all_query_tags: bool = True) -> bool:
        """Helper to check if item_tags match query_tags."""
        if query_tags is None: # No context query, always matches
            return True
        if not query_tags: # Empty list of query tags also means match all (no specific context requested)
             return True
        if not item_tags and query_tags: # Item has no tags, but query requires tags
            return False

        if match_all_query_tags:
            return all(q_tag in item_tags for q_tag in query_tags)
        else: # Match any query tag
            return any(q_tag in item_tags for q_tag in query_tags)

    def clear_context_by_tag(self, context_tag: str):
        """Clears entities, relationships, and rules associated with a specific context_tag."""
        self.entities = {
            eid: e for eid, e in self.entities.items() if context_tag not in e.context_tags
        }
        self.relationships = [
            r for r in self.relationships if context_tag not in r.context_tags
        ]
        self.rules = {
            rid: r for rid, r in self.rules.items() if context_tag not in r.context_tags
        }

    def clear_contexts_by_tags(self, context_tags_to_clear: List[str], match_all: bool = True):
        """
        Clears entities, relationships, and rules based on a list of context tags.
        Args:
            context_tags_to_clear (List[str]): The list of tags to check for clearing.
            match_all (bool): If True, an item is cleared if it has ALL tags in context_tags_to_clear.
                              If False, an item is cleared if it has ANY tag in context_tags_to_clear.
        """
        if not context_tags_to_clear:
            return

        self.entities = {
            eid: e for eid, e in self.entities.items() 
            if not self._matches_context(e.context_tags, context_tags_to_clear, match_all_query_tags=match_all)
        }
        # For relationships, if either subject or object is cleared, the relationship should be too.
        # However, simpler to just filter by tags on the relationship itself first.
        # Complexities arise if an entity is cleared but its relationships (not sharing all clear_tags) remain.
        # For now, filter relationships based on their own tags.
        # A more robust cleanup might be needed later if entities are removed and relationships become orphaned.
        self.relationships = [
            r for r in self.relationships 
            if not self._matches_context(r.context_tags, context_tags_to_clear, match_all_query_tags=match_all)
        ]
        self.rules = {
            rid: r for rid, r in self.rules.items() 
            if not self._matches_context(r.context_tags, context_tags_to_clear, match_all_query_tags=match_all)
        }
        self.solutions = {
            sid: s for sid, s in self.solutions.items()
            if not self._matches_context(s.context_tags, context_tags_to_clear, match_all_query_tags=match_all)
        }
        self.patterns = {
            pid: p for pid, p in self.patterns.items()
            if not self._matches_context(p.context_tags, context_tags_to_clear, match_all_query_tags=match_all)
        }
        self.solutions = {
            sid: s for sid, s in self.solutions.items()
            if context_tag not in s.context_tags
        }
        self.patterns = {
            pid: p for pid, p in self.patterns.items()
            if context_tag not in p.context_tags
        }
        
        # Secondary pass: remove relationships whose subject or object no longer exists
        # This handles cases where entities were removed due to context tags, but relationships
        # might not have had the exact same set of tags to be removed in the first pass.
        valid_entity_ids = set(self.entities.keys())
        self.relationships = [
            r for r in self.relationships 
            if r.subject.id in valid_entity_ids and (r.object is None or r.object.id in valid_entity_ids)
        ]


    def add_entity(self, entity: SymbolicEntity):
        """Adds a symbolic entity to the KB. Entity ID should be globally unique and context-aware."""
        if entity.id in self.entities:
            # Potentially update or merge. For now, overwrite.
            # Consider logging a warning if overwriting outside of a clear context operation.
            # print(f"Warning: Entity with ID {entity.id} already exists. Overwriting.")
            pass
        self.entities[entity.id] = entity
        return entity

    def add_arc_object_as_entity(self, arc_object: ARCObject, entity_id: str, context_tags: Optional[List[str]] = None) -> SymbolicEntity:
        """
        Converts an ARCObject into a SymbolicEntity and adds it to the KB.
        Args:
            arc_object (ARCObject): The source ARCObject.
            entity_id (str): A globally unique ID for this symbolic entity (e.g., "taskX_ex0_input_obj1").
            context_tags (Optional[List[str]]): Context tags for this entity.
        """
        if entity_id in self.entities:
            # print(f"Warning: Entity {entity_id} already exists. Overwriting.")
            pass

        entity = SymbolicEntity(id=entity_id, entity_type="arc_object", 
                                source_arc_object_id=arc_object.id,
                                context_tags=context_tags or [])
        
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

        self.add_entity(entity)
        return entity

    def add_grid_features_as_entity(self, grid_features: GridFeatures, entity_id: str, context_tags: Optional[List[str]] = None) -> SymbolicEntity:
        """
        Converts GridFeatures into a SymbolicEntity and adds it to the KB.
        Args:
            grid_features (GridFeatures): The source GridFeatures.
            entity_id (str): A globally unique ID for this grid entity (e.g., "taskX_ex0_input_grid").
            context_tags (Optional[List[str]]): Context tags for this entity.
        """
        if entity_id in self.entities:
            # print(f"Warning: Grid entity {entity_id} already exists. Overwriting.")
            pass
            
        entity = SymbolicEntity(id=entity_id, entity_type="arc_grid", 
                                source_grid_features_id=entity_id, # Using entity_id as source_grid_features_id
                                context_tags=context_tags or [])
        
        entity.add_property(name="height", value=grid_features.grid_height, value_type="integer")
        entity.add_property(name="width", value=grid_features.grid_width, value_type="integer")
        entity.add_property(name="background_color", value=grid_features.background_color, value_type="color")
        entity.add_property(name="num_objects", value=len(grid_features.objects), value_type="integer")
        # Could add more global features like total_pixels_of_color_X etc.
        
        self.add_entity(entity)
        return entity
    
    def add_relationship(self, relationship: SymbolicRelationship):
        """Adds a symbolic relationship to the KB."""
        # Ensure subject and object entities exist if relationship is not self-referential or special
        if not self.entities.get(relationship.subject.id):
            print(f"Warning: Subject entity {relationship.subject.id} for relationship {relationship.name} not found in KB.")
            return None
        if relationship.object and not self.entities.get(relationship.object.id):
            print(f"Warning: Object entity {relationship.object.id} for relationship {relationship.name} not found in KB.")
            return None

        # Could add duplicate checking if necessary, though context tags might differentiate them
        self.relationships.append(relationship)
        return relationship
        
    def create_relationship(self, subject_entity_id: str, object_entity_id: Optional[str], 
                            rel_type: str, rel_name: str, 
                            attributes: Optional[Dict[str, Any]] = None, 
                            context_tags: Optional[List[str]] = None) -> Optional[SymbolicRelationship]:
        """Helper to create and add a relationship."""
        subj = self.get_entity(subject_entity_id) # Use context-aware getter if filtering by context at entity level
        obj = self.get_entity(object_entity_id) if object_entity_id else None
        
        if not subj:
            print(f"Warning: Could not create relationship '{rel_name}' - subject entity '{subject_entity_id}' not found.")
            return None
        if object_entity_id and not obj:
            print(f"Warning: Could not create relationship '{rel_name}' - object entity '{object_entity_id}' not found.")
            return None

        attr_sym_values = {}
        if attributes:
            for k, v_tuple in attributes.items(): 
                val, val_type_str = v_tuple if isinstance(v_tuple, tuple) and len(v_tuple) == 2 else (v_tuple, None)
                if val_type_str is None: # Auto-detect type
                    if isinstance(val, ARCPixel): val_type_str = "color"
                    elif isinstance(val, int): val_type_str = "integer"
                    elif isinstance(val, float): val_type_str = "float"
                    elif isinstance(val, str): val_type_str = "string"
                    elif isinstance(val, bool): val_type_str = "boolean"
                    else: val_type_str = "unknown"
                attr_sym_values[k] = SymbolicValue(value_type=val_type_str, value=val)
        
        rel = SymbolicRelationship(
            type=rel_type, 
            name=rel_name, 
            subject=subj, 
            object=obj,
            attributes=attr_sym_values,
            context_tags=context_tags or []
        )
        self.relationships.append(rel) # Assuming direct append, can add duplicate checks
        return rel

    def add_rule(self, rule: SymbolicTransformationRule):
        """Adds a transformation rule to the KB."""
        if rule.id in self.rules:
            # print(f"Warning: Rule with ID {rule.id} already exists. Overwriting.")
            pass
        self.rules[rule.id] = rule
            
    def get_entity(self, entity_id: str, context_tags: Optional[List[str]] = None, match_all_query_tags: bool = True) -> Optional[SymbolicEntity]:
        entity = self.entities.get(entity_id)
        if entity and self._matches_context(entity.context_tags, context_tags, match_all_query_tags):
            return entity
        return None

    def query_entities(self, entity_type: Optional[str] = None, context_tags: Optional[List[str]] = None, match_all_query_tags: bool = True) -> List[SymbolicEntity]:
        """Queries entities by type and context."""
        results = []
        for entity in self.entities.values():
            type_match = (entity_type is None) or (entity.entity_type == entity_type)
            context_match = self._matches_context(entity.context_tags, context_tags, match_all_query_tags)
            if type_match and context_match:
                results.append(entity)
        return results

    def query_relationships(self, 
                            subject_id: Optional[str] = None, 
                            rel_name: Optional[str] = None, 
                            object_id: Optional[str] = None,
                            rel_type: Optional[str] = None,
                            context_tags: Optional[List[str]] = None,
                            match_all_query_tags: bool = True) -> List[SymbolicRelationship]:
        """Queries relationships with context awareness."""
        results = []
        for rel in self.relationships:
            if not self._matches_context(rel.context_tags, context_tags, match_all_query_tags):
                continue
            
            # Entity context matching for subject/object (optional, depends on desired strictness)
            # If query_tags are provided, we might want to ensure that the related entities also match the context.
            # However, get_entity used in create_relationship does not yet enforce this.
            # For now, relationship context tag matching is primary.

            subject_match = (subject_id is None) or (rel.subject.id == subject_id)
            name_match = (rel_name is None) or (rel.name == rel_name)
            object_match = (object_id is None) or (rel.object and rel.object.id == object_id)
            type_match = (rel_type is None) or (rel.type == rel_type)
            
            if subject_match and name_match and object_match and type_match:
                results.append(rel)
        return results
    
    def get_rules(self, context_tags: Optional[List[str]] = None, match_all_query_tags: bool = True) -> List[SymbolicTransformationRule]:
        """Gets rules, filtered by context tags."""
        return [
            rule for rule in self.rules.values()
            if self._matches_context(rule.context_tags, context_tags, match_all_query_tags)
        ]

    def add_stored_solution(self, solution: StoredTaskSolution):
        if solution.solution_id in self.solutions:
            # print(f"Warning: Solution with ID {solution.solution_id} already exists. Overwriting.")
            pass
        self.solutions[solution.solution_id] = solution

    def get_stored_solution(self, solution_id: str, context_tags: Optional[List[str]] = None, match_all_query_tags: bool = True) -> Optional[StoredTaskSolution]:
        solution = self.solutions.get(solution_id)
        if solution and self._matches_context(solution.context_tags, context_tags, match_all_query_tags):
            return solution
        return None

    def query_stored_solutions(self, task_id: Optional[str] = None, 
                               is_successful: Optional[bool] = None, 
                               context_tags: Optional[List[str]] = None, 
                               match_all_query_tags: bool = True) -> List[StoredTaskSolution]:
        results = []
        for solution in self.solutions.values():
            if not self._matches_context(solution.context_tags, context_tags, match_all_query_tags):
                continue
            
            task_match = (task_id is None) or (solution.task_id == task_id)
            success_match = (is_successful is None) or (solution.is_successful == is_successful)

            if task_match and success_match:
                results.append(solution)
        return results

    def add_arc_pattern(self, pattern: ARCPattern):
        if pattern.pattern_id in self.patterns:
            # print(f"Warning: Pattern with ID {pattern.pattern_id} already exists. Overwriting.")
            pass
        self.patterns[pattern.pattern_id] = pattern

    def get_arc_pattern(self, pattern_id: str, context_tags: Optional[List[str]] = None, match_all_query_tags: bool = True) -> Optional[ARCPattern]:
        pattern = self.patterns.get(pattern_id)
        if pattern and self._matches_context(pattern.context_tags, context_tags, match_all_query_tags):
            return pattern
        return None

    def query_arc_patterns(self, pattern_type: Optional[str] = None, 
                           context_tags: Optional[List[str]] = None, 
                           match_all_query_tags: bool = True) -> List[ARCPattern]:
        results = []
        for pattern in self.patterns.values():
            if not self._matches_context(pattern.context_tags, context_tags, match_all_query_tags):
                continue
            
            type_match = (pattern_type is None) or (pattern.pattern_type == pattern_type)
            if type_match:
                results.append(pattern)
        return results

    # TODO: More sophisticated methods for populating KB from ARCPuzzle (input/output pairs)
    # Need to update these to use new context-aware methods and entity ID conventions.

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
            # Consider using logging instead of print for library code
            # print(f"Inferred {len(newly_inferred_relationships)} new '{relation_name}' relationships.")
            pass


    def infer_object_correspondence_from_training_pairs(self, current_task_id: str, current_example_id: str):
        """
        Finds and records likely correspondences between objects in input and output grids
        of a specific training pair context.
        Entity IDs are expected to be structured to allow this, e.g., by including task/example info,
        or this method relies on context_tags.
        """
        inferred_correspondences = 0
        
        input_context_tags = [current_task_id, current_example_id, "input"]
        output_context_tags = [current_task_id, current_example_id, "output"]

        input_objects = self.query_entities(entity_type="arc_object", context_tags=input_context_tags)
        output_objects = self.query_entities(entity_type="arc_object", context_tags=output_context_tags)

        if not input_objects or not output_objects:
            # print(f"No input or output objects found for context: {current_task_id}, {current_example_id}")
            return

        # This matching logic is simplified; a more robust version might use features from
        # perception.UNetARCFeatureExtractor.analyze_object_transformations
        for input_obj_entity in input_objects:
            input_color_prop = next((p for p in input_obj_entity.properties if p.name == "color"), None)
            input_px_count_prop = next((p for p in input_obj_entity.properties if p.name == "pixel_count"), None)

            if not input_color_prop or not input_px_count_prop:
                continue

            candidate_output_objects = []
            for output_obj_entity in output_objects:
                output_color_prop = next((p for p in output_obj_entity.properties if p.name == "color"), None)
                output_px_count_prop = next((p for p in output_obj_entity.properties if p.name == "pixel_count"), None)

                if not output_color_prop or not output_px_count_prop:
                    continue
                
                if input_color_prop.value == output_color_prop.value and \
                   input_px_count_prop.value == output_px_count_prop.value:
                    candidate_output_objects.append(output_obj_entity)
            
            if len(candidate_output_objects) == 1: # Unique match based on simplified criteria
                output_obj_entity = candidate_output_objects[0]
                
                attrs = {"confidence": (1.0, "float"), 
                         "task_id": (current_task_id, "string"),
                         "example_id": (current_example_id, "string")}
                
                # Context for the relationship itself
                rel_context_tags = [current_task_id, current_example_id, "correspondence"]

                rel = self.create_relationship(
                    subject_entity_id=input_obj_entity.id,
                    object_entity_id=output_obj_entity.id,
                    rel_type="correspondence_logic",
                    rel_name="has_training_pair_correspondence",
                    attributes=attrs,
                    context_tags=rel_context_tags
                )
                if rel:
                    inferred_correspondences += 1
        
        if inferred_correspondences > 0:
            # print(f"Inferred {inferred_correspondences} object correspondences for {current_task_id}/{current_example_id}.")
            pass

    def infer_property_transfer_via_correspondence(self, current_task_id: str, current_example_id: str):
        """
        Identifies how properties change between corresponding input and output objects
        within a specific task/example context.
        """
        inferred_property_changes = 0
        # Query for correspondence relationships specific to this context
        correspondence_context_tags = [current_task_id, current_example_id, "correspondence"]
        correspondence_rels = self.query_relationships(
            rel_name="has_training_pair_correspondence", 
            context_tags=correspondence_context_tags
        )

        for corr_rel in correspondence_rels:
            # Ensure entities are fetched respecting their original contexts if necessary,
            # though direct references from `corr_rel` should be fine as they were added with context.
            input_obj_entity = corr_rel.subject 
            output_obj_entity = corr_rel.object

            if not input_obj_entity or not output_obj_entity: continue

            input_props_map: Dict[str, SymbolicProperty] = {p.name: p for p in input_obj_entity.properties}
            output_props_map: Dict[str, SymbolicProperty] = {p.name: p for p in output_obj_entity.properties}

            for prop_name, input_prop in input_props_map.items():
                output_prop = output_props_map.get(prop_name)
                if output_prop and input_prop.value != output_prop.value:
                    attrs = {
                        "property_name": (prop_name, "string"),
                        "input_value": (input_prop.value.value, input_prop.value.value_type),
                        "output_value": (output_prop.value.value, output_prop.value.value_type),
                        "task_id": (current_task_id, "string"),
                        "example_id": (current_example_id, "string")
                    }
                    
                    # Context for the "property_changed" relationship itself
                    change_rel_context_tags = [current_task_id, current_example_id, "transformation_observation"]

                    rel = self.create_relationship(
                        subject_entity_id=input_obj_entity.id,
                        object_entity_id=output_obj_entity.id,
                        rel_type="transformation_observation",
                        rel_name="property_changed",
                        attributes=attrs,
                        context_tags=change_rel_context_tags
                    )
                    if rel:
                        inferred_property_changes += 1
                # else:
                    # Property lost (input_prop exists, output_prop does not)
                    # Future: add "property_lost" relationship
                    # pass

            # Future: Iterate output_props_map to find "property_gained"
            # for prop_name, output_prop in output_props_map.items():
            #     if prop_name not in input_props_map:
            #         # Property gained
            #         pass
        
        if inferred_property_changes > 0:
            # print(f"Inferred {inferred_property_changes} property changes via correspondence.")
            pass

    def propagate_properties_via_symmetry(self, symmetry_relation_name: str = "is_symmetric_to"):
        """
        Infers properties for an object based on the properties of its symmetric counterpart.
        Assumes "is_symmetric_to" relationships exist in the KB.
        Properties are propagated if one entity has it and the other lacks it.
        """
        propagatable_properties = ["color", "pixel_count", "bbox_height", "bbox_width"] # Extendable list
        # Potentially "shape_type", "is_hollow" if these get added as standard properties.
        
        symmetry_relationships = self.query_relationships(rel_name=symmetry_relation_name)
        properties_propagated_count = 0

        for rel in symmetry_relationships:
            entity_A = rel.subject
            entity_B = rel.object

            if not entity_A or not entity_B: # Should not happen with valid relationships
                continue

            # Helper to get a map of existing properties for an entity
            def get_props_map(entity: SymbolicEntity) -> Dict[str, SymbolicProperty]:
                return {p.name: p for p in entity.properties}

            props_A = get_props_map(entity_A)
            props_B = get_props_map(entity_B)

            for prop_name in propagatable_properties:
                prop_A_exists = prop_name in props_A
                prop_B_exists = prop_name in props_B

                # Propagate A to B
                if prop_A_exists and not prop_B_exists:
                    source_prop = props_A[prop_name]
                    # SymbolicEntity.add_property handles creating new SymbolicValue and SymbolicProperty
                    entity_B.add_property(name=source_prop.name, 
                                          value=source_prop.value.value, 
                                          value_type=source_prop.value.value_type)
                    properties_propagated_count += 1
                    props_B = get_props_map(entity_B) # Update props_B map after adding

                # Propagate B to A
                elif prop_B_exists and not prop_A_exists:
                    source_prop = props_B[prop_name]
                    entity_A.add_property(name=source_prop.name, 
                                          value=source_prop.value.value, 
                                          value_type=source_prop.value.value_type)
                    properties_propagated_count += 1
                    props_A = get_props_map(entity_A) # Update props_A map
                
                # Conflict logging (simplified as per Phase 2 instructions to only add if missing)
                # elif prop_A_exists and prop_B_exists:
                #     if props_A[prop_name].value != props_B[prop_name].value:
                #         # print(f"Warning: Symmetric entities {entity_A.id} and {entity_B.id} have conflicting values for property '{prop_name}'.")
                #         pass # Using pass as per simplified instructions

        if properties_propagated_count > 0:
            # print(f"Propagated {properties_propagated_count} properties via symmetry.")
            pass

    def get_all_object_properties_for_prompt(self, context_tags: Optional[List[str]] = None, match_all_query_tags: bool = True) -> List[str]:
        """
        Formats properties of 'arc_object' entities for inclusion in a prompt, filtered by context.
        """
        object_property_strings = []
        
        relevant_entities = self.query_entities(entity_type="arc_object", 
                                                context_tags=context_tags, 
                                                match_all_query_tags=match_all_query_tags)

        for entity in relevant_entities:
            props_summary = []
            for prop in entity.properties:
                val_str = str(prop.value.value)
                if prop.value.value_type == "color":
                    val_str = f"Color({prop.value.value.value})" if hasattr(prop.value.value, 'value') else str(prop.value.value)
                
                # Selective property inclusion for brevity in prompt
                if prop.name in ["color", "pixel_count", "bbox_height", "bbox_width", "centroid_row", "centroid_col"] or "symmetry" in prop.name:
                     props_summary.append(f"{prop.name}: {val_str}")
            
            details = ", ".join(props_summary)
            object_property_strings.append(f"Object '{entity.id}': {details}")
        return object_property_strings

    def get_all_relationships_for_prompt(self, context_tags: Optional[List[str]] = None, match_all_query_tags: bool = True) -> List[str]:
        """
        Formats relationships for inclusion in a prompt, filtered by context.
        """
        relationship_strings = []
        relevant_relationships = self.query_relationships(context_tags=context_tags, 
                                                          match_all_query_tags=match_all_query_tags)
        for rel in relevant_relationships:
            subject_id_str = str(rel.subject.id) if isinstance(rel.subject, SymbolicEntity) else str(rel.subject)
            object_id_str = "None"
            if rel.object:
                object_id_str = str(rel.object.id) if isinstance(rel.object, SymbolicEntity) else str(rel.object)
            
            attrs_summary = [f"{name}: {val.value}" for name, val in rel.attributes.items()]
            attr_str = f" ({', '.join(attrs_summary)})" if attrs_summary else ""
            relationship_strings.append(f"Relationship: {subject_id_str} {rel.name} {object_id_str}{attr_str}")
        return relationship_strings


if __name__ == '__main__':
    kb = ARCKnowledgeBase()
    
    task_id = "task123"
    example_id_train0 = "train_0"
    example_id_train1 = "train_1"

    # Context tags for entities and relationships
    input_ctx_train0 = [task_id, example_id_train0, "input"]
    output_ctx_train0 = [task_id, example_id_train0, "output"]
    correspondence_ctx_train0 = [task_id, example_id_train0, "correspondence"]
    rule_ctx_general = ["general_rule"]


    # Example: Populate from a hypothetical ARCObject (normally from Perception module)
    obj_a_data = ARCObject(id=1, color=ARCPixel(1), pixels={(0,0),(0,1)}, bounding_box=(0,0,0,1), pixel_count=2, centroid=(0,0.5))
    obj_b_data = ARCObject(id=2, color=ARCPixel(2), pixels={(1,1)}, bounding_box=(1,1,1,1), pixel_count=1, centroid=(1,1))

    # Entity IDs should be unique and descriptive
    entity_a_id = f"{task_id}_{example_id_train0}_input_obj_1"
    entity_b_id = f"{task_id}_{example_id_train0}_input_obj_2"

    entity_a = kb.add_arc_object_as_entity(obj_a_data, entity_id=entity_a_id, context_tags=input_ctx_train0)
    entity_b = kb.add_arc_object_as_entity(obj_b_data, entity_id=entity_b_id, context_tags=input_ctx_train0)

    print(f"Entity A: {entity_a.id}, Tags: {entity_a.context_tags}, Properties: {[(p.name, p.value.value) for p in entity_a.properties]}")
    print(f"Entity B: {entity_b.id}, Tags: {entity_b.context_tags}, Properties: {[(p.name, p.value.value) for p in entity_b.properties]}")

    # Add a spatial relationship
    kb.create_relationship(entity_a.id, entity_b.id, 
                           rel_type="spatial", rel_name="is_left_of", 
                           attributes={"distance_pixels": (1, "integer")},
                           context_tags=input_ctx_train0)

    # Query with context
    left_of_rels_train0 = kb.query_relationships(rel_name="is_left_of", context_tags=input_ctx_train0)
    print(f"\n'is_left_of' relationships in context {input_ctx_train0}: {len(left_of_rels_train0)}")
    for rel in left_of_rels_train0:
        print(f"  - {rel.subject.id} {rel.name} {rel.object.id if rel.object else ''} (Tags: {rel.context_tags})")
    
    left_of_rels_task_only = kb.query_relationships(rel_name="is_left_of", context_tags=[task_id])
    print(f"\n'is_left_of' relationships in context [{task_id}] (any example): {len(left_of_rels_task_only)}")


    # Example Rule with variables
    # Rule: If object ?X has color Red (1), and ?X is_left_of ?Y,
    #       Then change color of ?X to Blue (2), and add property "processed" to ?Y.
    
    var_X = EntityPlaceholder("?X")
    var_Y = EntityPlaceholder("?Y")

    cond1 = RuleCondition(
        entity_var=var_X,
        property_pattern=SymbolicProperty(name="color", value=SymbolicValue("color", ARCPixel(1)))
    )
    # Note: SymbolicRelationship's subject/object can now be EntityPlaceholder
    cond2 = RuleCondition(
        relationship_pattern=SymbolicRelationship(
            subject=var_X, 
            name="is_left_of", 
            object=var_Y, 
            type="spatial", 
            context_tags=[] # Rule patterns usually don't have context tags themselves
        )
    )

    act1 = RuleAction(
        entity_var=var_X,
        property_assert=SymbolicProperty(name="color", value=SymbolicValue("color", ARCPixel(2)))
    )
    act2 = RuleAction(
        entity_var=var_Y,
        property_assert=SymbolicProperty(name="processed", value=SymbolicValue("boolean", True))
    )

    variable_rule = SymbolicTransformationRule(
        id="variable_color_change_and_tag",
        description="If a red object ?X is left of ?Y, ?X becomes blue and ?Y gets 'processed' tag.",
        conditions=[cond1, cond2],
        actions=[act1, act2],
        source="hypothetical_variable_rule",
        context_tags=rule_ctx_general
    )
    kb.add_rule(variable_rule)

    general_rules = kb.get_rules(context_tags=rule_ctx_general)
    print(f"\nGeneral rules: {len(general_rules)}")
    if general_rules:
        print(f"Rule '{general_rules[-1].id}': Tags: {general_rules[-1].context_tags}, "
              f"Num Conditions: {len(general_rules[-1].conditions)}, Num Actions: {len(general_rules[-1].actions)}")
        # Example of accessing a condition detail
        if general_rules[-1].conditions:
            first_cond = general_rules[-1].conditions[0]
            if first_cond.entity_var and first_cond.property_pattern:
                 print(f"  First condition: entity_var={first_cond.entity_var.variable_name}, prop_name='{first_cond.property_pattern.name}', prop_value='{first_cond.property_pattern.value.value}'")


    # Clear specific context
    kb.clear_context_by_tag(example_id_train0) # This should not clear the general_rule
    print(f"\nKB entities after clearing task context '{example_id_train0}': {len(kb.entities)}")
    # Check if general rule is still there
    general_rules_after_clear = kb.get_rules(context_tags=rule_ctx_general)
    print(f"General rules after clearing task context '{example_id_train0}': {len(general_rules_after_clear)}")
    assert len(general_rules_after_clear) > 0 
    
    print(f"\nKB entities after clearing task context '{example_id_train0}': {len(kb.entities)}")
    # Check if general rule is still there
    general_rules_after_clear = kb.get_rules(context_tags=rule_ctx_general)
    print(f"General rules after clearing task context '{example_id_train0}': {len(general_rules_after_clear)}")
    assert len(general_rules_after_clear) > 0 
    
    # Example of adding and querying solutions and patterns
    solution1 = StoredTaskSolution(
        solution_id=f"{task_id}_{example_id_train0}_sol1",
        task_id=task_id,
        example_id=example_id_train0,
        applied_rules_or_transformations=[{'type': 'rule_application', 'rule_id': variable_rule.id}],
        is_successful=True,
        context_tags=[task_id, example_id_train0]
    )
    kb.add_stored_solution(solution1)
    
    retrieved_sols = kb.query_stored_solutions(task_id=task_id, is_successful=True, context_tags=[example_id_train0])
    print(f"\nRetrieved successful solutions for {task_id}, {example_id_train0}: {len(retrieved_sols)}")
    if retrieved_sols:
        print(f"  - Solution ID: {retrieved_sols[0].solution_id}, Tags: {retrieved_sols[0].context_tags}")

    pattern1 = ARCPattern(
        pattern_id="checkerboard_2x2_red_blue",
        description="A 2x2 checkerboard pattern with red and blue.",
        pattern_type="visual_motif",
        representation={"type": "grid_patch", "patch": [[ARCPixel(1), ARCPixel(2)],[ARCPixel(2),ARCPixel(1)]]},
        context_tags=["common_visuals"]
    )
    kb.add_arc_pattern(pattern1)
    retrieved_patterns = kb.query_arc_patterns(pattern_type="visual_motif", context_tags=["common_visuals"])
    print(f"\nRetrieved visual motif patterns with tag 'common_visuals': {len(retrieved_patterns)}")
    if retrieved_patterns:
        print(f"  - Pattern ID: {retrieved_patterns[0].pattern_id}, Type: {retrieved_patterns[0].pattern_type}")


    # Clear general rule context
    kb.clear_context_by_tag("general_rule") # This should clear the variable_rule
    general_rules_after_clear_general = kb.get_rules(context_tags=rule_ctx_general)
    print(f"General rules after clearing 'general_rule' context: {len(general_rules_after_clear_general)}")
    assert len(general_rules_after_clear_general) == 0