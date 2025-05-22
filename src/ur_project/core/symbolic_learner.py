from typing import List, Dict, Any, Optional
import uuid

from .knowledge_base import (
    ARCKnowledgeBase, SymbolicTransformationRule, RuleCondition, RuleAction,
    SymbolicProperty, SymbolicValue, EntityPlaceholder, SymbolicEntity
)
# Assuming perception.py and its types might be needed for context or full analysis
# from .perception import GridFeatures, ARCObject 
from ..data_processing.arc_types import ARCPixel # For type hints if needed

class SimpleRuleInducer:
    """
    Induces simple symbolic transformation rules from observed ARC task examples.
    """

    def _get_stable_properties_for_entity(self, kb: ARCKnowledgeBase, entity_id: str,
                                         prop_names_to_exclude: Optional[List[str]] = None,
                                         context_tags: Optional[List[str]] = None) -> List[SymbolicProperty]:
        """
        Fetches stable properties of an entity (e.g., shape, initial size if they didn't change).
        These can serve as additional conditions for rules.
        """
        stable_props = []
        entity = kb.get_entity(entity_id, context_tags=context_tags)
        if not entity:
            return []

        exclude_list = prop_names_to_exclude or []

        for prop in entity.properties:
            if prop.name not in exclude_list:
                # We might want to be more selective, e.g. only non-positional properties for some rules
                stable_props.append(SymbolicProperty(name=prop.name, value=prop.value)) # Create copies
        return stable_props

    def induce_rules_from_task_example(
        self, 
        kb: ARCKnowledgeBase, 
        task_id: str, 
        example_id: str,
        transformation_analysis_results: Dict[str, Any] # Output from perception.analyze_object_transformations
    ) -> List[SymbolicTransformationRule]:
        """
        Induces rules from a single task example.
        Operates on information previously populated in the KB (entities, relationships)
        and the direct output of object transformation analysis.

        Args:
            kb: The knowledge base containing entities and relationships for the example.
            task_id: The ID of the ARC task.
            example_id: The ID of the specific training/test example (e.g., "train_0").
            transformation_analysis_results: The dictionary output from 
                                             UNetARCFeatureExtractor.analyze_object_transformations.
        
        Returns:
            A list of induced SymbolicTransformationRule objects.
        """
        induced_rules: List[SymbolicTransformationRule] = []
        
        # Define context tags for querying and for new rules
        # Input entities are tagged with [task_id, example_id, "input"]
        # Output entities with [task_id, example_id, "output"]
        # Correspondence relationships with [task_id, example_id, "correspondence"]
        # Property changed relationships with [task_id, example_id, "transformation_observation"]
        
        example_context_tags = [task_id, example_id]
        rule_context_tags = example_context_tags + ["induced"]

        # --- 1. Induce rules from property changes on persistent objects ---
        persistent_objects_info = transformation_analysis_results.get("persistent_objects", [])
        
        for item in persistent_objects_info:
            input_obj_id = item.get("input_object_id")
            output_obj_id = item.get("output_object_id") # In KB, these are full entity IDs
            
            # The transformation_analysis_results gives internal ARCObject IDs.
            # We need to map these to the global SymbolicEntity IDs in the KB.
            # For simplicity, assume the KB has been populated such that these IDs align or
            # that transformation_analysis_results provides the global KB entity IDs.
            # Let's assume item['input_object_id'] and item['output_object_id'] are the KB entity IDs.
            # Example: "task1_train0_input_obj_1", "task1_train0_output_obj_1"

            if not input_obj_id or not output_obj_id:
                # print(f"Warning: Missing input/output ID in persistent_objects_info item: {item}")
                continue

            input_entity = kb.get_entity(input_obj_id, context_tags=example_context_tags + ["input"])
            output_entity = kb.get_entity(output_obj_id, context_tags=example_context_tags + ["output"])

            if not input_entity or not output_entity:
                # print(f"Warning: Could not find input/output entity for {input_obj_id}/{output_obj_id} in KB for rule induction.")
                continue

            # Placeholder for the subject of the transformation
            subject_var = EntityPlaceholder(f"?subject_{input_entity.source_arc_object_id or 'entity'}")


            # Check for changed properties based on the analysis results
            # This is more direct than querying "property_changed" relationships for this specific task.
            input_props = item.get('input_properties', {})
            output_props = item.get('output_properties', {})
            
            changed_prop_names = []
            if item.get('color_changed'): changed_prop_names.append('color')
            if item.get('shape_changed'): changed_prop_names.append('shape_signature') # Assuming shape_signature is the key property for shape
            if item.get('size_changed'): changed_prop_names.append('pixel_count')
            # Add other properties if tracked, e.g., position (centroid), bounding_box dimensions

            for prop_name in changed_prop_names:
                original_value = input_props.get(prop_name)
                new_value = output_props.get(prop_name)

                if original_value is None or new_value is None: continue

                # Determine value_type (this is simplified, KB should store rich SymbolicValue)
                val_type = "unknown"
                if isinstance(original_value, ARCPixel) or isinstance(new_value, ARCPixel): val_type = "color"
                elif isinstance(original_value, int) or isinstance(new_value, int): val_type = "integer"
                elif isinstance(original_value, str) or isinstance(new_value, str): val_type = "string"
                elif isinstance(original_value, bool) or isinstance(new_value, bool): val_type = "boolean"
                elif isinstance(original_value, float) or isinstance(new_value, float): val_type = "float"
                
                # Condition: Input object has the original property
                cond_prop = SymbolicProperty(name=prop_name, value=SymbolicValue(val_type, original_value))
                condition = RuleCondition(entity_var=subject_var, property_pattern=cond_prop)
                
                conditions = [condition]
                # Add other stable properties of the input_entity as further conditions
                stable_input_props = self._get_stable_properties_for_entity(
                    kb, input_entity.id, 
                    prop_names_to_exclude=[prop_name, "centroid_row", "centroid_col"], # Exclude changing prop and position
                    context_tags=example_context_tags + ["input"]
                )
                for sp in stable_input_props:
                    conditions.append(RuleCondition(entity_var=subject_var, property_pattern=sp))

                # Action: Output object (same placeholder) has the new property
                action_prop = SymbolicProperty(name=prop_name, value=SymbolicValue(val_type, new_value))
                action = RuleAction(entity_var=subject_var, property_assert=action_prop)
                
                rule_id = f"rule_{task_id}_{example_id}_propchange_{input_entity.source_arc_object_id}_{prop_name}_{uuid.uuid4().hex[:4]}"
                rule_desc = (f"If {subject_var} has {prop_name}={original_value} (and other properties), "
                             f"then its {prop_name} becomes {new_value}.")
                
                new_rule = SymbolicTransformationRule(
                    id=rule_id, description=rule_desc,
                    conditions=conditions, actions=[action],
                    context_tags=rule_context_tags + ["property_change", "induced_rule_pool"], # Added general pool tag
                    source=f"induced_from_{task_id}_{example_id}"
                )
                induced_rules.append(new_rule)
                kb.add_rule(new_rule)

        # --- 2. Induce rules for vanished objects ---
        vanished_objects_info = transformation_analysis_results.get("vanished_objects", [])
        for vanished_obj_data in vanished_objects_info:
            # vanished_obj_data contains properties of the object that vanished
            # We need its original entity ID from the KB input context.
            # This assumes transformation_analysis_results uses KB entity IDs.
            original_entity_id = vanished_obj_data.get("kb_entity_id") # Assuming this key exists or adapt
            if not original_entity_id: # Fallback if ID not directly provided, try to find by source_arc_object_id
                 source_id = vanished_obj_data.get("object_id") # This is usually ARCObject.id
                 if source_id is not None:
                    input_entities = kb.query_entities(entity_type="arc_object", context_tags=example_context_tags + ["input"])
                    found_entity = next((e for e in input_entities if e.source_arc_object_id == source_id), None)
                    if found_entity: original_entity_id = found_entity.id

            if not original_entity_id: continue
            
            input_entity = kb.get_entity(original_entity_id, context_tags=example_context_tags + ["input"])
            if not input_entity: continue

            subject_var = EntityPlaceholder(f"?vanished_obj_{input_entity.source_arc_object_id or 'entity'}")
            
            conditions = []
            # Add all known properties of the input_entity as conditions
            for prop in input_entity.properties:
                 # Exclude positional properties if desired, as they might make rules too specific
                if "centroid" not in prop.name and "bbox" not in prop.name :
                    conditions.append(RuleCondition(entity_var=subject_var, property_pattern=SymbolicProperty(name=prop.name, value=prop.value)))
            
            if not conditions: # Avoid rule with no conditions if all props were positional
                conditions.append(RuleCondition(entity_var=subject_var, property_pattern=SymbolicProperty(name="exists", value=SymbolicValue("boolean",True))))


            # Action: Mark as "deleted" or "vanished"
            action = RuleAction(entity_var=subject_var, 
                                property_assert=SymbolicProperty(name="state", value=SymbolicValue("string", "deleted")))
            
            rule_id = f"rule_{task_id}_{example_id}_vanish_{input_entity.source_arc_object_id}_{uuid.uuid4().hex[:4]}"
            rule_desc = f"If an object {subject_var} with properties {[(c.property_pattern.name, c.property_pattern.value.value) for c in conditions if c.property_pattern]} exists, it is deleted."
            
            new_rule = SymbolicTransformationRule(
                id=rule_id, description=rule_desc,
                conditions=conditions, actions=[action],
                context_tags=rule_context_tags + ["vanish_object"],
                source=f"induced_from_{task_id}_{example_id}"
            )
            induced_rules.append(new_rule)
            kb.add_rule(new_rule)

        # --- 3. Induce rules for appeared objects ---
        appeared_objects_info = transformation_analysis_results.get("appeared_objects", [])
        for appeared_obj_data in appeared_objects_info:
            # appeared_obj_data contains properties of the new object
            # We need its entity ID from the KB output context.
            output_entity_id = appeared_obj_data.get("kb_entity_id") # Assuming this key exists
            if not output_entity_id:
                source_id = appeared_obj_data.get("object_id") # ARCObject.id
                if source_id is not None:
                    output_entities = kb.query_entities(entity_type="arc_object", context_tags=example_context_tags + ["output"])
                    found_entity = next((e for e in output_entities if e.source_arc_object_id == source_id), None)
                    if found_entity: output_entity_id = found_entity.id
            
            if not output_entity_id: continue

            output_entity = kb.get_entity(output_entity_id, context_tags=example_context_tags + ["output"])
            if not output_entity: continue

            # For "appeared" objects, conditions are harder. What triggered the appearance?
            # Simplest: unconditional appearance (no specific entity conditions, maybe grid conditions).
            # Or, condition on some global property of the input grid.
            # Let's start with a rule that's conditioned on the input grid existing.
            input_grid_entity_id = f"{task_id}_{example_id}_input_grid" # Convention
            input_grid_entity = kb.get_entity(input_grid_entity_id, context_tags=example_context_tags + ["input"])
            
            conditions = []
            if input_grid_entity:
                grid_var = EntityPlaceholder("?input_grid")
                # Example condition: number of objects in input grid
                num_obj_prop = next((p for p in input_grid_entity.properties if p.name == "num_objects"), None)
                if num_obj_prop:
                     conditions.append(RuleCondition(entity_var=grid_var, property_pattern=SymbolicProperty(name="num_objects", value=num_obj_prop.value)))
            
            if not conditions: # Fallback: rule is less specific
                 pass # No specific conditions from input grid, implies a more general "create"

            # Action: "create" the new object by asserting its properties on a new placeholder
            new_object_var_name = f"?new_obj_{output_entity.source_arc_object_id or 'entity'}"
            new_object_var = EntityPlaceholder(new_object_var_name)
            
            actions = [RuleAction(entity_var=new_object_var,
                                  property_assert=SymbolicProperty(name="state", value=SymbolicValue("string", "created")))]
            for prop in output_entity.properties:
                # Exclude positional properties for the creation rule, as position might be context-dependent
                # Or, include them if the appearance is always at a fixed relative spot (more complex rule)
                if "centroid" not in prop.name and "bbox" not in prop.name : # Basic filter
                    actions.append(RuleAction(entity_var=new_object_var, property_assert=SymbolicProperty(name=prop.name, value=prop.value)))
            
            rule_id = f"rule_{task_id}_{example_id}_appear_{output_entity.source_arc_object_id}_{uuid.uuid4().hex[:4]}"
            rule_desc = f"Under certain input grid conditions, create a new object {new_object_var} with specified properties."
            if not conditions:
                rule_desc = f"Create a new object {new_object_var} with specified properties."


            new_rule = SymbolicTransformationRule(
                id=rule_id, description=rule_desc,
                conditions=conditions, actions=actions,
                    context_tags=rule_context_tags + ["vanish_object", "induced_rule_pool"], # Added general pool tag
                source=f"induced_from_{task_id}_{example_id}"
            )
            induced_rules.append(new_rule)
            kb.add_rule(new_rule)

        return induced_rules

# Example Usage (conceptual, would be part of a larger workflow)
# if __name__ == '__main__':
#     kb = ARCKnowledgeBase()
#     # Populate KB with entities from input and output grids of an example
#     # (e.g., using UNetARCFeatureExtractor and ARCKnowledgeBase.add_arc_object_as_entity)
#     # ...
    
#     # Assume perception_analyzer is an instance of UNetARCFeatureExtractor
#     # input_grid_features = perception_analyzer.extract_features(input_grid_data)
#     # output_grid_features = perception_analyzer.extract_features(output_grid_data)
#     # transformation_results = perception_analyzer.analyze_object_transformations(input_grid_features, output_grid_features)
#     mock_transformation_results = {
#         "persistent_objects": [
#             {
#                 "input_object_id": "task1_train0_input_obj_1", # KB Global ID
#                 "output_object_id": "task1_train0_output_obj_1", # KB Global ID
#                 "input_properties": {"color": ARCPixel(1), "pixel_count": 5, "shape_signature": "rect_2x3"},
#                 "output_properties": {"color": ARCPixel(2), "pixel_count": 5, "shape_signature": "rect_2x3"},
#                 "color_changed": True, "shape_changed": False, "size_changed": False,
#                 "delta_centroid": (0,0)
#             }
#         ],
#         "vanished_objects": [
#             # {"kb_entity_id": "task1_train0_input_obj_2", "object_id": 2, ... other props ...}
#         ],
#         "appeared_objects": [
#             # {"kb_entity_id": "task1_train0_output_obj_3", "object_id": 3, ... other props ...}
#         ]
#     }
#     # Need to ensure KB is populated with these entities for _get_stable_properties_for_entity to work
#     # For example:
#     # obj_a_data_in = ARCObject(id=1, color=ARCPixel(1), pixels={(0,0)}, bounding_box=(0,0,0,0), pixel_count=5, centroid=(0,0.5), metadata={'shape_signature': "rect_2x3"})
#     # kb.add_arc_object_as_entity(obj_a_data_in, entity_id="task1_train0_input_obj_1", context_tags=["task1", "train0", "input"])
#     # obj_a_data_out = ARCObject(id=1, color=ARCPixel(2), pixels={(0,0)}, bounding_box=(0,0,0,0), pixel_count=5, centroid=(0,0.5), metadata={'shape_signature': "rect_2x3"})
#     # kb.add_arc_object_as_entity(obj_a_data_out, entity_id="task1_train0_output_obj_1", context_tags=["task1", "train0", "output"])


#     inducer = SimpleRuleInducer()
#     task_id = "task1"
#     example_id = "train0"
#     newly_induced_rules = inducer.induce_rules_from_task_example(kb, task_id, example_id, mock_transformation_results)
    
#     print(f"\nInduced {len(newly_induced_rules)} rules from example {task_id}_{example_id}:")
#     for rule in newly_induced_rules:
#         print(f"  Rule ID: {rule.id}, Description: {rule.description}")
#         print(f"    Conditions: {len(rule.conditions)}")
#         for cond in rule.conditions:
#             if cond.property_pattern:
#                 print(f"      - Entity: {cond.entity_var}, Property: {cond.property_pattern.name}={cond.property_pattern.value.value}")
#         print(f"    Actions: {len(rule.actions)}")
#         for act in rule.actions:
#             if act.property_assert:
#                 print(f"      - Entity: {act.entity_var}, Assert: {act.property_assert.name}={act.property_assert.value.value}")

print("SimpleRuleInducer structure defined in symbolic_learner.py")
