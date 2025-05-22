import unittest

from src.ur_project.core.knowledge_base import (
    ARCKnowledgeBase,
    SymbolicEntity,
    SymbolicRelationship,
    SymbolicValue
)
from src.ur_project.core.perception import ARCObject # For dummy object creation
from src.ur_project.data_processing.arc_types import ARCPixel # For dummy object creation

# Helper function to create a dummy ARCObject
def create_dummy_arc_object(obj_id: int, color_val: int = 0) -> ARCObject:
    """Creates a simple ARCObject for testing."""
    return ARCObject(
        id=obj_id,
        color=ARCPixel(color_val),
        pixels={(obj_id, 0)}, # Dummy pixel
        bounding_box=(obj_id, 0, obj_id, 0), # Dummy bbox
        pixel_count=1,
        centroid=(float(obj_id), 0.0)
    )

class TestARCKnowledgeBaseInference(unittest.TestCase):

    def setUp(self):
        self.kb = ARCKnowledgeBase()
        # Common entities for many tests
        self.obj_a_data = create_dummy_arc_object(1, 1)
        self.obj_b_data = create_dummy_arc_object(2, 2)
        self.obj_c_data = create_dummy_arc_object(3, 3)
        self.obj_d_data = create_dummy_arc_object(4, 4)
        self.obj_e_data = create_dummy_arc_object(5, 5)
        self.obj_f_data = create_dummy_arc_object(6, 6)

        self.entity_a = self.kb.add_arc_object_as_entity(self.obj_a_data, "input")
        self.entity_b = self.kb.add_arc_object_as_entity(self.obj_b_data, "input")
        self.entity_c = self.kb.add_arc_object_as_entity(self.obj_c_data, "input")
        self.entity_d = self.kb.add_arc_object_as_entity(self.obj_d_data, "input")
        self.entity_e = self.kb.add_arc_object_as_entity(self.obj_e_data, "input")
        self.entity_f = self.kb.add_arc_object_as_entity(self.obj_f_data, "input")
        
        # Clear relationships and entities for each test for better isolation
        self.kb.relationships.clear()
        self.kb.entities.clear()
        self.kb._temp_object_map.clear() # also clear the internal mapping

        # Re-add common entities after clearing
        self.entity_a = self.kb.add_arc_object_as_entity(self.obj_a_data, "input_obj_A")
        self.entity_b = self.kb.add_arc_object_as_entity(self.obj_b_data, "input_obj_B")
        self.entity_c = self.kb.add_arc_object_as_entity(self.obj_c_data, "input_obj_C")
        self.entity_d = self.kb.add_arc_object_as_entity(self.obj_d_data, "input_obj_D")
        self.entity_e = self.kb.add_arc_object_as_entity(self.obj_e_data, "input_obj_E")
        self.entity_f = self.kb.add_arc_object_as_entity(self.obj_f_data, "input_obj_F")


    def _add_rel(self, subj_entity: SymbolicEntity, obj_entity: SymbolicEntity, rel_name: str, rel_type: str = "spatial", attributes: dict = None):
        # Use the KB's own method for consistency, it handles attribute conversion
        # Ensure subj_entity and obj_entity are in self.kb.entities if not added via kb.add_arc_object_as_entity
        if subj_entity.id not in self.kb.entities: self.kb.entities[subj_entity.id] = subj_entity
        if obj_entity.id not in self.kb.entities: self.kb.entities[obj_entity.id] = obj_entity
        
        # Convert attributes to the expected tuple format (value, type_str)
        processed_attrs = {}
        if attributes:
            for k, v_tuple_or_val in attributes.items():
                if isinstance(v_tuple_or_val, tuple) and len(v_tuple_or_val) == 2:
                    processed_attrs[k] = v_tuple_or_val
                else: # Assume it's just the value, infer type (basic inference)
                    val_type = "string"
                    if isinstance(v_tuple_or_val, int): val_type = "integer"
                    elif isinstance(v_tuple_or_val, float): val_type = "float"
                    elif isinstance(v_tuple_or_val, bool): val_type = "boolean"
                    processed_attrs[k] = (v_tuple_or_val, val_type)

        return self.kb.add_relationship(subj_entity.id, obj_entity.id, rel_type, rel_name, processed_attrs)


    def _get_relationship(self, subj_id: str, obj_id: str, rel_name: str) -> SymbolicRelationship | None:
        """Gets a specific relationship from the KB if it exists."""
        for rel in self.kb.relationships:
            if rel.subject.id == subj_id and \
               rel.object and rel.object.id == obj_id and \
               rel.name == rel_name:
                return rel
        return None

    def _find_relationship(self, subj_id: str, obj_id: str, rel_name: str) -> bool:
        """Checks if a specific relationship exists in the KB."""
        return self._get_relationship(subj_id, obj_id, rel_name) is not None

    def _get_entity_property_value(self, entity_id: str, prop_name: str):
        """Helper to get a property value from an entity."""
        entity = self.kb.get_entity(entity_id)
        if entity:
            for prop in entity.properties:
                if prop.name == prop_name:
                    return prop.value.value
        return None
    
    def _entity_has_property(self, entity_id: str, prop_name: str) -> bool:
        """Checks if an entity has a specific property."""
        entity = self.kb.get_entity(entity_id)
        if entity:
            return any(prop.name == prop_name for prop in entity.properties)
        return False

    def _count_relationships(self, subj_id: str, obj_id: str, rel_name: str) -> int:
        """Counts occurrences of a specific relationship."""
        count = 0
        for rel in self.kb.relationships:
            if rel.subject.id == subj_id and \
               rel.object and rel.object.id == obj_id and \
               rel.name == rel_name:
                count += 1
        return count

    def test_basic_transitivity(self):
        """A is_left_of B, B is_left_of C => A is_left_of C."""
        self._add_rel(self.entity_a, self.entity_b, "is_left_of")
        self._add_rel(self.entity_b, self.entity_c, "is_left_of")
        self.assertEqual(len(self.kb.relationships), 2)

        self.kb.infer_spatial_relationships(relation_name="is_left_of")
        
        self.assertEqual(len(self.kb.relationships), 3)
        self.assertTrue(self._find_relationship(self.entity_a.id, self.entity_c.id, "is_left_of"))
        # Ensure original relationships are still there
        self.assertTrue(self._find_relationship(self.entity_a.id, self.entity_b.id, "is_left_of"))
        self.assertTrue(self._find_relationship(self.entity_b.id, self.entity_c.id, "is_left_of"))

    def test_no_transitivity(self):
        """A is_left_of B, C is_left_of B => No inference between A and C."""
        self._add_rel(self.entity_a, self.entity_b, "is_left_of")
        self._add_rel(self.entity_c, self.entity_b, "is_left_of")
        self.assertEqual(len(self.kb.relationships), 2)

        self.kb.infer_spatial_relationships(relation_name="is_left_of")
        
        self.assertEqual(len(self.kb.relationships), 2) # No new relationships
        self.assertFalse(self._find_relationship(self.entity_a.id, self.entity_c.id, "is_left_of"))
        self.assertFalse(self._find_relationship(self.entity_c.id, self.entity_a.id, "is_left_of"))

    def test_separate_relationship_types(self):
        """Inference for 'is_left_of' should not affect 'is_above' and vice-versa."""
        # Setup 'is_left_of' chain
        self._add_rel(self.entity_a, self.entity_b, "is_left_of")
        self._add_rel(self.entity_b, self.entity_c, "is_left_of")
        # Setup 'is_above' chain
        self._add_rel(self.entity_d, self.entity_e, "is_above")
        self._add_rel(self.entity_e, self.entity_f, "is_above")
        self.assertEqual(len(self.kb.relationships), 4)

        # Infer 'is_left_of'
        self.kb.infer_spatial_relationships(relation_name="is_left_of")
        self.assertEqual(len(self.kb.relationships), 5) # One new 'is_left_of'
        self.assertTrue(self._find_relationship(self.entity_a.id, self.entity_c.id, "is_left_of"))
        self.assertFalse(self._find_relationship(self.entity_d.id, self.entity_f.id, "is_above")) # 'is_above' not inferred yet

        # Infer 'is_above'
        self.kb.infer_spatial_relationships(relation_name="is_above")
        self.assertEqual(len(self.kb.relationships), 6) # One new 'is_above'
        self.assertTrue(self._find_relationship(self.entity_d.id, self.entity_f.id, "is_above"))
        # Ensure 'is_left_of' A-C is still there
        self.assertTrue(self._find_relationship(self.entity_a.id, self.entity_c.id, "is_left_of"))

    def test_duplicate_prevention(self):
        """A is_left_of B, B is_left_of C. A is_left_of C (pre-existing). Should not add duplicates."""
        self._add_rel(self.entity_a, self.entity_b, "is_left_of")
        self._add_rel(self.entity_b, self.entity_c, "is_left_of")
        self._add_rel(self.entity_a, self.entity_c, "is_left_of") # Pre-existing direct link
        initial_rels_count = len(self.kb.relationships)
        self.assertEqual(initial_rels_count, 3)
        self.assertEqual(self._count_relationships(self.entity_a.id, self.entity_c.id, "is_left_of"), 1)


        self.kb.infer_spatial_relationships(relation_name="is_left_of")
        
        # The number of relationships should remain the same as the inferred one was already there
        self.assertEqual(len(self.kb.relationships), initial_rels_count)
        # Crucially, ensure the A -> C link is still singular
        self.assertEqual(self._count_relationships(self.entity_a.id, self.entity_c.id, "is_left_of"), 1)

    def test_chain_with_preexisting_direct_link(self):
        """Same as duplicate_prevention, more explicit name for clarity."""
        self._add_rel(self.entity_a, self.entity_b, "is_left_of")
        self._add_rel(self.entity_b, self.entity_c, "is_left_of")
        # Explicitly add the direct link that would be inferred
        self._add_rel(self.entity_a, self.entity_c, "is_left_of")
        self.assertEqual(len(self.kb.relationships), 3)

        self.kb.infer_spatial_relationships(relation_name="is_left_of")
        
        self.assertEqual(len(self.kb.relationships), 3) # No new relationships should be added
        self.assertTrue(self._find_relationship(self.entity_a.id, self.entity_c.id, "is_left_of"))
        # Ensure it's only present once (assuming SymbolicRelationship implements __eq__ correctly)
        self.assertEqual(self._count_relationships(self.entity_a.id, self.entity_c.id, "is_left_of"), 1)

    def test_self_loop_prevention_direct(self):
        """If A is_left_of B and B is_left_of A, A is_left_of A should NOT be inferred."""
        # Create a cycle: A <-> B
        self._add_rel(self.entity_a, self.entity_b, "is_left_of")
        self._add_rel(self.entity_b, self.entity_a, "is_left_of") 
        self.assertEqual(len(self.kb.relationships), 2)

        self.kb.infer_spatial_relationships(relation_name="is_left_of")
        
        # The inference A -> B and B -> A might infer A -> A.
        # The current code has `A.id != C.id` which prevents this.
        self.assertEqual(len(self.kb.relationships), 2) # No new relationships should be added
        self.assertFalse(self._find_relationship(self.entity_a.id, self.entity_a.id, "is_left_of"))
        self.assertFalse(self._find_relationship(self.entity_b.id, self.entity_b.id, "is_left_of"))

    def test_transitivity_with_different_attributes_still_infers_base_relation(self):
        """
        A is_left_of B (dist 1), B is_left_of C (dist 1).
        Infer A is_left_of C. Attributes are currently not combined by infer_spatial_relationships.
        The inferred relationship will have empty attributes.
        """
        rel_ab = self._add_rel(self.entity_a, self.entity_b, "is_left_of")
        rel_ab.attributes = {"distance": SymbolicValue("integer", 1)}
        
        rel_bc = self._add_rel(self.entity_b, self.entity_c, "is_left_of")
        rel_bc.attributes = {"distance": SymbolicValue("integer", 1)}

        self.assertEqual(len(self.kb.relationships), 2)
        self.kb.infer_spatial_relationships(relation_name="is_left_of")
        
        self.assertEqual(len(self.kb.relationships), 3)
        
        inferred_rel_found = False
        for rel in self.kb.relationships:
            if rel.subject.id == self.entity_a.id and \
               rel.object and rel.object.id == self.entity_c.id and \
               rel.name == "is_left_of":
                inferred_rel_found = True
                # Current implementation adds inferred relationships with empty attributes
                self.assertEqual(rel.attributes, {}) 
                break
        self.assertTrue(inferred_rel_found, "Inferred A is_left_of C not found")

    # --- Tests for infer_object_correspondence_from_training_pairs ---

    def test_correspondence_basic(self):
        # Setup: train_0_input_obj_1 (color 5, px 10), train_0_output_obj_A (color 5, px 10)
        in_obj_data = create_dummy_arc_object(obj_id=101, color_val=5)
        in_obj_data.pixel_count = 10
        in_entity = self.kb.add_arc_object_as_entity(in_obj_data, grid_context="train_0_input")
        
        out_obj_data = create_dummy_arc_object(obj_id=201, color_val=5)
        out_obj_data.pixel_count = 10
        out_entity = self.kb.add_arc_object_as_entity(out_obj_data, grid_context="train_0_output")

        self.kb.infer_object_correspondence_from_training_pairs()

        corr_rel = self._get_relationship(in_entity.id, out_entity.id, "has_training_pair_correspondence")
        self.assertIsNotNone(corr_rel)
        self.assertEqual(corr_rel.attributes["pair_id"].value, "train_0")
        self.assertEqual(corr_rel.attributes["confidence"].value, 1.0)

    def test_correspondence_no_match_property_mismatch(self):
        in_obj_data = create_dummy_arc_object(obj_id=102, color_val=5)
        in_obj_data.pixel_count = 10
        in_entity = self.kb.add_arc_object_as_entity(in_obj_data, grid_context="train_0_input")
        
        out_obj_data = create_dummy_arc_object(obj_id=202, color_val=6) # Different color
        out_obj_data.pixel_count = 10
        out_entity = self.kb.add_arc_object_as_entity(out_obj_data, grid_context="train_0_output")

        self.kb.infer_object_correspondence_from_training_pairs()
        self.assertFalse(self._find_relationship(in_entity.id, out_entity.id, "has_training_pair_correspondence"))

    def test_correspondence_multiple_candidates_no_unique_match(self):
        in_obj_data = create_dummy_arc_object(obj_id=103, color_val=7)
        in_obj_data.pixel_count = 15
        in_entity = self.kb.add_arc_object_as_entity(in_obj_data, grid_context="train_0_input")
        
        out1_obj_data = create_dummy_arc_object(obj_id=203, color_val=7)
        out1_obj_data.pixel_count = 15
        out1_entity = self.kb.add_arc_object_as_entity(out1_obj_data, grid_context="train_0_output")

        out2_obj_data = create_dummy_arc_object(obj_id=204, color_val=7)
        out2_obj_data.pixel_count = 15
        out2_entity = self.kb.add_arc_object_as_entity(out2_obj_data, grid_context="train_0_output")

        self.kb.infer_object_correspondence_from_training_pairs()
        # Expect no relationship because the match is not unique
        self.assertFalse(self._find_relationship(in_entity.id, out1_entity.id, "has_training_pair_correspondence"))
        self.assertFalse(self._find_relationship(in_entity.id, out2_entity.id, "has_training_pair_correspondence"))

    def test_correspondence_different_training_pairs(self):
        # Pair 0
        in0_data = create_dummy_arc_object(10, 1); in0_data.pixel_count = 1
        in0 = self.kb.add_arc_object_as_entity(in0_data, "train_0_input")
        out0_data = create_dummy_arc_object(11, 1); out0_data.pixel_count = 1
        out0 = self.kb.add_arc_object_as_entity(out0_data, "train_0_output")
        # Pair 1
        in1_data = create_dummy_arc_object(20, 2); in1_data.pixel_count = 2
        in1 = self.kb.add_arc_object_as_entity(in1_data, "train_1_input")
        out1_data = create_dummy_arc_object(21, 2); out1_data.pixel_count = 2
        out1 = self.kb.add_arc_object_as_entity(out1_data, "train_1_output")

        self.kb.infer_object_correspondence_from_training_pairs()

        corr0 = self._get_relationship(in0.id, out0.id, "has_training_pair_correspondence")
        self.assertIsNotNone(corr0)
        self.assertEqual(corr0.attributes["pair_id"].value, "train_0")
        
        corr1 = self._get_relationship(in1.id, out1.id, "has_training_pair_correspondence")
        self.assertIsNotNone(corr1)
        self.assertEqual(corr1.attributes["pair_id"].value, "train_1")
        
        self.assertFalse(self._find_relationship(in0.id, out1.id, "has_training_pair_correspondence"))

    def test_correspondence_no_output_objects(self):
        in_obj_data = create_dummy_arc_object(obj_id=105, color_val=5)
        in_obj_data.pixel_count = 10
        self.kb.add_arc_object_as_entity(in_obj_data, grid_context="train_0_input")
        
        self.kb.infer_object_correspondence_from_training_pairs()
        self.assertEqual(len(self.kb.query_relationships(rel_name="has_training_pair_correspondence")), 0)

    # --- Tests for infer_property_transfer_via_correspondence ---

    def test_property_transfer_changed(self):
        in_obj = self.kb.add_arc_object_as_entity(create_dummy_arc_object(1,1), "pair1_input_obj_1")
        out_obj = self.kb.add_arc_object_as_entity(create_dummy_arc_object(2,2), "pair1_output_obj_1") # Different color
        
        # Manually add correspondence
        self._add_rel(in_obj, out_obj, "has_training_pair_correspondence", "correspondence_logic", 
                      attributes={"pair_id": ("pair1", "string"), "confidence": (1.0, "float")})
        
        # Add different height
        in_obj.add_property("height", 10, "integer")
        out_obj.add_property("height", 20, "integer")
        # Add same pixel_count
        in_obj.add_property("pixel_count", 5, "integer") # ARCObject default is 1, so this is an update
        out_obj.add_property("pixel_count", 5, "integer")


        self.kb.infer_property_transfer_via_correspondence()

        height_change_rel = self._get_relationship(in_obj.id, out_obj.id, "property_changed")
        found_height_change = False
        for rel in self.kb.query_relationships(subject_id=in_obj.id, object_id=out_obj.id, rel_name="property_changed"):
            if rel.attributes["property_name"].value == "height":
                self.assertEqual(rel.attributes["input_value"].value, 10)
                self.assertEqual(rel.attributes["output_value"].value, 20)
                self.assertEqual(rel.attributes["pair_id"].value, "pair1")
                found_height_change = True
            if rel.attributes["property_name"].value == "color": # color changed from 1 to 2
                 self.assertEqual(rel.attributes["input_value"].value.value, 1) 
                 self.assertEqual(rel.attributes["output_value"].value.value, 2)


        self.assertTrue(found_height_change, "Height property change not found")
        
        # Check that no change for pixel_count
        found_px_count_change = any(
            rel.attributes["property_name"].value == "pixel_count" 
            for rel in self.kb.query_relationships(subject_id=in_obj.id, object_id=out_obj.id, rel_name="property_changed")
        )
        self.assertFalse(found_px_count_change, "Pixel count property change should not have been inferred.")


    def test_property_transfer_same_value(self):
        in_obj = self.kb.add_arc_object_as_entity(create_dummy_arc_object(3,1), "pair2_input_obj_3")
        out_obj = self.kb.add_arc_object_as_entity(create_dummy_arc_object(4,1), "pair2_output_obj_4")
        self._add_rel(in_obj, out_obj, "has_training_pair_correspondence", "correspondence_logic", {"pair_id": "pair2"})
        in_obj.add_property("height", 10, "integer")
        out_obj.add_property("height", 10, "integer")

        self.kb.infer_property_transfer_via_correspondence()
        self.assertFalse(self._find_relationship(in_obj.id, out_obj.id, "property_changed"))

    def test_property_transfer_property_lost_gained(self):
        in_obj = self.kb.add_arc_object_as_entity(create_dummy_arc_object(5,1), "pair3_input_obj_5")
        out_obj = self.kb.add_arc_object_as_entity(create_dummy_arc_object(6,1), "pair3_output_obj_6")
        self._add_rel(in_obj, out_obj, "has_training_pair_correspondence", "correspondence_logic", {"pair_id": "pair3"})
        in_obj.add_property("texture", "rough", "string")
        out_obj.add_property("shape", "square", "string")

        self.kb.infer_property_transfer_via_correspondence()
        # No "property_changed" for texture (lost) or shape (gained) with current logic
        self.assertFalse(self._find_relationship(in_obj.id, out_obj.id, "property_changed"))

    def test_property_transfer_no_prior_correspondence(self):
        in_obj = self.kb.add_arc_object_as_entity(create_dummy_arc_object(7,1), "pair4_input_obj_7")
        out_obj = self.kb.add_arc_object_as_entity(create_dummy_arc_object(8,1), "pair4_output_obj_8")
        in_obj.add_property("height", 10, "integer")
        out_obj.add_property("height", 20, "integer")

        self.kb.infer_property_transfer_via_correspondence()
        self.assertFalse(self._find_relationship(in_obj.id, out_obj.id, "property_changed"))

    # --- Tests for propagate_properties_via_symmetry ---

    def test_symmetry_propagate_basic(self):
        # entity_a (input_obj_A) is defined in setUp, color 1
        self.entity_a.add_property("color", ARCPixel(1), "color") # Ensure it's there
        # entity_b (input_obj_B) has no color initially or default color 2
        # Remove color from B if it exists from setup
        self.entity_b.properties = [p for p in self.entity_b.properties if p.name != "color"]

        self._add_rel(self.entity_a, self.entity_b, "is_symmetric_to", "spatial")
        self.kb.propagate_properties_via_symmetry()
        
        self.assertEqual(self._get_entity_property_value(self.entity_b.id, "color").value, 1)

    def test_symmetry_propagate_both_ways(self):
        # entity_a (color 1), entity_b (pixel_count 10 from dummy obj, color 2)
        self.entity_a.properties = [p for p in self.entity_a.properties if p.name != "pixel_count"] # Remove existing pixel_count from A
        self.entity_b.add_property("pixel_count", 10, "integer") # ensure B has it

        self._add_rel(self.entity_a, self.entity_b, "is_symmetric_to", "spatial")
        self.kb.propagate_properties_via_symmetry()

        self.assertEqual(self._get_entity_property_value(self.entity_b.id, "color").value, 1) # A's color to B
        self.assertEqual(self._get_entity_property_value(self.entity_a.id, "pixel_count"), 10) # B's count to A

    def test_symmetry_propagate_already_exists_no_change(self):
        self.entity_a.add_property("color", ARCPixel(1), "color")
        self.entity_b.add_property("color", ARCPixel(1), "color")
        self._add_rel(self.entity_a, self.entity_b, "is_symmetric_to", "spatial")
        
        self.kb.propagate_properties_via_symmetry()
        self.assertEqual(self._get_entity_property_value(self.entity_b.id, "color").value, 1)

    def test_symmetry_propagate_conflict_no_overwrite(self):
        self.entity_a.add_property("color", ARCPixel(1), "color") # red
        self.entity_b.add_property("color", ARCPixel(2), "color") # blue
        self._add_rel(self.entity_a, self.entity_b, "is_symmetric_to", "spatial")

        self.kb.propagate_properties_via_symmetry()
        self.assertEqual(self._get_entity_property_value(self.entity_a.id, "color").value, 1) # Still red
        self.assertEqual(self._get_entity_property_value(self.entity_b.id, "color").value, 2) # Still blue

    def test_symmetry_propagate_no_symmetry_rel(self):
        self.entity_a.add_property("color", ARCPixel(1), "color")
        self.entity_b.properties = [p for p in self.entity_b.properties if p.name != "color"] # B has no color

        self.kb.propagate_properties_via_symmetry()
        self.assertFalse(self._entity_has_property(self.entity_b.id, "color"))

    def test_symmetry_propagate_non_propagatable_property(self):
        self.entity_a.add_property("unique_id", "special_A", "string")
        self._add_rel(self.entity_a, self.entity_b, "is_symmetric_to", "spatial")

        self.kb.propagate_properties_via_symmetry()
        self.assertFalse(self._entity_has_property(self.entity_b.id, "unique_id"))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
