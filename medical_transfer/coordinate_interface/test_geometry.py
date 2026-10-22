import dataclasses
import json
import unittest
from geometry import FrameContract

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.c = FrameContract(100, 200, 112, 224)

    def test_known_mapping_and_edges(self):
        self.assertEqual(self.c.to_original([28, 56, 84, 168]), [25, 50, 75, 150])
        self.assertEqual(self.c.to_original([0, 0, 112, 224]), [0, 0, 100, 200])

    def test_anisotropic_mapping(self):
        c = FrameContract(100, 200, 224, 224)
        self.assertEqual(c.to_original([56, 56, 168, 168]), [25, 50, 75, 150])

    def test_identity(self):
        c = FrameContract(112, 224, 112, 224)
        self.assertEqual(c.to_original([1, 2, 110, 220]), [1, 2, 110, 220])

    def test_no_clipping_or_invalid_numbers(self):
        for box in ([0,0,113,224],[-1,0,112,224],[0,0,0,3], [False,0,2,3], [0,0,float('nan'),3]):
            with self.assertRaises(ValueError): self.c.to_original(box)

    def test_unambiguous_wrappers_only(self):
        self.assertEqual(json.loads(self.c.normalize_answer('```json\n[{"bbox_2d":[28,56,84,168],"label":"polyp"}]\n```')), {'bbox':[25,50,75,150]})
        for raw in ('[{"bbox":[1,2,3,4]},{"bbox":[2,3,4,5]}]', '{"bbox":[1,2,3,4],"bbox_2d":[1,2,3,4]}', '{"bbox":[1,2,3,4],"answer":1}'):
            with self.assertRaises(ValueError): self.c.normalize_answer(raw)

    def test_immutable_contract_no_question_gate(self):
        with self.assertRaises(dataclasses.FrozenInstanceError): self.c.input_width = 1000
        self.assertEqual(set(self.c.visible_contract()), {'input_width','input_height','bbox_coordinate_system'})
        # No mutable question or answer argument is accepted by normalization.
        with self.assertRaises(TypeError): self.c.normalize_answer('{"bbox":[1,2,3,4]}', question='rewritten')

    def test_ambiguous_frame_rejected(self):
        with self.assertRaises(ValueError): FrameContract(100,200,112,224,'auto')
        with self.assertRaises(ValueError): FrameContract(100,200,100,200)

if __name__ == '__main__': unittest.main()
