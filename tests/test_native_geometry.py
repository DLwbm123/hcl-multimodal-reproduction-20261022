import json
import unittest
from unittest.mock import patch
from hcl.native_geometry import NativePixelModel
from hcl.models import LocalTransformersModel

class NativeGeometryContractTest(unittest.TestCase):
    def call(self, output, question):
        model=NativePixelModel(name='contract-test',path='/unused')
        with patch.object(LocalTransformersModel,'generate',return_value=output):
            return model.generate(question,state={'phase':'final_answer','question':question})
    def test_single_documented_box_can_be_wrapped_as_explicit_requested_schema(self):
        result=self.call('[{"bbox_2d":[10,20,30,40],"label":"ignored"}]','Return only one JSON object with bbox [x1,y1,x2,y2] in absolute pixels.')
        self.assertEqual(json.loads(result),{'bbox':[10,20,30,40]})
    def test_detection_array_and_invalid_boxes_are_not_repaired(self):
        raw='[{"bbox_2d":[10,20,30,40],"label":"person"}]'
        self.assertEqual(self.call(raw,'Return a JSON array with all detections.'),raw)
        for raw in ['[{"bbox_2d":[30,20,10,40]}]','[{"bbox_2d":[0,0,1,1]},{"bbox_2d":[2,2,3,3]}]','malformed']:
            self.assertEqual(self.call(raw,'Return only one JSON object with bbox in pixels.'),raw)
