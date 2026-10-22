import tempfile
import unittest
from pathlib import Path
from hcl.memory.anchor import AnchorMemory

class MultimodalAnchorIdentityTest(unittest.TestCase):
    def test_distinct_images_survive_identical_caption_instruction(self):
        records=[{'task_id':str(i),'task_name':'caption','question':'Describe the image.',
                  'metadata':{'image_id':i}} for i in range(20)]
        with tempfile.TemporaryDirectory() as d:
            anchors=AnchorMemory(Path(d)/'anchors.jsonl',capacity_per_task=25,identity_mode='visible_input')
            anchors.add_batch('caption',records[:10])
            anchors.add_batch('caption',records[10:])
            self.assertEqual(len(anchors.records),20)
            anchors.add_batch('caption',[dict(records[0],task_id='duplicate-alias')])
            self.assertEqual(len(anchors.records),20)
    def test_default_question_identity_remains_unchanged(self):
        records=[{'task_id':str(i),'task_name':'caption','question':'Describe the image.',
                  'metadata':{'image_id':i}} for i in range(2)]
        with tempfile.TemporaryDirectory() as d:
            anchors=AnchorMemory(Path(d)/'anchors.jsonl')
            anchors.add_batch('caption',records)
            self.assertEqual(len(anchors.records),1)
