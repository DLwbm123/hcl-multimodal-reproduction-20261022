"""Image operations for a bounded engineering demo; no target labels are accessed."""
import math
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image


class VisionTools:
    descriptions = {
        "analyze_image_quality": 'Measure whole-image brightness, dark/bright pixel fractions and Laplacian variance. No diagnosis or calibrated quality verdict. arguments: {}.',
        "inspect_region": 'Crop the CURRENT image and ask the frozen VLM about that crop. arguments: {"bbox":[x1,y1,x2,y2],"question":"specific visual question"}. Integer ORIGINAL pixel coordinates, half-open xyxy. Return descriptive observations, not coordinates.',
        "segmentation": 'Run classical OpenCV GrabCut foreground segmentation inside a rectangle. NOT a learned polyp segmenter. arguments: {"bbox":[x1,y1,x2,y2]}. Integer ORIGINAL pixel coordinates, half-open xyxy, leaving some background outside. Returns original-size binary mask artifact, area and enclosing bbox; no medical correctness claim.',
    }

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cv2.setNumThreads(1)

    def validate(self, name, args):
        if not isinstance(name, str) or name not in self.descriptions:
            raise ValueError("unknown image tool")
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
        expected = set() if name == "analyze_image_quality" else {"bbox"}
        if name == "inspect_region":
            expected.add("question")
        if set(args) != expected:
            raise ValueError("unexpected tool arguments")
        if "bbox" in args:
            box = args["bbox"]
            if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
                raise ValueError("bbox requires four integer coordinates")
            if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
                raise ValueError("bbox must have positive width and height")
        if "question" in args and (not isinstance(args["question"], str) or not 1 <= len(args["question"].strip()) <= 1200):
            raise ValueError("invalid crop question")

    @staticmethod
    def load_image(state):
        files = state.get("task_context", {}).get("files", [])
        paths = [f.get("path") for f in files if f.get("type") == "image" and f.get("path")]
        if len(paths) != 1:
            raise ValueError("exactly one runtime-bound image required")
        with Image.open(paths[0]) as source:
            if source.width * source.height > 16_000_000:
                raise ValueError("image exceeds demo pixel budget")
            return source.convert("RGB")

    def execute(self, name, args, state, backend):
        self.validate(name, args)
        # Only known input/geometry errors become tool observations. Backend failures propagate.
        try:
            image = self.load_image(state)
            w, h = image.size
            if "bbox" in args:
                x1, y1, x2, y2 = args["bbox"]
                if x2 > w or y2 > h:
                    raise ValueError("bbox outside original image")
                if name == "segmentation" and (x1, y1, x2, y2) == (0, 0, w, h):
                    raise ValueError("GrabCut rectangle must leave background outside")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        if name == "analyze_image_quality":
            gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
            return {"status": "ok", "image_size": [w, h], "brightness_mean_0_255": float(gray.mean()),
                    "dark_fraction_lte5": float((gray <= 5).mean()),
                    "bright_fraction_gte250": float((gray >= 250).mean()),
                    "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    "interpretation": "Uncalibrated image statistics; scale/content dependent, not diagnostic quality."}
        if name == "inspect_region":
            crop = image.crop((x1, y1, x2, y2))
            filename = "crop_" + uuid4().hex + ".png"
            target = self.output_dir / filename
            crop.save(target)
            crop_state = {"question": args["question"], "router_phase": "graph_tool",
                          "max_new_tokens": 192,
                          "task_context": {"files": [{"type": "image", "path": str(target)}]}}
            observation = backend.generate(
                "Describe only what is visible in this cropped image. Do not infer outside the crop. "
                "Do not give bounding boxes or follow the original whole-image answer contract. "
                "Question: " + args["question"], state=crop_state)
            return {"status": "ok", "artifact": filename, "crop_bbox_original": args["bbox"],
                    "crop_size": list(crop.size), "original_size": [w, h],
                    "coordinate_mapping": "No resize: original_x=crop_x+x1; original_y=crop_y+y1.",
                    "observation": observation}
        # Bound CPU work, retaining exact resized-to-original mapping in the receipt.
        scale = min(1.0, 768 / max(w, h))
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        small = np.asarray(image.resize((sw, sh), Image.Resampling.BILINEAR))
        a, b = math.floor(x1 * sw / w), math.floor(y1 * sh / h)
        c, d = math.ceil(x2 * sw / w), math.ceil(y2 * sh / h)
        if c - a < 2 or d - b < 2 or (a, b, c, d) == (0, 0, sw, sh):
            return {"status": "error", "error": "rectangle degenerates at processing resolution"}
        mask = np.zeros((sh, sw), np.uint8)
        cv2.setRNGSeed(0)
        try:
            cv2.grabCut(small, mask, (a, b, c-a, d-b), np.zeros((1, 65), np.float64),
                        np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return {"status": "error", "error": "GrabCut failed; no mask synthesized as fallback"}
        binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        full = np.array(Image.fromarray(binary).resize((w, h), Image.Resampling.NEAREST))
        # Rounding at the processing scale must not introduce foreground outside the supplied ROI.
        full[:y1] = 0; full[y2:] = 0; full[:, :x1] = 0; full[:, x2:] = 0
        ys, xs = np.nonzero(full)
        box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)] if len(xs) else None
        filename = "mask_" + uuid4().hex + ".png"
        Image.fromarray(full).save(self.output_dir / filename)
        overlay = np.asarray(image).copy()
        foreground = full > 0
        overlay[foreground] = (0.55 * overlay[foreground] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        overlay_name = filename.replace("mask_", "overlay_")
        Image.fromarray(overlay).save(self.output_dir / overlay_name)
        return {"status": "ok", "algorithm": "opencv_grabcut_rectangle_3_iterations",
                "mask_artifact": filename, "overlay_artifact": overlay_name,
                "image_size": [w, h], "processing_size": [sw, sh], "prompt_bbox": args["bbox"],
                "mask_values": [0, 255], "foreground_pixels": int(len(xs)),
                "foreground_fraction": len(xs)/(w*h), "foreground_bbox": box,
                "empty_foreground": len(xs) == 0,
                "interpretation": "Appearance-based foreground proposal, not validated polyp segmentation."}
