import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from ovlm._internal.output import serialize_result
from ovlm._types.common import ImageInferenceResponse


class VisualizationTests(unittest.TestCase):
    def test_bfloat16_tensor_results_are_json_serializable(self):
        import torch

        payload = json.loads(serialize_result("image.png", ImageInferenceResponse(
            masks=torch.ones((1, 1, 2, 2), dtype=torch.bool),
            boxes=torch.tensor([[0, 0, 2, 2]], dtype=torch.bfloat16),
            scores=torch.tensor([0.5], dtype=torch.bfloat16),
        )))
        self.assertEqual(payload["scores"], [0.5])
        self.assertEqual(payload["boxes"], [[0, 0, 2, 2]])

    def test_pipe_roundtrip_preserves_mask_and_emits_png(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "source.png"
            Image.new("RGB", (40, 40), "white").save(image)
            masks = np.zeros((1, 1, 40, 40), dtype=bool)
            masks[0, 0, 10:30, 10:30] = True
            payload = serialize_result(str(image), ImageInferenceResponse(
                masks=masks, boxes=np.array([[10, 10, 30, 30]]), scores=np.array([0.9]),
            ))
            rle = json.loads(payload)["masks"][0]
            decoded = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")})
            np.testing.assert_array_equal(decoded, masks[0, 0])
            result = subprocess.run(
                [sys.executable, "-m", "ovlm.tools.visualize"],
                input=payload.encode(), capture_output=True, check=True,
            )
            self.assertTrue(result.stdout.startswith(b"\x89PNG\r\n\x1a\n"))
            with Image.open(io.BytesIO(result.stdout)) as rendered:
                self.assertEqual(rendered.size, (40, 40))
                self.assertEqual(rendered.getpixel((0, 0)), (255, 255, 255))
                self.assertNotEqual(rendered.getpixel((20, 25)), (255, 255, 255))

    def test_empty_detections_and_direct_output(self):
        with tempfile.TemporaryDirectory() as directory:
            image, output = Path(directory) / "source.png", Path(directory) / "output.png"
            Image.new("RGB", (10, 10), "white").save(image)
            payload = serialize_result(str(image), ImageInferenceResponse(
                masks=np.zeros((0, 1, 10, 10)), boxes=np.zeros((0, 4)), scores=np.zeros(0),
            ))
            result = subprocess.run(
                [sys.executable, "-m", "ovlm.tools.visualize", "--output", str(output)],
                input=payload.encode(), capture_output=True, check=True,
            )
            self.assertEqual(result.stdout, b"")
            with Image.open(output) as rendered:
                self.assertEqual(rendered.getpixel((5, 5)), (255, 255, 255))

    def test_invalid_input_keeps_stdout_empty(self):
        result = subprocess.run(
            [sys.executable, "-m", "ovlm.tools.visualize"],
            input=b"not json", capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"Cannot visualize", result.stderr)


if __name__ == "__main__":
    unittest.main()
