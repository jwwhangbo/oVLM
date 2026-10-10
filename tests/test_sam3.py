import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image
from typer.testing import CliRunner

from ovlm._internal.registry import get_model
from ovlm._types.image import ImageInput, Prompt
from ovlm.main import app
from ovlm.vlms.sam3 import SAM3


class SAM3PullTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name)
        cache_patch = patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache))
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        self.checkpoint = self.cache / "models--facebook--sam3" / "snapshots" / "test-revision" / "sam3.pt"
        self.request = ImageInput(
            image=Image.new("RGB", (2, 2)), prompt=Prompt(texts=["bird"]),
        )
        builder_patch = patch(
            "ovlm.vlms.sam3.build_sam3_image_model",
            return_value=torch.nn.Linear(1, 1),
        )
        self.builder = builder_patch.start()
        self.addCleanup(builder_patch.stop)
        processor_patch = patch("ovlm.vlms.sam3.Sam3Processor")
        processor = processor_patch.start().return_value
        self.addCleanup(processor_patch.stop)
        processor.set_text_prompt.return_value = {
            "masks": torch.ones((1, 1, 2, 2), dtype=torch.bool),
            "boxes": torch.tensor([[0, 0, 2, 2]]),
            "scores": torch.tensor([0.9]),
        }

    def cache_checkpoint(self, **kwargs):
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint.write_bytes(b"test checkpoint")
        refs = self.cache / "models--facebook--sam3" / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text("test-revision")
        return str(self.checkpoint)

    def test_pull_records_weights_without_building_model(self):
        model = SAM3()
        with patch("ovlm.vlms.sam3.download_ckpt_from_hf", side_effect=self.cache_checkpoint):
            path = model.pull()
        self.assertEqual(model.weights_path, path)
        self.assertTrue(Path(path).is_file())
        self.builder.assert_not_called()

    def test_fresh_instance_uses_previously_pulled_weights_offline(self):
        with patch("ovlm.vlms.sam3.download_ckpt_from_hf", side_effect=self.cache_checkpoint):
            path = SAM3().pull()
        stderr = io.StringIO()
        with patch("ovlm.vlms.sam3.download_ckpt_from_hf") as download, redirect_stderr(stderr):
            model = get_model("sam3")
            result = model.infer_image(self.request)
        self.assertEqual(model.weights_path, path)
        self.assertEqual(result.masks.shape, (1, 1, 2, 2))
        self.assertEqual(stderr.getvalue(), "")
        download.assert_not_called()
        self.builder.assert_called_once_with(checkpoint_path=path, load_from_HF=False)

    def test_missing_weights_download_then_continue_with_notice_on_stderr(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        model = SAM3()
        with patch("ovlm.vlms.sam3.download_ckpt_from_hf", side_effect=self.cache_checkpoint) as download:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = model.infer_image(self.request)
                model.infer_image(self.request)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue().count("No local weights"), 1)
        self.assertEqual(result.scores.shape, (1,))
        download.assert_called_once()
        self.builder.assert_called_with(checkpoint_path=str(self.checkpoint), load_from_HF=False)

    def test_explicit_local_weights_skip_cache_and_download(self):
        path = self.cache / "custom.pt"
        path.write_bytes(b"test checkpoint")
        with patch("ovlm.vlms.sam3.hf_hub_download") as cache_lookup:
            with patch("ovlm.vlms.sam3.download_ckpt_from_hf") as download:
                SAM3(weights_path=str(path)).infer_image(self.request)
        cache_lookup.assert_not_called()
        download.assert_not_called()
        self.builder.assert_called_once_with(checkpoint_path=str(path), load_from_HF=False)

    def test_deleted_weights_are_downloaded_again(self):
        model = SAM3(weights_path=self.cache_checkpoint())
        self.checkpoint.unlink()
        with patch("ovlm.vlms.sam3.download_ckpt_from_hf", side_effect=self.cache_checkpoint) as download:
            with redirect_stderr(io.StringIO()):
                model.infer_image(self.request)
        download.assert_called_once()
        self.assertTrue(self.checkpoint.is_file())

    def test_download_failure_does_not_start_inference(self):
        model = SAM3()
        with patch("ovlm.vlms.sam3.download_ckpt_from_hf", side_effect=OSError("Access denied")):
            with redirect_stderr(io.StringIO()), self.assertRaisesRegex(OSError, "Access denied"):
                model.infer_image(self.request)
        self.assertIsNone(model.weights_path)
        self.builder.assert_not_called()

    def test_invalid_prompt_does_not_download_weights(self):
        with patch("ovlm.vlms.sam3.download_ckpt_from_hf") as download:
            with self.assertRaisesRegex(ValueError, "requires a prompt"):
                SAM3().infer_image(ImageInput(image=self.request.image))
        download.assert_not_called()
        self.builder.assert_not_called()

    def test_rm_removes_cached_checkpoint_without_downloading(self):
        path = self.cache_checkpoint()
        config = self.checkpoint.with_name("config.json")
        config.write_text("{}")
        with patch("ovlm.vlms.sam3.download_ckpt_from_hf") as download:
            model = get_model("sam3")
            self.assertEqual(model.rm(), path)
            self.assertIsNone(model.weights_path)
            self.assertIsNone(model.rm())
        self.assertFalse(self.checkpoint.exists())
        self.assertTrue(config.is_file())
        download.assert_not_called()
        self.builder.assert_not_called()

    def test_inference_downloads_again_after_rm(self):
        model = SAM3(weights_path=self.cache_checkpoint())
        model.rm()
        with patch("ovlm.vlms.sam3.download_ckpt_from_hf", side_effect=self.cache_checkpoint) as download:
            with redirect_stderr(io.StringIO()):
                model.infer_image(self.request)
        download.assert_called_once()
        self.assertTrue(self.checkpoint.is_file())
        self.builder.assert_called_once_with(checkpoint_path=str(self.checkpoint), load_from_HF=False)

    def test_rm_custom_checkpoint_does_not_remove_cached_checkpoint(self):
        self.cache_checkpoint()
        custom = self.cache / "custom.pt"
        custom.write_bytes(b"custom checkpoint")
        model = SAM3(weights_path=str(custom))
        self.assertEqual(model.rm(), str(custom))
        self.assertFalse(custom.exists())
        self.assertTrue(self.checkpoint.is_file())
        self.assertIsNone(model.weights_path)

    def test_rm_missing_custom_checkpoint_does_not_fall_back_to_cache(self):
        self.cache_checkpoint()
        model = SAM3(weights_path=str(self.cache / "missing.pt"))
        self.assertIsNone(model.rm())
        self.assertTrue(self.checkpoint.is_file())
        self.assertIsNone(model.weights_path)

    def test_rm_symlink_preserves_shared_checkpoint(self):
        target = self.cache / "shared.pt"
        target.write_bytes(b"shared checkpoint")
        link = self.cache / "custom.pt"
        try:
            link.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"Symlink creation unavailable: {exc}")
        model = SAM3(weights_path=str(link))
        self.assertEqual(model.rm(), str(link))
        self.assertFalse(link.is_symlink())
        self.assertTrue(target.is_file())

    def test_rm_failure_preserves_checkpoint_and_state(self):
        path = self.cache_checkpoint()
        model = SAM3(weights_path=path)
        with patch("ovlm.vlms.sam3.Path.unlink", side_effect=PermissionError("Access denied")):
            with self.assertRaisesRegex(PermissionError, "Access denied"):
                model.rm()
        self.assertEqual(model.weights_path, path)
        self.assertTrue(self.checkpoint.is_file())

    def test_rm_command_removes_cached_checkpoint(self):
        self.cache_checkpoint()
        result = CliRunner().invoke(app, ["rm", "sam3"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Removed checkpoint for sam3", result.stdout)
        self.assertFalse(self.checkpoint.exists())

    def test_rm_command_with_no_local_weights_does_not_download(self):
        with patch("ovlm.vlms.sam3.download_ckpt_from_hf") as download:
            result = CliRunner().invoke(app, ["rm", "sam3"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("No local weights are available", result.stdout)
        download.assert_not_called()


class PullCommandTests(unittest.TestCase):
    def test_pull_command_reports_checkpoint_location(self):
        model = Mock()
        model.pull.return_value = "models/sam3.pt"
        with patch("ovlm.main.get_model", return_value=model) as resolve:
            result = CliRunner().invoke(app, ["pull", "sam3"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("models/sam3.pt", result.stdout)
        resolve.assert_called_once_with("sam3")
        model.pull.assert_called_once_with()

    def test_unknown_model_is_a_cli_error(self):
        result = CliRunner().invoke(app, ["pull", "missing-model"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("Unknown VLM", result.output)

    def test_download_failure_is_a_cli_error(self):
        model = Mock()
        model.pull.side_effect = OSError("Access denied")
        with patch("ovlm.main.get_model", return_value=model):
            result = CliRunner().invoke(app, ["pull", "sam3"])
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("Cannot pull sam3: Access denied", result.stderr)

    def test_rm_unknown_model_is_a_cli_error(self):
        result = CliRunner().invoke(app, ["rm", "missing-model"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("Unknown VLM", result.output)

    def test_rm_failure_is_a_cli_error(self):
        model = Mock()
        model.rm.side_effect = PermissionError("Access denied")
        with patch("ovlm.main.get_model", return_value=model):
            result = CliRunner().invoke(app, ["rm", "sam3"])
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("Cannot remove sam3: Access denied", result.stderr)


if __name__ == "__main__":
    unittest.main()
