import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ovlm._internal.registry import list_local_models
from ovlm.api import app
from ovlm.vlms.sam3 import SAM3


async def get_models_response():
    """Exercise the real ASGI route without an HTTP client dependency."""
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/model",
            "raw_path": b"/api/model",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
        },
        receive,
        send,
    )
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return status, json.loads(body)


class LocalModelsTests(unittest.TestCase):
    def test_registry_filters_missing_weights_and_returns_sorted_names(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            checkpoint.write_bytes(b"test checkpoint")
            paths = {
                "zebra": str(checkpoint),
                "absent": None,
                "deleted": str(Path(directory) / "missing.pt"),
                "directory": directory,
                "alpha": str(checkpoint),
            }
            factories = {
                name: Mock(return_value=Mock(find_cached_weights=Mock(return_value=path)))
                for name, path in paths.items()
            }
            with patch("ovlm._internal.registry._discover_models", return_value=factories):
                self.assertEqual(list_local_models(), ["alpha", "zebra"])
            for factory in factories.values():
                factory.return_value.pull.assert_not_called()
                factory.return_value.infer_image.assert_not_called()


class ModelsAPITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name)
        cache_patch = patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache))
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        self.checkpoint = self.cache / "models--facebook--sam3" / "snapshots" / "test-revision" / "sam3.pt"
        builder_patch = patch("ovlm.vlms.sam3.build_sam3_image_model")
        self.builder = builder_patch.start()
        self.addCleanup(builder_patch.stop)
        download_patch = patch("ovlm.vlms.sam3.download_ckpt_from_hf", side_effect=self.cache_checkpoint)
        self.download = download_patch.start()
        self.addCleanup(download_patch.stop)

    def cache_checkpoint(self, **kwargs):
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint.write_bytes(b"test checkpoint")
        refs = self.cache / "models--facebook--sam3" / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text("test-revision")
        return str(self.checkpoint)

    def test_endpoint_returns_empty_list_without_downloading(self):
        self.assertEqual(asyncio.run(get_models_response()), (200, []))
        self.download.assert_not_called()
        self.builder.assert_not_called()

    def test_endpoint_reflects_pull_and_rm_without_restart(self):
        self.assertEqual(asyncio.run(get_models_response()), (200, []))
        SAM3().pull()
        self.assertEqual(asyncio.run(get_models_response()), (200, ["sam3"]))
        self.assertEqual(asyncio.run(get_models_response()), (200, ["sam3"]))
        SAM3().rm()
        self.assertEqual(asyncio.run(get_models_response()), (200, []))
        self.download.assert_called_once()
        self.builder.assert_not_called()

    def test_config_without_checkpoint_is_not_an_available_model(self):
        self.cache_checkpoint()
        self.checkpoint.with_name("config.json").write_text("{}")
        self.checkpoint.unlink()
        self.assertEqual(asyncio.run(get_models_response()), (200, []))
        self.download.assert_not_called()
        self.builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
