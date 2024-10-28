"""Full tests of training and prediction.

This runs five epochs of training over a small toy data set, attempting to
overfit, then compares the resubstitution predictions on this set to
previously computed results. As such this is essentially a change-detector
test. Currently, English (en), Greek (el), and Russian (ru) are supported."""

import difflib
import os
import tempfile
import unittest

from parameterized import parameterized

from udtube import cli

# Directory the unit test is located in, relative to the working directory.
DIR = os.path.relpath(os.path.dirname(__file__), os.getcwd())
CONFIG_PATH = os.path.join(DIR, "testdata/udtube_config.yaml")
DATASETS_DIR = os.path.join(DIR, "testdata/datasets")


class UDTubeTest(unittest.TestCase):
    def assertFileExists(self, path: str):
        self.assertTrue(os.path.exists(path), msg=f"file {path} not found")

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="udtube_test-")
        self.assertFileExists(CONFIG_PATH)

    def tearDown(self):
        self.tempdir.cleanup()

    @parameterized.expand(
        [
            # TODO: commented out to test CircleCI test resource limitations.
            # ("el", True),
            ("en", True),
            # TODO: ditto.
            # ("ru", False),  # Russian doesn't have XPOS.
        ]
    )
    def test_model(self, langcode: str, use_xpos: bool):
        # Fits model.
        train_path = os.path.join(DATASETS_DIR, f"{langcode}_train.conllu")
        self.assertFileExists(train_path)
        expected_path = os.path.join(
            DATASETS_DIR, f"{langcode}_expected.conllu"
        )
        self.assertFileExists(expected_path)
        model_dir = os.path.join(self.tempdir.name, "model")
        cli.udtube_python_interface(
            [
                "fit",
                f"--config={CONFIG_PATH}",
                f"--data.model_dir={model_dir}",
                f"--data.train={train_path}",
                # We are trying to overfit on the training data.
                f"--data.val={train_path}",
                f"--model.use_xpos={use_xpos}",
            ]
        )
        checkpoint_path = (
            f"{self.tempdir.name}/model/lightning_logs/"
            "version_0/checkpoints/last.ckpt"
        )
        self.assertFileExists(checkpoint_path)
        # Predicts on "expected" data.
        predicted_path = os.path.join(
            self.tempdir.name, f"{langcode}_predicted.conllu"
        )
        cli.udtube_python_interface(
            [
                "predict",
                f"--config={CONFIG_PATH}",
                f"--data.model_dir={model_dir}",
                f"--data.predict={expected_path}",
                f"--model.use_xpos={use_xpos}",
                f"--prediction.path={predicted_path}",
            ]
        )
        self.assertFileExists(predicted_path)
        diff = self._diff(predicted_path, expected_path)
        self.assertEqual(diff, [], f"Differences found:\n{diff}")

    @staticmethod
    def _diff(predicted_path: str, expected_path: str) -> list:
        with (
            open(predicted_path, "r") as predicted,
            open(expected_path, "r") as expected,
        ):
            diff = list(
                difflib.unified_diff(
                    predicted.readlines(),
                    expected.readlines(),
                    fromfile=predicted_path,
                    tofile=expected_path,
                    n=1,
                )
            )
        return diff


if __name__ == "__main__":
    unittest.main()
