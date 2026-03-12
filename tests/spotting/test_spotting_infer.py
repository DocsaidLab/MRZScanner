from pathlib import Path
from typing import List

import capybara as cb
import cv2
import numpy as np
import pytest

import mrzscanner.spotting.infer as spotting_infer
from mrzscanner.spotting.infer import Inference


class DummyONNXEngine:
    def __init__(self, model_path, gpu_id, backend, **kwargs):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.backend = backend

    def summary(self):
        return {
            "inputs": [{"name": "input", "dtype": "", "shape": []}],
            "outputs": [{"name": "output", "dtype": "", "shape": []}],
        }

    def __call__(self, **kwargs):
        # Return a dummy output array with shape (1, 10, 20)
        return {"output": np.zeros((1, 10, 20), dtype=np.float32)}


def dummy_imresize(img, size):
    # Return an array of the given size with the same number of channels as the input.
    channels = img.shape[2] if img.ndim == 3 else 1
    return np.zeros((size[0], size[1], channels), dtype=img.dtype)


def dummy_download(file_id, file_name, target_dir):
    pass


class DummyPath:
    def __init__(self, path, exists_flag=True):
        self.path = path
        self.exists_flag = exists_flag

    def exists(self):
        return self.exists_flag

    def __truediv__(self, other):
        return f"{self.path}/{other}"


# --- Tests ---


def test_init_no_download(tmp_path, monkeypatch):
    monkeypatch.setattr(spotting_infer, "DIR", tmp_path)
    monkeypatch.setattr(
        spotting_infer, "Path", lambda p: DummyPath(p, exists_flag=True)
    )
    download_called = {"called": False}

    def dummy_download_google(file_id, file_name, target_dir):
        download_called["called"] = True

    monkeypatch.setattr(spotting_infer, "download_from_google", dummy_download_google)
    monkeypatch.setattr(spotting_infer, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)

    inf = Inference()
    assert inf.image_size == (512, 512)
    assert inf.input_key == "input"
    assert inf.output_key == "output"
    assert not download_called["called"]
    assert hasattr(inf, "text_dec")


def test_init_with_download(tmp_path, monkeypatch):
    monkeypatch.setattr(spotting_infer, "DIR", tmp_path)
    monkeypatch.setattr(
        spotting_infer, "Path", lambda p: DummyPath(p, exists_flag=False)
    )
    download_called = {"called": False}

    def dummy_download_google(file_id, file_name, target_dir):
        download_called["called"] = True

    monkeypatch.setattr(spotting_infer, "download_from_google", dummy_download_google)
    monkeypatch.setattr(spotting_infer, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)

    Inference()
    assert download_called["called"]


def test_preprocess_padding_horizontal(tmp_path, monkeypatch):
    # Test when image height < width (H < W)
    monkeypatch.setattr(spotting_infer, "DIR", tmp_path)
    monkeypatch.setattr(
        spotting_infer, "Path", lambda p: DummyPath(p, exists_flag=True)
    )
    monkeypatch.setattr(spotting_infer, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)

    record = {}

    def dummy_copyMakeBorder(img, top, bottom, left, right, borderType, value):
        record["top"] = top
        record["bottom"] = bottom
        record["left"] = left
        record["right"] = right
        return img

    monkeypatch.setattr(cv2, "copyMakeBorder", dummy_copyMakeBorder)

    inf = Inference()
    # Create an image with shape (200, 300, 3) where H < W.
    img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    tensor_dict = inf.preprocess(img, normalize=True)
    # Expected padding: pad = (300 - 200) // 2 = 50 -> top=50, bottom=50, left=0, right=0.
    assert record["top"] == 50
    assert record["bottom"] == 50
    assert record["left"] == 0
    assert record["right"] == 0
    tensor = tensor_dict[inf.input_key]
    assert tensor.shape == (1, 3, 512, 512)
    # Since dummy_imresize returns zeros, after normalization the tensor remains zeros.
    assert np.all(tensor == 0)


def test_preprocess_padding_vertical(tmp_path, monkeypatch):
    # Test when image height >= width (H >= W)
    monkeypatch.setattr(spotting_infer, "DIR", tmp_path)
    monkeypatch.setattr(
        spotting_infer, "Path", lambda p: DummyPath(p, exists_flag=True)
    )
    monkeypatch.setattr(spotting_infer, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)

    record = {}

    def dummy_copyMakeBorder(img, top, bottom, left, right, borderType, value):
        record["top"] = top
        record["bottom"] = bottom
        record["left"] = left
        record["right"] = right
        return img

    monkeypatch.setattr(cv2, "copyMakeBorder", dummy_copyMakeBorder)

    inf = Inference()
    # Create an image with shape (300, 200, 3) where H >= W.
    img = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
    tensor_dict = inf.preprocess(img, normalize=False)
    # Expected padding: pad = (300 - 200) // 2 = 50 -> left=50, right=50, top=0, bottom=0.
    assert record["top"] == 0
    assert record["bottom"] == 0
    assert record["left"] == 50
    assert record["right"] == 50
    tensor = tensor_dict[inf.input_key]
    assert tensor.shape == (1, 3, 512, 512)
    # Without normalization, dummy_imresize still returns zeros.
    assert np.all(tensor == 0)


def test_call(tmp_path, monkeypatch):
    monkeypatch.setattr(spotting_infer, "DIR", tmp_path)
    monkeypatch.setattr(
        spotting_infer, "Path", lambda p: DummyPath(p, exists_flag=True)
    )

    class DummyONNXEngineCall:
        def __init__(self, model_path, gpu_id, backend, **kwargs):
            pass

        def summary(self):
            return {
                "inputs": [{"name": "input", "dtype": "", "shape": []}],
                "outputs": [{"name": "output", "dtype": "", "shape": []}],
            }

        def __call__(self, **kwargs):
            # Return a dummy output array.
            return {"output": np.zeros((1, 10, 20), dtype=np.float32)}

    monkeypatch.setattr(spotting_infer, "ONNXEngine", DummyONNXEngineCall)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)

    inf = Inference()
    # Override text_dec to return a fixed string.
    inf.text_dec = lambda x: ("HELLO&WORLD",)
    img = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
    result = inf(img, normalize=True)
    assert isinstance(result, list)
    assert result == ["HELLO", "WORLD"]
