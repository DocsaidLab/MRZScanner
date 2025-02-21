from pathlib import Path

import capybara as cb
import numpy as np
import pytest

from mrzscanner.det.infer import Inference


class DummyONNXEngine:
    def __init__(self, model_path, gpu_id, backend, **kwargs):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.backend = backend
        self.input_infos = {"input": "dummy"}
        self.output_infos = {"output": "dummy"}

    def __call__(self, **kwargs):
        # Return a dummy heatmap (value 0.5) with shape (256,256)
        return {"output": [np.full((256, 256), 0.5, dtype=np.float32)]}

# Dummy image resize function


def dummy_imresize(img, size):
    # For a 3D image (H,W,C), force output shape to (size[0], size[1], C)
    if img.ndim == 3:
        return np.resize(img, (size[0], size[1], img.shape[2]))
    else:
        return np.resize(img, (size[0], size[1]))

# Dummy binarization function


def dummy_imbinarize(img):
    # Threshold at 127: values above become 255, below become 0
    return (img > 127).astype(np.uint8) * 255

# Dummy polygon classes to simulate cb.Polygons.from_image behavior


class DummyPolygon:
    def __init__(self, area, points):
        self.area = area
        self.points = points

    def to_min_boxpoints(self):
        return self.points


class DummyPolygons:
    def __init__(self, polygons):
        self.polygons = polygons

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, idx):
        # Support slicing and boolean array indexing
        if isinstance(idx, slice):
            return DummyPolygons(self.polygons[idx])
        elif isinstance(idx, np.ndarray):
            filtered = [p for p, flag in zip(self.polygons, idx) if flag]
            return DummyPolygons(filtered)
        else:
            return self.polygons[idx]

    @property
    def area(self):
        return np.array([p.area for p in self.polygons])

# --- Test cases using pytest ---

# Test that __init__ does not call download_from_google when file exists.


def test_init_no_download(tmp_path, monkeypatch):
    # Use tmp_path as the current directory.
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)
    # Simulate that the model file exists.

    class DummyPathExists:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return True
    monkeypatch.setattr(cb, "Path", lambda path: DummyPathExists(path))
    # Override ONNXEngine with our dummy version.
    monkeyatch = monkeypatch  # (alias for clarity)
    monkeyatch.setattr(cb, "ONNXEngine", DummyONNXEngine)
    monkeyatch.setattr(cb, "imresize", dummy_imresize)
    monkeyatch.setattr(cb, "imbinarize", dummy_imbinarize)
    # Provide a dummy Polygons.from_image (not used in __init__).
    monkeyatch.setattr(cb.Polygons, "from_image",
                       lambda hmap: DummyPolygons([]))

    inf = Inference()
    assert inf.image_size == (256, 256)
    assert inf.input_key == "input"
    assert inf.output_key == "output"
    assert isinstance(inf.model, DummyONNXEngine)

# Test that __init__ calls download_from_google when file does not exist.


def test_init_with_download(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)
    # Simulate that the model file does not exist.

    class DummyPathNotExists:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return False
    monkeypatch.setattr(cb, "Path", lambda path: DummyPathNotExists(path))
    download_called = {"called": False}

    def dummy_download(file_id, file_name, target_dir):
        download_called["called"] = True
    monkeypatch.setattr(cb, "download_from_google", dummy_download)
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)
    monkeypatch.setattr(cb, "imbinarize", dummy_imbinarize)
    monkeypatch.setattr(cb.Polygons, "from_image",
                        lambda hmap: DummyPolygons([]))

    Inference()
    assert download_called["called"] is True

# Test the preprocess method with normalization enabled.


def test_preprocess_normalize(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)

    class DummyPathExists:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return True
    monkeypatch.setattr(cb, "Path", lambda path: DummyPathExists(path))
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)
    monkeypatch.setattr(cb, "imbinarize", dummy_imbinarize)
    monkeypatch.setattr(cb.Polygons, "from_image",
                        lambda hmap: DummyPolygons([]))

    inf = Inference()
    # Create a dummy image of shape (300, 400, 3)
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    tensor_dict, orig_size = inf.preprocess(img, normalize=True)
    tensor = tensor_dict[inf.input_key]
    # Expect shape to be (1, channels, 256, 256)
    assert tensor.shape == (1, 3, 256, 256)
    # With normalization, values should be in the range [0, 1].
    assert tensor.max() <= 1.0
    assert orig_size == (300, 400)

# Test the preprocess method with normalization disabled.


def test_preprocess_no_normalize(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)

    class DummyPathExists:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return True
    monkeypatch.setattr(cb, "Path", lambda path: DummyPathExists(path))
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)
    monkeypatch.setattr(cb, "imbinarize", dummy_imbinarize)
    monkeypatch.setattr(cb.Polygons, "from_image",
                        lambda hmap: DummyPolygons([]))

    inf = Inference()
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    tensor_dict, orig_size = inf.preprocess(img, normalize=False)
    tensor = tensor_dict[inf.input_key]
    # Without normalization, the values should remain above 1.0.
    assert tensor.max() > 1.0
    assert orig_size == (300, 400)

# Test the postprocess method when no polygons are detected.


def test_postprocess_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)

    class DummyPathExists:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return True
    monkeypatch.setattr(cb, "Path", lambda path: DummyPathExists(path))
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)
    monkeypatch.setattr(cb, "imbinarize", dummy_imbinarize)
    # Force Polygons.from_image to return an empty container.
    monkeypatch.setattr(cb.Polygons, "from_image",
                        lambda hmap: DummyPolygons([]))

    inf = Inference()
    dummy_hmap = np.ones((256, 256), dtype=np.float32) * 0.5
    result = inf.postprocess(dummy_hmap, (300, 400))
    assert result.size == 0
    assert result.dtype == np.float32

# Test the postprocess method when polygons are detected.


def test_postprocess_polygon(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)

    class DummyPathExists:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return True
    monkeypatch.setattr(cb, "Path", lambda path: DummyPathExists(path))
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)
    monkeypatch.setattr(cb, "imbinarize", dummy_imbinarize)
    # Create two dummy polygons: one with smaller area and one with larger area.
    poly1 = DummyPolygon(10, [(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = DummyPolygon(20, [(0, 0), (2, 0), (2, 2), (0, 2)])
    dummy_polys = DummyPolygons([poly1, poly2])
    monkeypatch.setattr(cb.Polygons, "from_image", lambda hmap: dummy_polys)

    inf = Inference()
    dummy_hmap = np.ones((256, 256), dtype=np.float32) * 0.5
    result = inf.postprocess(dummy_hmap, (300, 400))
    expected = np.array([(0, 0), (2, 0), (2, 2), (0, 2)], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)

# Test the __call__ method to ensure the full pipeline works.


def test_call(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)

    class DummyPathExists:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return True
    monkeypatch.setattr(cb, "Path", lambda path: DummyPathExists(path))
    # Use a custom dummy ONNXEngine that returns a dummy heatmap.

    class DummyONNXEngineCall:
        def __init__(self, model_path, gpu_id, backend, **kwargs):
            self.input_infos = {"input": "dummy"}
            self.output_infos = {"output": "dummy"}

        def __call__(self, **kwargs):
            return {"output": [np.full((256, 256), 0.5, dtype=np.float32)]}
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngineCall)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)
    monkeypatch.setattr(cb, "imbinarize", dummy_imbinarize)
    # Set up Polygons.from_image to return two dummy polygons.
    poly1 = DummyPolygon(10, [(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = DummyPolygon(20, [(0, 0), (2, 0), (2, 2), (0, 2)])
    dummy_polys = DummyPolygons([poly1, poly2])
    monkeypatch.setattr(cb.Polygons, "from_image", lambda hmap: dummy_polys)

    inf = Inference()
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    result = inf(img, normalize=True)
    expected = np.array([(0, 0), (2, 0), (2, 2), (0, 2)], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
