from pathlib import Path

import capybara as cb
import numpy as np
import pytest

from mrzscanner.rec.infer import Inference


class DummyONNXEngine:
    def __init__(self, model_path, gpu_id, backend, **kwargs):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.backend = backend
        self.input_infos = {"input": "dummy"}
        self.output_infos = {"output": "dummy"}

    def __call__(self, **kwargs):
        # 傳回一個 dummy 預測結果，其 shape 為 (1, 序列長度, 類別數)
        # 這裡使用全 0 陣列即可，因為後續會用 argmax 處理
        return {"output": np.zeros((1, 5, 10), dtype=np.float32)}


def dummy_imresize(img, size):
    # 強制將輸入圖片調整為 (size[0], size[1], channels)
    if img.ndim == 3:
        return np.resize(img, (size[0], size[1], img.shape[2]))
    return np.resize(img, (size[0], size[1]))


def dummy_download(file_id, file_name, target_dir):
    # 紀錄下載呼叫，不做實際動作
    pass

# Dummy Path 物件


class DummyPath:
    def __init__(self, path, exists_flag=True):
        self.path = path
        self.exists_flag = exists_flag

    def exists(self):
        return self.exists_flag

# --- 測試案例 ---

# 測試 __init__ 當模型檔案存在時不會呼叫下載


def test_init_no_download(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)
    monkeypatch.setattr(cb, "Path", lambda p: DummyPath(p, exists_flag=True))
    download_called = {"called": False}

    def dummy_download_google(file_id, file_name, target_dir):
        download_called["called"] = True
    monkeypatch.setattr(cb, "download_from_google", dummy_download_google)
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)

    inf = Inference()
    # 驗證模型設定與屬性
    assert inf.image_size == (64, 640)
    assert inf.input_key == "input"
    assert inf.output_key == "output"
    assert not download_called["called"]
    # 驗證 text_dec 已正確建立
    assert hasattr(inf, "text_dec")

# 測試 __init__ 當模型檔案不存在時會呼叫下載


def test_init_with_download(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)
    monkeypatch.setattr(cb, "Path", lambda p: DummyPath(p, exists_flag=False))
    download_called = {"called": False}

    def dummy_download_google(file_id, file_name, target_dir):
        download_called["called"] = True
    monkeypatch.setattr(cb, "download_from_google", dummy_download_google)
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)

    Inference()
    assert download_called["called"]

# 測試 preprocess 方法 (啟用 normalization)


def test_preprocess_normalize(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)
    monkeypatch.setattr(cb, "Path", lambda p: DummyPath(p, exists_flag=True))
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)

    inf = Inference()
    # 建立一個 shape 為 (100, 200, 3) 的 dummy 圖片
    img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
    tensors = inf.preprocess(img, normalize=True)
    tensor = tensors[inf.input_key]
    # 預期 tensor 經過轉置後 shape 為 (1, 3, 64, 640)
    assert tensor.shape == (1, 3, 64, 640)
    # normalization 應該將數值壓縮至 [0, 1]
    assert tensor.max() <= 1.0

# 測試 preprocess 方法 (未啟用 normalization)


def test_preprocess_no_normalize(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)
    monkeypatch.setattr(cb, "Path", lambda p: DummyPath(p, exists_flag=True))
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngine)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)

    inf = Inference()
    img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
    tensors = inf.preprocess(img, normalize=False)
    tensor = tensors[inf.input_key]
    assert tensor.shape == (1, 3, 64, 640)
    # 未 normalization 時，tensor 中應有值大於 1
    assert tensor.max() > 1.0

# 測試 postprocess 方法


def test_postprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)
    monkeypatch.setattr(cb, "Path", lambda p: DummyPath(p, exists_flag=True))
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngine)

    inf = Inference()
    # 覆寫 text_dec，使其回傳固定字串
    inf.text_dec = lambda x: ("ABC<SEP>DEF",)
    # 建立 dummy 預測資料
    dummy_pred_array = np.zeros((1, 5, 10), dtype=np.float32)
    dummy_pred = {inf.output_key: dummy_pred_array}
    result = inf.postprocess(dummy_pred)
    assert result == "ABC<SEP>DEF"

# 測試 __call__ 方法，檢查整個流程


def test_call(monkeypatch, tmp_path):
    monkeypatch.setattr(cb, "get_curdir", lambda _: tmp_path)
    monkeypatch.setattr(cb, "Path", lambda p: DummyPath(p, exists_flag=True))
    # 使用一個自訂的 Dummy ONNXEngine

    class DummyONNXEngineCall:
        def __init__(self, model_path, gpu_id, backend, **kwargs):
            self.input_infos = {"input": "dummy"}
            self.output_infos = {"output": "dummy"}

        def __call__(self, **kwargs):
            return {"output": np.zeros((1, 5, 10), dtype=np.float32)}
    monkeypatch.setattr(cb, "ONNXEngine", DummyONNXEngineCall)
    monkeypatch.setattr(cb, "imresize", dummy_imresize)

    inf = Inference()
    # 覆寫 text_dec 使其回傳含有分隔符號的字串
    inf.text_dec = lambda x: ("XYZ<SEP>123",)
    img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
    result = inf(img, normalize=True)
    assert isinstance(result, list)
    assert result == ["XYZ", "123"]
