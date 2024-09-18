from pathlib import Path
from typing import List

import cv2
import docsaidkit as D
import numpy as np
import pytest
from mrzscanner.spotting import Inference


@pytest.fixture
def mock_inference(monkeypatch):
    """Fixture to初始化推理類並模擬需要的依賴。"""
    # 模擬 get_curdir
    monkeypatch.setattr(D, 'get_curdir', lambda x: Path('/mock/path'))

    # 模擬下載模型
    def mock_download(file_id, file_name, model_path):
        pass  # 模擬下載，不實際進行下載
    monkeypatch.setattr(D, 'download_from_docsaid', mock_download)

    # 初始化推理
    inference = Inference(gpu_id=0, backend=D.Backend.cpu,
                          model_cfg='20240917')
    return inference


def test_init(mock_inference):
    """測試初始化和模型加載。"""
    assert isinstance(mock_inference, Inference)
    assert mock_inference.model_cfg == '20240917'
    assert mock_inference.image_size == (512, 512)


def test_preprocess(mock_inference, monkeypatch):
    """測試 preprocess 函數是否正確處理圖片。"""
    # 創建一個隨機的圖像（300x400，3通道）
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    # 模擬 centercrop 函數
    monkeypatch.setattr(D, 'centercrop', lambda x: x)

    # 不進行 centercrop
    processed_img = mock_inference.preprocess(img, do_centercrop=False)
    assert processed_img.shape == (3, 512, 512)  # 確認輸出大小是否正確

    # 測試 centercrop
    processed_img_with_crop = mock_inference.preprocess(
        img, do_centercrop=True)
    assert processed_img_with_crop.shape == (3, 512, 512)  # 確認大小是否正確


def test_model_download(mock_inference, monkeypatch):
    """測試模型文件不存在時是否會觸發下載。"""
    # 模擬路徑不存在
    monkeypatch.setattr(Path, 'exists', lambda x: False)

    # 模擬下載函數
    download_called = []

    def mock_download(file_id, file_name, model_path):
        download_called.append(True)

    monkeypatch.setattr(D, 'download_from_docsaid', mock_download)

    # 初始化推理類
    mock_inference.__init__(
        gpu_id=0, backend=D.Backend.cpu, model_cfg='20240917')

    # 確認是否觸發下載
    assert download_called, "Should trigger model download"


def test_engine(mock_inference, monkeypatch):
    """測試 engine 函數。"""
    # 模擬 ONNX 模型輸出
    mock_result = np.random.rand(1, 10, 38)

    # 模擬模型引擎
    monkeypatch.setattr(mock_inference.model, '__call__',
                        lambda img: {'text': mock_result})

    # 創建一個假定的 tensor
    dummy_tensor = np.random.rand(3, 512, 512).astype('float32')

    # 運行推理
    result = mock_inference.engine(dummy_tensor)

    # 確保返回值是 numpy array
    assert isinstance(result, np.ndarray)


def test_inference_pipeline(mock_inference, monkeypatch):
    """測試整個推理流程。"""
    # 模擬 ONNX 模型輸出
    mock_result = np.random.rand(1, 10, 38)

    # 模擬模型引擎
    monkeypatch.setattr(mock_inference.model, '__call__',
                        lambda img: {'text': mock_result})

    # 創建一個假定的圖像
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    # 運行整個推理管道
    result = mock_inference(img, do_centercrop=False)

    # 確保輸出是預期的列表
    assert isinstance(result, list)
