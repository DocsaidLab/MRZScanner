from unittest.mock import MagicMock

import numpy as np
import pytest

from mrzscanner import ErrorCodes, ModelType, MRZScanner


@pytest.fixture
def mock_scanner():
    """建立預設為 spotting 模式的 MRZScanner."""
    return MRZScanner(model_type=ModelType.spotting)


@pytest.fixture
def mock_image():
    """建立隨機 100x100 測試影像."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_invalid_image():
    """提供不合法的影像資料格式."""
    return "invalid_image_format"


def test_mrzscanner_initialization():
    """確認 MRZScanner 初始化正確."""
    scanner = MRZScanner(model_type=ModelType.spotting)
    assert isinstance(scanner, MRZScanner)
    assert scanner.model_type == ModelType.spotting


def test_invalid_model_type():
    """確認傳入不合法的 model_type 會拋出 ValueError."""
    with pytest.raises(ValueError, match="is not correct for ModelType"):
        MRZScanner(model_type=99)


def test_list_models(mock_scanner):
    """測試 list_models() 可以回傳包含 spotting / detection / recognition 的字典."""
    models = mock_scanner.list_models()
    assert isinstance(models, dict)
    assert "spotting" in models
    assert "detection" in models
    assert "recognition" in models


def test_invalid_image_format(mock_scanner, mock_invalid_image):
    """測試傳入不合法影像格式時, 回傳錯誤碼 INVALID_INPUT_FORMAT."""
    result = mock_scanner(mock_invalid_image)
    # 根據實際程式碼邏輯, 回傳類型應該是 dict, 其中包含 'msg' 欄位
    assert isinstance(result, dict)
    assert result["msg"] == ErrorCodes.INVALID_INPUT_FORMAT


def test_postprocess_invalid_line_count(mock_scanner):
    """測試 postprocess 輸入行數不為 2 或 3 時, 回傳 POSTPROCESS_FAILED_LINE_COUNT."""
    invalid_results = np.array(["ABCDE", "FGHIJ", "KLMNO", "PQRST"])
    processed, error = mock_scanner.postprocess(invalid_results)
    assert error == ErrorCodes.POSTPROCESS_FAILED_LINE_COUNT


def test_postprocess_td1_invalid_length(mock_scanner):
    """測試三行 (TD1) 但行長不正確時, 回傳 POSTPROCESS_FAILED_TD1_LENGTH."""
    invalid_results = np.array(["A" * 30, "B" * 30, "C" * 31])
    processed, error = mock_scanner.postprocess(invalid_results)
    assert error == ErrorCodes.POSTPROCESS_FAILED_TD1_LENGTH


def test_postprocess_td2_td3_invalid_length(mock_scanner):
    """測試兩行 (TD2/TD3) 但行長不符 36 或 44 時, 回傳 POSTPROCESS_FAILED_TD2_TD3_LENGTH."""
    invalid_results = np.array(["A" * 35, "B" * 35])
    processed, error = mock_scanner.postprocess(invalid_results)
    assert error == ErrorCodes.POSTPROCESS_FAILED_TD2_TD3_LENGTH


def test_valid_image_processing(mock_scanner, mock_image, monkeypatch):
    """測試有效影像傳入 spotting 模式, 模擬返回 2 行 (44 字元) 的 MRZ."""
    mock_scanner.scanner = MagicMock(
        return_value=np.array([
            "P<USATEST1234567890123456789012345<<<<<<<<<<",  # 44 characters
            "6408127M1406222USA00000000000000<<<<<<<<<<<0"  # 44 characters
        ])
    )
    result = mock_scanner(mock_image)
    assert result["msg"] == ErrorCodes.NO_ERROR


def test_detection_model(mock_image, monkeypatch):
    """測試 detection 模式, 模擬偵測器返回座標."""
    mock_detector = MRZScanner(model_type=ModelType.detection)
    mock_detector.detector = MagicMock(return_value=[[0, 0], [10, 10]])
    result = mock_detector(mock_image)
    assert result["mrz_polygon"] is not None


def test_two_stage_model(mock_image, monkeypatch):
    """測試 two_stage 模式, 同時模擬 detector 與 recognizer."""
    mock_two_stage = MRZScanner(model_type=ModelType.two_stage)
    mock_two_stage.detector = MagicMock(
        return_value=np.array([[0, 0], [10, 0], [10, 10], [0, 10]]))
    mock_two_stage.recognizer = MagicMock(
        return_value=["MRZLINE1", "MRZLINE2"])

    result = mock_two_stage(mock_image)
    assert result["mrz_polygon"] is not None
    assert result["mrz_texts"] is not None
