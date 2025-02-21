from unittest.mock import MagicMock

import numpy as np
import pytest

from mrzscanner import ErrorCodes, ModelType, MRZScanner


@pytest.fixture
def mock_scanner():
    return MRZScanner(model_type=ModelType.spotting)


@pytest.fixture
def mock_image():
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_invalid_image():
    return "invalid_image_format"


def test_mrzscanner_initialization():
    scanner = MRZScanner(model_type=ModelType.spotting)
    assert isinstance(scanner, MRZScanner)
    assert scanner.model_type == ModelType.spotting


def test_invalid_model_type():
    with pytest.raises(ValueError, match="is not correct for ModelType"):
        MRZScanner(model_type=99)


def test_list_models(mock_scanner):
    models = mock_scanner.list_models()
    assert isinstance(models, dict)
    assert "spotting" in models
    assert "detection" in models
    assert "recognition" in models


def test_invalid_image_format(mock_scanner, mock_invalid_image):
    result = mock_scanner(mock_invalid_image)
    assert isinstance(result, tuple)
    assert result[1] == ErrorCodes.INVALID_INPUT_FORMAT


def test_postprocess_invalid_line_count(mock_scanner):
    invalid_results = np.array(["ABCDE", "FGHIJ", "KLMNO", "PQRST"])
    processed, error = mock_scanner.postprocess(invalid_results)
    assert error == ErrorCodes.POSTPROCESS_FAILED_LINE_COUNT


def test_postprocess_td1_invalid_length(mock_scanner):
    invalid_results = np.array(["A" * 30, "B" * 30, "C" * 31])
    processed, error = mock_scanner.postprocess(invalid_results)
    assert error == ErrorCodes.POSTPROCESS_FAILED_TD1_LENGTH


def test_postprocess_td2_td3_invalid_length(mock_scanner):
    invalid_results = np.array(["A" * 35, "B" * 35])
    processed, error = mock_scanner.postprocess(invalid_results)
    assert error == ErrorCodes.POSTPROCESS_FAILED_TD2_TD3_LENGTH


def test_valid_image_processing(mock_scanner, mock_image, monkeypatch):
    mock_scanner.scanner = MagicMock(
        return_value=np.array([
            "P<USATEST1234567890123456789012345<<<<<<<<<<",  # 44 characters
            "6408127M1406222USA00000000000000<<<<<<<<<<<0"  # 44 characters, 保證符合數字/字母要求
        ])
    )
    result = mock_scanner(mock_image)
    assert result["msg"] == ErrorCodes.NO_ERROR


def test_detection_model(mock_image, monkeypatch):
    mock_detector = MRZScanner(model_type=ModelType.detection)
    mock_detector.detector = MagicMock(return_value=[[0, 0], [10, 10]])
    result = mock_detector(mock_image)
    assert result["mrz_polygon"] is not None


def test_two_stage_model(mock_image, monkeypatch):
    mock_two_stage = MRZScanner(model_type=ModelType.two_stage)
    mock_two_stage.detector = MagicMock(return_value=np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10]]))  # 確保是 ndarray
    mock_two_stage.recognizer = MagicMock(
        return_value=["MRZLINE1", "MRZLINE2"])
    result = mock_two_stage(mock_image)
    assert result["mrz_polygon"] is not None
    assert result["mrz_texts"] is not None
