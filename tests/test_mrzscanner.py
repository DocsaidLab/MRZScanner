import docsaidkit as D
import numpy as np
import pytest
from mrzscanner import ErrorCodes, ModelType, MRZScanner


@pytest.fixture
def mock_mrz_scanner(monkeypatch):
    """Fixture 用來初始化 MRZScanner 並模擬依賴。"""
    # 初始化 MRZScanner
    return MRZScanner(model_type=ModelType.spotting)


def test_mrz_scanner_init(mock_mrz_scanner):
    """測試 MRZScanner 的初始化。"""
    assert isinstance(mock_mrz_scanner, MRZScanner)
    assert mock_mrz_scanner.scanner is not None

# 測試無效輸入格式


def test_invalid_input_format(mock_mrz_scanner):
    """測試當輸入格式無效時的錯誤處理。"""
    invalid_input = "This is not an image"
    result, error = mock_mrz_scanner(invalid_input)
    assert error == ErrorCodes.INVALID_INPUT_FORMAT

# 測試圖片處理和輸出


def test_mrz_scanner_call(mock_mrz_scanner):
    """測試 MRZScanner 進行推理和處理的完整流程。"""
    # 創建一個假的圖像
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    # 執行 MRZ 掃描
    result, error = mock_mrz_scanner(img)

    # 檢查結果
    assert error == ErrorCodes.POSTPROCESS_FAILED_LINE_COUNT

# 測試 TD1 後處理失敗


def test_postprocess_failed_td1_length(mock_mrz_scanner, monkeypatch):
    """測試 TD1 後處理時行數不符的錯誤處理。"""
    # 模擬 SpottingInference 返回一個無效的 TD1 格式
    monkeypatch.setattr('mrzscanner.SpottingInference.__call__', lambda self, img, do_center_crop: np.array([
        "ABCDEFGHIJKLMNO1234567890ABCDEFGHIJKLMNO",  # 長度不符合 TD1 規定
        "1234567890ABCDE1234567890ABCDEF",
        "ABCDEFGHIJKLMNO1234567890ABCDEFGHIJKLMNO"
    ]))

    # 創建一個假的圖像
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    # 執行 MRZ 掃描並檢查錯誤代碼
    result, error = mock_mrz_scanner(img, do_postprocess=True)

    assert error == ErrorCodes.POSTPROCESS_FAILED_TD1_LENGTH

# 測試 TD2/TD3 後處理失敗


def test_postprocess_failed_td2_td3_length(mock_mrz_scanner, monkeypatch):
    """測試 TD2 或 TD3 後處理時長度不符的錯誤處理。"""
    # 模擬 SpottingInference 返回一個無效的 TD2/TD3 格式
    monkeypatch.setattr('mrzscanner.SpottingInference.__call__', lambda self, img, do_center_crop: np.array([
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO6908061F9406236ZE184226B<<<<<10L898902C36UTO"  # TD3 長度超過標準
    ]))

    # 創建一個假的圖像
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    # 執行 MRZ 掃描並檢查錯誤代碼
    result, error = mock_mrz_scanner(img, do_postprocess=True)
    assert error == ErrorCodes.POSTPROCESS_FAILED_TD2_TD3_LENGTH

# 測試無效模型配置


def test_invalid_model_config():
    """測試無效的 model_cfg 時是否會拋出異常。"""
    with pytest.raises(ValueError):
        MRZScanner(model_type=ModelType.spotting, model_cfg='invalid_cfg')
