import numpy as np
import pytest
from capybara.runtime import Backend

from mrzscanner.det.infer import Inference


@pytest.fixture
def fake_inference():
    # 建立 Inference 物件，方便後續測試使用
    # 這邊的 GPU ID 與 backend 可根據環境需求自行調整
    return Inference(gpu_id=0, backend=Backend.cpu, model_cfg="20250222")


def test_inference_init(fake_inference):
    # 測試初始化參數是否正確
    assert fake_inference.model_cfg == "20250222"
    assert fake_inference.image_size == (256, 256)
    assert fake_inference.input_key is not None
    assert fake_inference.output_key is not None


def test_preprocess_padding_width_greater_than_height(fake_inference):
    # 模擬一張寬大於高的假圖 (H < W)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor_dict, (h, w), (shift_w, shift_h) = fake_inference.preprocess(img)

    # 檢查 key 是否正確
    assert fake_inference.input_key in tensor_dict

    # 檢查 tensor shape: (1, 3, 256, 256) or (batch, channel, height, width)
    assert tensor_dict[fake_inference.input_key].shape == (1, 3, 256, 256)

    # 原圖高寬
    assert (h, w) == (200, 200)

    # 確認是上下 padding，因此 shift_h 應大於 0，shift_w 應該是 0
    assert shift_h > 0
    assert shift_w == 0


def test_preprocess_padding_height_greater_than_width(fake_inference):
    # 模擬一張高大於寬的假圖 (H > W)
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    tensor_dict, (h, w), (shift_w, shift_h) = fake_inference.preprocess(img)

    # 檢查 key 是否正確
    assert fake_inference.input_key in tensor_dict

    # 檢查 tensor shape
    assert tensor_dict[fake_inference.input_key].shape == (1, 3, 256, 256)

    # 原圖高寬
    assert (h, w) == (200, 200)

    # 確認是左右 padding，因此 shift_w 應大於 0，shift_h 應該是 0
    assert shift_w > 0
    assert shift_h == 0


def test_preprocess_normalize(fake_inference):
    # 測試 normalize 的情況
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    tensor_dict, _, _ = fake_inference.preprocess(img, normalize=True)
    tensor = tensor_dict[fake_inference.input_key]

    # 確認值域應該介於 0~1 之間
    assert np.all(tensor >= 0) and np.all(tensor <= 1)


def test_postprocess_empty_heatmap(fake_inference):
    # 模擬空的 heatmap，預期輸出空的 polygon array
    hmap = np.zeros((256, 256), dtype=np.float32)
    poly = fake_inference.postprocess(hmap, (256, 256), (0, 0))
    assert poly.shape == (0,), f"預期空陣列, 但得到 shape={poly.shape}"


def test_postprocess_single_polygon(fake_inference):
    # 模擬簡單單一 polygon 的 heatmap，使用中心畫一個白色方塊
    hmap = np.zeros((256, 256), dtype=np.float32)
    hmap[100:150, 100:150] = 1.0  # 中心 50x50 區域
    poly = fake_inference.postprocess(hmap, (256, 256), (0, 0))

    # 預期會回傳 4 個點的外接矩形
    # 注意實際測試時可能要容忍一些誤差
    assert poly.shape == (4, 2)


def test_call_with_mock(fake_inference, monkeypatch):
    # 假設不想依賴真實模型推理結果，我們可以 mock 其輸出
    mock_output = {
        fake_inference.output_key: np.random.rand(1, 256, 256).astype(np.float32)
    }

    def mock_model_call(*args, **kwargs):
        return mock_output

    monkeypatch.setattr(fake_inference.model, "__call__", mock_model_call)

    # 隨意建立一張圖片測試
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

    poly = fake_inference(img)

    # 只要確認最後不會報錯，且 polygon 有正確回傳即可
    assert isinstance(poly, np.ndarray), "最終輸出必須為 Numpy ndarray"
