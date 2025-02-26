**[English](./README.md)** | [中文](./README_tw.md)

# MRZScanner

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href="https://github.com/DocsaidLab/MRZScanner/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/MRZScanner?color=ffa"></a>
    <a href="https://pypi.org/project/mrzscanner_docsaid/"><img src="https://img.shields.io/pypi/v/mrzscanner_docsaid.svg"></a>
    <a href="https://pypi.org/project/mrzscanner_docsaid/"><img src="https://img.shields.io/pypi/dm/mrzscanner_docsaid?color=9cf"></a>
</p>

## Introduction

<div align="center">
   <img src="https://github.com/DocsaidLab/MRZScanner/raw/main/docs/title.webp?raw=true" width="800">
</div>

MRZ (Machine Readable Zone) refers to a specific area on travel documents such as passports, visas, and identity cards, where information can be quickly read by machines. MRZ is designed and generated according to the specifications in ICAO Document 9303, to expedite border checks and improve the accuracy of information processing.

People may not be familiar with MRZ, but most likely they have a passport that contains an MRZ section, which looks like the red-boxed part shown below:

<div align="center">
   <img src="https://github.com/DocsaidLab/MRZScanner/raw/main/docs/img1.jpg?raw=true" width="60%">
</div>

## Technical Documentation

For complete details and instructions on using the model, please refer to [**MRZScanner Documents**](https://docsaid.org/en/docs/mrzscanner/).

## Model Testing

We provide a web-based model testing tool for preliminary model evaluation.

- [**MRZScanner Web Demo**](https://docsaid.org/en/playground/mrzscanner-demo/)

If you have custom requirements, please feel free to contact us:

- **docsaidlab@gmail.com**

## Installation

### Install via PyPI

1. Install `mrzscanner_docsaid`:

   ```bash
   pip install mrzscanner-docsaid
   ```

2. Verify installation:

   ```bash
   python -c "import mrzscanner; print(mrzscanner.__version__)"
   ```

3. If you see the version number, the installation is successful.

### Install from GitHub

1. Download the project from GitHub:

   ```bash
   git clone https://github.com/DocsaidLab/MRZScanner.git
   ```

2. Install the wheel package:

   ```bash
   pip install wheel
   ```

3. Build the whl file:

   ```bash
   cd MRZScanner
   python setup.py bdist_wheel
   ```

4. Install the whl file:

   ```bash
   pip install dist/mrzscanner_docsaid-*-py3-none-any.whl
   ```

## Model Inference

First, don't worry about anything and just try running the following code to see if it executes properly:

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner

# Create model
model = MRZScanner()

# Read image from the web
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Model inference
result = model(img, do_center_crop=True, do_postprocess=False)

# Output result
print(result)
# {
#     'mrz_polygon':
#         array(
#             [
#                 [ 158.536 , 1916.3734],
#                 [1682.7792, 1976.1683],
#                 [1677.1018, 2120.8926],
#                 [ 152.8586, 2061.0977]
#             ],
#             dtype=float32
#         ),
#     'mrz_texts': [
#         'PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#         'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#     ],
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

If it runs successfully, let’s take a look at the details of the code below.

> [!TIP]
> MRZScanner has been wrapped with `__call__`, so you can directly call the instance for inference.

> [!NOTE]
> We have designed an automatic model download feature. When the program detects that you are missing the model, it will automatically connect to our server to download it.

## Using the `do_center_crop` Parameter

This image is likely taken with a mobile device, with a narrow aspect ratio. If you directly feed it to the model without modification, it may cause excessive text distortion. Therefore, we added the `do_center_crop` parameter for inference, which is used to crop the center of the image.

This parameter defaults to `False` because we believe no modifications should be made to the image without the user's knowledge. However, in real-world applications, images are often not in the standard square shape.

Images come in various sizes and aspect ratios, such as:

- Photos taken with mobile phones commonly have a 9:16 aspect ratio;
- Scanned documents are often in the A4 paper ratio;
- Webpage screenshots mostly have a 16:9 aspect ratio;
- Images taken with webcams are typically 4:3.

These non-square images, when directly used for inference without proper processing, often contain irrelevant areas or blank spaces, which negatively affect the model’s inference. Center cropping can effectively reduce these irrelevant regions, focusing on the central part of the image, thereby improving the accuracy and efficiency of inference.

Here’s how to use it:

```python
from mrzscanner import MRZScanner

model = MRZScanner()

result = model(img, do_center_crop=True) # Use center cropping
```

## Using the `do_postprocess` Parameter

In addition to center cropping, we also offer a post-processing option, `do_postprocess`, to further improve the model’s accuracy.

This parameter also defaults to `False`, for the same reason as before: we believe no modifications should be made to the recognition results without the user's knowledge.

In practical applications, there are certain rules in MRZ blocks, such as: country codes must be uppercase letters, gender can only be `M` or `F`, and fields related to dates can only contain numbers. These rules can be used to standardize MRZ blocks.

Therefore, we perform manual correction on standardizable blocks. Below is an example code snippet for correcting misidentified characters, replacing digits with the correct characters in fields where digits are not possible:

```python
import re

def replace_digits(text: str):
    text = re.sub('0', 'O', text)
    text = re.sub('1', 'I', text)
    text = re.sub('2', 'Z', text)
    text = re.sub('4', 'A', text)
    text = re.sub('5', 'S', text)
    text = re.sub('8', 'B', text)
    return text

if doc_type == 3:  # TD1
    if len(results[0]) != 30 or len(results[1]) != 30 or len(results[2]) != 30:
        return [''], ErrorCodes.POSTPROCESS_FAILED_TD1_LENGTH
    # Line1
    doc = results[0][0:2]
    country = replace_digits(results[0][2:5])
```

Although this post-processing did not significantly improve the accuracy in our project, retaining this functionality can still help correct erroneous recognition results in some cases.

You can consider setting `do_postprocess` to `True` during inference for potentially better results:

```python
result = model(img, do_postprocess=True)
```

Or, if you prefer to see the raw model output, simply use the default value.

## Advanced Settings

When calling the `MRZScanner` model, you can pass parameters for advanced configuration.

### Initialization

Here are the advanced configuration options during initialization:

- **Backend**

  Backend is an enumeration type used to specify the computation backend for `MRZScanner`.

  It includes the following options:

  - **cpu**: Use CPU for computation.
  - **cuda**: Use GPU for computation (requires appropriate hardware support).

  ```python
  from capybara import Backend

  model = MRZScanner(backend=Backend.cuda) # Use CUDA backend
  #
  # or
  #
  model = MRZScanner(backend=Backend.cpu) # Use CPU backend
  ```

  We use ONNXRuntime as the model inference engine. While ONNXRuntime supports various backend engines (including CPU, CUDA, OpenCL, DirectX, TensorRT, etc.), due to the environment we commonly use, we have slightly wrapped it and currently only provide CPU and CUDA as backend engines. Additionally, using CUDA computation requires appropriate hardware support and the installation of the corresponding CUDA drivers and toolkit.

  If your system does not have CUDA installed, or if the version is incorrect, you will not be able to use the CUDA computation backend.

- **ModelType**

  ModelType is an enumeration used to specify the type of model used by `MRZScanner`.

  The following options are currently available:

  - **spotting**: Uses an end-to-end model architecture and loads a single model.
  - **two_stage**: Uses a two-stage model architecture and loads two models.
  - **detection**: Loads only the MRZ detection model.
  - **recognition**: Loads only the MRZ recognition model.

  You can specify the model to use with the `model_type` parameter.

  ```python
  from mrzscanner import MRZScanner

  model = MRZScanner(model_type=MRZScanner.spotting)
  ```

- **ModelCfg**

  You can use `list_models` to view all available models.

  ```python
  from mrzscanner import MRZScanner

  print(MRZScanner().list_models())
  # {
  #    'spotting': ['20240919'],
  #    'detection': ['20250222'],
  #    'recognition': ['20250221']
  # }
  ```

  Choose the version you want, and use parameters like `spotting_cfg`, `detection_cfg`, `recognition_cfg`, etc., along with `ModelType` to specify the model.

  1. **spotting**:

     ```python
     model = MRZScanner(
        model_type=ModelType.spotting,
        spotting_cfg='20240919'
     )
     ```

  2. **two_stage**:

     ```python
     model = MRZScanner(
        model_type=ModelType.two_stage,
        detection_cfg='20250222',
        recognition_cfg='20250221'
     )
     ```

  3. **detection**:

     ```python
     model = MRZScanner(
        model_type=ModelType.detection,
        detection_cfg='20250222'
     )
     ```

  4. **recognition**:

     ```python
     model = MRZScanner(
        model_type=ModelType.recognition,
        recognition_cfg='20250221'
     )
     ```

  You can also skip specifying these parameters, as we have already configured default versions for each model.

### ModelType.spotting

This model is an end-to-end model that directly detects the location of the MRZ and performs recognition. Its downside is lower accuracy and it does not return the MRZ coordinates.

Example usage:

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner, ModelType

# Create model
model = MRZScanner(
   model_type=ModelType.spotting,
   spotting_cfg='20240919'
)

# Read image from the web
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Model inference
result = model(img, do_center_crop=True, do_postprocess=False)

# Output result
print(result)
# {
#    'mrz_polygon': None,
#    'mrz_texts': [
#        'PCAZEQAOARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#        'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#    ],
#    'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

### ModelType.two_stage

This is a two-stage model that first detects the MRZ location and then performs recognition. Its advantage is higher accuracy, and it also returns the MRZ coordinates.

Here’s an example, where we can also draw the MRZ location:

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner, ModelType

# Create model
model = MRZScanner(
   model_type=ModelType.two_stage,
   detection_cfg='20250222',
   recognition_cfg='20250221'
)

# Read image from the web
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Model inference
result = model(img, do_center_crop=True, do_postprocess=False)

# Output result
print(result)
# {
#     'mrz_polygon':
#         array(
#             [
#                 [ 158.536 , 1916.3734],
#                 [1682.7792, 1976.1683],
#                 [1677.1018, 2120.8926],
#                 [ 152.8586, 2061.0977]
#             ],
#             dtype=float32
#         ),
#     'mrz_texts': [
#         'PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#         'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#     ],
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }

# Draw MRZ location
from capybara import draw_polygon, imwrite, centercrop

poly_img = draw_polygon(img, result['mrz_polygon'], color=(0, 0, 255), thickness=5)
imwrite(centercrop(poly_img))
```

<div align="center">
   <img src="https://github.com/DocsaidLab/MRZScanner/raw/main/docs/demo_two_stage.jpg?raw=true" width="60%">
</div>

### ModelType.detection

This model only detects the MRZ location and does not perform recognition.

Example usage:

```python
import cv2
from skimage import io
from mrzscanner import MRZScanner, ModelType

# Create model
model = MRZScanner(
   model_type=ModelType.detection,
   detection_cfg='20250222',
)

# Read image from the web
img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Model inference
result = model(img, do_center_crop=True)

# Output result
print(result)
# {
#     'mrz_polygon':
#         array(
#             [
#                 [ 158.536 , 1916.3734],
#                 [1682.7792, 1976.1683],
#                 [1677.1018, 2120.8926],
#                 [ 152.8586, 2061.0977]
#             ],
#             dtype=float32
#         ),
#     'mrz_texts': None,
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

The MRZ detection result is the same as before, so we won't repeat the drawing here.

### ModelType.recognition

This model only performs MRZ recognition and does not detect the MRZ location.

To run this model, you need to first prepare the cropped MRZ image and pass it into the model.

Let’s prepare the cropped MRZ image by using the coordinates from the detection:

```python
import numpy as np
from skimage import io
from capybara import imwarp_quadrangle, imwrite

polygon = np.array([
    [ 158.536 , 1916.3734],
    [1682.7792, 1976.1683],
    [1677.1018, 2120.8926],
    [ 152.8586, 2061.0977]
], dtype=np.float32)

img = io.imread('https://github.com/DocsaidLab/MRZScanner/blob/main/docs/test_mrz.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

mrz_img = imwarp_quadrangle(img, polygon)
imwrite(mrz_img)
```

After running the above code, we can extract the cropped MRZ image:

<div align="center">
   <img src="https://github.com/DocsaidLab/MRZScanner/raw/main/docs/demo_recognition_warp.jpg?raw=true" width="90%">
</div>

With the image ready, we can now run the recognition model:

```python
from mrzscanner import MRZScanner, ModelType

# Create model
model = MRZScanner(
   model_type=ModelType.recognition,
   recognition_cfg='20250221'
)

# Input the cropped MRZ image
result = model(mrz_img, do_center_crop=False)

# Output result
print(result)
# {
#     'mrz_polygon': None,
#     'mrz_texts': [
#         'PCAZEQAQARIN<<FIDAN<<<<<<<<<<<<<<<<<<<<<<<<<',
#         'C946302620AZE6707297F23031072W12IMJ<<<<<<<40'
#     ],
#     'msg': <ErrorCodes.NO_ERROR: 'No error.'>
# }
```

> [!WARNING]
> Note that the parameter is set to `do_center_crop=False` here because we’ve already performed the cropping.

## Model Design

### Two-Stage Recognition Model

The two-stage model divides MRZ recognition into two phases: localization and recognition.

With this approach in mind, we can start designing the related models. Let's first take a look at the localization model.

### Localization Model

The localization of the MRZ region can be divided into two directions:

1. **Localization of MRZ Region Corners:**

    <div align="center">
      <img src="https://github.com/DocsaidLab/MRZScanner/raw/main/docs/img2.jpg?raw=true" width="50%">
    </div>

   This approach is similar to our previous document localization projects, except here we replace the document with the MRZ region.

   The difference is that the corners in document localization "physically" exist in the image, so the model doesn’t have to "imagine" a corner. However, in MRZ localization, we need the model to "guess" the corners.

   It turns out that this approach produces an unstable model. A slight movement of the passport results in the predicted corners of the MRZ region shifting all over the place.

   ***

2. **Segmentation of the MRZ Region:**

    <div align="center">
        <img src="https://github.com/DocsaidLab/MRZScanner/raw/main/docs/img3.jpg?raw=true" width="50%">
    </div>

   This method is more stable, as we can directly use a segmentation model to predict the range of the MRZ region. The text in the MRZ region is physically present in the image, so the model doesn’t need to make unnecessary guesses. As a result, we can directly segment the MRZ region without worrying about corner issues.

---

We adopted the segmentation approach.

In real-world scenarios, the passport held by the user will inevitably be tilted, so we need to correct the MRZ region to form a proper rectangle.

For the loss function, we referred to a survey paper:

- [**[20.06] A survey of loss functions for semantic segmentation**](https://arxiv.org/abs/2006.14822)

This paper provides a unified comparison and introduction of various segmentation loss functions proposed in recent years, and offers a solution to existing problems: **Log-Cosh Dice Loss**.

Interested readers can refer to this paper; we won’t go into further details here.

![Log-Cosh Dice Loss](https://github.com/DocsaidLab/MRZScanner/raw/main/docs/img4.jpg?raw=true)

In our experiments, using only the `Log-Cosh Dice Loss` did not provide satisfactory results, so we combined it with pixel classification loss `CrossEntropyLoss` and pixel regression loss `SmoothL1Loss` for training.

### Recognition Model

The recognition model is simpler because we’ve already segmented the MRZ region, and we only need to feed this region into a text recognition model to get the final result.

In this phase, we can have several design directions:

1. **Split the string and recognize each part:**

   Some MRZs consist of two lines, such as the TD2 and TD3 formats, while others have three lines, such as the TD1 format. We can split these texts into parts and recognize them one by one.

   The recognition model needs to handle transforming an image of text into text output. There are many methods to achieve this, such as the early popular CRNN+CTC or the more recent CLIP4STR.

   This method has many drawbacks, such as the need for additional logic to handle two or three lines in the MRZ, or the difficulty in distinguishing text due to narrow spacing in some documents.

2. **Recognize the entire cropped MRZ image at once:**

   Since the aspect ratio of the MRZ region does not vary much, we can crop the entire MRZ region and recognize the entire image at once. This approach is particularly suited for Transformer models.

   For example, if you only want to use the Transformer Encoder architecture, the model design could look like this:

    <div align="center">
      <img src="https://github.com/DocsaidLab/MRZScanner/raw/main/docs/img6.jpg?raw=true" width="80%">
    </div>

   Due to the self-attention mechanism, multiple tokens may point to the same text in the image. In this case, using a standard decoding method might confuse the model:

   > "This image clearly shows this text; why decode it into another text?"

   Using CTC for text decoding works better here, because each token comes from "a specific" image region of text. We only need to merge the output at the final stage to get the final text result.

   ***

   Of course, if you don’t like CTC and find it cumbersome, you can use the Encoder-Decoder architecture. The model design could be as follows:

    <div align="center">
      <img src="https://github.com/DocsaidLab/MRZScanner/raw/main/docs/img7.jpg?raw=true" width="80%">
    </div>

   This approach allows direct decoding of the string without going through an additional CTC layer, as the tokens input to the Decoder are queries for text. Each token is responsible for finding the corresponding text in sequence.

   The Decoder can output in parallel, without the need for autoregression.

   On further reflection, the reason for using autoregression is because we need to "predict the next based on the previous prediction." However, in this case, such an operation is unnecessary.

   Since each MRZ character is independent, the prediction of the first position does not affect the prediction of the second position. All the relevant results are already in the Encoder output, and the Decoder’s job is simply to query them.

   Of course, we’ve actually tested both parallel output and autoregressive training methods. The result was that parallel output converged faster, achieved higher performance, and had better generalization.

### Error Propagation

At this point, we can revisit the issue of corner estimation.

All two-stage models face a common problem: **error propagation**.

We all believe that there is no such thing as a 100% accurate model, because we can never model the statistical population perfectly, so there will always be exceptions to rules and errors in models. Regardless of which method is chosen above, we ultimately face the same challenge:

- **Inaccurate corner estimation**

Due to the inaccurate estimation of the corners, the corrected MRZ region becomes inaccurate. As a result, the MRZ region itself becomes unreliable, which leads to inaccurate text recognition. This creates a classic case of error propagation.

## Single-Stage Recognition Model

The primary challenge of a single-stage model is handling multi-scale features.

The MRZ region changes depending on the angle at which the user takes the photo. This means that before detecting the text, we must perform multi-scale processing on the image.

### Model Architecture

![single-stage](https://github.com/DocsaidLab/MRZScanner/raw/main/docs/img9.jpg?raw=true)

### Backbone

![backbone](https://github.com/DocsaidLab/MRZScanner/raw/main/docs/img8.jpg?raw=true)

Recently, Google released a new paper: **MobileNet-V4**, which optimizes performance on mobile devices, which is great news for us, as we can directly use it.

For this, we are using it as our Backbone, utilizing pre-trained weights from timm, with an input image size of 512 x 512 RGB images.

- [**[24.04] MobileNet-V4: Five years of legacy**](https://docsaid.org/en/papers/lightweight/mobilenet-v4/)
- [**huggingface/pytorch-image-models**](https://github.com/huggingface/pytorch-image-models)

Through testing, we found that with an input image resolution of 512 x 512, the size of each MRZ character is about 4-8 pixels. Lowering the resolution further causes the MRZ text to become blurry, leading to poor recognition results.

### Neck

![neck](https://github.com/DocsaidLab/MRZScanner/raw/main/docs/img10.jpg?raw=true)

To better fuse multi-scale features, we introduced BiFPN. By allowing bidirectional flow of contextual information, BiFPN enhances the expression of features. BiFPN generates a series of scale-rich and semantically strong feature maps, which are highly effective for capturing objects at different scales and positively impact the final prediction accuracy.

In our ablation experiments, we tried removing this part and directly using the feature maps from the Backbone output, but the model failed to train.

- [**[19.11] EfficientDet: BiFPN is the key**](https://docsaid.org/en/papers/feature-fusion/bifpn/)

### Patchify

The previous steps are standard procedures.

Next, we’ll move on to our own creative attempts.

---

First, we need to convert the feature maps from each stage into the input format for a Transformer. Here, we use standard convolution operations to transform the feature maps into patches.

Here are some of our settings:

1. **Patch Size: 4 x 4.**

   We manually measured the size of the MRZ text and found that small text is approximately 4-8 pixels. Smaller than that, the text becomes unreadable. The size of larger text varies based on the shooting distance. Considering these factors, we set the Patch Size to 4 x 4.

2. **Each feature map has a corresponding Patch Embedding and Position Embedding.**

   Since each feature map has a different scale, they cannot share the same Embedding. If they did, the feature maps with different scales would fail to exchange information correctly. We did consider designing a shared Embedding, but it turned out to be more complicated, so we temporarily abandoned the idea.

   We also tested a Shared Weight approach, where all feature maps shared the same Conv2d layer for Embedding, but the results were poor.

### Cross-Attention

Finally, we used Cross-Attention for text recognition.

We randomly initialized 93 tokens.

- **Why 93 tokens?**

This is because the longest MRZ format, TD1, contains 90 characters. TD1 has three lines, so we need 2 "separator" tokens. We also need one "end" token, resulting in a total of 93 tokens.

We use `&` for separator tokens and `[EOS]` for the end token. If there are extra positions, we use `[EOS]` as the boundary, and any subsequent tokens are not supervised. The model is free to predict whatever it likes, and we don’t care about it.

---

For the Transformer decoder, we used the following basic settings:

- Dimensions: 256
- Layers: 6
- Attention heads: 4
- Dropout: 0
- Normalization: Post-LN

The main design philosophy behind this architecture: we provide the Decoder with a "multi-scale" feature space, allowing the Decoder to freely choose features from different scales for text recognition. We don't need to worry about the position of text in the image, as this problem is entirely up to the model to solve.

### And More

Throughout the experimental process, we kept some records that we’ve written down here, which might be helpful for you.

1. **Models with dimensions of 64 and 128 can converge, but each time you halve the dimension, the convergence time doubles.**

   Our training equipment is an RTX4090, and training a model with 256 dimensions takes about 50 hours; a model with 128 dimensions takes about 100 hours; and a model with 64 dimensions takes about 200 hours.

   Why didn’t we try 512 dimensions? Because the model would become too large and exceed 100MB, which is not the size we want.

---

2. **Adding extra branches, such as the Polygon or the center points of the text, can speed up model convergence.**

   But it’s not very practical! Collecting data is already challenging, and finding MRZ regions in the data and labeling them is not suitable for broad application.

   In the end, the convergence effect was similar, with little contribution to the overall performance.

---

3. **Removing the Neck.**

   The model can still converge, but it takes three times longer, so we need to think carefully about it.

---

4. **Removing position encoding.**

   The model fails to converge.

---

5. **Adjusting Weight Decay from $10^{-5}$ to $10^{-2}$.**

   The model converges earlier, but the generalization ability is reduced.

   Smaller models naturally have some regularization effects, so they don’t need a very strong Weight Decay.

---

6. **Using Pre-LN.**

   The model converges earlier, but the generalization ability is reduced.

   Pre-LN slightly reduces the depth of the model, which is not ideal for smaller models.

---

7. **Adding more image augmentations.**

   To speed up the experiments, we controlled the rotation angle of the MRZ images between -45 and 45 degrees.

   We tried full-range rotation and more augmentations, but this scale of model couldn't handle so many augmentations, which led to failure to converge.

## Conclusion

We believe that the current single-stage model design still lacks some key components, and we will continue to read more literature and conduct further experiments.

Perhaps scaling up the model will definitely be the most effective way forward. The challenge lies in how to meet all the requirements above using a "lightweight" parameter scale, which is the issue we need to think about next.

However, as we mentioned earlier, a "two-stage" solution can reliably solve almost all scenarios. If you really want to proceed, we would still recommend going back to developing a two-stage model to avoid unnecessary trouble.

## Citations

We are grateful to all those who have paved the way before us, as their work has been invaluable to our research.

If you find our work helpful, please cite our work:

```bibtex
@misc{yuan2024mrzscanner,
  author = {Ze Yuan},
  title = {MRZScanner},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/DocsaidLab/MRZScanner},
  note = {GitHub repository}
}
```
