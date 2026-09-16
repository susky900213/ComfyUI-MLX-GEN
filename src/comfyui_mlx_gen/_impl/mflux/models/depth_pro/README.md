# Depth Pro
This directory contains MLX-Gen's MLX implementation of **Depth Pro** (Apple's monocular depth model), used for generating depth maps from images. Depth maps can be used by FLUX depth workflows to constrain image generation.

![Depth Pro Example](../../assets/depth_pro_example.jpg)
*Example images from [Unsplash](https://unsplash.com/photos/VotK70bRo0U) and [Unsplash](https://unsplash.com/photos/Q3QJbt9f54g)*

## Export a depth map
Depth Pro weights are downloaded explicitly:

```sh
mlxgen download --model depth-pro
```

To generate and export the depth map from an image without running image generation, use the Python API:

<details>
<summary>Python API</summary>

```python
from mflux.models.depth_pro.model.depth_pro import DepthPro

model = DepthPro(quantize=8)
result = model.create_depth_map("your_image.jpg")
result.depth_image.save("your_image_depth.png")
```
</details>

This will create a depth map and save it with the same name as your image but with a `_depth` suffix (e.g., `your_image_depth.png`).

## Notes
- Quantization is supported for the Depth Pro model, however output quality can vary a lot depending on the input image.

> [!WARNING]
> Note: The Depth Pro model requires an additional ~1.9GB download from Apple. MLX-Gen will not download it during generation; run `mlxgen download --model depth-pro` first.
