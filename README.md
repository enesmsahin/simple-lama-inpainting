# simple-lama-inpainting

<div align="center">
Simple pip package for LaMa[1] inpainting.<br>
<a href="https://badge.fury.io/py/simple-lama-inpainting"><img src="https://badge.fury.io/py/simple-lama-inpainting.svg" alt="PyPI version" height="18"></a>
</div>

## Installation
```
pip install simple-lama-inpainting
```

## Usage
### CLI
```
simple_lama <path_to_input_image> <path_to_mask_image> <path_to_output_image>
```

### Integration to Your Code
Input formats: `np.ndarray` or `PIL.Image.Image`. (3 channel input image & 1 channel binary mask image where pixels with 255 will be inpainted). \
Output format: `PIL.Image.Image`
```python
from simple_lama_inpainting import SimpleLama
from PIL import Image

simple_lama = SimpleLama()

img_path = "image.png"
mask_path = "mask.png"

image = Image.open(img_path)
mask = Image.open(mask_path)

result = simple_lama(image, mask)
result.save("inpainted.png")
```

## Sources
[1] Suvorov, R., Logacheva, E., Mashikhin, A., Remizova, A., Ashukha, A., Silvestrov, A., Kong, N., Goka, H., Park, K., & Lempitsky, V. (2021). Resolution-robust Large Mask Inpainting with Fourier Convolutions. arXiv preprint arXiv:2109.07161. \
[2] https://github.com/saic-mdal/lama \
[3] https://github.com/Sanster/lama-cleaner
