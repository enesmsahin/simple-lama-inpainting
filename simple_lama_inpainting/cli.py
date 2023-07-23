from simple_lama_inpainting.models.model import SimpleLama
from PIL import Image
from pathlib import Path
import fire


def main(image_path: str, mask_path: str, out_path: str | None = None):
    """Apply lama inpainting using given image and mask.

    Args:
        img_path (str): Path to input image (RGB)
        mask_path (str): Path to input mask (Binary 1-CH Image.
                        Pixels with value 255 will be inpainted)
        out_path (str, optional): Optional output imaga path.
                        If not provided it will be saved to the same
                            path as input image.
                        Defaults to None.
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    assert img.mode == "RGB" and mask.mode == "L"

    lama = SimpleLama()
    result = lama(img, mask)
    if out_path is None:
        out_path = image_path.with_stem(image_path.stem + "_out")

    Path.mkdir(Path(out_path).parent, exist_ok=True, parents=True)
    result.save(out_path)
    print(f"Inpainted image is saved to {out_path}")


def lama_cli():
    fire.Fire(main)


if __name__ == "__main__":
    fire.Fire(main)
