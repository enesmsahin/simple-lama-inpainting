import pytest
import torch
from pathlib import Path
from PIL import Image
from typing import Tuple
from simple_lama_inpainting import SimpleLama


@pytest.fixture(scope="session")
def image_paths() -> Tuple[Path, Path, Path]:
    image_path = Path(__file__).parent / "data" / "image_1.png"
    mask_path = Path(__file__).parent / "data" / "mask_1.png"
    out_path = Path(__file__).parent / "data" / "out_1.png"

    return image_path, mask_path, out_path


@pytest.fixture(scope="session")
def images(
    image_paths: Tuple[Path, Path, Path]
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    image = Image.open(image_paths[0])
    mask = Image.open(image_paths[1])
    out = Image.open(image_paths[2])

    return image, mask, out


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def simple_lama(device) -> SimpleLama:
    return SimpleLama(device)
