from typing import Tuple
from PIL import Image
from pathlib import Path
import numpy as np
from simple_lama_inpainting import SimpleLama
import subprocess
import tempfile


def test_lama(
    images: Tuple[Image.Image, Image.Image, Image.Image], simple_lama: SimpleLama
):
    out = simple_lama(*images[:-1])
    np.testing.assert_array_equal(np.array(out), np.array(images[-1]))


def test_lama_cli(image_paths: Tuple[Path, Path, Path]):
    temp_dir = tempfile.TemporaryDirectory()
    out_file_path = Path(temp_dir.name) / "tmp_out.png"
    subprocess.run(
        ["simple_lama", str(image_paths[0]), str(image_paths[1]), str(out_file_path)],
        check=True,
    )
    np.testing.assert_array_equal(
        np.array(Image.open(out_file_path)), np.array(Image.open(image_paths[-1]))
    )
    temp_dir.cleanup()
