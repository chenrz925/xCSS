import pytest
import tempfile
import numpy as np
from imageio import v3 as iio
from random import choice
from shutil import rmtree
from pathlib import Path
from xcss.loader.image import ImageLoader


def test_image_loader():
    work_dir = tempfile.mkdtemp(prefix="xCSS_")
    image_dir = Path(tempfile.mkdtemp(prefix="xCSS_"))
    exts = [".jpg", ".png", ".bmp"]

    for idx in range(10):
        ext = choice(exts)
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        image_path = image_dir / f"{idx}{ext}"
        iio.imwrite(str(image_path), image)

    loader = ImageLoader(work_dir, str(image_dir), exts)

    assert loader.work_dir.exists()
    assert loader.images_dir.exists()
    assert len(loader) == 10
    assert len(tuple(loader.images_dir.glob("*.png"))) == 10
    assert loader[0][1].shape == (720, 1280, 3)

    rmtree(work_dir)
    rmtree(image_dir)
