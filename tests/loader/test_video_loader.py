import pytest
import tempfile
from shutil import rmtree
from pathlib import Path
from xcss.loader.video import VideoLoader


def test_video_loader():
    work_dir = tempfile.mkdtemp(prefix="xCSS_")

    loader = VideoLoader(work_dir, "imageio:cockatoo.mp4")

    assert len(loader) == len(tuple(loader.images_dir.glob("*.png")))
    assert loader[0][1].shape == (720, 1280, 3)

    rmtree(work_dir)
