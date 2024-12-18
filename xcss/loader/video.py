from typing import Text
from dependency_injector import containers, providers
from pathlib import Path
import imageio.v3 as iio
from .base import BaseLoader


class VideoLoader(BaseLoader):
    def __init__(self, work_dir: Text, video_path: Text):
        super().__init__(work_dir)
        self.video_path = Path(video_path)
        self._read_write_video()

    def _read_write_video(self):
        video = iio.imread(str(self.video_path), plugin="pyav")
        work_dir = self.work_dir
        images_dir = work_dir / "images"
        self._images_dir = images_dir
        self._length = video.shape[0]

        if not images_dir.exists():
            images_dir.mkdir(parents=True)

        for idx, frame in enumerate(video):
            iio.imwrite(str(images_dir / f"frame_{idx:09d}.png"), frame)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index < 0 or index >= self._length:
            raise IndexError("Index out of range")

        image = iio.imread(str(self._images_dir / f"frame_{index:09d}.png"))
        return self._images_dir / f"frame_{index:09d}.png", image

    def __iter__(self):
        for index in range(self._length):
            yield self[index]

    @property
    def images_dir(self):
        return self._images_dir
