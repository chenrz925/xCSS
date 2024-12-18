from typing import Text, List
from pathlib import Path
from imageio import v3 as iio

from .base import BaseLoader


class ImageLoader(BaseLoader):
    def __init__(
        self, work_dir: Text, images_dir: Text, ext: List[Text], *args, **kwargs
    ):
        super().__init__(work_dir, *args, **kwargs)
        self._source_images_dir = Path(images_dir)
        self._ext = ext
        self._images = []

        # Ensure the target images directory exists
        self._images_dir = self.work_dir / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)

        # Process images: filter, convert, and rename
        self._process_images()

    def _process_images(self):
        # Collect all image files matching the extensions
        image_files = [
            file
            for file in self._source_images_dir.iterdir()
            if file.suffix.lower() in self._ext and file.is_file()
        ]

        for idx, image_file in enumerate(sorted(image_files)):
            target_file = self._images_dir / f"frame_{idx:09d}.png"

            # Read and write image using imageio
            img = iio.imread(image_file)
            iio.imwrite(target_file, img)

            self._images.append(target_file)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        if index < 0 or index >= len(self._images):
            raise IndexError("Index out of range")

        image = iio.imread(self._images[index])

        return self._images[index], image

    def __iter__(self):
        return iter(self._images)

    @property
    def images_dir(self):
        return self._images_dir
