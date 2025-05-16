from typing import Dict, Literal
from nerfstudio.data.datasets.base_dataset import InputDataset
import torch
from imageio import v3 as iio


class XCSSDataset(InputDataset):
    def get_data(
        self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32"
    ) -> Dict:
        data_item = super().get_data(image_idx, image_type)

        if (
            "depth_filenames" in self.metadata
            and self.metadata["depth_filenames"] is not None
        ):
            depth_path = self.metadata["depth_filenames"][image_idx]
            depth = iio.imread(str(depth_path))
            data_item["depth"] = torch.from_numpy(depth)

        return data_item
