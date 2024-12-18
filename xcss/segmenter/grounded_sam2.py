from typing import Text
from collections import namedtuple
from dependency_injector import containers, providers
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class GroundedSAM2Segmenter(object):
    GroundingDINO = namedtuple("GroundingDINO", ["processor", "model"])
    SAM2 = namedtuple("SAM2", ["video_predictor", "image_predictor"])

    def __init__(
        self,
        sam2_ckpt_path: Text,
        sam2_config: Text,
        grounding_dino_config: Text,
        device: Text = "cuda",
    ):
        self._grounding_dino = self._create_grounding_dino(grounding_dino_config)
        self._sam2 = self._create_sam2(sam2_ckpt_path, sam2_config)
        self._device = device

    def _create_grounding_dino(self, grounding_dino_config: Text):
        processor = AutoProcessor.from_pretrained(grounding_dino_config)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_dino_config
        )
        return self.GroundingDINO(processor=processor, model=model)

    def _create_sam2(self, sam2_ckpt_path: Text, sam2_config: Text):
        video_predictor = build_sam2_video_predictor(sam2_config, sam2_ckpt_path)
        image_model = build_sam2(sam2_config, sam2_ckpt_path)
        image_predictor = SAM2ImagePredictor(image_model)
        return self.SAM2(video_predictor=video_predictor, image_predictor=image_predictor)

    def track(self, video_path: Text, text: Text):
        pass


SEGMENTER_CLASSES = {"grounded_sam2": GroundedSAM2Segmenter}
