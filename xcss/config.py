"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from .model import XCSSModelConfig
from .dataparser import XCSSDataParserConfig
from .datamanager import XCSSDataManagerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

xcss = MethodSpecification(
    TrainerConfig(
        method_name="xcss",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=XCSSDataManagerConfig(
                dataparser=XCSSDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=XCSSModelConfig(
                sh_degree=10,
                output_depth_during_training=True,
                cull_alpha_thresh=0.005,
                cull_scale_thresh=2.,
                cull_screen_size=0.5,
                num_downscales=0,
                refine_every=100,
                warmup_length=5000,
                reset_alpha_every=100,
                stop_split_at=50000,
                n_split_samples=8,
                max_gauss_ratio=40.,
                dump_illumination=True
            ),
        ),
        gradient_accumulation_steps={
            "means": 64,
            "features_dc": 64,
            "features_rest": 1,
            "view_shs": 1,
            "opacities": 1,
            "scales": 64,
            "quats": 64,
            "camera_opt": 1,
            "bilateral_grid": 1,
        },
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "view_shs": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="eXpandable Crowdsourced Scene Splatting (XCSS)",
)
