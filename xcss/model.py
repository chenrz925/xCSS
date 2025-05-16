"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from pathlib import Path

import torch
from torch.nn import Parameter
from torch.nn import functional as F
from torchvision.io import write_png
from pytorch_msssim import SSIM

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.model_components.lib_bilagrid import (
    BilateralGrid,
    total_variation_loss,
)
from nerfstudio.utils.colors import get_color
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import get_viewmat
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.rendering import rasterization
from msplat.compute_sh import compute_sh
from kornia.geometry.quaternion import Quaternion

from .strategy import XCSSStrategy


@dataclass
class XCSSModelConfig(SplatfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: XCSSModel)    
    dump_illumination: bool = False
    depth_constraint: bool = False

def init_scales_quats(indices, means, distances):
    with torch.no_grad():
        indiced_means = means.data[indices]
        _, scales, rotations = torch.svd(indiced_means)
        # scales = scales.sqrt()
        scales = torch.where(scales > scales.mean(0, keepdim=True) + 3 * scales.std(0, keepdim=True), scales.mean(0, keepdim=True), scales)
        scales = torch.clip(scales, scales.median(1).values[..., None], scales.min(1).values[..., None])
        scales = torch.log(scales)
        quats = Quaternion.from_matrix(rotations.transpose(-1, -2)).data.data
        
        return scales, quats

class XCSSModel(SplatfactoModel):
    """Template Model."""

    config: XCSSModelConfig

    # @torch.no_grad()
    # def render_illumination(
    #     self,
    #     fx: float,
    #     fy: float,
    #     cx: float,
    #     cy: float,
    #     depth: torch.Tensor,  # (H, W)
    #     viewmat: torch.Tensor,  # (4, 4)
    # ):
    #     """基于深度图生成视线方向并计算光照
        
    #     Args:
    #         fx, fy: 焦距
    #         cx, cy: 主点坐标
    #         depth: 深度图 (H, W)
    #         viewmat: 视图矩阵 (4, 4)
        
    #     Returns:
    #         illumination: 光照图 (H, W, 3)
    #     """
    #     H, W = depth.shape
    #     device = depth.device
        
    #     # 生成像素坐标网格
    #     i, j = torch.meshgrid(torch.arange(H, device=device),
    #                         torch.arange(W, device=device), 
    #                         indexing="ij")  # (H, W)
        
    #     # 转换为相机坐标系3D坐标
    #     x = (j - cx) * depth / fx  # X分量 (H, W)
    #     y = (i - cy) * depth / fy  # Y分量 (H, W)
    #     z = depth                  # Z分量 (H, W)
    #     points_cam = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
        
    #     # 转换到世界坐标系
    #     homogeneous_points = torch.cat([
    #         points_cam.reshape(-1, 3), 
    #         torch.ones((H*W, 1), device=device)
    #     ], dim=-1)  # (H*W, 4)
    #     points_world = (torch.inverse(viewmat) @ homogeneous_points.T).T[..., :3]  # (H*W, 3)
        
    #     # 计算视线方向（从相机原点指向场景点）
    #     cam_pos = torch.inverse(viewmat)[:3, 3]  # 相机原点位置
    #     view_dirs = points_world - cam_pos       # (H*W, 3)
    #     view_dirs = F.normalize(view_dirs, p=2, dim=-1).reshape(H, W, 3)  # (H, W, 3)
        
    #     # 计算球谐光照
    #     illu_sh = self.view_shs[0][None, None, ...].expand(H, W, *self.view_shs.shape[-2:])  # (H, W, 3, sh_dim)
    #     illu_sh = illu_sh.permute(0, 1, 3, 2)  # (H, W, sh_dim, 3)
    #     illu = F.softplus(compute_sh(illu_sh, view_dirs + 1e-5))  # (H, W, 3)
        
    #     return illu

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter(
                (torch.rand((self.config.num_random, 3)) - 0.5)
                * self.config.random_scale
            )
        distances, indices = k_nearest_sklearn(means.data, 5)
        # find the average of the three nearest neighbors for each point and use that as the scale
        scales, quats = init_scales_quats(indices, means, distances)
        scales = torch.nn.Parameter(scales)
        quats = torch.nn.Parameter(quats)
        # num_points = means.shape[0]
        # avg_dist = distances.mean(dim=-1, keepdim=True)
        # scales1 = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        # quats1 = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = (self.config.sh_degree + 1) ** 2

        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            features_dc = torch.nn.Parameter(
                # (self.seed_points[1] / 255.0).clone().to(torch.float32)
                torch.ones_like(self.seed_points[1], dtype=torch.float32)
            )
            features_rest = torch.nn.Parameter(
                torch.zeros(num_points, 0, 3, dtype=torch.float32)
            )
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(
                torch.zeros(num_points, 0, 3, dtype=torch.float32)
            )

        opacities = torch.nn.Parameter(torch.logit(0.5 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        view_shs = torch.zeros(self.num_train_data, dim_sh, 3)
        view_shs[:, 0, :] = RGB2SH(
            0.5 + torch.log(torch.e - torch.ones(self.num_train_data, 3))
            # torch.ones(self.num_train_data, 3)
        )
        self.view_shs = torch.nn.Parameter(view_shs)

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

        # Strategy for GS densification
        if self.config.strategy == "default":
            # Strategy for GS densification
            self.strategy = XCSSStrategy(
                prune_opa=self.config.cull_alpha_thresh,
                grow_grad2d=self.config.densify_grad_thresh,
                grow_scale3d=self.config.densify_size_thresh,
                grow_scale2d=self.config.split_screen_size,
                prune_scale3d=self.config.cull_scale_thresh,
                prune_scale2d=self.config.cull_screen_size,
                refine_scale2d_stop_iter=self.config.stop_screen_size_at,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                reset_every=self.config.reset_alpha_every * self.config.refine_every,
                refine_every=self.config.refine_every,
                pause_refine_after_reset=self.num_train_data + self.config.refine_every,
                absgrad=self.config.use_absgrad,
                revised_opacity=False,
                verbose=True,
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        elif self.config.strategy == "mcmc":
            self.strategy = MCMCStrategy(
                cap_max=self.config.max_gs_num,
                noise_lr=self.config.noise_lr,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                refine_every=self.config.refine_every,
                min_opacity=self.config.cull_alpha_thresh,
                verbose=False,
            )
            self.strategy_state = self.strategy.initialize_state()
        else:
            raise ValueError(
                f"""Splatfacto does not support strategy {self.config.strategy} 
                             Currently, the supported strategies include default and mcmc."""
            )
        if self.config.dump_illumination:
            self.illu_counter = {}
            self.dump_illumination_per = 20


    @torch.no_grad()
    def view_dirs(self, width, height, fx, fy, cx, cy):
        """生成相机坐标系下的单位视线方向向量
        Args:
            width: 图像宽度（像素）
            height: 图像高度（像素）
            fx, fy: 焦距（像素单位）
            cx, cy: 主点坐标（像素单位）

        Returns:
            directions: (H, W, 3) 的Tensor，每个像素对应的单位视线方向
        """
        # 生成像素坐标网格
        i, j = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

        # 转换为相机坐标系下的坐标
        x = (j - cx) / fx  # X方向分量
        y = (i - cy) / fy  # Y方向分量
        z = torch.ones_like(x)  # Z方向分量设为1

        # 组合成方向向量并归一化
        directions = torch.stack([x, y, z], dim=-1)
        directions = F.normalize(directions, p=2, dim=-1)
        directions = directions.reshape(-1, 3)

        return directions.to(torch.float32)

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color,
                )
        else:
            crop_ids = torch.ones(self.opacities.shape[0], dtype=torch.bool, device=self.opacities.device)
        
        crop_ids = crop_ids & torch.isfinite(self.means).any(-1) & torch.isfinite(self.scales).any(-1)

        opacities_crop = self.opacities[crop_ids]
        means_crop = self.means[crop_ids]
        features_dc_crop = self.features_dc[crop_ids]
        scales_crop = self.scales[crop_ids]
        quats_crop = self.quats[crop_ids]

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        if self.training:
            homogeneous_means = torch.cat(
                [
                    means_crop,
                    torch.ones((means_crop.shape[0], 1), device=means_crop.device),
                ],
                dim=-1,
            )
            camera_coords = (viewmat.squeeze(0) @ homogeneous_means.T).T[
                ..., :3
            ]  # [N, 3]
            view_dirs = F.normalize(camera_coords, p=2, dim=-1)  # [N, 3]
            illu_sh = (
                self.view_shs[camera.metadata["cam_idx"]][None]
                .expand(view_dirs.shape[0], *self.view_shs.shape[-2:])
                .permute(0, 2, 1)
            )
            illu = F.softplus(compute_sh(illu_sh, view_dirs + 1e-5))
            colors_crop = F.relu(features_dc_crop) * illu
            # del illu_sh, view_dirs
        else:
            colors_crop = features_dc_crop
            illu = None

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=None,
            sparse_grad=False,
            absgrad=self.strategy.absgrad
            if isinstance(self.strategy, DefaultStrategy)
            else False,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        if self.training and self.config.dump_illumination:
            self.illu_counter[camera.metadata["cam_idx"]] = self.illu_counter.get(camera.metadata["cam_idx"], 0) + 1
            if self.illu_counter[camera.metadata["cam_idx"]] % self.dump_illumination_per == 0:
                with torch.no_grad():
                    illu_render, _, _ = rasterization(
                        means=means_crop,
                        quats=quats_crop,  # rasterization does normalization internally
                        scales=torch.exp(scales_crop),
                        opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                        colors=illu,
                        viewmats=viewmat,  # [1, 4, 4]
                        Ks=K,  # [1, 3, 3]
                        width=W,
                        height=H,
                        packed=False,
                        near_plane=0.01,
                        far_plane=1e10,
                        render_mode='RGB',
                        sh_degree=None,
                        sparse_grad=False,
                        absgrad=self.strategy.absgrad
                        if isinstance(self.strategy, DefaultStrategy)
                        else False,
                        rasterize_mode=self.config.rasterize_mode,
                        # set some threshold to disregrad small gaussians for faster rendering.
                        # radius_clip=3.0,
                    )
                    
                    illu_dir = Path.cwd() / "illumination" / f'L{self.config.sh_degree}'
                    if not illu_dir.exists():
                        illu_dir.mkdir(parents=True, exist_ok=True)
                    
                    illu_render = illu_render / illu_render.max()
                    write_png(
                        (illu_render.squeeze().permute(2, 0, 1) * 255).to(torch.uint8).cpu(),
                        str(illu_dir / f"{camera.metadata['cam_idx']}.png"),
                    )

                    ref_render, _, _ = rasterization(
                        means=means_crop,
                        quats=quats_crop,  # rasterization does normalization internally
                        scales=torch.exp(scales_crop),
                        opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                        colors=F.relu(features_dc_crop),
                        viewmats=viewmat,  # [1, 4, 4]
                        Ks=K,  # [1, 3, 3]
                        width=W,
                        height=H,
                        packed=False,
                        near_plane=0.01,
                        far_plane=1e10,
                        render_mode='RGB',
                        sh_degree=None,
                        sparse_grad=False,
                        absgrad=self.strategy.absgrad
                        if isinstance(self.strategy, DefaultStrategy)
                        else False,
                        rasterize_mode=self.config.rasterize_mode,
                    )
                    ref_dir = Path.cwd() / "illumination" / f'R{self.config.sh_degree}'
                    if not ref_dir.exists():
                        ref_dir.mkdir(parents=True, exist_ok=True)
                    write_png(
                        (torch.clamp(ref_render, 0, 1).squeeze().permute(2, 0, 1) * 255).to(torch.uint8).cpu(),
                        str(ref_dir / f"{camera.metadata['cam_idx']}.png"),
                    )
                    im_dir = Path.cwd() / "illumination" / f"P{self.config.sh_degree}"
                    if not im_dir.exists():
                        im_dir.mkdir(parents=True, exist_ok=True)
                    write_png(
                        (torch.clamp(render[..., :3], 0., 1.).squeeze().permute(2, 0, 1) * 255).to(torch.uint8).cpu(),
                        str(im_dir / f"{camera.metadata['cam_idx']}.png"),
                    )


        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params,
                self.optimizers,
                self.strategy_state,
                self.step,
                self.info,
            )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, 0.0).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            # "depth": None,
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        if self.config.use_bilateral_grid:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        self.camera_optimizer.get_param_groups(param_groups=gps)
        gps["view_shs"] = [self.view_shs]
        return gps

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        if outputs['depth'] is not None and self.config.depth_constraint:
            with torch.no_grad():
                gt_depth = F.interpolate(
                    batch["depth"].unsqueeze(0).unsqueeze(0),
                    size=gt_img.shape[:2],
                    mode="area",
                ).squeeze()
                gt_depth = torch.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)
            pred_depth = outputs["depth"].squeeze(-1)

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        if self.config.dump_illumination and self.illu_counter[batch['image_idx']] == 1:
            image_dir = Path.cwd() / "illumination" / f"I{self.config.sh_degree}"
            if not image_dir.exists():
                image_dir.mkdir(parents=True, exist_ok=True)
            write_png(
                (gt_img.permute(2, 0, 1) * 255).to(torch.uint8).cpu(),
                str(image_dir / f"{batch['image_idx']}.png"),
            )

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]
        )
        if outputs['depth'] is not None and self.config.depth_constraint:
            abs_rel = torch.abs(gt_depth - pred_depth) / gt_depth
            abs_rel = torch.nan_to_num(
                abs_rel[(gt_depth > 1e-5) & (mask.squeeze(-1) > 0.5)].mean(), 0.0
            )

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        if outputs['depth'] is not None and self.config.depth_constraint:
            loss_dict = {
                "main_loss": (1 - self.config.ssim_lambda) * Ll1
                + self.config.ssim_lambda * simloss
                + abs_rel,
                "scale_reg": scale_reg,
            }
        else:
            loss_dict = {
                "main_loss": (1 - self.config.ssim_lambda) * Ll1
                + self.config.ssim_lambda * simloss,
                "scale_reg": scale_reg,
            }

        # Losses for mcmc
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                mcmc_opacity_reg = (
                    self.config.mcmc_opacity_reg
                    * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
                loss_dict["mcmc_opacity_reg"] = mcmc_opacity_reg
            if self.config.mcmc_scale_reg > 0.0:
                mcmc_scale_reg = (
                    self.config.mcmc_scale_reg
                    * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                )
                loss_dict["mcmc_scale_reg"] = mcmc_scale_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict
