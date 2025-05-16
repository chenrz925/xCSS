import torch
from typing import Any, Dict, Union
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import reset_opa

class XCSSStrategy(DefaultStrategy):
    pass
    # def step_post_backward(
    #     self,
    #     params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    #     optimizers: Dict[str, torch.optim.Optimizer],
    #     state: Dict[str, Any],
    #     step: int,
    #     info: Dict[str, Any],
    #     packed: bool = False,
    # ):
    #     """Callback function to be executed after the `loss.backward()` call."""
    #     if step >= self.refine_stop_iter:
    #         return

    #     self._update_state(params, state, info, packed=packed)

    #     if (
    #         step > self.refine_start_iter
    #         and step % self.refine_every == 0
    #         and step % self.reset_every >= self.pause_refine_after_reset
    #     ):
    #         # grow GSs
    #         n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
    #         if self.verbose:
    #             print(
    #                 f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
    #                 f"Now having {len(params['means'])} GSs."
    #             )

    #         # prune GSs
    #         n_prune = self._prune_gs(params, optimizers, state, step)
    #         if self.verbose:
    #             print(
    #                 f"Step {step}: {n_prune} GSs pruned. "
    #                 f"Now having {len(params['means'])} GSs."
    #             )

    #         # reset running stats
    #         state["grad2d"].zero_()
    #         state["count"].zero_()
    #         if self.refine_scale2d_stop_iter > 0:
    #             state["radii"].zero_()
    #         torch.cuda.empty_cache()

    #     if step % self.reset_every == 0:
    #         reset_opa(
    #             params=params,
    #             optimizers=optimizers,
    #             state=state,
    #             value=0.5,
    #         )