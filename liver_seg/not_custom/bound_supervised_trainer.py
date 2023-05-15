####
# CHANG 20210817
####
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

#from monai.config import IgniteInfo
from monai.engines import SupervisedTrainer
from monai.engines.utils import (
        IterationEvents,
        )
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import PT_BEFORE_1_7, min_version, optional_import

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ignite.metrics import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", "0.4.4", min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", "0.4.4", min_version, "EventEnum")

class CommonKeys:
    IMAGE = "image"
    LABEL = "label"
    PRED = "pred"
    LOSS = "loss"
    DISTMAP = "dist_map"

def custom_prepare_batch(
        batchdata: Dict[str, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:

    if not isinstance(batchdata, dict):
        raise AssertionError("default prepare_batch expects dictionary input data")
    if isinstance(batchdata.get(CommonKeys.DISTMAP), torch.Tensor):
        return(
                batchdata[CommonKeys.IMAGE].to(device=device, non_blocking=non_blocking),
                batchdata[CommonKeys.LABEL].to(device=device, non_blocking=non_blocking),
                batchdata[CommonKeys.DISTMAP].to(device=device, non_blocking=non_blocking),
                )
    if isinstance(batchdata.get(CommonKeys.LABEL), torch.Tensor):
        return(
                batchdata[CommonKeys.IMAGE].to(device=device, non_blocking=non_blocking),
                batchdata[CommonKeys.LABEL].to(device=device, non_blocking=non_blocking),
                None
                )
    return batchdata[CommonKeys.IMAGE].to(device=device, non_blocking=non_blocking), None, None

class CustomSupervisedTrainer(SupervisedTrainer):
    def __init__(self, 
            device: torch.device, 
            max_epochs: int,
            train_data_loader: Union[Iterable, DataLoader], 
            network: torch.nn.Module,
            optimizer: Optimizer,
            loss_function: Callable,
            epoch_length: Optional[int] = None,
            non_blocking: bool = False,
            prepare_batch: Callable = custom_prepare_batch,
            iteration_update: Optional[Callable] = None,
            inferer: Optional[Inferer] = None,
            post_transform: Optional[Transform] = None,
            key_train_metric: Optional[Dict[str, Metric]] = None,
            additional_metrics: Optional[Dict[str, Metric]] = None,
            #metric_cmp_fn: Callable = default_metric_cmp_fn,
            train_handlers: Optional[Sequence] = None,
            amp: bool = False,
            event_names: Optional[List[Union[str, EventEnum]]] = None,
            event_to_attr: Optional[dict] = None,
            decollate: bool = True,
            optim_set_to_none: bool = False,
            ) -> None:
        super().__init__(
                device=device,
                max_epochs=max_epochs,
                train_data_loader=train_data_loader,
                epoch_length=epoch_length,
                non_blocking=non_blocking,
                prepare_batch=prepare_batch,
                iteration_update=iteration_update,
                post_transform=post_transform,
                key_metric=key_train_metric,
                additional_metrics=additional_metrics,
                #metric_cmp_fn=metric_cmp_fn,
                handlers=train_handlers,
                amp=amp,
                event_names=event_names,
                event_to_attr=event_to_attr,
                decollate=decollate,
                )

        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inferer = SimpleInferer() if inferer is None else inferer
        self.optim_set_to_none = optim_set_to_none

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 3:
            inputs, targets, distmap = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, distmap, args, kwargs = batch

        # put iteration outputs into engine.state
        engine.state.output = {CommonKeys.IMAGE: inputs, CommonKeys.LABEL: targets}

        def _compute_pred_loss():
            engine.state.output[CommonKeys.PRED] = self.inferer(inputs, self.network, *args, **kwargs)
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            engine.state.output[CommonKeys.LOSS] = self.loss_function(engine.state.output[CommonKeys.PRED], targets).mean()
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

        self.network.train()
        # `set_to_none` only work from PyTorch 1.7.0
        if PT_BEFORE_1_7:
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad(set_to_none=self.optim_set_to_none)

        if self.amp and self.scalar is not None:
            with torch.cuda.amp.autocast():
                _compute_pred_loss()
            self.scalar.scale(engine.state.output[CommonKeys.LOSS]).backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            self.scalar.step(self.optimizer)
            self.scalar.update()
        else:
            _compute_pred_loss()
            engine.state.output[CommonKeys.LOSS].backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            self.optimizer.step()
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output
