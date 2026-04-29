import os
import torch
import importlib
from torch import nn


class CaptionNet(nn.Module):

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_detector is True:
            self.detector.eval()
            for param in self.detector.parameters():
                param.requires_grad = False
        return self

    def pretrained_parameters(self):
        if hasattr(self.captioner, 'pretrained_parameters'):
            return self.captioner.pretrained_parameters()
        else:
            return []

    def __init__(self, args, dataset_config, train_dataset):
        super(CaptionNet, self).__init__()

        self.freeze_detector = args.freeze_detector
        self.detector = None
        self.captioner = None

        if args.detector is not None:
            detector_name = args.detector
            requested_module = f"models.{detector_name}"
            try:
                detector_module = importlib.import_module(
                    f"{requested_module}.CoCa3d"
                )
            except ModuleNotFoundError as exc:
                if exc.name not in {requested_module, f"{requested_module}.CoCa3d"}:
                    raise
                fallback = "3D_coca"
                fallback_dir = os.path.join(os.path.dirname(__file__), fallback)
                if detector_name != fallback and os.path.isdir(fallback_dir):
                    detector_name = fallback
                    args.detector = detector_name
                    detector_module = importlib.import_module(
                        f"models.{detector_name}.CoCa3d"
                    )
                else:
                    raise
            self.detector = detector_module.detector(args, dataset_config,train_dataset)

        if args.captioner is not None:
            captioner_module = importlib.import_module(
                f'models.{args.captioner}.captioner'
            )
            self.captioner = captioner_module.captioner(args, train_dataset)

        self.train()

    def forward(self, batch_data_label: dict, is_eval: bool = False) -> dict:

        outputs = {'loss': torch.zeros(1)[0].cuda()}

        if self.detector is not None:
            if self.freeze_detector is True:
                outputs = self.detector(batch_data_label, is_eval=True)
            else:
                outputs = self.detector(batch_data_label, is_eval=is_eval)

        if (
            not is_eval
            and isinstance(outputs, dict)
            and "loss" not in outputs
            and "box_outputs" in outputs
            and isinstance(outputs["box_outputs"], dict)
            and "loss" in outputs["box_outputs"]
        ):
            outputs["loss"] = outputs["box_outputs"]["loss"]

        if self.freeze_detector is True:
            outputs['loss'] = torch.zeros(1)[0].cuda()

        # if self.captioner is not None:
        #     outputs = self.captioner(outputs, batch_data_label, is_eval=is_eval)
        # else:
        #     batch, nproposals, _, _ = outputs['box_outputs']['box_corners'].shape
        #     outputs['lang_cap'] = [
        #                               ["this is a valid match!"] * nproposals
        #                           ] * batch
        return outputs
