import time
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer


class SlidingWindowTrainer(Trainer):
    def calculate_class_weights(self):
        train_labels = self.train_dataset["labels"]
        self.class_weights = np.asarray(
            compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
        ).astype(np.float32)

        print(f"class_weights: {self.class_weights}")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        number_of_chunks = [len(x) for x in input_ids]

        input_ids_combined = []
        for id in input_ids:
            input_ids_combined.extend(id)

        input_ids_combined_tensors = torch.stack(
            [torch.tensor(x).to(self.model.device) for x in input_ids_combined]
        )

        attention_mask_combined = []
        for mask in attention_mask:
            attention_mask_combined.extend(mask)

        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(self.model.device) for x in attention_mask_combined]
        )

        outputs = model(input_ids_combined_tensors, attention_mask_combined_tensors)
        logits = outputs.get("logits")

        logits_split = logits.split(number_of_chunks)

        pooled_logits = torch.cat(
            [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
        )

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights)
        ).to(self.model.device)
        loss = loss_fct(
            pooled_logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        return 0

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )
                loss = loss.mean().detach()

                logits = outputs.get("logits")
                number_of_chunks = [len(x) for x in inputs["input_ids"]]
                logits_split = logits.split(number_of_chunks)

                pooled_logits = torch.cat(
                    [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
                )
                logits = pooled_logits

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()


def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    - num_steps: number of steps processed
    - num_tokens: number of tokens processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if runtime == 0:
        return result
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    if num_tokens is not None:
        tokens_per_second = num_tokens / runtime
        result[f"{split}_tokens_per_second"] = round(tokens_per_second, 3)
    return result


# flos and floating_point_ops is set to only the first input_ids shape... not sure how this will affect the whole thing

# need to modify evaluate function now... it never ends
# theoretically you just need to modify data loader and data collator(?) try this later
# clean up
