"""Controlled replay of the TIAToolbox 1.6/2.1 MicroNet padding difference.

This diagnostic changes only the out-of-image padding constant (0 or 255).
It does not alter the primary 2.1.3 benchmark or tune a decoding parameter.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import micronet_consep as old
from . import micronet_consep_v2 as current


def run(limit: int | None) -> None:
    import torch
    from scipy.io import loadmat
    from tiatoolbox.models.architecture import get_pretrained_model
    from tiatoolbox.tools.patchextraction import PatchExtractor
    from tiatoolbox.utils.metrics import dice
    from tiatoolbox.wsicore.wsireader import WSIReader

    runtime = current.runtime()
    data = current.inputs()
    model, config = get_pretrained_model("micronet-consep", old.CHECKPOINT)
    model = model.to("cuda").eval()
    inputs, outputs = PatchExtractor.get_coordinates(
        image_shape=(1000, 1000),
        patch_input_shape=config.patch_input_shape,
        patch_output_shape=config.patch_output_shape,
        stride_shape=config.stride_shape,
    )
    if not np.array_equal(inputs, outputs) or len(inputs) != 49:
        raise ValueError("Unexpected registered MicroNet tiling")
    rows = []
    for item in data[:limit]:
        name = item["id"]
        gt = loadmat(old.TEST / "Labels" / f"{name}.mat")["inst_map"] > 0
        reader = WSIReader.open(old.TEST / "Images" / f"{name}.png", mpp=0.25)
        row = {"id": name, "padding": {}}
        for padding, reference_path in (
            (0, old.BASE / "probabilities" / f"{name}.npy"),
            (255, current.BASE / "probabilities" / f"{name}.npy"),
        ):
            total = np.zeros((1000, 1000, 2), dtype=np.float32)
            count = np.zeros((1000, 1000, 1), dtype=np.float32)
            for start in range(0, len(inputs), 4):
                patches = [
                    model.preproc(
                        reader.read_bounds(
                            bounds,
                            resolution=0.25,
                            units="mpp",
                            coord_space="resolution",
                            pad_constant_values=padding,
                        )
                    )
                    for bounds in inputs[start : start + 4]
                ]
                predictions = model.infer_batch(model, torch.from_numpy(np.stack(patches)), device="cuda")[0]
                for pred, (x1, y1, x2, y2) in zip(predictions, outputs[start : start + 4], strict=True):
                    x2, y2 = min(x2, 1000), min(y2, 1000)
                    total[y1:y2, x1:x2] += pred[: y2 - y1, : x2 - x1]
                    count[y1:y2, x1:x2] += 1
            if not np.all(count > 0):
                raise ValueError("Patch replay did not cover the image")
            probabilities = total / count
            reference = np.load(reference_path)
            error = np.abs(probabilities - reference)
            replay_instances = current.postprocess(model, probabilities)
            reference_instances = current.postprocess(model, reference)
            row["padding"][str(padding)] = {
                "reference_sha256": old.digest(reference_path),
                "max_absolute_error": float(error.max()),
                "mean_absolute_error": float(error.mean()),
                "argmax_disagreement_pixels": int((probabilities.argmax(-1) != reference.argmax(-1)).sum()),
                "replay_dice": float(dice(gt, replay_instances > 0)),
                "reference_dice": float(dice(gt, reference_instances > 0)),
            }
        rows.append(row)
        print(f"{name}: {row['padding']}", flush=True)
    current.write_json(
        current.BASE / "padding_audit.json",
        {
            "runtime": runtime,
            "checkpoint_sha256": old.CHECKPOINT_SHA256,
            "inputs": data,
            "cases": rows,
            "complete": len(rows) == 14,
            "patch_input_shape": list(config.patch_input_shape),
            "stride_shape": list(config.stride_shape),
            "old_padding_source": "TIAToolbox 1.6.0 WSIStreamDataset.__getitem__: pad_constant_values=0",
            "new_padding_source": "TIAToolbox 2.1.3 WSIPatchDataset.__getitem__: pad_constant_values=255",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 14:
        parser.error("--limit must be between 1 and 14")
    run(args.limit)
