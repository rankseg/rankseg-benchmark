#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${RANKSEG_NNUNET_PYTHON:-python}"
WORKSPACE="${RANKSEG_NNUNET_WORKSPACE:-$ROOT}"
export nnUNet_raw_data_base="$WORKSPACE/work/v1/nnUNet_raw_data_base"
export nnUNet_preprocessed="$WORKSPACE/work/v1/nnUNet_preprocessed"
export RESULTS_FOLDER="$WORKSPACE/work/v1/RESULTS_FOLDER"
export MPLCONFIGDIR=/tmp/matplotlib_task008_pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model="$WORKSPACE/artifacts/models/Task008_HepaticVessel.zip"
if [[ ! -f "$model" ]]; then
    echo "Missing checksum-verified model archive: $model" >&2
    exit 1
fi

nnUNet_install_pretrained_model_from_zip "$model"

bash "$WORKSPACE/work/v1/oof/Task008_HepaticVessel_3d_lowres/run_oof_inference.sh"
bash "$WORKSPACE/work/v1/oof/Task008_HepaticVessel_3d_fullres/run_oof_inference.sh"
bash "$WORKSPACE/work/v1/oof/Task008_HepaticVessel_ensemble/run_oof_ensemble.sh"

native="$WORKSPACE/work/v1/RESULTS_FOLDER/nnUNet/ensembles/Task008_HepaticVessel/ensemble_3d_lowres__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/ensembled_postprocessed"
mkdir -p "$native"
for source in "$WORKSPACE"/work/v1/oof/Task008_HepaticVessel_ensemble/fold_predictions/fold_*/hepaticvessel_*.nii.gz; do
    destination="$native/${source##*/}"
    if [[ -e "$destination" || -L "$destination" ]]; then
        [[ "$(readlink -f "$destination")" == "$(readlink -f "$source")" ]]
    else
        ln -s "$(readlink -f "$source")" "$destination"
    fi
done

"$PYTHON" -m rankseg_benchmark.nnunet.cli evaluate \
    "$ROOT/configs/nnunet/Task008_HepaticVessel_ensemble_oof.yaml" --device cuda

echo "Task008 evaluation complete; rebuild the main result with scripts/nnunet/build_full_aggregate.sh"
