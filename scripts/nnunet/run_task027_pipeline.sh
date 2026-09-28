#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${RANKSEG_NNUNET_PYTHON:-python}"
WORKSPACE="${RANKSEG_NNUNET_WORKSPACE:-$ROOT}"
export nnUNet_raw_data_base="$WORKSPACE/work/v1/nnUNet_raw_data_base"
export nnUNet_preprocessed="$WORKSPACE/work/v1/nnUNet_preprocessed"
export RESULTS_FOLDER="$WORKSPACE/work/v1/RESULTS_FOLDER"
export MPLCONFIGDIR=/tmp/matplotlib_task027_pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

archive="$WORKSPACE/artifacts/datasets/ACDC_training.zip"
model="$WORKSPACE/artifacts/models/Task027_ACDC.zip"
source="$WORKSPACE/work/v1/source_data/ACDC_official_training"
raw="$WORKSPACE/work/v1/nnUNet_raw_data_base/nnUNet_raw_data/Task027_ACDC"
splits="$WORKSPACE/work/v1/nnUNet_preprocessed/Task027_ACDC/splits_final.pkl"
two_d="$WORKSPACE/work/v1/oof/Task027_ACDC_2d"
fullres="$WORKSPACE/work/v1/oof/Task027_ACDC_3d_fullres"
ensemble="$WORKSPACE/work/v1/oof/Task027_ACDC_ensemble"
pp="$WORKSPACE/work/v1/RESULTS_FOLDER/nnUNet/ensembles/Task027_ACDC/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json"

[[ -f "$model" ]] || { echo "Missing Task027 model archive" >&2; exit 1; }
[[ "$(stat -c %s "$model")" == 1830026265 ]] || { echo "Task027 model size mismatch" >&2; exit 1; }
[[ "$(md5sum "$model" | cut -d' ' -f1)" == 70723b2d5dd7237e38461128b5d5c601 ]] || {
    echo "Task027 model MD5 mismatch" >&2
    exit 1
}

mkdir -p "$source"
if [[ "$(find "$source" -type d -name 'patient???' | wc -l)" -ne 100 ]]; then
    [[ -f "$archive" ]] || { echo "Missing official ACDC training files/archive" >&2; exit 1; }
    unzip -tqq "$archive"
    unzip -q "$archive" -d "$source"
fi
"$PYTHON" "$ROOT/scripts/nnunet/prepare_task027_acdc.py" \
    --source-dir "$source" --output-dir "$raw" --splits-file "$splits"
nnUNet_install_pretrained_model_from_zip "$model"

[[ -f "$pp" ]] || { echo "Missing official Task027 ensemble postprocessing file" >&2; exit 1; }

"$PYTHON" -m rankseg_benchmark.nnunet.cli prepare-oof-v1 \
    --task Task027_ACDC --images-dir "$raw/imagesTr" --output-dir "$two_d" \
    --splits-file "$splits" --model 2d --trainer nnUNetTrainerV2 --plans nnUNetPlansv2.1 \
    --nifti-save-threads 6
"$PYTHON" -m rankseg_benchmark.nnunet.cli prepare-oof-v1 \
    --task Task027_ACDC --images-dir "$raw/imagesTr" --output-dir "$fullres" \
    --splits-file "$splits" --model 3d_fullres --trainer nnUNetTrainerV2 --plans nnUNetPlansv2.1 \
    --nifti-save-threads 6
bash "$two_d/run_oof_inference.sh"
bash "$fullres/run_oof_inference.sh"
"$PYTHON" -m rankseg_benchmark.nnunet.cli prepare-oof-ensemble-v1 \
    --first-oof-dir "$two_d" --second-oof-dir "$fullres" --output-dir "$ensemble" \
    --threads 6 --postprocessing-file "$pp"
bash "$ensemble/run_oof_ensemble.sh"

"$PYTHON" -m rankseg_benchmark.nnunet.cli evaluate \
    "$ROOT/configs/nnunet/Task027_ACDC_ensemble_oof.yaml" --device cuda

echo "Task027 evaluation complete; rebuild the main result with scripts/nnunet/build_full_aggregate.sh"
