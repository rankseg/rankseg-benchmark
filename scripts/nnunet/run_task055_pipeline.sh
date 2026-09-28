#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${RANKSEG_NNUNET_PYTHON:-python}"
WORKSPACE="${RANKSEG_NNUNET_WORKSPACE:-$ROOT}"
export nnUNet_raw_data_base="$WORKSPACE/work/v1/nnUNet_raw_data_base"
export nnUNet_preprocessed="$WORKSPACE/work/v1/nnUNet_preprocessed"
export RESULTS_FOLDER="$WORKSPACE/work/v1/RESULTS_FOLDER"
export MPLCONFIGDIR=/tmp/matplotlib_task055_pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

archive="$WORKSPACE/artifacts/datasets/SegTHOR_train.zip"
model="$WORKSPACE/artifacts/models/Task055_SegTHOR.zip"
source="$WORKSPACE/work/v1/source_data/SegTHOR_official"
raw="$WORKSPACE/work/v1/nnUNet_raw_data_base/nnUNet_raw_data/Task055_SegTHOR"
fullres="$WORKSPACE/work/v1/oof/Task055_SegTHOR_3d_fullres"
cascade="$WORKSPACE/work/v1/oof/Task055_SegTHOR_3d_cascade_fullres"
ensemble="$WORKSPACE/work/v1/oof/Task055_SegTHOR_ensemble"
native="$WORKSPACE/work/v1/oof/Task055_SegTHOR_native_postprocessed"
pp="$WORKSPACE/work/v1/RESULTS_FOLDER/nnUNet/ensembles/Task055_SegTHOR/ensemble_3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1--3d_cascade_fullres__nnUNetTrainerV2CascadeFullRes__nnUNetPlansv2.1/postprocessing.json"

[[ -f "$archive" ]] || { echo "Missing SegTHOR training archive" >&2; exit 1; }
[[ "$(stat -c %s "$archive")" == 1820142473 ]] || { echo "SegTHOR archive size mismatch" >&2; exit 1; }
unzip -tqq "$archive"
[[ -f "$model" ]] || { echo "Missing checksum-verified Task055 model" >&2; exit 1; }
[[ "$(stat -c %s "$model")" == 5019434005 ]] || { echo "Task055 model size mismatch" >&2; exit 1; }
[[ "$(md5sum "$model" | cut -d' ' -f1)" == ae464374487596ab54f9d24bfd14b767 ]] || {
    echo "Task055 model MD5 mismatch" >&2
    exit 1
}

mkdir -p "$source"
if [[ ! -f "$source/.extracted_from_verified_archive" ]]; then
    unzip -oq "$archive" -d "$source"
    touch "$source/.extracted_from_verified_archive"
fi
"$PYTHON" "$ROOT/scripts/nnunet/prepare_task055_segthor.py" \
    --source-dir "$source" --output-dir "$raw"

nnUNet_install_pretrained_model_from_zip "$model"
[[ -f "$pp" ]] || { echo "Missing selected official ensemble postprocessing: $pp" >&2; exit 1; }

"$PYTHON" -m rankseg_benchmark.nnunet.cli prepare-oof-v1 \
    --task Task055_SegTHOR --images-dir "$raw/imagesTr" --output-dir "$fullres" \
    --model 3d_fullres
"$PYTHON" -m rankseg_benchmark.nnunet.cli prepare-oof-v1 \
    --task Task055_SegTHOR --images-dir "$raw/imagesTr" --output-dir "$cascade" \
    --model 3d_cascade_fullres
"$PYTHON" -m rankseg_benchmark.nnunet.cli prepare-oof-ensemble-v1 \
    --first-oof-dir "$fullres" --second-oof-dir "$cascade" --output-dir "$ensemble" \
    --threads 2 --postprocessing-file "$pp"

bash "$fullres/run_oof_inference.sh"
bash "$cascade/run_oof_inference.sh"
bash "$ensemble/run_oof_ensemble.sh"

# nnUNet_ensemble stores probabilities below not_postprocessed/, while the
# official postprocessed native masks live one directory higher. Mirror the
# latter into the probability tree shape so evaluation compares like for like.
for fold in 0 1 2 3 4; do
    mkdir -p "$native/fold_$fold/not_postprocessed"
    for source_path in "$ensemble/fold_predictions/fold_$fold"/Patient_*.nii.gz; do
        destination="$native/fold_$fold/not_postprocessed/${source_path##*/}"
        ln -sfn "$source_path" "$destination"
    done
done

"$PYTHON" -m rankseg_benchmark.nnunet.cli evaluate \
    "$ROOT/configs/nnunet/Task055_SegTHOR_ensemble_oof.yaml" --device cuda

echo "Task055 evaluation complete; rebuild the main result with scripts/nnunet/build_full_aggregate.sh"
