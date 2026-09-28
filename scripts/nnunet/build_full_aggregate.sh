#!/usr/bin/env bash
set -euo pipefail

PYTHON="${RANKSEG_NNUNET_PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${RANKSEG_NNUNET_WORKSPACE:-$ROOT}"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
    -m rankseg_benchmark.nnunet.cli aggregate \
    "$WORKSPACE/outputs/Task001_BrainTumour_3d_fullres_oof/summary.json" \
    "$WORKSPACE/outputs/Task002_Heart_3d_fullres_oof/summary.json" \
    "$WORKSPACE/outputs/Task003_Liver_ensemble_oof/summary.json" \
    "$WORKSPACE/outputs/Task004_Hippocampus_oof/summary.json" \
    "$WORKSPACE/outputs/Task005_Prostate_ensemble_postprocessed_oof/summary.json" \
    "$WORKSPACE/outputs/Task006_Lung_ensemble_oof/summary.json" \
    "$WORKSPACE/outputs/Task007_Pancreas_ensemble_oof/summary.json" \
    "$WORKSPACE/outputs/Task008_HepaticVessel_ensemble_oof/summary.json" \
    "$WORKSPACE/outputs/Task009_Spleen_ensemble_postprocessed_oof/summary.json" \
    "$WORKSPACE/outputs/Task010_Colon_3d_cascade_fullres_oof/summary.json" \
    "$WORKSPACE/outputs/Task027_ACDC_ensemble_oof/summary.json" \
    "$WORKSPACE/outputs/Task055_SegTHOR_ensemble_oof/summary.json" \
    "$WORKSPACE/outputs/Task075_Fluo_C3DH_A549_3d_fullres_oof/summary.json" \
    "$WORKSPACE/outputs/Task024_PROMISE12_independent_test/summary.json" \
    "$WORKSPACE/outputs/Task038_CHAOS_ensemble_oof/summary.json" \
    "$WORKSPACE/outputs/CHAOS_CT_TotalSegmentator_v2_external/summary.json" \
    --output-dir "$WORKSPACE/outputs/full_aggregate" \
    --overall-tests

echo "Full 16-dataset benchmark aggregate complete"
