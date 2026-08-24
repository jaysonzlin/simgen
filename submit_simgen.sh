#!/usr/bin/env bash
#SBATCH --job-name=simgen_panda_ball_can
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:30:00
#SBATCH --array=0-499%4
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/simgen/logs/simgen_%A_%a.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/simgen/logs/simgen_%A_%a.err

set -euo pipefail

PROJECT_DIR="${SIMGEN_PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/simgen}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
SEED="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set by the job array}"
SCENE_TEMPLATE="${PROJECT_DIR}/examples/panda_ball_can.yaml"
if [[ ! -f "${SCENE_TEMPLATE}" || ! -f "${PROJECT_DIR}/simgen.sif" ]]; then
    echo "SimGen project directory is invalid: ${PROJECT_DIR}" >&2
    exit 2
fi
cd "${PROJECT_DIR}"
mkdir -p logs
echo "Job ID: ${SLURM_JOB_ID:-not set}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi
export PYTHONUNBUFFERED=1
TASK_SCENE="$(mktemp "${PROJECT_DIR}/examples/.panda_ball_can_seed_${SEED}.XXXXXX")"
CONTAINER_SCENE="/workspace/simgen/examples/$(basename "${TASK_SCENE}")"
CONTAINER_OUTPUT="runs/panda_ball_can/sample_${SEED}"

trap 'rm -f "${TASK_SCENE}"' EXIT

sed -E \
    -e "s/^seed:[[:space:]]*[0-9]+[[:space:]]*$/seed: ${SEED}/" \
    -e "s/^([[:space:]]*keep_simulation:)[[:space:]]*(true|false)[[:space:]]*$/\\1 false/" \
    "${SCENE_TEMPLATE}" > "${TASK_SCENE}"

grep -qx "seed: ${SEED}" "${TASK_SCENE}"
grep -Eq "^[[:space:]]*keep_simulation:[[:space:]]*false$" "${TASK_SCENE}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: generating seed ${SEED}"
echo "Output: ${CONTAINER_OUTPUT}"

singularity exec --nv \
    -B /n/holylabs \
    -B /net/holy-isilon \
    --bind "${WORKSPACE_DIR}:/workspace" \
    -B /tmp:/dev/shm \
    "${PROJECT_DIR}/simgen.sif" \
    bash -c 'cd /workspace/simgen && /opt/conda/envs/app/bin/python -m simgen.generate --scene "$1" --output "$2"' \
    simgen-submit "${CONTAINER_SCENE}" "${CONTAINER_OUTPUT}"
