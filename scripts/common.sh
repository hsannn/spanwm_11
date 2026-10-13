# Shared settings. Override PYTHON / OUT from the environment if needed.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-python}
OUT=${OUT:-outputs}
mkdir -p $OUT/logs
MODELS=(meta-llama/Llama-3.2-3B meta-llama/Llama-3.1-8B Qwen/Qwen3-4B Qwen/Qwen3-8B)
DATASETS=(c4 wmt16 cnn)
