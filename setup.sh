#!/usr/bin/env bash
# Create .venv, fetch the NLA inference client, the artefacts and the AR.
#
#   ./setup.sh                          download everything (reproduction from artefacts)
#   ./setup.sh --local-artefacts DIR    use artefacts already on disk
#   ./setup.sh --ar-dir DIR             use an AR checkpoint already on disk
#   ./setup.sh --with-av                reproduction from scratch: also the AV,
#                                       spacy and its English model
#
# TORCH_INDEX_URL picks the torch build (default: CUDA 12.8 wheels). If torch
# is already importable in .venv it is left alone.
set -euo pipefail
cd "$(dirname "$0")"

HF_DATASET_ID=${HF_DATASET_ID:-Near32/OnMonitorabilityOfNLAs-artefacts}
TORCH_VERSION=${TORCH_VERSION:-2.9.1}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
NLA_INFERENCE_SHA=38b802a33d1d317f21b6825a9116f388c2141f86
NLA_INFERENCE_SHA256=45cbf64489dc8f1daa8c9e98fe4dd4e881e4fd743fd783d6128c070cc0677f26
AR_REPO=kitft/nla-qwen2.5-7b-L20-ar  AR_REV=e2c9e57eac213d37a31612087f645ab6332c1bb6
AV_REPO=kitft/nla-qwen2.5-7b-L20-av  AV_REV=b88469162777ae6553bc14208eb0cb579336f8f4

local_artefacts="" ar_dir="" with_av=0
while [ $# -gt 0 ]; do
  case "$1" in
    --local-artefacts) local_artefacts=$2; shift 2 ;;
    --ar-dir) ar_dir=$2; shift 2 ;;
    --with-av) with_av=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Debian/Ubuntu ship venv without pip unless python3.11-venv is installed.
[ -d .venv ] || python3.11 -m venv .venv || { rm -rf .venv; virtualenv -p python3.11 .venv; }
. .venv/bin/activate
python -m pip install -q --upgrade pip
python -c "import torch" 2>/dev/null \
  || pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
pip install -r requirements.txt

# The reference NLA client (Apache-2.0, kitft/nla-inference), pinned by commit
# and checked by hash.
curl -fsSL -o nla/nla_inference.py \
  "https://raw.githubusercontent.com/kitft/nla-inference/$NLA_INFERENCE_SHA/nla_inference.py"
echo "$NLA_INFERENCE_SHA256  nla/nla_inference.py" | sha256sum -c -

if [ -n "$local_artefacts" ]; then
  rm -rf artefacts && cp -r "$local_artefacts" artefacts
else
  hf download "$HF_DATASET_ID" --repo-type dataset --local-dir artefacts || {
    echo "the artefacts are gated: accept the terms at" \
         "https://huggingface.co/datasets/$HF_DATASET_ID, then run hf auth login" >&2
    exit 1; }
fi

mkdir -p models
if [ -n "$ar_dir" ]; then
  ln -sfn "$(realpath "$ar_dir")" models/ar
else
  hf download "$AR_REPO" --revision "$AR_REV" --local-dir models/ar
fi
if [ "$with_av" = 1 ]; then
  hf download "$AV_REPO" --revision "$AV_REV" --local-dir models/av
  pip install spacy==3.8.16 \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
fi

python - <<'EOF'
import torch
print("torch", torch.__version__, "| GPU available:", torch.cuda.is_available())
EOF
echo "setup done; next: bash exp1_identification/run_rescore.sh"
