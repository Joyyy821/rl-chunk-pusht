#!/usr/bin/env bash
set -euo pipefail

POD_NAME="${POD_NAME:-yjiao-rl-chunk-pusht-dev}"
NAMESPACE="${NAMESPACE:-}"
REPO_URL="${REPO_URL:-https://github.com/Joyyy821/rl-chunk-pusht.git}"
REPO_DIR="${REPO_DIR:-/workspace/src/rl-chunk-pusht}"
TIMEOUT="${TIMEOUT:-20m}"

kubectl_cmd=(kubectl)
if [[ -n "${NAMESPACE}" ]]; then
  kubectl_cmd+=(--namespace "${NAMESPACE}")
fi

echo "Waiting for pod/${POD_NAME} to become Ready..."
"${kubectl_cmd[@]}" wait --for=condition=Ready "pod/${POD_NAME}" --timeout="${TIMEOUT}"

echo "Bootstrapping ${REPO_DIR} on the PVC..."
"${kubectl_cmd[@]}" exec "${POD_NAME}" -- /bin/bash -lc "
set -euo pipefail
export HOME=\"\${HOME:-/workspace/home}\"
export XDG_CACHE_HOME=\"\${XDG_CACHE_HOME:-/workspace/cache}\"
export UV_CACHE_DIR=\"\${UV_CACHE_DIR:-/workspace/cache/uv}\"
export TMPDIR=\"\${TMPDIR:-/workspace/tmp}\"
mkdir -p /workspace/src /workspace/outputs /workspace/cache /workspace/home /workspace/tmp
id
touch /workspace/write-test
if ! command -v git >/dev/null 2>&1; then
  echo 'git is not available in this image. Try a different public image or one that includes git.' >&2
  exit 1
fi
if [[ -d '${REPO_DIR}/.git' ]]; then
  cd '${REPO_DIR}'
  git fetch --all --prune
  git pull --ff-only
else
  rm -rf '${REPO_DIR}'
  git clone '${REPO_URL}' '${REPO_DIR}'
fi
cd '${REPO_DIR}'
bash nautilus/scripts/setup_env.sh
"

echo "Bootstrap complete."
