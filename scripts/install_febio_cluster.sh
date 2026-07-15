#!/usr/bin/env bash
# Install a user/project-local FEBio binary for the dataset pipeline.
#
# Run this once on the cluster login node:
#
#   cd /path/to/febio-foot-surrogate-pipeline
#   bash scripts/install_febio_cluster.sh
#
# If the default URL does not work, pass another official FEBio tarball URL:
#
#   FEBIO_URL=https://repo.febio.org/download/FEBio4.tar.gz bash scripts/install_febio_cluster.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/febio-foot-surrogate-pipeline}"
INSTALL_DIR="${FEBIO_INSTALL_DIR:-${PROJECT_DIR}/third_party/febio}"
DOWNLOAD_DIR="${PROJECT_DIR}/third_party/downloads"
FEBIO_URL="${FEBIO_URL:-https://repo.febio.org/download/febio4HPC.tar.gz}"
ARCHIVE="${DOWNLOAD_DIR}/$(basename "${FEBIO_URL}")"

mkdir -p "${INSTALL_DIR}" "${DOWNLOAD_DIR}"

echo "[INFO] Project dir : ${PROJECT_DIR}"
echo "[INFO] Install dir : ${INSTALL_DIR}"
echo "[INFO] FEBio URL   : ${FEBIO_URL}"

if [[ ! -f "${ARCHIVE}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L "${FEBIO_URL}" -o "${ARCHIVE}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${ARCHIVE}" "${FEBIO_URL}"
  else
    echo "[ERROR] Neither curl nor wget is available." >&2
    exit 1
  fi
else
  echo "[INFO] Reusing downloaded archive: ${ARCHIVE}"
fi

echo "[INFO] Extracting FEBio..."
tar -xzf "${ARCHIVE}" -C "${INSTALL_DIR}" --strip-components=0

FEBIO_EXE="$(find "${INSTALL_DIR}" -type f -name febio4 -perm -u+x | head -n 1 || true)"
if [[ -z "${FEBIO_EXE}" ]]; then
  FEBIO_EXE="$(find "${INSTALL_DIR}" -type f -name febio4 | head -n 1 || true)"
fi

if [[ -z "${FEBIO_EXE}" ]]; then
  echo "[ERROR] Could not find febio4 after extraction." >&2
  echo "[DEBUG] Extracted files:" >&2
  find "${INSTALL_DIR}" -maxdepth 4 -type f | head -n 50 >&2
  exit 2
fi

chmod +x "${FEBIO_EXE}"
echo "${FEBIO_EXE}" > "${PROJECT_DIR}/.febio_exe"

echo "[OK] FEBio executable: ${FEBIO_EXE}"
"${FEBIO_EXE}" -v || true

