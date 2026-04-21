#!/usr/bin/env bash
# Install FFmpeg shared libraries for torchcodec.
#
# torchcodec requires FFmpeg shared libraries (libavutil, libavcodec, etc.)
# at runtime. This script installs them via miniforge/conda-forge, which
# works on all platforms without requiring system package manager access.
#
# Usage:
#   bash scripts/install-ffmpeg.sh
#   # or with a custom prefix:
#   CONDA_PREFIX=/path/to/miniforge bash scripts/install-ffmpeg.sh
#
# After installation, set LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS)
# to include the conda lib directory so torchcodec can find the libraries.

set -euo pipefail

CONDA_PREFIX="${CONDA_PREFIX:-/opt/miniforge}"
FFMPEG_VERSION="${FFMPEG_VERSION:-<8}"  # torchcodec 0.7-0.11 supports FFmpeg 4-7

echo "=== FFmpeg installer for torchcodec ==="
echo "Conda prefix: ${CONDA_PREFIX}"
echo "FFmpeg version constraint: ${FFMPEG_VERSION}"

# Step 1: Check if FFmpeg shared libs are already available
if python -c "import torchcodec" 2>/dev/null; then
    echo "torchcodec already works — FFmpeg shared libs are available."
    exit 0
fi

# Step 2: Install miniforge if not present
if ! command -v "${CONDA_PREFIX}/bin/conda" &>/dev/null; then
    echo "Installing miniforge..."
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)  INSTALLER="Miniforge3-Linux-x86_64.sh" ;;
        Linux-aarch64) INSTALLER="Miniforge3-Linux-aarch64.sh" ;;
        Darwin-arm64)  INSTALLER="Miniforge3-MacOSX-arm64.sh" ;;
        Darwin-x86_64) INSTALLER="Miniforge3-MacOSX-x86_64.sh" ;;
        *)             echo "Unsupported platform: $(uname -s)-$(uname -m)"; exit 1 ;;
    esac
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}" \
        -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "${CONDA_PREFIX}"
    rm /tmp/miniforge.sh
fi

# Step 3: Install FFmpeg
echo "Installing FFmpeg ${FFMPEG_VERSION} via conda-forge..."
"${CONDA_PREFIX}/bin/conda" install -y -p "${CONDA_PREFIX}" "ffmpeg${FFMPEG_VERSION}"

# Step 4: Make libraries findable
LIB_DIR="${CONDA_PREFIX}/lib"
echo ""
echo "=== FFmpeg installed ==="
echo "Libraries are in: ${LIB_DIR}"
echo ""

case "$(uname -s)" in
    Linux)
        echo "Add to your environment:"
        echo "  export LD_LIBRARY_PATH=\"${LIB_DIR}:\${LD_LIBRARY_PATH}\""
        echo ""
        echo "Or add to system ldconfig (requires sudo):"
        echo "  echo '${LIB_DIR}' | sudo tee /etc/ld.so.conf.d/miniforge.conf && sudo ldconfig"
        # Try ldconfig if we have sudo
        if command -v sudo &>/dev/null; then
            echo "${LIB_DIR}" | sudo tee /etc/ld.so.conf.d/miniforge.conf >/dev/null
            sudo ldconfig
            echo "(ldconfig updated)"
        fi
        ;;
    Darwin)
        echo "Add to your environment:"
        echo "  export DYLD_LIBRARY_PATH=\"${LIB_DIR}:\${DYLD_LIBRARY_PATH}\""
        ;;
esac

# Step 5: Verify
echo ""
echo "Verifying torchcodec can load..."
if LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH:-}" \
   DYLD_LIBRARY_PATH="${LIB_DIR}:${DYLD_LIBRARY_PATH:-}" \
   python -c "import torchcodec; print(f'torchcodec {torchcodec.__version__} OK')" 2>/dev/null; then
    echo "Success!"
else
    echo "torchcodec still cannot load. You may need to set the library path manually."
    echo "See instructions above."
fi
