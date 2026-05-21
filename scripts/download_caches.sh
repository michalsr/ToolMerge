#!/usr/bin/env bash
# Pre-built caches are NOT redistributed; build them locally with cache_build/.
#
# See README.md "Building the per-video caches" and `cache_build/README.md`
# for the full instructions.

set -euo pipefail

cat <<EOF
Per-video caches are not redistributed. Build them locally with:

  python -m cache_build.build_caches \\
      --video_dir <path-to-videos> \\
      --dataset_json <path-to-dataset.json> \\
      --tools siglip tren_per_frame ocr \\
      --siglip_output_dir \${TOOLMERGE_CACHE_DIR}/siglip/<dataset> \\
      --tren_per_frame_output_dir \${TOOLMERGE_CACHE_DIR}/tren/<dataset> \\
      --ocr_output_dir \${TOOLMERGE_CACHE_DIR}/ocr/<dataset> \\
      --video_backend cv2

The OCR-judge cache is per-question and built lazily by the runtime on first
inference — no pre-build step required.
EOF
