#!/usr/bin/env bash
# Download the full COCO val2017 images + annotations for accuracy evaluation.
set -e
DIR="${1:-val2017}"
mkdir -p "$DIR" && cd "$DIR"
echo "Downloading COCO val2017 (images ~778MB + annotations ~241MB)..."
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o val2017.zip
unzip -o annotations_trainval2017.zip
echo "Done: $(pwd)/val2017 with $(ls val2017 | wc -l) images."
