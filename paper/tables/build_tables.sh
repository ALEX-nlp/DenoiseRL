#!/usr/bin/env bash
set -euo pipefail

TABLE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$TABLE_DIR/../.." && pwd)"
BUILD_DIR="$REPO_DIR/tmp/table-build"
PDF_DIR="$REPO_DIR/output/pdf/tables"
PNG_DIR="$PDF_DIR/png"
PDFTOPPM="${PDFTOPPM:-pdftoppm}"

mkdir -p "$BUILD_DIR" "$PDF_DIR" "$PNG_DIR"

for source in "$TABLE_DIR"/standalone/*.tex; do
  name="$(basename "$source" .tex)"
  item_build="$BUILD_DIR/$name"
  mkdir -p "$item_build"
  (
    cd "$TABLE_DIR/standalone"
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -output-directory="$item_build" "$name.tex"
  )
  cp "$item_build/$name.pdf" "$PDF_DIR/$name.pdf"
  "$PDFTOPPM" -png -r 200 -singlefile \
    "$PDF_DIR/$name.pdf" "$PNG_DIR/$name"
done

overview_build="$BUILD_DIR/all_tables"
mkdir -p "$overview_build"
(
  cd "$TABLE_DIR"
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -output-directory="$overview_build" all_tables.tex
)
cp "$overview_build/all_tables.pdf" "$PDF_DIR/all_tables.pdf"
