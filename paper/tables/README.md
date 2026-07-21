# DenoiseRL table sources

This directory contains reusable LaTeX versions of all six tables in the submitted paper and the eight unique, completed tables added in the rebuttal.

- `fragments/`: table bodies that can be included in the camera-ready paper.
- `standalone/`: independently compilable sources, one per table.
- `all_tables.tex`: a review document containing every table on its own page.
- `../../output/pdf/tables/`: generated standalone PDFs and a combined overview PDF.
- `../../output/pdf/tables/png/`: generated PNG previews.

Rebuttal tables repeated across multiple reviewer responses are deduplicated using `rebuttal/rebuttal/reviewer1.md`, which contains the complete filled set. Draft tables containing `TBD` values in `rebuttal/test/test.md` are intentionally excluded.

Run `./build_tables.sh` from this directory to rebuild all PDFs and PNG previews. The script uses `latexmk` and Poppler's `pdftoppm`.

