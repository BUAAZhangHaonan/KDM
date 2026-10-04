#!/bin/sh
set -eu
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -halt-on-error paper_v3_zh.tex
bibtex paper_v3_zh
xelatex -interaction=nonstopmode -halt-on-error paper_v3_zh.tex
xelatex -interaction=nonstopmode -halt-on-error paper_v3_zh.tex
