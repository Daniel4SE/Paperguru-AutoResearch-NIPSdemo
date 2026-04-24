#!/bin/bash
# post_run.sh
# Run from the repo root AFTER the server's v6 watcher has finished
# (i.e. results/E1_logs/ALL_DONE exists on the server).
#
# Pulls fresh metrics, rebuilds the paper, syncs logs, commits, and
# (optionally) pushes to Daniel4SE/Paperguru-AutoResearch-NIPSdemo.

set -e
SERVER=ubuntu@217.18.55.93
REMOTE=/home/ubuntu/vq-rotation
REPO=$(cd "$(dirname "$0")" && pwd)

echo "=== post_run.sh @ $(date +%T) ==="

# 1. Regenerate paper_numbers.tex on the server
ssh -o StrictHostKeyChecking=no "$SERVER" \
  "bash -lc 'source ~/venv/bin/activate && unset LD_LIBRARY_PATH && cd $REMOTE && python scripts/collect_results.py > /dev/null'"

# 2. Rsync numbers, figures, logs, TB events, paper_numbers.json
rsync -az -e 'ssh -o StrictHostKeyChecking=no' \
  "$SERVER:$REMOTE/results/paper_numbers.tex" \
  "$REPO/paper/results_numbers.tex"

rsync -az -e 'ssh -o StrictHostKeyChecking=no' \
  --include='*/' \
  --include='events.out.tfevents.*' --include='config.yaml' --include='*.json' \
  --exclude='ckpt_*.pt' --exclude='*' \
  "$SERVER:$REMOTE/results/" \
  "$REPO/experiments/results/"

rsync -az -e 'ssh -o StrictHostKeyChecking=no' \
  "$SERVER:$REMOTE/results/E1_logs/" \
  "$REPO/experiments/logs/"

rsync -az -e 'ssh -o StrictHostKeyChecking=no' \
  "$SERVER:$REMOTE/results/E4_logs/" \
  "$REPO/experiments/logs_E4/" 2>/dev/null || true

# 3. Optionally regenerate paper-quality figures on the server and pull
ssh -o StrictHostKeyChecking=no "$SERVER" \
  "bash -lc 'source ~/venv/bin/activate && unset LD_LIBRARY_PATH && cd $REMOTE && python scripts/viz_paper.py > /dev/null'" || true
rsync -az -e 'ssh -o StrictHostKeyChecking=no' \
  "$SERVER:$REMOTE/results/figures_paper/" \
  "$REPO/paper/figures/" 2>/dev/null || true

# 4. Recompile paper
cd "$REPO/paper"
rm -f main.aux main.log main.bbl main.blg main.out
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex   main                               > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pages=$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')
size=$(stat -f%z main.pdf 2>/dev/null || stat -c%s main.pdf)
echo "main.pdf: $pages pages, $size bytes"

# 5. Commit and optionally push
cd "$REPO"
git add -A
if ! git diff --cached --quiet; then
  git -c user.email="noreply@paperguru.ai" -c user.name="Paperguru AutoResearch" \
    commit -m "Results update $(date -u +%Y-%m-%dT%H:%MZ): ${pages}-page PDF, fresh numbers"
  if [ "${1:-}" = "--push" ]; then
    git push origin main
    echo "=== pushed to origin ==="
  else
    echo "=== committed locally; re-run with --push to publish ==="
  fi
else
  echo "=== no changes to commit ==="
fi
