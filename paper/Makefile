# Build the paper with auto-refreshed results numbers.
#
#   make          -> pull latest paper_numbers.tex from server, compile
#   make pdf      -> compile only (no rsync)
#   make refresh  -> regenerate paper_numbers.tex on server and pull
#   make clean    -> remove build artifacts

SERVER := ubuntu@217.18.55.93
REMOTE_DIR := /home/ubuntu/vq-rotation

.PHONY: all pdf refresh clean

all: refresh pdf

refresh:
	ssh -o StrictHostKeyChecking=no $(SERVER) \
	  "bash -lc 'source ~/venv/bin/activate && unset LD_LIBRARY_PATH && cd $(REMOTE_DIR) && python scripts/collect_results.py > /dev/null'"
	rsync -az -e 'ssh -o StrictHostKeyChecking=no' \
	  $(SERVER):$(REMOTE_DIR)/results/paper_numbers.tex ./results_numbers.tex
	@echo "=== refreshed results_numbers.tex ==="

pdf:
	pdflatex -interaction=nonstopmode main.tex > /dev/null
	bibtex main > /dev/null
	pdflatex -interaction=nonstopmode main.tex > /dev/null
	pdflatex -interaction=nonstopmode main.tex > /dev/null
	@errs=$$(grep -c '^!' main.log); \
	 warn=$$(grep -c 'Warning: Citation' main.log); \
	 undef=$$(grep -c 'Reference.*undefined' main.log); \
	 pages=$$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/ {print $$2}'); \
	 size=$$(stat -f%z main.pdf 2>/dev/null || stat -c%s main.pdf); \
	 echo "=== main.pdf $$pages pages, $$size bytes, errors=$$errs citations=$$warn undef=$$undef ==="

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.pdf
