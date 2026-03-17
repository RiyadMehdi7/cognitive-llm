PYTHON ?= python3
LATEXMK ?= latexmk
TECTONIC ?= tectonic

.PHONY: figures pdf clean

figures:
	$(PYTHON) scripts/make_figures.py

pdf: figures
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex; \
	elif command -v $(TECTONIC) >/dev/null 2>&1; then \
		$(TECTONIC) main.tex; \
	else \
		echo "No TeX engine found. Install latexmk or tectonic."; \
		exit 1; \
	fi

clean:
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -C >/dev/null 2>&1 || true; \
	fi
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.xdv
