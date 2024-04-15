######################################################################
# Edit here
######################################################################
# Name of document
TEX = oth-aw-beamer.tex

# Additional sources
SOURCES = $(wildcard *.tex images/* *sty)
SVGS = $(shell find images -name "*.svg")

# binaries
BINPATH = bin
######################################################################
# No editing should be needed below
######################################################################
LATEXCMD = lualatex

BASE = $(basename $(TEX))
PDF = $(BASE).pdf

SVGPDFS = $(SVGS:%.svg=%.pdf)

.PHONY: all $(PDF) $(LTEX) $(LPDF) touch clean texclean images

all: hello $(PDF) bye

# Display doc
view:: $(PDF)
	@evince $(PDF) &

hello::
	@echo "#####################################"
	@echo "### Hi, you look gorgeous today!  ###" 
	@echo "#####################################"

bye::
	@echo "###########################"
	@echo "### Have a nice day :)  ###" 
	@echo "###########################"

# force build
touch::
	touch $(TEX)

clean:: texclean
	if [ ! -z "$(LOCALBUILDDIR)" -a -d $(LOCALBUILDDIR) ]; then \
		rm -rf $(LOCALBUILDDIR)/* ;\
	fi

texclean::
	rm -f *.gls *.glo *.aux *.log *.toc *.ilg *.glg *.ist *.toc *.pdf *.snm *.out *.nav *.tex~ *.vrb

LOCALBUILDDIR=/tmp/texbuild
LTEX = $(LOCALBUILDDIR)/$(TEX)
LPDF = $(LOCALBUILDDIR)/$(PDF)

images: $(SVGPDFS)

images/%.pdf: images/%.svg
	cd $(LOCALBUILDDIR) && \
	$(BINPATH)/svg2pdf $<


$(LPDF): $(LTEX) images $(SOURCES)
	cd $(LOCALBUILDDIR) && \
	$(LATEXCMD) -jobname $(BASE) $(TEX) && \
	$(LATEXCMD) -jobname $(BASE) $(TEX)

$(PDF): $(LPDF) $(SOURCES)
	rsync -at $(LPDF) .

$(LTEX): $(SOURCES) $(LOCALBUILDDIR)
	@rsync -at --exclude ".*" ./ $(LOCALBUILDDIR)
	@echo Transferred files to local build dir

$(LOCALBUILDDIR):
	mkdir -p $(LOCALBUILDDIR)
	chmod 700 $(LOCALBUILDDIR)
	@echo Created local directory for fast TeX build: $(LOCALBUILDDIR)
