run : 171860624_4_v1.tex preamble.tex
	xelatex -shell-escape -synctex=1 -interaction=nonstopmode 171860624_4_v1.tex
	open 171860624_4_v1.pdf
	notifyloop 171860624_4_v1.tex make lop

lop : 171860624_4_v1.tex preamble.tex
	xelatex -shell-escape -synctex=1 -interaction=nonstopmode 171860624_4_v1.tex

clean :
	rm -r *.bib *.blg *.bbl *.aux *.log *.brf *.nlo *.out *.dvi *.ps *.lof *.toc *.fls *.fdb_latexmk *.pdfsync *.synctex*.gz *.ind *.ilg *.idx
