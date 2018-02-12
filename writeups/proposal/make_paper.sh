pdflatex main
bibtex main
pdflatex main
pdflatex main

mv main.pdf paper_draft.pdf

rm *.aux *.log *.out *.dvi *.blg *.bbl
