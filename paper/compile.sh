#!/bin/bash

[ -z "$1" ] && echo "Provide TeX filename WITHOUT extension!" && exit 1

echo -e "\033[0;32m\nCompile bibliography...\033[0m"
pdflatex $1.tex &> /dev/null
bibtex $1.aux

echo -e "\033[0;32m\nCompile document...\033[0m"
pdflatex $1.tex &> /dev/null
pdflatex $1.tex

/bin/rm *.aux
/bin/rm *.bbl
/bin/rm *.blg
