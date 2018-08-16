(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("report" "10pt" "twoside" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("packages/ethasl" "mt" "hs" "english")))
   (TeX-run-style-hooks
    "latex2e"
    "preambles/asl_preamble"
    "chapters/preface"
    "chapters/abstract"
    "chapters/symbols"
    "chapters/introduction"
    "chapters/chapter"
    "chapters/appendix_text"
    "chapters/appendix_datasheets"
    "rep10"
    "pdfpages"
    "packages/ethasl"
    "mathrsfs"
    "bm")
   (LaTeX-add-bibliographies
    "bibliography/references"))
 :latex)

