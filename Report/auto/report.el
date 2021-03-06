(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("report" "10pt" "twoside" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("packages/ethasl" "mt" "hs" "english") ("algorithm2e" "ruled" "vlined")))
   (TeX-run-style-hooks
    "latex2e"
    "preambles/asl_preamble"
    "chapters/declaration"
    "chapters/abstract"
    "chapters/symbols"
    "chapters/introduction"
    "chapters/chapter"
    "chapters/appendix_text"
    "chapters/preface"
    "rep10"
    "pdfpages"
    "packages/ethasl"
    "mathrsfs"
    "algorithm2e"
    "bm"
    "makecell"
    "siunitx"
    "float"
    "caption")
   (LaTeX-add-bibliographies
    "bibliography/references"))
 :latex)

