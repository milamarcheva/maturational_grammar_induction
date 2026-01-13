# maturational_grammar_induction
Testing maturational hypotheses of syntactic acquisition via grammar induction

## running inside outside
./inside-outside/io -V -n100 -g ../morphemic_tokenisation/data/filtered_ctb/filtered_ctb_grammar_min2_lexiconCHILDES_TB.lt ../morphemic_tokenisation/data/filtered_ctb/filtered_ctb_yields.txt > ../morphemic_tokenisation/data/filtered_ctb/results_min2.lt 

## inside_outside

The `inside_outside` folder contains code originally developed by 
[Mark Johnson and obtained from his website: https://web.science.mq.edu.au/~mjohnson/Software.htm.

Original copyright notices:

(c) Mark_Johnson@Brown.edu, 21 August 2000  
(c) modified 12 July 2004  
(c) modified 2 September 2007 (Variational Bayes)

### License and attribution

According to the original distribution website:

> "This software is open-source, but I do request acknowledgement whenever
> this software is used to produce published results or incorporated into
> other software."

This software is redistributed here in accordance with that request.
All credit for the original implementation belongs to Mark Johnson.

-------------------------------------------------------------------

Extracting grammars and yields
------------------------------

Use `morphemic_tokenisation/extract_grammar_from_tagged.py` to build a
grammar or extract yields from PTB-style trees (one tree per line).

Grammar extraction:

  python morphemic_tokenisation/extract_grammar_from_tagged.py \
    --input trees.txt \
    --output grammar.lt \
    --weight-mode uniform_vb \
    --weight 1.0 \
    --pseudocount 0.1

Common options:
- `--weight-mode` controls the output values:
  `percentage` (probabilities), `counts`, `uniform`, `uniform_vb`, or `none`.
- `--weight` and `--pseudocount` fill the `[Weight [Pseudocount]]` fields for
  `uniform`/`uniform_vb` and the VB-style default line format.
- `--min-freq` filters rules by count.
- `--rule-type` selects `full`, `productions`, or `lexicon`.
- `--drop-unary-nt` removes unary NT->NT rules and prunes undefined NTs.

Yield extraction (lowercased):

  python morphemic_tokenisation/extract_grammar_from_tagged.py \
    --input trees.txt \
    --extract-yields \
    --yields-output yields.txt \
    --skip-bad
