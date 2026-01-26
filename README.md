# maturational_grammar_induction
Testing maturational hypotheses of syntactic acquisition via grammar induction

## running inside outside
./inside-outside/io -V -n 100 -d 1000 -g ../morphemic_tokenisation/data/filtered_ctb/filtered_ctb_grammar_min2_lexiconCHILDES_TB.lt ../morphemic_tokenisation/data/filtered_ctb/filtered_ctb_yields.txt > ../morphemic_tokenisation/data/filtered_ctb/results_min2.lt 

./inside-outside/io -V -n 100 -d 1000 -g ../morphemic_tokenisation/data/filtered_ctb/filtered_ctb_grammar_withUnary_min3.lt  ../morphemic_tokenisation/data/filtered_ctb/filtered_ctb_yields.txt > ../morphemic_tokenisation/data/filtered_ctb/results_withUnary_min3.lt

## running create_grammar

milamarcheva@Milas-MacBook-Air-3 maturational_grammar_induction % python3 scripts/create_grammar.py --grammar ../morphemic_tokenisation/data/filtered_ctb/filtered_ctb_grammar_withUnary_min5.lt --s
entences ../morphemic_tokenisation/data/brown_sents.txt --lexicalisation postagged
Wrote grammar to ../morphemic_tokenisation/data/filtered_ctb/filtered_ctb_grammar_withUnary_min5__lex-postagged__wp-1p0__wl-1p0__pp-0p1__pl-0p1.lt
- Productions: 1692
- Preterminals: 41
- Vocab size: 4602
- Lexical rules: 25152


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

Change to the code: 
 -- I have added a timing functionality in expected-counts.c, which prints an extra column in the debugging output: the time (in minutes) that each iteration takes is printed under "iter_min". 

 
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


## procedure
1. extract grammars or use the already extracted grammars in childes_tb_extracted_grammars
2. create mature grammars using scripts/create_grammar.py