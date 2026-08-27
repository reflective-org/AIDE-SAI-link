"""Import-path setup for the src/ tree, and the two repo-relative anchors.

src/ is deliberately NOT a Python package. Every module in it is a flat,
top-level name -- `coupling`, `settling`, `radiation`, `fct_lr`, `fct_fast`,
`tomas_fast` -- and the subdirectories are organisational only. Importing this
module puts each of them on sys.path, so `import settling` keeps working from
anywhere in the tree with no package prefix and no import statement had to
change when the files moved into src/.

TWO RULES, both load-bearing:

1. DO NOT add __init__.py to a src/ subdirectory without converting all the
   imports at the same time. src/settling/ and settling.py share a name, as do
   src/radiation/ and radiation.py. A bare directory is only a namespace-package
   candidate, which loses to a real module, so `import settling` finds the file
   today. An __init__.py would make it a regular package, which WINS -- and
   `settling.tang_density` would then be an AttributeError on an empty package.
   The subdirectories are inserted ahead of src/ itself below for the same
   reason: the file is found before the directory is ever considered.

2. DO NOT let an "organize imports" pass reorder the modules that import this
   one. They run path setup before importing what that setup makes importable,
   so hoisting the imports breaks the tree. This is why ruff's I001 is off.

Converting src/ into a proper installable package removes the need for all of
the above; see docs/DEFERRED.md for why that was not done here.
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC)
MODELS = os.path.join(REPO_ROOT, 'models')
INPUTS = os.path.join(REPO_ROOT, 'inputs')

# Subdirectories FIRST, src/ itself last -- see rule 1 above.
for _d in ('advection', 'radiation', 'settling', 'microphysics', ''):
    _p = os.path.join(SRC, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
