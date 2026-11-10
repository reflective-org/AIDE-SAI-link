#!/usr/bin/env python3
"""Copy every figure on the page (site/img) into presentation/<section>/ for slides: PNGs as-is (white background),
GIFs as-is plus a still PNG of their LAST frame. usage: collect_presentation.py [site/img dir] [out dir]"""
import os, shutil, sys
from PIL import Image
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/site/img'
OUT = sys.argv[2] if len(sys.argv) > 2 else f'{ROOT}/presentation'
def section(n):
    if n.startswith(('climate_', 'aoa_')): return '1_climate_state'
    if n.startswith('inj_'): return '2_advection'
    if n.startswith('micro_'): return '3_microphysics'
    return '4_coupled'
for n in sorted(os.listdir(SRC)):
    d = f'{OUT}/{section(n)}'; os.makedirs(d, exist_ok=True)
    shutil.copy2(f'{SRC}/{n}', f'{d}/{n}')
    if n.endswith('.gif'):
        im = Image.open(f'{SRC}/{n}'); im.seek(im.n_frames - 1)
        bg = Image.new('RGB', im.size, 'white'); bg.paste(im.convert('RGBA'), mask=im.convert('RGBA'))
        bg.save(f'{d}/{n[:-4]}_lastframe.png')
    print(section(n), n)
