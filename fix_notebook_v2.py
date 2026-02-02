#!/usr/bin/env python3
"""
Fix the notebook formatting - ensure proper line breaks
"""

import json

# Read the notebook
with open('yield_curve.ipynb', 'r') as f:
    nb = json.load(f)

# Fix formatting for all code cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell:
        # Ensure each line ends with \n and is a separate string
        source = ''.join(cell['source'])
        lines = source.split('\n')
        # Rebuild with proper formatting
        cell['source'] = [line + '\n' for line in lines[:-1]] + ([''] if lines[-1] == '' else [lines[-1] + '\n'])

# Write back
with open('yield_curve.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook formatting fixed!")
