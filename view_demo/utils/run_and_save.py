"""
Register a cell magic to allow executing the code in a notebook cell
And write it out(sync). It seems to be a good way to avoid copying code blocks
With version control, manual edits overwriting can be recovered from.

Following code snippet is from:
https://datascience.stackexchange.com/questions/13669/how-to-export-one-cell-of-a-jupyter-notebook
"""
from IPython.core.magic import register_cell_magic
@register_cell_magic
def run_and_save(line, cell):
    'Run and save python code block to a file'
    with open(line, 'wt') as fd:
        fd.write(cell)
    code = compile(cell, line, 'exec')
    exec(code, globals())
