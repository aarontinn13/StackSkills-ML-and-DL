import os

print('absolute path:', os.path.abspath(os.path.dirname(__file__)))
print('relative path:', os.path.relpath(os.path.dirname(__file__)))