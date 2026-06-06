import os, sys, importlib
print('PYTHON', sys.executable)
print('PREFIX', sys.prefix)
print('PATH[0]', sys.path[0])
print('sys.path contains site-packages:')
for p in sys.path:
    if 'site-packages' in p:
        print('  ', p)

for name in ['pkg_resources', 'setuptools', 'python_multipart', 'multipart']:
    try:
        mod = importlib.import_module(name)
        print('IMPORT_OK', name, '->', getattr(mod, '__file__', None), 'version=', getattr(mod, '__version__', None))
        if name == 'multipart':
            print('  HAS MultipartSegment', hasattr(mod, 'MultipartSegment'))
            if hasattr(mod, 'MultipartSegment'):
                print('  type(MultipartSegment)=', type(mod.MultipartSegment))
    except Exception as exc:
        print('IMPORT_FAIL', name, exc)

root = sys.prefix
sp = os.path.join(root, 'Lib', 'site-packages')
print('\nSITE-PACKAGES LISTING:')
for entry in sorted(os.listdir(sp)):
    if entry.lower().startswith(('pkg', 'setuptools', 'python', 'multipart')):
        print('  ', entry)

for root_dir, dirs, files in os.walk(sp):
    if 'pkg_resources' in dirs or 'pkg_resources.py' in files or 'pkg_resources' in root_dir:
        print('FOUND pkg_resources path', root_dir)
        break

print('\nSETUPTOOLS DIR EXISTS', os.path.isdir(os.path.join(sp, 'setuptools')))
print('python_multipart dir exists', os.path.isdir(os.path.join(sp, 'python_multipart')))
print('multipart dir exists', os.path.isdir(os.path.join(sp, 'multipart')))
print('multipart.py exists', os.path.isfile(os.path.join(sp, 'multipart.py')))
