import sys
import importlib
print('PYTHON', sys.version)
print('PATHS', sys.path)
for name in ['setuptools', 'pkg_resources', 'mlflow', 'evidently', 'python_multipart', 'multipart']:
    try:
        mod = importlib.import_module(name)
        print('MODULE', name, '->', getattr(mod, '__file__', None), 'version=', getattr(mod, '__version__', 'unknown'))
    except Exception as exc:
        print('MODULE_FAIL', name, exc)
print('\nSITE_PACKAGES:')
import site
for p in site.getsitepackages():
    print('  ', p)
