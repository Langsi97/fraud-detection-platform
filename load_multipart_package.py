import importlib.util
import sys
from pathlib import Path
import types
import traceback

pkg = types.ModuleType('multipart')
pkg.__path__ = [str(Path('.venv/Lib/site-packages/multipart').resolve())]
sys.modules['multipart'] = pkg
path = Path('.venv/Lib/site-packages/multipart/multipart.py').resolve()
spec = importlib.util.spec_from_file_location('multipart.multipart', path)
mod = importlib.util.module_from_spec(spec)
sys.modules['multipart.multipart'] = mod
try:
    spec.loader.exec_module(mod)
    print('loaded directly', mod.__file__)
    print('has', hasattr(mod, 'MultipartSegment'))
    if hasattr(mod, 'MultipartSegment'):
        print(mod.MultipartSegment)
except Exception:
    traceback.print_exc()
