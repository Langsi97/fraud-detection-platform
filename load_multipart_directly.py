import importlib.util
from pathlib import Path
import traceback

path = Path('.venv/Lib/site-packages/multipart/multipart.py').resolve()
print('path', path)
spec = importlib.util.spec_from_file_location('test_multipart', path)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print('loaded directly')
    print('has', hasattr(mod, 'MultipartSegment'))
    if hasattr(mod, 'MultipartSegment'):
        print(mod.MultipartSegment)
except Exception:
    traceback.print_exc()
