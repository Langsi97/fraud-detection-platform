import importlib.util
import sys
from pathlib import Path
import types

pkg = types.ModuleType('multipart')
pkg.__path__ = [str(Path('.venv/Lib/site-packages/multipart').resolve())]
sys.modules['multipart'] = pkg
path = Path('.venv/Lib/site-packages/multipart/multipart.py').resolve()
spec = importlib.util.spec_from_file_location('multipart.multipart', path)
mod = importlib.util.module_from_spec(spec)
sys.modules['multipart.multipart'] = mod
spec.loader.exec_module(mod)
print('names containing Multipart:', [name for name in dir(mod) if 'Multipart' in name])
print('has MultipartSegment:', hasattr(mod, 'MultipartSegment'))
print('len names', len([name for name in dir(mod) if 'Multipart' in name]))
