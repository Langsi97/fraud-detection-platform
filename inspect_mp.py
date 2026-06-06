import inspect
import python_multipart.multipart as m
print('HAS MultipartSegment in python_multipart.multipart', hasattr(m, 'MultipartSegment'))
print('NAMES containing Multipart:', [name for name in dir(m) if 'Multipart' in name])
source = inspect.getsource(m)
for i, line in enumerate(source.splitlines(), 1):
    if 'MultipartSegment' in line or 'class Multipart' in line:
        print(i, line)
