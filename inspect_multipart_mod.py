import importlib
m = importlib.import_module('multipart.multipart')
print('module', m)
print('file', getattr(m, '__file__', None))
print('has MultipartSegment', hasattr(m, 'MultipartSegment'))
if hasattr(m, 'MultipartSegment'):
    print('MultipartSegment', m.MultipartSegment)
print('has create_form_parser', hasattr(m, 'create_form_parser'))
