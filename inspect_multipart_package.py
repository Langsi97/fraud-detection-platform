import importlib, traceback
try:
    import multipart
    print('multipart package file', multipart.__file__)
    print('package version', getattr(multipart, '__version__', None))
    print('has segment in package', hasattr(multipart, 'MultipartSegment'))
    print('package attrs', [name for name in dir(multipart) if 'Multipart' in name])
    import multipart.multipart as mm
    print('multipart.mult file', mm.__file__)
    print('has segment in module', hasattr(mm, 'MultipartSegment'))
    if hasattr(mm, 'MultipartSegment'):
        print(mm.MultipartSegment)
except Exception:
    traceback.print_exc()
