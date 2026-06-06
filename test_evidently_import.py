import traceback

try:
    import evidently
    print('evidently imported')
except Exception:
    traceback.print_exc()
