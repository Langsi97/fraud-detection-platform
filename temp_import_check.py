import importlib
pkgs = ['fastapi', 'uvicorn', 'pandas', 'numpy', 'xgboost', 'mlflow', 'sqlalchemy', 'evidently', 'pydantic', 'sklearn']
for p in pkgs:
    try:
        importlib.import_module(p)
        print('IMPORT_OK', p)
    except Exception as e:
        print('IMPORT_FAIL', p, e)

try:
    import src.api.main as m
    print('IMPORT_OK src.api.main')
except Exception as e:
    print('IMPORT_FAIL src.api.main', e)

try:
    import src.pipelines.training_pipeline as tp
    print('IMPORT_OK src.pipelines.training_pipeline')
except Exception as e:
    print('IMPORT_FAIL src.pipelines.training_pipeline', e)
