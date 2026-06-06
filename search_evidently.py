import os
root = os.path.join(os.getcwd(), '.venv', 'Lib', 'site-packages', 'evidently')
for dirpath, dirs, files in os.walk(root):
    for name in files:
        if name.endswith('.py'):
            path = os.path.join(dirpath, name)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            if 'MultipartSegment' in text:
                print(path)
                for i, line in enumerate(text.splitlines(), 1):
                    if 'MultipartSegment' in line:
                        print(i, line)
