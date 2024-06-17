import os
def copy(src, dest):
    if type(src) is list:
        for s in src:
            copy(s, dest)
        return dest
    if not os.path.exists(src):
        raise Exception(f'{src} does not exist')
    print(f'Copying {src} to {dest}')
    os.system('mkdir -p ' + os.path.dirname(dest))
    if os.path.isdir(src):
        os.system('cp -r ' + src + ' ' + dest)
    else:
        os.system('cp ' + src + ' ' + dest)
    return dest

def write(text, dest):
    os.system('mkdir -p ' + os.path.dirname(dest))
    with open(dest, 'w') as f:
        f.write(text)

def read(src):
    with open(src, 'r') as f:
        return f.read()

def remove(path):
    os.system('rm -rf ' + path)

def zip(output_dir):
    dir_name = os.path.basename(output_dir)
    files = get_files(output_dir)
    print(f'Zipping {files} to {dir_name}.zip')
    os.system(f'cd {output_dir} && zip -r {dir_name}.zip {' '.join(files)}')
    return f'{output_dir}/{dir_name}.zip'

def get_files(output_dir):
    return os.listdir(output_dir)