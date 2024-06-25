import os
import platform
import shutil


def copy(src, dest):
    if isinstance(src, list):
        for s in src:
            copy(s, dest)
        return dest
    if not os.path.exists(src):
        raise Exception(f"{src} does not exist")
    print(f"Copying {src} to {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isdir(src):
        if platform.system() == "Windows":
            os.system(
                f'powershell Copy-Item -Path "{src}" -Destination "{dest}" -Recurse -Force'
            )
        else:
            os.system(f'cp -r "{src}" "{dest}"')
    else:
        if platform.system() == "Windows":
            os.system(f'copy "{src}" "{dest}"')
        else:
            os.system(f'cp "{src}" "{dest}"')
    return dest


def write(text, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(text)


def read(src):
    with open(src, "r") as f:
        return f.read()


def remove(path):
    if platform.system() == "Windows":
        os.system(f'rmdir /S /Q "{path}"')
    else:
        os.system(f'rm -rf "{path}"')


def zip(output_dir):
    dir_name = os.path.basename(output_dir)
    zip_path = os.path.join(output_dir, f"{dir_name}.zip")
    print(f"Zipping {output_dir} to {zip_path}")
    if platform.system() == "Windows":
        shutil.make_archive(zip_path[:-4], "zip", output_dir)
    else:
        os.system(f'cd "{output_dir}" && zip -r "{dir_name}.zip" *')
    return zip_path


def get_files(output_dir):
    return os.listdir(output_dir)
