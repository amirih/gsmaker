# import utils.file as file
# import utils.gs.java as gs_java
import zipfile
import os
import io
from flask import send_file


def build(configs):
    language = configs["language"]
    if language == "java":
        return build_java(configs)
    # elif language == "python":
    #     return build_python(configs)
    else:
        raise Exception("Language not supported yet")


def zip_files(file_paths, zip_name):
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w") as zipf:
        for file_path in file_paths:
            if isinstance(file_path, tuple):
                file_name, file_content = file_path
                zipf.writestr(file_name, file_content)
            else:
                zipf.write(file_path, os.path.basename(file_path))
    memory_file.seek(0)
    return memory_file


def build_java(configs):
    file_list = []

    # Add backend files
    backend_files = [
        configs["run_autograder_bash"],
        configs["compile_bash"],
        configs["lib_dir"],
        configs["run_bash"],
        configs["setup_bash"],
        configs["gs_lib_dir"],
    ]

    for file_path in backend_files:
        with open(file_path, "rb") as f:
            file_content = f.read()
            file_list.append((os.path.basename(file_path), file_content))

    # Handle files uploaded from frontend
    def add_files_from_frontend(zip_file_path):
        if zip_file_path is None:
            return
        try:
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    with zip_ref.open(file_info) as f:
                        file_content = f.read()
                        file_list.append((file_info.filename, file_content))
        except zipfile.BadZipFile as e:
            file_content = zip_file_path.read()
            file_list.append(
                (os.path.basename(zip_file_path.name), file_content)
            )
        except Exception as e:
            print(
                f"An error occurred while processing {zip_file_path}. Exception: {e}"
            )

    # Add student submission files
    # add_files_from_frontend(configs["student_submission_files"])
    add_files_from_frontend(configs["unit_tests_files"])
    add_files_from_frontend(configs["starter_code"])
    add_files_from_frontend(configs["data_files"])

    assignment_name = configs["assignment_name"]

    zip_memory = zip_files(file_list, f"{assignment_name}.zip")
    return send_file(
        io.BytesIO(
            zip_memory.getvalue()
        ),  # Ensure proper content is read from BytesIO
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{assignment_name}.zip",
    )


# def build_python(configs, output_dir):
#     file.copy(configs["run_tests_template"], output_dir + "/run_tests.py")
#     file.copy(configs["setup_bash"], output_dir + "/setup.sh")
#     file.copy(configs["starter_code"], output_dir)
#     file.copy(configs["requirements"], output_dir)
#     file.copy(configs["data_files"], output_dir + "/data/")

#     # Open a new .sh file in write mode
#     with open(output_dir + "/run_autograder", "w") as autograder_file:
#         autograder_file.write("#!/bin/bash\n")

#     required_files = configs["student_submission_files"]

#     for file_name in required_files:
#         with open(output_dir + "/run_autograder", "a") as autograder_file:
#             autograder_file.write(
#                 f"cp /autograder/submission/{file_name} /autograder/source/{file_name}\n"
#             )

#     with open(output_dir + "/run_autograder", "a") as autograder_file:
#         autograder_file.write("cd /autograder/source\npython3 run_tests.py\n")

#     unit_test_file_names = configs["unit_tests_files"]

#     for test_case in unit_test_file_names:
#         path = os.path.join(configs["unit_tests_dir"], test_case)
#         file.copy(path, output_dir + "/tests/")

#     file.zip(output_dir)
#     return output_dir
