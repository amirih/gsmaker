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
    elif language == "python":
        return build_python(configs)
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
    backend_files_paths = [
        configs["run_autograder_bash"],
        configs["compile_bash"],
        configs["lib_dir"],
        configs["run_bash"],
        configs["setup_bash"],
        configs["gs_lib_dir"],
    ]

    test_files = []  # will be used later to replace replace_me in RunTests.java

    # modify the run_autograder file to copy the required files
    with open(configs["run_autograder_bash"], "r") as f:
        run_autograder_content = f.read()
        required_files = configs["student_submission_files"]
        for file_name in required_files:
            run_autograder_content += f"cp /autograder/submission/{file_name} /autograder/source/{file_name}\n"
        run_autograder_content += "bash ./compile.sh\n"
        run_autograder_content += (
            "bash ./run.sh >/autograder/results/results.json\n"
        )
        file_list.append(("run_autograder", run_autograder_content))

    for file_path in backend_files_paths:
        # skip run_autograder_bash as it is modified above
        if file_path == configs["run_autograder_bash"]:
            continue
        # if com, unzip it and put it under src
        if file_path == configs["gs_lib_dir"]:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    with zip_ref.open(file_info) as f:
                        f.seek(0)
                        file_content = f.read()
                        file_list.append(
                            (f"src/com/{file_info.filename}", file_content)
                        )
            continue
        # if lib, unzip it
        if file_path == configs["lib_dir"]:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    with zip_ref.open(file_info) as f:
                        f.seek(0)
                        file_content = f.read()
                        file_list.append(
                            (f"lib/{file_info.filename}", file_content)
                        )
            continue
        if file_path == configs["run_tests_template"]:
            continue
        with open(file_path, "rb") as f:
            file_content = f.read()
            file_list.append((os.path.basename(file_path), file_content))

    # Handle files uploaded from frontend
    def add_files_from_frontend(zip_file):
        if zip_file is None:
            return
        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    with zip_ref.open(file_info) as f:
                        f.seek(0)
                        file_content = f.read()
                        if zip_file == configs["unit_tests_files"]:
                            print(f"tests/{file_info.filename}")
                            file_list.append(
                                (f"src/{file_info.filename}", file_content)
                            )
                            test_files.append(file_info.filename)
                        elif zip_file == configs["starter_code"]:
                            file_list.append(
                                (f"src/{file_info.filename}", file_content)
                            )
                        elif zip_file == configs["data_files"]:
                            file_list.append(
                                (f"data/{file_info.filename}", file_content)
                            )
                        else:
                            file_list.append((file_info.filename, file_content))
        except zipfile.BadZipFile as e:
            if zip_file == configs["unit_tests_files"]:
                configs["unit_tests_files"].seek(0)
                file_list.append(
                    (
                        "src/" + configs["unit_tests_files"].filename,
                        configs["unit_tests_files"].read(),
                    )
                )
                test_files.append(configs["unit_tests_files"].filename)
            elif zip_file == configs["starter_code"]:
                configs["starter_code"].seek(0)
                file_list.append(
                    (
                        "src/" + configs["starter_code"].filename,
                        configs["starter_code"].read(),
                    )
                )
            elif zip_file == configs["data_files"]:
                configs["data_files"].seek(0)
                file_list.append(
                    (
                        "data/" + configs["data_files"].filename,
                        configs["data_files"].read(),
                    )
                )
            else:
                zip_file.seek(0)
                file_content = zip_file.read()
                file_list.append(
                    (os.path.basename(zip_file.name), file_content)
                )
        except Exception as e:
            print(
                f"An error occurred while processing {zip_file}. Exception: {e}"
            )

    # Add student submission files
    # add_files_from_frontend(configs["student_submission_files"])
    # print(f"Unit Test files: {configs['unit_tests_files'].read()}")
    add_files_from_frontend(configs["unit_tests_files"])
    add_files_from_frontend(configs["starter_code"])
    add_files_from_frontend(configs["data_files"])

    assignment_name = configs["assignment_name"]

    # Replace replace_me in RunTests.java
    with open("templates/java/RunTests.java", "r") as f:
        test_files = [file.replace(".java", ".class") for file in test_files]
        run_tests_content = f.read()
        run_tests_content = run_tests_content.replace(
            "replace_me", ", ".join(test_files)
        )
        file_list.append(("src/RunTests.java", run_tests_content))

    zip_memory = zip_files(file_list, f"{assignment_name}.zip")
    return send_file(
        io.BytesIO(
            zip_memory.getvalue()
        ),  # Ensure proper content is read from BytesIO
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{assignment_name}.zip",
    )


def build_python(configs):
    file_list = []

    # Add backend files
    backend_files_paths = [
        configs["setup_bash"],
        configs["run_tests_template"],
        configs["requirements"],
    ]

    # modify the run_autograder file to copy the required files
    with open(configs["run_autograder_bash"], "r") as f:
        run_autograder_content = f.read()
        required_files = configs["student_submission_files"]
        for file_name in required_files:
            run_autograder_content += f"cp /autograder/submission/{file_name} /autograder/source/{file_name}\n"
        run_autograder_content += "cd /autograder/source\n"
        run_autograder_content += "python3 run_tests.py\n"
        file_list.append(("run_autograder", run_autograder_content))

    for file_path in backend_files_paths:
        with open(file_path, "rb") as f:
            file_content = f.read()
            file_list.append((os.path.basename(file_path), file_content))

    # Handle files uploaded from frontend
    def add_files_from_frontend(zip_file):
        if zip_file is None:
            return
        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                print(f"zip_ref: {zip_ref}")
                print(f"zip_ref.infolist(): {zip_ref.infolist()}")
                for file_info in zip_ref.infolist():
                    print(f"file_info: {file_info}")
                    with zip_ref.open(file_info) as f:
                        f.seek(0)
                        file_content = f.read()
                        if zip_file == configs["unit_tests_files"]:
                            file_list.append(
                                (f"tests/{file_info.filename}", file_content)
                            )
                        elif zip_file == configs["starter_code"]:
                            file_list.append(
                                (f"src/{file_info.filename}", file_content)
                            )
                        elif zip_file == configs["data_files"]:
                            file_list.append(
                                (f"data/{file_info.filename}", file_content)
                            )
                        else:
                            file_list.append((file_info.filename, file_content))
        except zipfile.BadZipFile as e:
            if zip_file == configs["unit_tests_files"]:
                configs["unit_tests_files"].seek(0)
                file_list.append(
                    (
                        "tests/" + configs["unit_tests_files"].filename,
                        configs["unit_tests_files"].read(),
                    )
                )
            elif zip_file == configs["starter_code"]:
                configs["starter_code"].seek(0)
                file_list.append(
                    (
                        "src/" + configs["starter_code"].filename,
                        configs["starter_code"].read(),
                    )
                )
            elif zip_file == configs["data_files"]:
                configs["data_files"].seek(0)
                file_list.append(
                    (
                        "data/" + configs["data_files"].filename,
                        configs["data_files"].read(),
                    )
                )
            else:
                zip_file.seek(0)
                file_content = zip_file.read()
                file_list.append(
                    (os.path.basename(zip_file.name), file_content)
                )
        except Exception as e:
            print(
                f"An error occurred while processing {zip_file}. Exception: {e}"
            )

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
