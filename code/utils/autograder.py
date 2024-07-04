import utils.file as file
import utils.gs.java as gs_java
import os


def build(configs):
    output_dir = configs["output_dir"]
    language = configs["language"]
    file.remove(output_dir)
    if language == "java":
        return build_java(configs, output_dir)
    elif language == "python":
        return build_python(configs, output_dir)
    else:
        raise Exception("Language not supported yet")


def build_java(configs, output_dir):
    file.copy(configs["compile_bash"], output_dir + "/compile.sh")
    file.copy(configs["lib_dir"], output_dir + "/lib")
    file.copy(configs["run_bash"], output_dir + "/run.sh")
    file.copy(configs["setup_bash"], output_dir + "/setup.sh")
    file.copy(configs["gs_lib_dir"], output_dir + "/src/")
    file.copy(configs["test_cases"], output_dir + "/src/")
    file.copy(configs["starter_code"], output_dir + "/src/")

    file.write(
        gs_java.get_run_autograder(configs), output_dir + "/run_autograder"
    )
    file.write(
        gs_java.get_run_tests(configs), output_dir + "/src/RunTests.java"
    )
    file.zip(output_dir)
    return output_dir


def build_python(configs, output_dir):
    file.copy(configs["run_tests_template"], output_dir + "/run_tests.py")
    file.copy(configs["setup_bash"], output_dir + "/setup.sh")
    file.copy(configs["starter_code"], output_dir)
    file.copy(configs["requirements"], output_dir)
    file.copy(configs["data_files"], output_dir + "/data/")

    # Open a new .sh file in write mode
    with open(output_dir + "/run_autograder", "w") as autograder_file:
        autograder_file.write("#!/bin/bash\n")

    required_files = configs["student_submission_files"]

    for file_name in required_files:
        with open(output_dir + "/run_autograder", "a") as autograder_file:
            autograder_file.write(
                f"cp /autograder/submission/{file_name} /autograder/source/{file_name}\n"
            )

    with open(output_dir + "/run_autograder", "a") as autograder_file:
        autograder_file.write("cd /autograder/source\npython3 run_tests.py\n")

    unit_test_file_names = configs["unit_tests_files"]

    for test_case in unit_test_file_names:
        path = os.path.join(configs["unit_tests_dir"], test_case)
        file.copy(path, output_dir + "/tests/")

    file.zip(output_dir)
    return output_dir
