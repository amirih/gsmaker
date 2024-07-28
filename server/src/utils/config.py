def get_java_configs(
    assignment_name,
    required_files,
    test_cases_list,
    starter_code_list,
    data_list,
):
    configs = {}
    # dont modify or do by caution:
    configs["assignment_name"] = assignment_name
    configs["language"] = "java"
    configs["lib_dir"] = "templates/java/lib.zip"
    configs["compile_bash"] = "templates/java/compile.sh"
    configs["run_bash"] = "templates/java/run.sh"
    configs["setup_bash"] = "templates/java/setup.sh"
    configs["run_tests_template"] = "templates/java/RunTests.java"
    configs["gs_lib_dir"] = "templates/java/com.zip"
    configs["run_autograder_bash"] = "templates/java/run_autograder"

    # update these to your own files:
    configs["student_submission_files"] = required_files
    configs["unit_tests_files"] = test_cases_list
    configs["starter_code"] = starter_code_list
    configs["data_files"] = data_list

    return configs


def get_python_configs(
    assignment_name,
    required_files,
    test_cases_list,
    starter_code_list,
    data_list,
):
    configs = {}
    # dont modify or do by caution:
    configs["assignment_name"] = assignment_name
    configs["language"] = "python"
    configs["setup_bash"] = "templates/python/setup.sh"
    configs["run_tests_template"] = "templates/python/run_tests.py"
    configs["run_autograder_bash"] = "templates/python/run_autograder"
    configs["requirements"] = "templates/python/requirements.txt"

    # update these to your own files:
    configs["student_submission_files"] = required_files
    configs["unit_tests_files"] = test_cases_list
    configs["starter_code"] = starter_code_list
    configs["data_files"] = data_list
    return configs


def get_configs(
    assignment_name,
    language,
    required_files,
    test_cases_list,
    starter_code_list,
    data_list,
):
    if language == "java":
        return get_java_configs(
            assignment_name,
            required_files,
            test_cases_list,
            starter_code_list,
            data_list,
        )
    elif language == "python" or language == "":
        return get_python_configs(
            assignment_name,
            required_files,
            test_cases_list,
            starter_code_list,
            data_list,
        )
    else:
        raise Exception("Language not supported yet")
