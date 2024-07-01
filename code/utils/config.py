import os


def get_java_configs(assignment_name):
    configs = {}
    # dont modify or do by caution:
    configs["language"] = "java"
    configs["output_dir"] = os.path.join("out", assignment_name)
    configs["lib_dir"] = os.path.join("templates", "java", "lib")
    configs["compile_bash"] = os.path.join("templates", "java", "compile.sh")
    configs["run_bash"] = os.path.join("templates", "java", "run.sh")
    configs["setup_bash"] = os.path.join("templates", "java", "setup.sh")
    configs["run_tests_template"] = os.path.join(
        "templates", "java", "RunTests.java"
    )
    configs["gs_lib_dir"] = os.path.join("templates", "java", "com")
    configs["run_autograder_bash"] = os.path.join(
        "templates", "java", "run_autograder"
    )
    configs["zip_bash"] = os.path.join("templates", "java", "zip.sh")
    configs["zip_dir"] = os.path.join("out", "autograd.zip")

    # update these to your own files:
    configs["students_file_names"] = ["Playlist.java"]
    configs["test_cases"] = [
        os.path.join(
            "assignments",
            "cs171",
            "test_cases",
            "PlaylistTest.java",
        )
    ]

    # add option to include starter code here
    configs["starter_code"] = [
        os.path.join("assignments", "cs171", "starter_code", "Episode.java")
    ]

    return configs


def get_python_configs(assignment_name):
    configs = {}
    # dont modify or do by caution:
    configs["language"] = "python"
    configs["output_dir"] = os.path.join("out", assignment_name)
    configs["setup_bash"] = os.path.join("templates", "python", "setup.sh")
    configs["run_tests_template"] = os.path.join(
        "templates", "python", "run_tests.py"
    )
    configs["run_autograder_bash"] = os.path.join(
        "templates", "python", "run_autograder"
    )
    configs["zip_dir"] = os.path.join("out", "autograd.zip")

    # update these to your own files:
    configs["students_file_names"] = []
    configs["test_cases"] = [
        os.path.join(
            "assignments",
            "cs334-hw3",
            "test_cases",
            "test_assess.py",
        ),
        os.path.join(
            "assignments",
            "cs334-hw3",
            "test_cases",
            "test_dt.py",
        ),
        os.path.join(
            "assignments",
            "cs334-hw3",
            "test_cases",
            "test_files.py",
        ),
    ]

    # add option to include starter code here
    configs["starter_code"] = [
        # os.path.join("assignments", "cs171", "starter_code", "episode.py")
    ]

    configs["data_files"] = [
        os.path.join(
            "assignments",
            "cs334-hw3",
            "data",
            "space_testx.csv",
        ),
        os.path.join(
            "assignments",
            "cs334-hw3",
            "data",
            "space_testy.csv",
        ),
        os.path.join(
            "assignments",
            "cs334-hw3",
            "data",
            "space_trainx.csv",
        ),
        os.path.join(
            "assignments",
            "cs334-hw3",
            "data",
            "space_trainy.csv",
        ),
    ]

    configs["requirements"] = os.path.join(
        "templates", "python", "requirements.txt"
    )

    return configs


def get_configs(assignment_name="autograd", language="java"):
    if language == "java":
        return get_java_configs(assignment_name)
    elif language == "python":
        return get_python_configs(assignment_name)
    else:
        raise Exception("Language not supported yet")
