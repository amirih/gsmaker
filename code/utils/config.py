import os


def get_java_configs(course_name):
    configs = {}
    # dont modify or do by caution:
    configs["language"] = "java"
    configs["output_dir"] = os.path.join("out", course_name)
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
            "courses",
            "cs171",
            "test_cases",
            "PlaylistTest.java",
        )
    ]

    # add option to include starter code here
    configs["starter_code"] = [
        os.path.join("courses", "cs171", "starter_code", "Episode.java")
    ]

    return configs


def get_configs(course_name="autograd", language="java"):
    if language == "java":
        return get_java_configs(course_name)
    else:
        raise Exception("Language not supported yet")
