import utils.file as file
import utils.gs.java as gs_java


def build(configs):
    output_dir = configs["output_dir"]
    language = configs["language"]
    file.remove(output_dir)
    if language == "java":
        return build_java(configs, output_dir)
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
