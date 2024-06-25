import utils.autograder as autograder
import utils.config as config

if __name__ == "__main__":
    course_name = "cs171"  # change this to assignment name maybe?
    language = "java"
    configuration = config.get_configs(course_name, language)
    out_dir = autograder.build(configuration)
    print(f"{out_dir} created")
