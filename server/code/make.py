import utils.autograder as autograder
import utils.config as config

if __name__ == "__main__":
    assignment_name = "cs171"
    language = "java"
    configuration = config.get_configs(assignment_name, language)
    out_dir = autograder.build(configuration)
    print(f"{out_dir} created")
