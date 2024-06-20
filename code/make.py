
import utils.autograder as autograder
import utils.config as config
import utils.file as file

if __name__ == '__main__':
    course_name = 'gs101'
    language = 'java'
    configuration = config.get_configs(course_name, language)
    out_dir = autograder.build(configuration)
    print(f'{out_dir} created')
    
