import utils.file as file
def get_run_autograder(configs):
    run_autograder_template = file.read(configs['run_autograder_bash'])

    update_text = get_copy_files(configs['students_file_names'])
    run_autograder_template = run_autograder_template.replace('replace_me', update_text)
    return run_autograder_template


def get_copy_files(files):
    text = ''
    for file in files:
        text += f'cp /autograder/submission/{file} /autograder/source/src/\n'
    return text

def get_run_tests(configs):
    text = file.read(configs['run_tests_template'])
    suit_classes = ''
    for test_file in configs['test_cases']:
        file_name = test_file.split('/')[-1].split('.')[0]
        suit_classes += f'        {file_name}.class,\n'
    text = text.replace('replace_me', suit_classes)
    return text

