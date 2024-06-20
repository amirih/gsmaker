
def get_java_configs(course_name):
    configs= {}
    # dont modify or do by caution:
    configs['language'] = 'java'
    configs['output_dir'] = f'out/{course_name}'
    configs['lib_dir'] = 'templates/java/lib'
    configs['compile_bash'] = 'templates/java/compile.sh'
    configs['run_bash'] = 'templates/java/run.sh'
    configs['setup_bash'] = 'templates/java/setup.sh'
    configs['run_tests_template'] = 'templates/java/RunTests.java'
    configs['gs_lib_dir'] = 'templates/java/com'
    configs['run_autograder_bash'] = 'templates/java/run_autograder'
    configs['zip_bash'] = 'templates/java/zip.sh'
    configs['zip_dir'] = 'out/autograd.zip'

    # update these to your own files:
    configs['students_file_names'] = ['HelloGradeScope.java', 'HelloGradeScope2.java']
    configs['test_cases'] = ['courses/hello_gradescope/test_cases/HelloGradeScopeTest.java', 'courses/hello_gradescope/test_cases/HelloGradeScopeTest2.java']
    return configs

def get_configs(course_name='autograd',language='java'):
    if language == 'java':
        return get_java_configs(course_name)
    else:
        raise Exception('Language not supported yet')