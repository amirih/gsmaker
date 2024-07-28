import zipfile
import tempfile


def extract_zip(zip_file):
    temp_dir = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir.name)
    return temp_dir


def process_data(required_files, test_cases, starter_code, data):
    required_files = required_files.replace(" ", "").split(",")

    # if test_cases.filename.endswith(".zip"):
    #     test_cases_dir = extract_zip(test_cases)
    #     print(test_cases)
    #     print(test_cases_dir)
    # else:
    test_cases_dir = test_cases

    if starter_code:
        # if starter_code.filename.endswith(".zip"):
        #     starter_code_dir = extract_zip(starter_code)
        # else:
        starter_code_dir = starter_code
    else:
        starter_code_dir = None

    if data:
        # if data.filename.endswith(".zip"):
        #     data_dir = extract_zip(data)
        # else:
        data_dir = data
    else:
        data_dir = None

    return required_files, test_cases_dir, starter_code_dir, data_dir
