from flask import Flask, request, jsonify
from flask_cors import CORS
import zipfile
import io
import src.utils.autograder as autograder
import src.utils.config as config
import src.utils.data_utils as data_utils

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Welcome to the Flask backend!"


@app.route("/api/form-submit", methods=["POST"])
def form_submit():
    assignment_name = request.form.get("assignment_name")
    language = request.form.get("language")
    required_files = request.form.get("required_files")
    test_cases = request.files.getlist("test_cases")
    starter_code = request.files.getlist("starter_code")
    data = request.files.getlist("data")

    # Create a zip file
    test_cases_zip_file = io.BytesIO()
    with zipfile.ZipFile(test_cases_zip_file, "w") as zipf:
        for file in test_cases:
            zipf.writestr(file.filename, file.read())

    if not starter_code:
        starter_code_zip_file = None
    else:
        starter_code_zip_file = io.BytesIO()
        with zipfile.ZipFile(starter_code_zip_file, "w") as zipf:
            for file in starter_code:
                zipf.writestr(file.filename, file.read())

    if not data:
        data_zip_file = None
    else:
        data_zip_file = io.BytesIO()
        with zipfile.ZipFile(data_zip_file, "w") as zipf:
            for file in data:
                zipf.writestr(file.filename, file.read())

    print(
        assignment_name,
        language,
        required_files,
        test_cases,
        starter_code,
        data,
    )

    required_files, test_cases_dir, starter_code_dir, data_dir = (
        data_utils.process_data(
            required_files,
            test_cases_zip_file,
            starter_code_zip_file,
            data_zip_file,
        )
    )

    configuration = config.get_configs(
        assignment_name,
        language,
        required_files,
        test_cases_dir,
        starter_code_dir,
        data_dir,
    )

    out_dir = autograder.build(configuration)

    return out_dir


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
