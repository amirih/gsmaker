from flask import Flask, request, jsonify
from flask_cors import CORS
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
    test_cases = request.files.get("test_cases")
    starter_code = request.files.get("starter_code")
    data = request.files.get("data")

    required_files, test_cases_dir, starter_code_dir, data_dir = (
        data_utils.process_data(required_files, test_cases, starter_code, data)
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
