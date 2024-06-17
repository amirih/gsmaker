# GradeScope Autograder Setup

This repository contains the necessary files and scripts to set up an autograder for GradeScope. It is designed to be flexible, supporting multiple programming languages through template-based setups.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. This project is compatible with Linux. Windows users may need to adapt the file handling utilities in `code/utils/file.py` using Python’s `os` module.

### Installation

Clone the repository to your local machine using the following command:

git clone https://github.com/amirih/gsmaker.git



### Project Structure

Here's an overview of the project directory structure:

- `code/`
  - `make.py`: Entry point script to generate necessary files from templates.
  - `utils/`
    - `config.py`: Contains configuration settings for the autograder.
    - `file.py`: File handling utilities for Linux. Windows users should adapt this using Python’s `os` module.
    - `gs/`
      - `java.py`: Java-specific grading utilities.
      - `autograder.py`: Core autograding script.
- `test_cases/`: Directory to place your test cases.
- `templates/`: Contains language-specific templates.
  - `java/`: Java template with necessary scripts and test runner.

### Configuration

Configure the autograder by editing the `config.py` file under `code/utils/`.

### Running

To set up the autograder, run:


python code/make.py


This script initializes the grading environment based on the templates and configurations defined.

## Deployment

Output will be generated in the `out/` directory. Here is what each sub-directory contains:
- `gs101/`: Example course directory.
  - `lib/`: Libraries and dependencies.
  - `src/`: Source files.
  - `compile.sh`: Script to compile the source code.
  - `run_autograder`: Script to execute the autograder.
  - `run.sh`: General run script.
  - `setup.sh`: Setup script for environment configuration.

## Contribution

Contributions to this project are welcome. Please fork the repository and submit a pull request.
