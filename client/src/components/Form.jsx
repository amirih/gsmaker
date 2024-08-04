import React, { useState } from 'react';
import axios from 'axios';

function Form() {
    const [formData, setFormData] = useState({
        assignment_name: '',
        language: '',
        required_files: '',
        test_cases: null,
        starter_code: null,
        data: null,
    });

    const handleChange = (e) => {
        const { name, value, files } = e.target;
        if (files) {
            setFormData({ ...formData, [name]: files[0] });
        } else {
            setFormData({ ...formData, [name]: value });
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        const data = new FormData();
        data.append('assignment_name', formData.assignment_name);
        data.append('language', formData.language);
        data.append('required_files', formData.required_files);
        data.append('test_cases', formData.test_cases);
        data.append('starter_code', formData.starter_code);
        data.append('data', formData.data);

        axios.post('https://autograder.mathcs.emory.edu//api/form-submit', data, {
            responseType: 'blob' // Ensure response is treated as a Blob
        })
            .then(response => {
                const url = window.URL.createObjectURL(new Blob([response.data]));
                const a = document.createElement('a');
                a.href = url;
                a.download = `${formData.assignment_name}.zip`;
                document.body.appendChild(a);
                a.click();
                a.remove();
            })
            .catch((err) => {
                console.log(err);
            });

        setFormData({
            assignment_name: '',
            language: '',
            required_files: '',
            test_cases: null,
            starter_code: null,
            data: null,
        });
    };

    return (
        <form onSubmit={handleSubmit}>
            <label>
                Enter your assignment name:<span style={{ color: 'red' }}>*</span>
                <input required type="text" name="assignment_name" value={formData.assignment_name} onChange={handleChange} placeholder="CS326-HW1" />
            </label>
            <br />
            <label>
                What language is the assignment in?<span style={{ color: 'red' }}>*</span>
                <select required name="language" value={formData.language} onChange={handleChange}>
                    <option value="python">Python</option>
                    <option value="java">Java</option>
                    {/* <option value="c">C</option> */}
                </select>
            </label>
            <br />
            <label>
                What file are the students supposed to submit? (Enter a comma separated list with the file extension)<span style={{ color: 'red' }}>*</span>
                <input required type="text" name="required_files" value={formData.required_files} onChange={handleChange} placeholder="model.py, README.txt" />
            </label>
            <br />
            <label>
                Enter a .zip file containing test cases. See the grading handbook for more details<span style={{ color: 'red' }}>*</span>
                <input required type="file" name="test_cases" onChange={handleChange} />
            </label>
            <br />
            <label>
                (OPTIONAL) Does your assignment come with a starter code? If yes, upload it here as a .zip file.
                <input type="file" name="starter_code" onChange={handleChange} />
            </label>
            <br />
            <label>
                (OPTIONAL) Does your assignment require any data files? If yes, upload them here as a .zip file.
                <input type="file" name="data" onChange={handleChange} />
            </label>
            <button type="submit">Submit</button>
        </form>
    );
}

export default Form;
