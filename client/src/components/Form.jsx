import React, { useState } from 'react';
import axios from 'axios';
import JSZip from 'jszip';

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
        if (e.target.name === 'test_cases' || e.target.name === 'starter_code' || e.target.name === 'data') {
            setFormData({
                ...formData,
                [e.target.name]: e.target.files
            });
        } else {
            setFormData({
                ...formData,
                [e.target.name]: e.target.value
            });
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const data = new FormData();
        data.append('assignment_name', formData.assignment_name);
        data.append('language', formData.language);
        data.append('required_files', formData.required_files);

        // Append each file in the test_cases FileList
        if (formData.test_cases) {
            for (let i = 0; i < formData.test_cases.length; i++) {
                data.append('test_cases', formData.test_cases[i]);
            }
        }

        // Append each file in the starter_code FileList
        if (formData.starter_code) {
            for (let i = 0; i < formData.starter_code.length; i++) {
                data.append('starter_code', formData.starter_code[i]);
            }
        }

        // Append each file in the data FileList
        if (formData.data) {
            for (let i = 0; i < formData.data.length; i++) {
                data.append('data', formData.data[i]);
            }
        }

        for (let [key, value] of data.entries()) {
            console.log(`${key}:`, value);
        }

        // axios.post('http://localhost:5000/api/form-submit', data, {
        axios.post('https://autograder.mathcs.emory.edu/api/form-submit', data, {
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
            })
            .finally(() => {
                window.location.reload();
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

    // INCORRECT ZIPPING LOGIC
    // const handleSubmit = async (e) => {
    //     e.preventDefault();

    //     let updatedFormData = { ...formData };
    //     // console.log(updatedFormData);
    //     // console.log(updatedFormData.test_cases);
    //     // Zip the test cases
    //     if (formData.test_cases) {
    //         // test cases is a file list, so we need to zip them
    //         const testCasesZip = new JSZip();
    //         for (let i = 0; i < formData.test_cases.length; i++) {
    //             testCasesZip.file(formData.test_cases[i].name, formData.test_cases[i]);
    //         }
    //         const testBlob = await testCasesZip.generateAsync({ type: 'blob' });
    //         updatedFormData.test_cases = new File([testBlob], 'test_cases.zip', { type: 'application/zip' });
    //     }
    //     if (formData.test_cases) {
    //         const testCasesZip = new JSZip();
    //         for (let i = 0; i < formData.test_cases.length; i++) {
    //             testCasesZip.file(formData.test_cases[i].name, formData.test_cases[i]);
    //         }
    //         const testBlob = await testCasesZip.generateAsync({ type: 'blob' });
    //         updatedFormData.test_cases = new File([testBlob], 'test_cases.zip', { type: 'application/zip' });
    //     }


    //     // Zip the starter code
    //     if (formData.starter_code) {
    //         const starterZip = new JSZip();
    //         for (let i = 0; i < formData.starter_code.length; i++) {
    //             starterZip.file(formData.starter_code[i].name, formData.starter_code[i]);
    //         }
    //         const starterBlob = await starterZip.generateAsync({ type: 'blob' });
    //         updatedFormData.starter_code = new File([starterBlob], 'starter_code.zip', { type: 'application/zip' });
    //     }

    //     // Zip the data
    //     if (formData.data) {
    //         const dataZip = new JSZip();
    //         for (let i = 0; i < formData.data.length; i++) {
    //             dataZip.file(formData.data[i].name, formData.data[i]);
    //         }
    //         const dataBlob = await dataZip.generateAsync({ type: 'blob' });
    //         updatedFormData.data = new File([dataBlob], 'data.zip', {
    //             type: 'application/zip'
    //         });
    //     }

    //     const data = new FormData();
    //     data.append('assignment_name', formData.assignment_name);
    //     data.append('language', formData.language);
    //     data.append('required_files', formData.required_files);
    //     data.append('test_cases', formData.test_cases);
    //     data.append('starter_code', formData.starter_code);
    //     data.append('data', formData.data);
    //     for (let [key, value] of data.entries()) {
    //         console.log(`${key}:`, value);
    //     }
    //     axios.post('http://localhost:5000/api/form-submit', data, {
    //         // axios.post('https://autograder.mathcs.emory.edu/api/form-submit', data, {
    //         responseType: 'blob' // Ensure response is treated as a Blob
    //     })
    //         .then(response => {
    //             const url = window.URL.createObjectURL(new Blob([response.data]));
    //             const a = document.createElement('a');
    //             a.href = url;
    //             a.download = `${formData.assignment_name}.zip`;
    //             document.body.appendChild(a);
    //             a.click();
    //             a.remove();
    //         })
    //         .catch((err) => {
    //             console.log(err);
    //         })
    //         .finally(() => {
    //             window.location.reload();
    //         })
    //         ;

    //     setFormData({
    //         assignment_name: '',
    //         language: '',
    //         required_files: '',
    //         test_cases: null,
    //         starter_code: null,
    //         data: null,
    //     });
    // };

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
                <input required multiple type="file" name="test_cases" onChange={handleChange} />
            </label>
            <br />
            <label>
                (OPTIONAL) Does your assignment need any auxiliary files? If yes, upload them here.
                <input multiple type="file" name="starter_code" onChange={handleChange} />
            </label>
            <br />
            <label>
                (OPTIONAL) Does your assignment require any data files? If yes, upload them here.
                <input multiple type="file" name="data" onChange={handleChange} />
            </label>
            <button type="submit">Submit</button>
        </form>
    );
}

export default Form;
