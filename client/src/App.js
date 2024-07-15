import './App.css';

function App() {
  return (
    <div className="App">
      <div className="title">
        Gradescope Autograder Generator
      </div>
      <div className='body'>
        <div className="instructions">
          <p>Instructions:</p>
        </div>
        <div className="form">
          <form method="POST" action="">
            <label>
              <p>Enter your assignment name: </p>
              <input type="text" name="assignment-name" placeholder="CS326-HW1" />
            </label>
            <label>
              <p>What language is the assignment in?</p>
              <select name="assignment-language">
                <option value="python">Python</option>
                <option value="java">Java</option>
                <option value="c">C</option>
              </select>
            </label>
            <label>
              <p>What file are the students supposed to submit? (Enter a comma separated list with the file extension)</p>
              <input type="text" name="required-files" placeholder="model.py, README.txt" />
            </label>
            <label>
              <p>Enter a .zip file containing test cases. See the grading handbook for more details</p>
              <input type="file" name="test-cases" />
            </label>
            <label>
              <p>(OPTIONAL) Does your assignment come with a starter code? If yes, upload it here as a .zip file.</p>
              <input type="file" name="starter-code" />
            </label>
            <label>
              <p>(OPTIONAL) Does your assignment require any data files? If yes, upload them here as a .zip file.</p>
              <input type="file" name="data" />
            </label>
            <br />
            <br />
            <input type="submit" value="Generate Autograder" />
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
