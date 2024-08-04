import Form from './components/Form';
import './styles/App.css';

function App() {
  return (
    <div className="container">
      <div className="content">
        <div className="title">
          Gradescope Autograder Generator
        </div>
        <div className='body'>
          <div className="instructions">
            <p>Instructions:</p>
            <div>
              <p>1. Current version supports only Python and Java. Support for other languages coming soon!</p>
              <p>2. When uploading multiple files, zip them from within the directory. See <a target="_blank" href="https://imgur.com/a/gradescope-zipping-tutorial-uPWTpYf">here.</a></p>
            </div>
          </div>
          <div className="form">
            <Form />
          </div>
          <span style={{ color: 'red' }}>*</span> Indicates required field
        </div>
      </div>
    </div>
  );
}

export default App;
