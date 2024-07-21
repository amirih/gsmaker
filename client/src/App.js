import Form from './components/Form';

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
          <Form />
        </div>
      </div>
    </div>
  );
}

export default App;
