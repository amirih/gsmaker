import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import com.gradescope.jh61b.grader.GradedTestListenerJSON;

@RunWith(Suite.class)
@Suite.SuiteClasses({         HelloGradeScopeTest.class,
        HelloGradeScopeTest2.class,
 })
public class RunTests {
    public static void main(String[] args) {
        JUnitCore runner = new JUnitCore();
        runner.addListener(new GradedTestListenerJSON());
        Result r = runner.run(RunTests.class);
    }
}
