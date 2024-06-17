import org.junit.Test;
import static org.junit.Assert.*;
import com.gradescope.jh61b.grader.GradedTest;

public class HelloGradeScopeTest {
    @Test
    @GradedTest(name="Testing HelloGradeScope()", max_score=1)
    public void testHelloGradeScope() {
        HelloGradeScope testObject = new HelloGradeScope();
        assertEquals("Hello GradeScope", testObject.sayHi());
    }
}
