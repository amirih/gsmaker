import org.junit.Test;
import static org.junit.Assert.*;
import com.gradescope.jh61b.grader.GradedTest;

public class HelloGradeScopeTest2 {
    @Test
    @GradedTest(name = "Testing HelloGradeScope()", max_score = 1)
    public void testHelloGradeScope2() {
        HelloGradeScope2 testObject = new HelloGradeScope2();
        assertEquals("Hello GradeScope", testObject.sayHi());
    }
}
