import org.junit.Test;
import static org.junit.Assert.*;
// This is an annotation for assigning point values to tests
import com.gradescope.jh61b.grader.GradedTest;

public class PlaylistTest {
    @Test
    @GradedTest(name = "Testing addFirst(A, 24) to empty playlist", max_score =
    2)
    public void test_addFirst() {
        Playlist playlist = new Playlist();
        try {
            playlist.addFirst("A", 24);
        } catch (Exception e) {
            fail("Unexpected exception adding to empty Playlist");
        }
        assertEquals("[HEAD] (A|24.0MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name = "Testing addFirst with 3 elements", max_score = 2)
    public void test_addFirst2() {
        Playlist playlist = new Playlist();
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 31.5);
            playlist.addFirst("C", 40.2);
        } catch (Exception e) {
            fail("Unexpected exception adding to Playlist");
        }
        // check that format is correct
        assertEquals("[HEAD] (C|40.2MIN) -> (B|31.5MIN) -> (A|24.0MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name = "Testing addFirst with 3 elements, print via toReverseString() to test prev links", max_score = 2)
    public void test_addFirst2_printreverse() {
        Playlist playlist = new Playlist();
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 31.5);
            playlist.addFirst("C", 40.2);
        } catch (Exception e) {
            fail("Unexpected exception adding to Playlist");
        }
        // check that format is correct
        assertEquals("[END] (A|24.0MIN) -> (B|31.5MIN) -> (C|40.2MIN) [HEAD]\n",
                playlist.toReverseString());
    }

    @Test
    @GradedTest(name = "Testing addFirst with 7 elements", max_score = 2,
    visibility = "after_published")
    public void test_addFirst3() {
        Playlist playlist = new Playlist();
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 31.5);
            playlist.addFirst("C", 40.2);
            playlist.addFirst("D", 24.5);
            playlist.addFirst("E", 10.2);
            playlist.addFirst("F", 13.1);
            playlist.addFirst("G", 5.9);
        } catch (Exception e) {
            fail("Unexpected exception adding to Playlist");
        }
        // check that format is correct
        assertEquals(
                "[HEAD] (G|5.9MIN) -> (F|13.1MIN) -> (E|10.2MIN) -> (D|24.5MIN) -> (C|40.2MIN) -> (B|31.5MIN) -> (A|24.0MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name = "Testing addFirst with 7 elements, print via toReverseString() to test prev links", max_score = 2,
    visibility = "after_published")
    public void test_addFirst3_printreverse() {
        Playlist playlist = new Playlist();
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 31.5);
            playlist.addFirst("C", 40.2);
            playlist.addFirst("D", 24.5);
            playlist.addFirst("E", 10.2);
            playlist.addFirst("F", 13.1);
            playlist.addFirst("G", 5.9);
        } catch (Exception e) {
            fail("Unexpected exception adding to Playlist");
        }
        // check that format is correct
        assertEquals(
                "[END] (A|24.0MIN) -> (B|31.5MIN) -> (C|40.2MIN) -> (D|24.5MIN) -> (E|10.2MIN) -> (F|13.1MIN) -> (G|5.9MIN) [HEAD]\n",
                playlist.toReverseString());
    }

    @Test
    @GradedTest(name = "Testing addLast(A, 24) to empty playlist", max_score = 2)
    public void test_addLast() {
        Playlist playlist = new Playlist();
        try {
            playlist.addLast("A", 24);
        } catch (Exception e) {
            fail("Unexpected exception adding to empty Playlist");
        }
        assertEquals("[HEAD] (A|24.0MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name = "Testing addLast with 3 elements", max_score = 2,
    visibility = "after_published")
    public void test_addLast2() {
        Playlist playlist = new Playlist();
        try {
            playlist.addLast("A", 24);
            playlist.addLast("B", 31.5);
            playlist.addLast("C", 40.2);
        } catch (Exception e) {
            fail("Unexpected exception adding to Playlist");
        }
        // check that format is correct
        assertEquals("[HEAD] (A|24.0MIN) -> (B|31.5MIN) -> (C|40.2MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name = "Testing addLast with 7 elements", max_score = 2,
    visibility = "after_published")
    public void test_addLast3() {
        Playlist playlist = new Playlist();
        try {
            playlist.addLast("A", 24);
            playlist.addLast("B", 31.5);
            playlist.addLast("C", 40.2);
            playlist.addLast("D", 24.5);
            playlist.addLast("E", 10.2);
            playlist.addLast("F", 13.1);
            playlist.addLast("G", 5.9);
        } catch (Exception e) {
            fail("Unexpected exception adding to Playlist");
        }
        // check that format is correct
        assertEquals(
                "[HEAD] (A|24.0MIN) -> (B|31.5MIN) -> (C|40.2MIN) -> (D|24.5MIN) -> (E|10.2MIN) -> (F|13.1MIN) -> (G|5.9MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name = "Testing addLast with 7 elements, print via toReverseString() to test prev links", max_score = 2,
    visibility = "after_published")
    public void test_addLast3_printreverse() {
        Playlist playlist = new Playlist();
        try {
            playlist.addLast("A", 24);
            playlist.addLast("B", 31.5);
            playlist.addLast("C", 40.2);
            playlist.addLast("D", 24.5);
            playlist.addLast("E", 10.2);
            playlist.addLast("F", 13.1);
            playlist.addLast("G", 5.9);
        } catch (Exception e) {
            fail("Unexpected exception adding to Playlist");
        }
        // check that format is correct
        assertEquals(
                "[END] (G|5.9MIN) -> (F|13.1MIN) -> (E|10.2MIN) -> (D|24.5MIN) -> (C|40.2MIN) -> (B|31.5MIN) -> (A|24.0MIN) [HEAD]\n",
                playlist.toReverseString());
    }

    @Test
    @GradedTest(name = "Testing deleteFirst on playlist (B|27.5) -> (A|24)",
    max_score = 2)
    public void test_deleteFirst() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 27.5);
            tmp = playlist.deleteFirst();
        } catch (Exception e) {
            fail("Unexpected exception in deleteFirst from Playlist of size=2");
        }
        assertEquals("B", tmp.title);
        assertEquals(27.5, tmp.length, 0.01);
        assertEquals(
                "[HEAD] (A|24.0MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name = "Testing deleteFirst on playlist (B|27.5) -> (A|24), print via toReverseString()",
    max_score = 2, visibility = "after_published")
    public void test_deleteFirst_printreverse() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 27.5);
            tmp = playlist.deleteFirst();
        } catch (Exception e) {
            fail("Unexpected exception in deleteFirst from Playlist of size=2");
        }
        assertEquals("B", tmp.title);
        assertEquals(27.5, tmp.length, 0.01);
        assertEquals(
                "[END] (A|24.0MIN) [HEAD]\n",
                playlist.toReverseString());
    }

    @Test
    @GradedTest(name = "Testing deleteFirst on playlist (A|24)", max_score = 2,
    visibility = "after_published")
    public void test_deleteFirst1() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            tmp = playlist.deleteFirst();
        } catch (Exception e) {
            fail("Unexpected exception in deleteFirst from Playlist of size=1");
        }
        assertEquals("A", tmp.title);
        assertEquals(24.0, tmp.length, 0.01);
        assertTrue(playlist.isEmpty());
    }

    @Test
    @GradedTest(name = "Testing deleteFirst on playlist (D|24.5) -> (C|23) -> (B|27.5) -> (A|24)", max_score = 2, visibility = "after_published")
    public void test_deleteFirst2() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 27.5);
            playlist.addFirst("C", 23);
            playlist.addFirst("D", 24.5);
            tmp = playlist.deleteFirst();
        } catch (Exception e) {
            fail("Unexpected exception in deleteFirst from Playlist of size=4");
        }
        assertEquals("D", tmp.title);
        assertEquals(24.5, tmp.length, 0.01);
        assertEquals("[HEAD] (C|23.0MIN) -> (B|27.5MIN) -> (A|24.0MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name = "Testing deleteFirst on playlist (D|24.5) -> (C|23) -> (B|27.5) -> (A|24), print via toReverseString()", max_score = 2, visibility = "after_published")
    public void test_deleteFirst2_printreverse() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 27.5);
            playlist.addFirst("C", 23);
            playlist.addFirst("D", 24.5);
            tmp = playlist.deleteFirst();
        } catch (Exception e) {
            fail("Unexpected exception in deleteFirst from Playlist of size=4");
        }
        assertEquals("D", tmp.title);
        assertEquals(24.5, tmp.length, 0.01);
        assertEquals("[END] (A|24.0MIN) -> (B|27.5MIN) -> (C|23.0MIN) [HEAD]\n",
                playlist.toReverseString());
    }


    @Test(expected = RuntimeException.class)
    @GradedTest(name = "Testing deleteFirst on empty playlist", max_score = 2)
    public void test_deleteFirst3() {
        Playlist playlist = new Playlist();
        playlist.deleteFirst();
    }

    @Test
    @GradedTest(name = "Testing deleteLast on playlist (B|27.5) -> (A|24)",
    max_score = 2)
    public void test_deleteLast() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 27.5);
            tmp = playlist.deleteLast();
        } catch (Exception e) {
            fail("Unexpected exception in deleteLast from Playlist of size=2");
        }
        assertEquals("A", tmp.title);
        assertEquals(24, tmp.length, 0.01);
        assertEquals("[HEAD] (B|27.5MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name = "Testing deleteLast on playlist (B|27.5) -> (A|24), print via toReverseString()",
    max_score = 2, visibility = "after_published")
    public void test_deleteLast_printreverse() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 27.5);
            tmp = playlist.deleteLast();
        } catch (Exception e) {
            fail("Unexpected exception in deleteLast from Playlist of size=2");
        }
        assertEquals("A", tmp.title);
        assertEquals(24, tmp.length, 0.01);
        assertEquals("[END] (B|27.5MIN) [HEAD]\n",
                playlist.toReverseString());
    }

    @Test
    @GradedTest(name = "Testing deleteLast on playlist (A|24)", max_score = 2,
    visibility = "after_published")
    public void test_deleteLast1() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            tmp = playlist.deleteLast();
        } catch (Exception e) {
            fail("Unexpected exception in deleteLast from Playlist of size=1");
        }
        assertEquals("A", tmp.title);
        assertEquals(24.0, tmp.length, 0.01);
        assertTrue(playlist.isEmpty());
    }

    @Test
    @GradedTest(name = "Testing deleteLast on playlist (D|24.5) -> (C|23) -> (B|27.5) -> (A|24)", max_score = 2, visibility = "after_published")
    public void test_deleteLast2() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 27.5);
            playlist.addFirst("C", 23);
            playlist.addFirst("D", 24.5);
            tmp = playlist.deleteLast();
        } catch (Exception e) {
            fail("Unexpected exception in deleteLast from Playlist of size=4");
        }
        assertEquals("A", tmp.title);
        assertEquals(24.0, tmp.length, 0.01);
        assertEquals("[HEAD] (D|24.5MIN) -> (C|23.0MIN) -> (B|27.5MIN) [END]\n",
                playlist.toString());
    }

    @Test(expected = RuntimeException.class)
    @GradedTest(name = "Testing deleteLast on empty playlist", max_score = 2,
    visibility = "after_published")
    public void test_deleteLast3() {
        Playlist playlist = new Playlist();
        playlist.deleteLast();
    }

    @Test
    @GradedTest(name = "Testing deleteEpisode A on playlist (A|24)", max_score = 2, visibility = "after_published")
    public void test_deleteEpisode() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            tmp = playlist.deleteEpisode("A");
        } catch (Exception e) {
            fail("Unexpected exception in deleteEpisode(A)");
        }
        assertEquals("A", tmp.title);
        assertEquals(24, tmp.length, 0.01);
        assertTrue(playlist.isEmpty());
    }

    @Test
    @GradedTest(name = "Testing deleteEpisode B on playlist (B|27.5) -> (A|24)",
    max_score = 2)
    public void test_deleteEpisode1() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 27.5);
            tmp = playlist.deleteEpisode("B");
        } catch (Exception e) {
            fail("Unexpected exception in deleteEpisode(B)");
        }
        assertEquals("B", tmp.title);
        assertEquals(27.5, tmp.length, 0.01);
        assertEquals("[HEAD] (A|24.0MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name = "Testing deleteEpisode C on playlist (D|24.5) -> (C|23) -> (B|27.5) -> (A|24)", max_score = 2, visibility = "after_published")
    public void test_deleteEpisode2() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 27.5);
            playlist.addFirst("C", 23);
            playlist.addFirst("D", 24.5);
            tmp = playlist.deleteEpisode("C");
        } catch (Exception e) {
            fail("Unexpected exception in deleteEpisode(C)");
        }
        assertEquals("C", tmp.title);
        assertEquals(23.0, tmp.length, 0.01);
        assertEquals("[HEAD] (D|24.5MIN) -> (B|27.5MIN) -> (A|24.0MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name = "Testing deleteEpisode C on playlist (D|24.5) -> (C|23) -> (B|27.5) -> (A|24), print via toReverseString()", max_score = 2, visibility = "after_published")
    public void test_deleteEpisode2_printreverse() {
        Playlist playlist = new Playlist();
        Episode tmp = new Episode("Test", 0.0, null, null);
        try {
            playlist.addFirst("A", 24);
            playlist.addFirst("B", 27.5);
            playlist.addFirst("C", 23);
            playlist.addFirst("D", 24.5);
            tmp = playlist.deleteEpisode("C");
        } catch (Exception e) {
            fail("Unexpected exception in deleteEpisode(C)");
        }
        assertEquals("C", tmp.title);
        assertEquals(23.0, tmp.length, 0.01);
        assertEquals("[END] (A|24.0MIN) -> (B|27.5MIN) -> (D|24.5MIN) [HEAD]\n",
                playlist.toReverseString());
    }

    @Test(expected = RuntimeException.class)
    @GradedTest(name = "Testing deleteEpisode on empty playlist", max_score = 2,
    visibility = "after_published")
    public void test_deleteEpisode3() {
        Playlist playlist = new Playlist();
        playlist.deleteEpisode("A");
    }

    @Test(expected = RuntimeException.class)
    @GradedTest(name = "Testing deleteEpisode on playlist without the episode",
    max_score = 2, visibility = "after_published")
    public void test_deleteEpisode4() {
        Playlist playlist = new Playlist();
        playlist.addFirst("A", 24);
        playlist.addFirst("B", 27.5);
        playlist.addFirst("C", 23);
        playlist.addFirst("D", 24.5);
        playlist.deleteEpisode("E");
    }


	@Test
    @GradedTest(name="Testing getSize() on empty playlist", max_score=1)
    public void test_getSize() {
        Playlist playlist = new Playlist();
        assertTrue(playlist.isEmpty());
        assertEquals(0,playlist.getSize());
    }


    @Test
    @GradedTest(name="Testing getSize() on playlist (A|24) -> (B|27.5) using addFirst",
                max_score=2)
    public void test_getSize1() {
        Playlist playlist = new Playlist();
        playlist.addFirst("A", 24);
        playlist.addFirst("B", 27.5);
        assertEquals(2, playlist.getSize());
    }

    @Test
    @GradedTest(name="Testing getSize() on playlist (A|24) -> (B|27.5) using addLast",
                max_score=2,
                visibility="after_published")
    public void test_getSize1A() {
        Playlist playlist = new Playlist();
        playlist.addLast("A", 24);
        playlist.addLast("B", 27.5);
        assertEquals(2, playlist.getSize());
    }

    @Test
    @GradedTest(name="Testing getSize() on playlist after 2 addFirst and 2 deleteFirst",
                max_score=2, visibility = "after_published")
    public void test_getSize2() {
        Playlist playlist = new Playlist();
        playlist.addFirst("A", 24);
        playlist.addFirst("B", 27.5);
        playlist.deleteFirst();
        playlist.deleteFirst();
        assertTrue(playlist.isEmpty());
        assertEquals(0, playlist.getSize());
    }

    @Test
    @GradedTest(name="Testing getSize() on playlist after 2 addLast and 2 deleteLast",
                max_score=2, visibility = "after_published")
    public void test_getSize3() {
        Playlist playlist = new Playlist();
        playlist.addLast("A", 24);
        playlist.addLast("B", 27.5);
        playlist.deleteLast();
        playlist.deleteLast();
        assertTrue(playlist.isEmpty());
        assertEquals(0, playlist.getSize());
    }

    @Test
    @GradedTest(name="Testing getSize() on playlist after 3 addLast and 2 deleteEpisode",
                max_score=2, visibility = "after_published")
    public void test_getSize4() {
        Playlist playlist = new Playlist();
        playlist.addLast("A", 24);
        playlist.addLast("B", 27.5);
        playlist.addLast("C", 23);
        playlist.deleteEpisode("C");
        playlist.deleteEpisode("A");
        assertEquals(1, playlist.getSize());
    }


    // Part 2: MergeSort
    @Test
    @GradedTest(name="Testing merge(null,episode) returns episode", max_score=1)
    public void testMergeNullArgA() {
        Playlist playlist = new Playlist();
        Episode episode = new Episode("Single Episode", 60, null, null);
        Episode resultNullArgA = playlist.merge(null, episode);
        assertEquals("Single Episode", resultNullArgA.title);
    }

    @Test
    @GradedTest(name="Testing merge(episode,null) returns episode", max_score=1, visibility = "after_published")
    public void testMergeNullArgB() {
        Playlist playlist = new Playlist();
        Episode episode = new Episode("Single Episode", 60, null, null);
        Episode resultNullArgB = playlist.merge(episode, null);
        assertEquals("Single Episode", resultNullArgB.title);
    }

    @Test
    @GradedTest(name="Testing merge(e1,e2) with e1.title < e2.title", max_score=1)
    public void testMergeSingleEpisode() {
        Playlist playlist = new Playlist();
        // duration of A is less than B
        Episode episodeA = new Episode("Episode A", 20, null, null);
        Episode episodeB = new Episode("Episode B", 30, null, null);
        Episode result = playlist.merge(episodeA, episodeB);
        assertEquals("Episode A", result.title);
        assertEquals("Episode B", result.next.title);
    }

    @Test
    @GradedTest(name="Testing merge(e2,e1) with e1.title < e2.title", max_score=1, visibility = "after_published")
    public void testMergeSingleEpisodeArgsOrder() {
        Playlist playlist = new Playlist();
        // duration of A is less than B
        Episode episodeA = new Episode("Episode A", 20, null, null);
        Episode episodeB = new Episode("Episode B", 30, null, null);
        Episode result = playlist.merge(episodeB, episodeA);
        assertEquals("Episode A", result.title);
        assertEquals("Episode B", result.next.title);
    }

    @Test
    @GradedTest(name="Testing merge(A1->A2, B1->B2)", max_score=1)
    public void testMergeTwoEpisodes() {
        Playlist playlist = new Playlist();
        Episode episodeA1 = new Episode("Episode A1", 90, null, null);
        Episode episodeA2 = new Episode("Episode A2", 70, null, episodeA1);
        episodeA1.next = episodeA2;

        Episode episodeB1 = new Episode("Episode B1", 80, null, null);
        Episode episodeB2 = new Episode("Episode B2", 60, null, episodeB1);
        episodeB1.next = episodeB2;

        Episode result = playlist.merge(episodeA1, episodeB1);
        assertEquals("Episode A1", result.title);
        assertEquals("Episode A2", result.next.title);
        assertEquals("Episode B1", result.next.next.title);
        assertEquals("Episode B2", result.next.next.next.title);
    }

    @Test
    @GradedTest(name="Testing merge(B1->B2, A1->A2)", max_score=1, visibility = "after_published")
    public void testMergeTwoEpisodesArgsOrder() {
        Playlist playlist = new Playlist();
        Episode episodeA1 = new Episode("Episode A1", 90, null, null);
        Episode episodeA2 = new Episode("Episode A2", 70, null, episodeA1);
        episodeA1.next = episodeA2;

        Episode episodeB1 = new Episode("Episode B1", 80, null, null);
        Episode episodeB2 = new Episode("Episode B2", 60, null, episodeB1);
        episodeB1.next = episodeB2;

        Episode result = playlist.merge(episodeB1, episodeA1);
        assertEquals("Episode A1", result.title);
        assertEquals("Episode A2", result.next.title);
        assertEquals("Episode B1", result.next.next.title);
        assertEquals("Episode B2", result.next.next.next.title);
    }

    @Test
    @GradedTest(name="Testing multiple merge() calls to merge A1->A2, B1->B2, C1->C2", max_score=2)
    public void testMergeThreeEpisodes() {
        Playlist playlist = new Playlist();
        Episode episodeA1 = new Episode("Episode A1", 90, null, null);
        Episode episodeA2 = new Episode("Episode A2", 70, null, episodeA1);
        episodeA1.next = episodeA2;

        Episode episodeB1 = new Episode("Episode B1", 80, null, null);
        Episode episodeB2 = new Episode("Episode B2", 60, null, episodeB1);
        episodeB1.next = episodeB2;

        Episode episodeC1 = new Episode("Episode C1", 50, null, null);
        Episode episodeC2 = new Episode("Episode C2", 30, null, episodeC1);
        episodeC1.next = episodeC2;

        Episode result = playlist.merge(episodeA1, episodeB1);
        result = playlist.merge(result, episodeC1);

        assertEquals("Episode A1", result.title);
        assertEquals("Episode A2", result.next.title);
        assertEquals("Episode B1", result.next.next.title);
        assertEquals("Episode B2", result.next.next.next.title);
        assertEquals("Episode C1", result.next.next.next.next.title);
        assertEquals("Episode C2", result.next.next.next.next.next.title);
    }

    @Test
    @GradedTest(name="Testing multiple merge() calls to merge B1->B2, A1->A2, C1->C2", max_score=2, visibility = "after_published")
    public void testMergeThreeEpisodesArgsOrder() {
        Playlist playlist = new Playlist();
        Episode episodeA1 = new Episode("Episode A1", 90, null, null);
        Episode episodeA2 = new Episode("Episode A2", 70, null, episodeA1);
        episodeA1.next = episodeA2;

        Episode episodeB1 = new Episode("Episode B1", 80, null, null);
        Episode episodeB2 = new Episode("Episode B2", 60, null, episodeB1);
        episodeB1.next = episodeB2;

        Episode episodeC1 = new Episode("Episode C1", 50, null, null);
        Episode episodeC2 = new Episode("Episode C2", 30, null, episodeC1);
        episodeC1.next = episodeC2;

        Episode result = playlist.merge(episodeB1, episodeA1);
        result = playlist.merge(result, episodeC1);
        assertEquals("Episode A1", result.title);
        assertEquals("Episode A2", result.next.title);
        assertEquals("Episode B1", result.next.next.title);
        assertEquals("Episode B2", result.next.next.next.title);
        assertEquals("Episode C1", result.next.next.next.next.title);
        assertEquals("Episode C2", result.next.next.next.next.next.title);
    }

    @Test
    @GradedTest(name="Testing multiple merge() calls to merge A1->A2, B1->B2, C1->C2, D1->D2", max_score=2, visibility = "after_published")
    public void testMergeFourEpisodes() {
        Playlist playlist = new Playlist();
        Episode episodeA1 = new Episode("Episode A1", 90, null, null);
        Episode episodeA2 = new Episode("Episode A2", 70, null, episodeA1);
        episodeA1.next = episodeA2;

        Episode episodeB1 = new Episode("Episode B1", 80, null, null);
        Episode episodeB2 = new Episode("Episode B2", 60, null, episodeB1);
        episodeB1.next = episodeB2;

        Episode episodeC1 = new Episode("Episode C1", 80, null, null);
        Episode episodeC2 = new Episode("Episode C2", 60, null, episodeC1);
        episodeC1.next = episodeC2;

        Episode episodeD1 = new Episode("Episode D1", 85, null, null);
        Episode episodeD2 = new Episode("Episode D2", 65, null, episodeD1);
        episodeD1.next = episodeD2;

        Episode result1 = playlist.merge(episodeA1, episodeB1);
        Episode result2 = playlist.merge(episodeC1, episodeD1);
        Episode result = playlist.merge(result1, result2);

        assertEquals("Episode A1", result.title);
        assertEquals("Episode A2", result.next.title);
        assertEquals("Episode B1", result.next.next.title);
        assertEquals("Episode B2", result.next.next.next.title);
        assertEquals("Episode C1", result.next.next.next.next.title);
        assertEquals("Episode C2", result.next.next.next.next.next.title);
        assertEquals("Episode D1", result.next.next.next.next.next.next.title);
        assertEquals("Episode D2", result.next.next.next.next.next.next.next.title);
    }

    @Test
    @GradedTest(name="Testing multiple merge() calls to merge B1->B2, D1->D2, A1->A2, C1->C2", max_score=2, visibility = "after_published")
    public void testMergeFourEpisodesArgsOrder() {
        Playlist playlist = new Playlist();
        Episode episodeA1 = new Episode("Episode A1", 90, null, null);
        Episode episodeA2 = new Episode("Episode A2", 70, null, episodeA1);
        episodeA1.next = episodeA2;

        Episode episodeB1 = new Episode("Episode B1", 80, null, null);
        Episode episodeB2 = new Episode("Episode B2", 60, null, episodeB1);
        episodeB1.next = episodeB2;

        Episode episodeC1 = new Episode("Episode C1", 80, null, null);
        Episode episodeC2 = new Episode("Episode C2", 60, null, episodeC1);
        episodeC1.next = episodeC2;

        Episode episodeD1 = new Episode("Episode D1", 85, null, null);
        Episode episodeD2 = new Episode("Episode D2", 65, null, episodeD1);
        episodeD1.next = episodeD2;

        Episode result1 = playlist.merge(episodeB1, episodeD1);
        Episode result2 = playlist.merge(episodeA1, episodeC1);
        Episode result = playlist.merge(result1, result2);

        assertEquals("Episode A1", result.title);
        assertEquals("Episode A2", result.next.title);
        assertEquals("Episode B1", result.next.next.title);
        assertEquals("Episode B2", result.next.next.next.title);
        assertEquals("Episode C1", result.next.next.next.next.title);
        assertEquals("Episode C2", result.next.next.next.next.next.title);
        assertEquals("Episode D1", result.next.next.next.next.next.next.title);
        assertEquals("Episode D2", result.next.next.next.next.next.next.next.title);
    }

    @Test
    @GradedTest(name="Testing mergeSort() on one node: A", max_score=1, visibility = "after_published")
    public void testMergeSortOneEpisode() {
        Playlist playlist = new Playlist();
        playlist.addLast("A",1);
        playlist.mergeSort();
        assertEquals("[HEAD] (A|1.0MIN) [END]\n", playlist.toString());
    }

    @Test
    @GradedTest(name="Testing mergeSort() on two nodes: B, A", max_score=2)
    public void testMergeSortTwoEpisodes() {
        Playlist playlist = new Playlist();
        playlist.addLast("B",1);
        playlist.addLast("A",2);
        playlist.mergeSort();
        assertEquals("[HEAD] (A|2.0MIN) -> (B|1.0MIN) [END]\n", playlist.toString());
    }

    @Test
    @GradedTest(name="Testing mergeSort() on 4 nodes: R, S, G, Z", max_score=3, visibility = "after_published")
    public void testMergeSortFourEpisodes() {
        Playlist playlist = new Playlist();
        playlist.addLast("R",1);
        playlist.addLast("S",2);
        playlist.addLast("G",3);
        playlist.addLast("Z",4);
        playlist.mergeSort();
        assertEquals("[HEAD] (G|3.0MIN) -> (R|1.0MIN) -> (S|2.0MIN) -> (Z|4.0MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name="Testing mergeSort() on 5 nodes: R, S, G, Z, A", max_score=3)
    public void testMergeSortFiveEpisodes() {
        Playlist playlist = new Playlist();
        playlist.addLast("R",1);
        playlist.addLast("S",2);
        playlist.addLast("G",3);
        playlist.addLast("Z",4);
        playlist.addLast("A",5);
        playlist.mergeSort();
        assertEquals("[HEAD] (A|5.0MIN) -> (G|3.0MIN) -> (R|1.0MIN) -> (S|2.0MIN) -> (Z|4.0MIN) [END]\n",
                playlist.toString());
    }

    @Test
    @GradedTest(name="Testing mergeSort() on 5 ordered nodes: A, B, C, D, E", max_score=3, visibility = "after_published")
    public void testMergeSortFiveOrderedEpisodes() {
        Playlist playlist = new Playlist();
        playlist.addLast("A",5);
        playlist.addLast("B",4);
        playlist.addLast("C",3);
        playlist.addLast("D",2);
        playlist.addLast("E",1);
        playlist.mergeSort();
        assertEquals("[HEAD] (A|5.0MIN) -> (B|4.0MIN) -> (C|3.0MIN) -> (D|2.0MIN) -> (E|1.0MIN) [END]\n",
                playlist.toString());
    }

}
