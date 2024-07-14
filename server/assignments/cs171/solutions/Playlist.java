/*  This class represents a Playlist of podcast episodes, where each
/*  episode is implemented as an object of type Episode. A user navigating
/*  a Playlist should be able to move between songs using next or previous references.
/*
/*  To enable flexible navigation, the Playlist is implemented as
/*  a Doubly Linked List where each episode has a link to both
/*  the next and the prev episodes in the list.
*/
import java.util.*;

public class Playlist
{
	private Episode head;
	private int size;

	public Playlist() {
		head = null;
		size = 0;
	}

	public boolean isEmpty() {
		return head == null;
	}

	// Ensure that "size" is updated properly in other methods, to always
	// reflect the correct number of episodes in the current Playlist
	public int getSize() {
		return size;
	}

	// Our implementation of toString() displays the Playlist forward,
	// starting at the first episode (i.e. head) and ending at the last episode,
	// while utilizing the "next" reference in each episode
	@Override
	public String toString()
	{
		String output = "[HEAD] ";
		Episode current = head;
		if ( ! isEmpty() ) {
			while( current.next != null ) {
				output += current + " -> ";
				current = current.next;
			}
			output += current + " [END]\n";
		}
		else {
			output += " [END]\n";
		}
		return output;
	}


	// This method displays the Playlist backward, starting at
	// the last episode and ending at the first episode (i.e. head),
	// while utilizing the "prev" reference in each episode
	public String toReverseString()
	{
		String output = "[END] ";
		Episode current = head;
		if( ! isEmpty() ) {
			while(current.next != null)
				current = current.next;
			// current is now pointing to last node

			while( current.prev != null ) {
				output += current + " -> ";
				current = current.prev;
			}
			output += current + " [HEAD]\n";
		}
		else {
			output += " [HEAD]\n";
		}
		return output;
	}


	// -------- Part 1: Implementing add/delete methods for a doubly linked list -------- //
	public void addFirst( String title, double duration )
	{
		Episode newEpisode = new Episode(title, duration, head, null); // next is current head, prev is null
		if( !isEmpty() ) head.prev = newEpisode;
		head = newEpisode;
		size++;
	}

	public void addLast( String title, double duration )
	{
		if( isEmpty() )
			addFirst(title, duration);
		else {
			Episode temp = head;
			while(temp.next != null)
				temp = temp.next;
			// temp is now pointing to last node
			temp.next = new Episode(title, duration, null, temp); // next is null, prev is last node (temp)
			size++;
		}
	}

	public Episode deleteFirst() {
		if( isEmpty() ) throw new NoSuchElementException("Attempting to deleteFirst() from an empty list.");

		Episode temp = head;
		if( temp.next == null ) { // one-node list!
			head = null;
		} else {
			head = head.next;
			head.prev = null;
		}
		size--;
		return temp;
	}

	public Episode deleteLast()
	{
		if( isEmpty() ) throw new NoSuchElementException("Attempting to deleteLast() from an empty list.");
		Episode temp = head;
		if( temp.next == null ) { // one-node list!
			head = null;
		} else {
			while(temp.next != null)
				temp = temp.next;
			// temp is now pointing to last node
			Episode prevNode = temp.prev;
			prevNode.next = null;
			temp.prev = null;
		}
		size--;
		return temp;
	}

	public Episode deleteEpisode(String title)
	{
		if( isEmpty() ) throw new NoSuchElementException("Attempting to deleteEpisode(title) from an empty list.");
		Episode temp = head;
		Episode prevNode = null;

		// if title is in first node:
		if( head.title.equals(title) )
			return deleteFirst();

		if( temp.next == null ) { // one-node list but we know title is not in head by now
			throw new NoSuchElementException("Episode does not exist.");
		} else { // multi-node list: search for title
			while(temp.next != null ) {
				if(temp.title.equals(title)) { // delete and return this episode
					prevNode = temp.prev;
					prevNode.next = temp.next;
					temp.next.prev = prevNode;
					temp.prev = null;
					size--;
					return temp;
				}
				temp = temp.next;
			}
		}
		if( temp.next == null ) { // reached end of list
			if(temp.title.equals(title)) { // if title is in last node
				prevNode = temp.prev;
				prevNode.next = null;
				temp.prev = null;
				size--;
			}
			else throw new NoSuchElementException("Episode does not exist.");
		}
		return temp;
	}



	// ----------------- Part 2: Sorting the Playlist using Merge Sort --------------- //
	private Episode getMiddleEpisode(Episode node) {
		if(node == null) return node;
		Episode slow = node;
		Episode fast = node;
			while(fast.next != null && fast.next.next != null) {
					 slow = slow.next;
					 fast = fast.next.next;
			}
			return slow;
	 }

		// MergeSort starting point
		public void mergeSort() {
			 if( isEmpty() ) throw new RuntimeException("Cannot sort empty list.");
			 head = sort(head);
		}

	public Episode sort(Episode node) {
			 if(node == null || node.next == null)
					 return node;
			 Episode middle = getMiddleEpisode(node); //get the middle of the list
			 Episode left_head = node;
			 Episode right_head = middle.next;

			 // split the list into two halves:
			 if(right_head != null) right_head.prev = null;
			 middle.next = null;

			 Episode left = sort(left_head);
			 Episode right = sort(right_head);
			 return merge(left, right);
	 }

		 // Merge two sorted lists (can be non-recursive)
	 public Episode merge(Episode a, Episode b) {
			 // Base cases
			 if (a == null) return b;
			 else if (b == null) return a;
			 Episode result;
			 // Pick either a or b, and recur
			 if (a.compareTo(b) < 0)
			 {
				 result = a;
				 result.next = merge(a.next, b);
				 result.next.prev = result;
			 }
			 else
			 {
				 result = b;
				 result.next = merge(a, b.next);
				 result.next.prev = result;
			 }
			 return result;
	 }


} // End of Playlist class
