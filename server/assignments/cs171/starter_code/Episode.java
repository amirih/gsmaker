
class Episode implements Comparable<Episode> {
	
		public String title; // episode title
		public double length; // episode length (minutes)
		public Episode next; // reference to next episode in playlist
		public Episode prev; // reference to previous episode in playlist

		public Episode(String title, double length, Episode next, Episode prev) {
			this.title = title;
			this.length = length;
			this.next = next;
			this.prev = prev;
		}

		public String toString() {
			return "("+ this.title + "|" + this.length + "MIN)" ;
		}
		
		@Override
		public int compareTo(Episode o) {
			if(this.title.compareTo(o.title) < 0 ) return -1;
			else if(this.title.compareTo(o.title) > 0 ) return +1;
			else return 0; 
		}
}
	