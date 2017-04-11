/* Program to find labels for x-ray data for Auckland AI Meetup using Java version 1.8
 * and the Jaunt web-scraping package */


import java.io.IOException;
import java.io.PrintWriter;

import com.jaunt.*; // Free web-scraping package

public class GetXRayDataLabels {

	public static void main(String[] args) {
		
		// Set number of records to get
		int n = 7470;
		
		// Variable and counter for Medical Subject Heading node values
		JNode MeSHMajors;
		JNode ImageLarge;
		String MeSHMajorString;
		String label;
		int j = 0;
		
		// Number of "negative" labels
		int negatives = 0;
		
		try {
			
			// Open files to write data, labels, and record of requests
		    PrintWriter XRayData = new PrintWriter("XRayData.json", "UTF-8");
		    PrintWriter XRayLabels = new PrintWriter("XRayLabels.csv", "UTF-8");
		    PrintWriter XRayLabelsDetailed = new PrintWriter("XRayLabelsDetailed.json",
		    		"UTF-8");
			
		    // Write header for labels .csv file
			XRayLabels.println("Record number, imgLarge, Label");
			
			// Initialize json file
			XRayLabelsDetailed.println("{");
			XRayLabelsDetailed.println("\t\"list\":[");
		    
			// Create scraping agent
			UserAgent agent = new UserAgent();
			
			// Send requests for 30 records at a time
			for (int i = 1; i <= n; i += 30) {
				agent.sendGET("https://openi.nlm.nih.gov/retrieve.php?"
						+ "query=&coll=iu&m=" + i + "&n=" + (i + 29));
				
				// Write raw data to file
				XRayData.print(agent.json);
				
				// Find Medical Subject Heading node values
				MeSHMajors = agent.json.findEvery("major");
				ImageLarge = agent.json.findEvery("imgLarge");

				for (int k = 0; k < MeSHMajors.size(); k++){
					// Increment record counter
					j++;

					// Print to detailed json file (I'm sure there's an easier way haha!)
					XRayLabelsDetailed.println("\t\t{");
					XRayLabelsDetailed.println("\t\t\"record_number\": " + j + ",");
					XRayLabelsDetailed.println("\t\t\"MeSH\":{");
					XRayLabelsDetailed.print("\t\t\t\"major\": " + MeSHMajors.get(k));
					XRayLabelsDetailed.println("\t\t},");
					XRayLabelsDetailed.println("\t\t\"imgLarge\": \"" + ImageLarge.get(k) +
							"\",");
					
					// Remove carriage return from Medical Subject Heading major node value
					MeSHMajorString = MeSHMajors.get(k).toString();
					MeSHMajorString = MeSHMajorString.substring(0, MeSHMajorString.length()
							- 1);
					
					/* Label "negative" if Medical Subject Heading major value is "normal"
					 * or "No Indexing, otherwise "positive" */
					if (MeSHMajorString.equals("[\"normal\"]")
							|| MeSHMajorString.equals("[\"No Indexing\"]")) {
						label = "negative";
						negatives++;
					} else {
						label = "positive";
					}
								
					// Add label to json file
					XRayLabelsDetailed.println("\t\t\"label\": " + label);
					XRayLabelsDetailed.println("\t\t}");
					
					// Add line to .csv file
					XRayLabels.println(j + ", " + ImageLarge.get(k) + ", " + label);
				
				} // End of node-finding loop
			} // End of request-sending loop
			
			// Finalize json and csv files
			XRayLabelsDetailed.println("\t]");
			XRayLabelsDetailed.println("}");
			XRayLabels.println(negatives + " negative labels and " + (j - negatives) +
					" positive labels produced");
			
			// Close data and label files
		    XRayData.close();
		    XRayLabels.close();
		    XRayLabelsDetailed.close();
		} 
		
		catch (JauntException e) {
			System.err.println(e);
		} 
		
		catch (IOException e) {
			System.err.println(e);
		}
	}
}
