This is a java app I wrote to get the metadata for the larger image set and make the labels.  I wrote it with java 1.8 and the Jaunt web scraping and JSON querying api.  To use it download the GetXRayDataLabels source code and get the free monthly trial of the Jaunt api from http://jaunt-api.com/download.htm.  This api is not available on Maven but is very easy to learn and use.  Extract the jaunt files and copy the .jar package into the same directory as the GetXRayDataLabels source code.  

When you compile and run the app it will send 249 queries to the website, download 7470 records, and create three data files.  It took my i3-5005U CPU with 4Gb RAM and fiber internet just a few minutes.

The small csv file is just the record number, large image filename, and positive/negative label for each record.  I hope that this format will be the easiest way to get the labels into your models and start training.

The smaller JSON file is a summary of each record including the major diagnosis, large image filename, and positive/negative label for each record.  The diagnoses are in Medical Subject Heading JSON form.  This means they are awkward to convert to csv, but they carry structured information which may be useful if we end up trying to classify images in more detail, eg. to particular medical conditions.

The largest data file (~15Mb) is the raw JSON delivered by the restful endpoint.  It is best to open it with a very lightweight text editor and it will probably still take a while.  However we can mine different combinations of label data directly from this file once we have it.

I have been told that other fields will be useful in the classification process, eg. age, gender, etc.  I hope the code is fairly readable and will be relatively easy to extend in this way.  I also expect to work on this myself and will be happy to accomodate requests for particular fields, formats, etc.

This is my first attempt at writing something like this and there are probably many errors, sorry about that!
