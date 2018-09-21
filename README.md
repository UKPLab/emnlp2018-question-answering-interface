# Wikidata access

## Simple API for a Wikidata SPARQL endpoint

This is a simple API for accessing a local Wikidata SPARQL endpoint. 
All methods generate SPARQL queries and send them to the endpoint, the result is preprocessed and returned.

As usual:
> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
 
### Project structure:

<table>
    <tr>
        <th>File</th><th>Description</th>
    </tr>
    <tr>
        <td>wikidata/resources</td><td>Preloaded information on properties</td>
    </tr>
    <tr>
        <td>wikidata/endpoint_access.py</td><td>Send a request to the KB</td>
    </tr>
    <tr>
        <td>wikidata/queries.py</td><td>Main interface for KB queries</td>
    </tr>
    <tr>
        <td>wikidata/scheme.py</td><td>Information about the structure of the KB, e.g frequency of properties</td>
    </tr>
    <tr>
        <td>test/</td><td>Unit tests</td>
    </tr>
</table>


#### Requirements:
* Python 3.6

### Usage

1. Clone and install the wikipedia-access project with pip from our internal git:

	```
	pip install git+https://git.ukp.informatik.tu-darmstadt.de/sorokin/wikidata-access.git
	```

2. Use the following code to access the local Wikidata endpoint:

	```python
	from wikidata import queries, endpoint_access
	
	sparql_query = queries.query_get_entity_by_label("Barack Obama")  # Get a sparql query to retrieve all entities with "Barack Obama" in the label
	entities = endpoint_access.query_wikidata(sparql_query)  # Execute the query against Wikidata
	```

### Further usage notes

 - The current endpoint is "http://knowledgebase:8890/sparql" that is only accessible from inside the UKP network
 - You can change the sparql endpoint with `endpoint_access.set_backend("new_url")`
 - The Wikidata dump was created with [Wikidata Toolkit](http://tools.wmflabs.org/wikidata-exports/rdf/). You can read more about different Wikidata RDF dumps [here](https://www.wikidata.org/wiki/Wikidata:Database_download).
 - It won't work with the official online Wikidata endpoint for SPAQRL queries, since they have a different dump structure! 
 - The API is not stable, but generally methods fall into two categories:
 	- Methods that start with `get_` or `map_` will send a query, preprocess the results and return them in some structured form
 	- Methods that start with `query_` generate a SPARQL query and return it as a string. You have to use `endpoint_access.query_wikidata` to execute the query yourself.
 	- (DS: I might streamline it at some point this summer)
 - Most methods in `query.py` have doctest that you consult to get an idea of how to use the method
 
 
 

### Contacts:
If you have any questions regarding the code, please, don't hesitate to contact the authors or report an issue.
  * Daniil Sorokin, \<lastname\>@ukp.informatik.tu-darmstadt.de
  * https://www.ukp.tu-darmstadt.de
  * https://www.tu-darmstadt.de


### License:
* Apache License Version 2.0
