# Stylometric self of Old German Children's Literature
## Comparison of Female and Male Authors
Stylometry based on the [stylometry](https://github.com/nohstns/stylometry) package.
Scraper forked from [gutenberg_scraper](https://www.katherinepully.com/project-gutenberg-scraper/)

Project by Ellen Jones, Nafal Ossandón Hostens and Nanjun Zhou in the context of the GER 389K course at UT Austin (Spring 22).

### The scraper
To be found under `collect_data`
- Includes a tiny bash script to bulk download a list of book ids and store them in a separate directory. 
- This means that only the `.txt` file needs to be modified to specify what books need to be downloaded.

### The feature extractor
