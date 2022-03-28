# Stylometric self of Old German Children's Literature
## Comparison of Female and Male Authors
Stylometry based on the [stylometry](https://github.com/jpotts18/stylometry) package.
Scraper forked from [gutenberg_scraper](https://www.katherinepully.com/project-gutenberg-scraper/)

Project by Ellen Jones, Nafal Ossandón Hostens and Nanjun Zhou in the context of the GER 389K course at UT Austin (Spring 22).

The purpose of the project is getting participants acquainted with some basic NLP with Python for Humanities research 
purposes, in this case literary studies, and the usage of the command line.

### Requirements
- lxml==4.8.0
- beautifulsoup4==4.10.0
- ftfy==6.1.1
- requests==2.27.1
- spacy==3.2
- nltk==3.2.5

Download the SpaCy German (small) language model:
`python -m spacy download de_core_news_sm`

### The scraper
To be found under `collect_data`
- Written to be run as a bash script from the terminal.
- The tiny bash script bulk downloads a list of book ids and store them in a separate directory. 
- This means that only the `.txt` file needs to be modified to specify what books need to be downloaded.

```shell
cd collect_data
./get-books.sh
```

### The feature extractor
To be found under `stylometry`
- Written to be run as a python script from the terminal.
- Requires the user to specify the subset of authors from which we want to extract the stylometry features from with the use of the flag `-g` or `--gender_author`, which needs to be binary (i.e. either `male` or `female`).
- Optionally: specify whether it is necessary to store an individual overview with the features for each text (flag `-e` or `--export`)
- Returns a dataframe formatted in `.json` with all the features for the specified subset of authors.

Example:
```shell
cd stylometry
python main.py -g female -e
```
