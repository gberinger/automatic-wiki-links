# automatic-wiki-links
Find closest Wiki article for words/entities in text

### Installation
```bash
pip install -r requirements.txt
python -m spacy download en
python -m spacy download en_core_web_lg
```

### Simple usage
```bash
python query_text.py --text texts/mars_1.txt --context 2
```

See ```python query_text.py --help``` for descriptions of available options.

### Optimization equations

Move correct keyword embedding (k) into the direction of the given context embedding (c) by a factor (alpha):

<a href="https://www.codecogs.com/eqnedit.php?latex=k_{i&plus;1}&space;=&space;k_{i}&space;&plus;&space;\alpha&space;*&space;(c&space;-&space;k_{i})" target="_blank"><img src="https://latex.codecogs.com/png.latex?k_{i&plus;1}&space;=&space;k_{i}&space;&plus;&space;\alpha&space;*&space;(c&space;-&space;k_{i})" title="k_{i+1} = k_{i} + \alpha * (c - k_{i})" /></a>

Move top-5 incorrect keyword embeddings away from the given context embedding by a factor (beta):

<a href="https://www.codecogs.com/eqnedit.php?latex=k_{i&plus;1}&space;=&space;k_{i}&space;-&space;\beta&space;*&space;(c&space;-&space;k_{i})" target="_blank"><img src="https://latex.codecogs.com/png.latex?k_{i&plus;1}&space;=&space;k_{i}&space;-&space;\beta&space;*&space;(c&space;-&space;k_{i})" title="k_{i+1} = k_{i} - \beta * (c - k_{i})" /></a>
