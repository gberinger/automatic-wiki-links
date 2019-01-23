# automatic-wiki-links
Find closest Wiki article for words/entities in text

### Installation
```bash
pip install -r requirements.txt
python -m spacy download en
python -m spacy download en_core_web_lg
```

### Keywords
Keywords are the expressions that we look for in the text, which in final product would link to their Wiki articles.
In this repository we try to distinguish between 20 keywords, which can be found (with Wiki links) in `keywords.csv`.

#### Keyword embeddings
To get the embedding of the keyword, we simply average the embeddings of every word in keyword, e.g.

```
embed("tree structure") = avg(embed("tree"), embed("structure"))
```

where `embed` is the embedding function.

### Data
We gathered 5 training and 5 (or 7) test samples per each keyword, stored in `texts/train` and `texts/test` respectively.
Paths, together with correct keyword and source link for each sample, are stored in `train.csv` and `test.csv`.

Example:
Path: `texts/train/bolt_climbing_1.txt`
Keyword: `bolt climbing`
Source: https://en.wikipedia.org/wiki/Climbing
Text:
```
Traditional climbing (more casually known as Trad climbing) is a form of climbing without fixed anchors and *bolts*. Climbers place removable protection such as camming devices, nuts, and other passive and active protection that holds the rope to the rock (via the use of carabiners and webbing/slings) in the event of a fall and/or when weighted by a climber.
```

Word of interest is denoted by `*` on each side of the word. The goal of this experiment is to predict which keyword
is the correct one, given the context.

### Check keyword embeddings on single sample
```bash
python query_text.py --kw_embeds kw_embeds.pickle --text texts/test/mars_1.txt --context 2
```

Look for word of interest (surrounded by `*` in text) in `texts/test/mars_1.txt` and use keyword embeddings from
`kw_embeds.pickle` to find the closest keyword based on the context embedding, created by taking `context` number
of words from each side of the word of interest. The output returns closest 5 keywords and displays their cosine
distance, e.g.

```
--------------------------
Index: 54
Word: tree
Context: languages that naturally embody tree structures for example lisp
Closest keywords:
        tree parse (0.321444)
        tree structure (0.337480)
        tree command (0.399569)
        tree decision (0.442712)
        tree family (0.446275)
```

See ```python query_text.py --help``` for descriptions of available options.

### Optimization equations

Move correct keyword embedding (k) into the direction of the given context embedding (c) by a factor (alpha):

<a href="https://www.codecogs.com/eqnedit.php?latex=k_{i&plus;1}&space;=&space;k_{i}&space;&plus;&space;\alpha&space;*&space;(c&space;-&space;k_{i})" target="_blank"><img src="https://latex.codecogs.com/png.latex?k_{i&plus;1}&space;=&space;k_{i}&space;&plus;&space;\alpha&space;*&space;(c&space;-&space;k_{i})" title="k_{i+1} = k_{i} + \alpha * (c - k_{i})" /></a>

Move top-5 incorrect keyword embeddings away from the given context embedding by a factor (beta):

<a href="https://www.codecogs.com/eqnedit.php?latex=k_{i&plus;1}&space;=&space;k_{i}&space;-&space;\beta&space;*&space;(c&space;-&space;k_{i})" target="_blank"><img src="https://latex.codecogs.com/png.latex?k_{i&plus;1}&space;=&space;k_{i}&space;-&space;\beta&space;*&space;(c&space;-&space;k_{i})" title="k_{i+1} = k_{i} - \beta * (c - k_{i})" /></a>
