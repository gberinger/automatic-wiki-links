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