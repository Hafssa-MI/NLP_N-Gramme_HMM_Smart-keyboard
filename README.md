# Smart Keyboard

A next-word prediction engine built from classical NLP components: an interpolated Kneser-Ney trigram language model combined with a part-of-speech Hidden Markov Model (HMM). The engine is served through a FastAPI backend and a browser interface.

## Features

- Next-word suggestions and completion of the word being typed
- Syntactic reweighting of candidates using an HMM over part-of-speech tags
- Phrase completion by beam search
- Per-user personal profile that learns new vocabulary at runtime, stored in memory only
- Part-of-speech tagging with the Viterbi algorithm
- Evaluation dashboard (perplexity, top-k accuracy, MRR, keystroke savings, latency)
- JSON API and a dependency-free web front end
- Docker support

## Screenshots

### Writer
![Writer view](docs/screenshots/writer.png)

### Analysis
![Analysis view](docs/screenshots/analysis.png)

### Evaluation
![Evaluation view](docs/screenshots/evaluation.png)

## How it works

1. **Kneser-Ney trigram model.** Estimates the probability of the next word from the previous two words. The highest order uses raw counts and the lower orders use continuation counts. Trigrams seen fewer than twice are pruned.
2. **POS HMM.** A first-order HMM over universal part-of-speech tags, trained on the Brown corpus. A forward pass gives the distribution over the next tag. Unknown words are handled with a suffix model.
3. **Combination.** Each candidate word is scored as `p_ngram(w) * f(w)^beta`, where `f(w)` measures how compatible the word is with the predicted next tag. `beta` is tuned on a validation set.
4. **Personal cache.** A per-user model over word strings blends in words and phrases from the user's own text, including words outside the training vocabulary.
5. **Phrase completion.** Beam search chains word predictions into short completions.

Training data: the IMDB review corpus (4,000 sampled reviews, about 690,000 training tokens), split by document into 80% training, 10% validation and 10% test.

## Results

Measured on the held-out test split. The HMM weight and the add-k smoothing constants were selected on the validation split only.

| Model | Perplexity | Top-1 | Top-3 | Top-5 | MRR | Keystroke savings |
|---|---|---|---|---|---|---|
| Unigram, add-k (k=0.001) | 502.41 | - | - | - | - | - |
| Bigram, add-k (k=0.01) | 287.16 | - | - | - | - | - |
| Trigram, add-k (k=0.01) | 1209.07 | - | - | - | - | - |
| Kneser-Ney trigram | 208.49 | 18.0% | 31.0% | 37.1% | 0.27 | 51.2% |
| Kneser-Ney trigram + HMM (beta=0.2) | 208.08 | 18.0% | 30.9% | 37.0% | 0.27 | 51.1% |

- POS tagger accuracy (Viterbi, held-out Brown sentences): 96.36%, against 93.91% for a most-frequent-tag baseline.
- Average latency: about 1 ms per suggestion request.

Kneser-Ney smoothing clearly outperforms plain add-k smoothing. The HMM reweighting gives only a marginal perplexity improvement and no measurable gain in ranking metrics on this data.

## Project structure

```
.
├── app/
│   ├── __init__.py
│   └── main.py            FastAPI application
├── data/
│   ├── corpus.csv         Local corpus (not versioned)
│   └── make_data.py       Corpus preparation script
├── docs/
│   └── screenshots/       Images used in this README
├── models/
│   └── engine.joblib      Trained model bundle
├── results/
│   ├── metrics.json       Evaluation metrics read by the API
│   ├── failures.txt
│   ├── ngram_results.png
│   └── run_log.txt
├── src/
│   ├── data.py            Tokenisation, splits, vocabulary
│   ├── ngram.py           Kneser-Ney and add-k models
│   ├── hmm.py             POS HMM (Viterbi, forward algorithm)
│   ├── engine.py          Predictor, beam search, personal cache
│   ├── evaluate.py        Evaluation metrics
│   └── pipeline.py        Training and evaluation pipeline
├── web/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── guide.css
│   └── guide.js
├── app.py
├── config.py              Dataset and model settings
├── Dockerfile
├── notebook.ipynb         Full walkthrough: data, models, evaluation, API, front end
└── requirements.txt
```

## Getting started

### Run the application

The trained model is included, so no training is needed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Open http://localhost:8000.

### Retrain the model

Open `notebook.ipynb` and run all cells from top to bottom. The notebook downloads the corpus, trains and evaluates the models, and rewrites `models/engine.joblib` and `results/metrics.json`. Additional packages are required for training:

```bash
pip install pandas nltk datasets huggingface_hub jupyter
```

### Docker

```bash
docker build -t smart-keyboard .
docker run -p 7860:7860 smart-keyboard
```

Open http://localhost:7860.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness check |
| `/api/info` | GET | Vocabulary size, selected `beta`, tag set |
| `/api/suggest` | POST | Next-word suggestions or prefix completions |
| `/api/complete` | POST | Beam-search phrase completions |
| `/api/learn` | POST | Learn from a user's text |
| `/api/forget` | POST | Delete a user's profile |
| `/api/metrics` | GET | Evaluation results |

Example:

```bash
curl -X POST http://localhost:8000/api/suggest \
  -H "Content-Type: application/json" \
  -d '{"text": "i thought the acting was ", "k": 3}'
```

## Limitations

- The training corpus is movie reviews, so predictions reflect that domain and register.
- The vocabulary is limited to about 9,000 words. Other words are treated as unknown, except through the personal cache.
- The HMM contributes little on this dataset, as shown in the results.
- User profiles are kept in memory and are lost when the server restarts.
