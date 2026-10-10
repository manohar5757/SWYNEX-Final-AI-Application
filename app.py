"""
Mood Meter - a beginner-friendly AI application.

It learns to label short English messages as POSITIVE, NEGATIVE or NEUTRAL
using a Naive Bayes classifier written from scratch (standard library only).

Usage:
    python app.py            # train, show accuracy, then chat in the terminal
    python app.py --web      # same model, but with a web page at http://localhost:8000
    python app.py --eval     # only print evaluation results
"""
import math
import random
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------- 1. DATA
# A tiny hand-written dataset. Real projects need far more (and more varied) data.
DATA = {
    "positive": [
        "I love this, it works great",
        "What a wonderful day",
        "This is amazing and so helpful",
        "I am really happy with the service",
        "Fantastic experience, highly recommend",
        "The team was friendly and kind",
        "Great job, I am very pleased",
        "Absolutely delicious food",
        "I enjoyed every minute of it",
        "Thank you, this made my day",
        "Excellent quality and fast delivery",
        "The app is easy and fun to use",
        "I feel proud and excited",
        "Best purchase I have made this year",
        "Beautiful design and smooth performance",
        "She was so supportive and caring",
        "Really good value, I am satisfied",
        "Such a lovely surprise",
        "Brilliant idea, it works perfectly",
        "I am grateful for your help",
        "Wonderful staff, lovely atmosphere",
        "I am delighted with the result",
        "Super helpful and very polite support",
        "Great food, great music, great night",
        "The update is fantastic and fast",
        "Happy to recommend this to everyone",
        "A really nice and relaxing trip",
        "Impressive work, truly excellent",
        "I adore this little cafe",
        "Smooth, simple and enjoyable",
        "The teacher was patient and inspiring",
        "Perfect gift, she loved it",
        "Cheerful, warm and welcoming place",
        "This phone is awesome, the camera is superb",
        "Good news, everything went well",
        "What a pleasant and comfortable room",
        "I am thrilled about the new job",
        "Reliable, quick and affordable",
        "Gorgeous view and amazing sunset",
        "Great support, problem solved quickly",
        "Not bad at all, actually quite good",
        "Never been happier with a purchase",
        "Nothing to complain about, it is perfect",
        "The book was captivating and moving",
        "Lovely people and a lovely city",
        "So glad I tried it, truly enjoyable",
        "Neat, clean and very comfortable",
        "Delightful service from start to finish",
        "Outstanding performance, I am impressed",
        "Warm thanks, you were wonderful",
    ],
    "negative": [
        "I hate this, it is terrible",
        "What a horrible day",
        "This is awful and completely useless",
        "I am very unhappy with the service",
        "Worst experience ever, do not buy",
        "The staff were rude and unhelpful",
        "Bad job, I am really disappointed",
        "Disgusting food and cold",
        "I regret buying this",
        "This ruined my whole day",
        "Poor quality and late delivery",
        "The app is slow and keeps crashing",
        "I feel angry and tired",
        "Waste of money, totally broken",
        "Ugly design and laggy performance",
        "He was so dismissive and cruel",
        "Overpriced and not worth it",
        "Such a nasty surprise",
        "Terrible idea, nothing works",
        "I am frustrated by the delays",
        "Awful staff, filthy atmosphere",
        "I am disgusted with the result",
        "Unhelpful and very impolite support",
        "Bad food, bad music, bad night",
        "The update is broken and slow",
        "Sad to say I cannot recommend this",
        "A really stressful and exhausting trip",
        "Disappointing work, truly poor",
        "I dislike this noisy cafe",
        "Clunky, confusing and annoying",
        "The teacher was impatient and boring",
        "Terrible gift, she hated it",
        "Cold, hostile and unwelcoming place",
        "This phone is garbage, the camera is dreadful",
        "Bad news, everything went wrong",
        "What an unpleasant and dirty room",
        "I am worried and upset about the job",
        "Unreliable, slow and expensive",
        "Gloomy view and miserable weather",
        "Useless support, problem still unsolved",
        "Not good at all, actually quite bad",
        "Never been more disappointed with a purchase",
        "Nothing works, it is a disaster",
        "The book was dull and depressing",
        "Rude people and a dirty city",
        "So sorry I tried it, truly regrettable",
        "Messy, dirty and very uncomfortable",
        "Dreadful service from start to finish",
        "Embarrassing performance, I am appalled",
        "Bitter complaint, you were horrible",
    ],
    "neutral": [
        "The meeting is at three o'clock",
        "I bought a laptop yesterday",
        "The package arrived on Tuesday",
        "She walked to the station",
        "The report has ten pages",
        "We are going to the market",
        "The store opens at nine",
        "He reads the news every morning",
        "The file is saved in the folder",
        "It is a blue notebook",
        "The bus leaves from platform two",
        "I will send the document tomorrow",
        "They live near the river",
        "The class starts next week",
        "There are five chairs in the room",
        "The phone is on the table",
        "He works at the library",
        "The train stops in two cities",
        "I need to update my address",
        "The form asks for a name and date",
        "The office is on the second floor",
        "I will call the plumber on Monday",
        "The recipe uses two eggs and flour",
        "Water boils at one hundred degrees",
        "She is a student at the university",
        "The meeting was moved to Friday",
        "He parked the car near the bank",
        "The museum has three exhibitions",
        "I changed my password this morning",
        "The road is closed for repairs",
        "There is a parcel at the front desk",
        "My flight departs at noon",
        "The invoice is attached to the email",
        "We watched a documentary about birds",
        "The shop sells shoes and bags",
        "The printer is next to the window",
        "He moved to a new apartment in May",
        "The lecture covers chapter four",
        "I ordered a coffee and a sandwich",
        "The website lists the opening hours",
        "The garden has a small pond",
        "She asked about the schedule",
        "The cable connects to the back of the monitor",
        "They discussed the budget for next year",
        "The library closes at six",
        "I took the metro to the airport",
        "The software runs on Windows and Linux",
        "A new bus route starts in March",
        "The instructions are on page eight",
        "He wrote his name on the list",
    ],
}

STOPWORDS = {"i", "am", "is", "are", "was", "were", "the", "a", "an", "it", "this", "that", "and", "to", "of",
             "in", "on", "at", "for", "we", "he", "she", "they", "my", "me", "so", "with", "be", "have", "has"}
NEGATORS = {"not", "no", "never", "dont", "don't", "isnt", "isn't", "cant", "can't", "wont", "won't", "nothing"}



# A small starter vocabulary ("lexicon"). Each word is counted as if it had been seen a few times
# in its class. This gives the model prior knowledge that 150 sentences cannot teach it.
LEXICON = {
    "positive": "good great love loved lovely wonderful amazing fantastic excellent happy glad pleased delighted "
                "enjoy enjoyed enjoyable best perfect awesome superb brilliant beautiful nice pleasant friendly "
                "kind helpful thanks thank grateful proud excited impressive impressed satisfied recommend "
                "fun easy smooth fast comfortable welcoming warm cheerful adore gorgeous outstanding delicious "
                "reliable affordable supportive caring polite inspiring thrilled delightful captivating".split(),
    "negative": "bad terrible awful horrible hate hated worst poor disappointed disappointing unhappy angry sad "
                "rude useless broken slow ugly nasty dirty filthy dreadful disgusting regret annoying boring "
                "frustrated frustrating worried upset miserable garbage disaster waste crashing laggy overpriced "
                "unhelpful impolite unreliable expensive uncomfortable unpleasant cruel dismissive stressful "
                "exhausting hostile gloomy depressing dull appalled embarrassing complaint terrible".split(),
}
LEXICON_WEIGHT = 3

# ------------------------------------------------------------ 2. PREPROCESS
def tokenize(text):
    """Lowercase, split into words, drop filler words, and mark the word after a negator
    ('not good' -> 'NOT_good') so negation isn't lost."""
    words = re.findall(r"[a-z']+", text.lower())
    tokens, negate = [], False
    for w in words:
        if w in NEGATORS:
            negate = True
            continue
        if w in STOPWORDS:
            continue
        tokens.append("NOT_" + w if negate else w)
        negate = False
    return tokens


# ------------------------------------------------------------------ 3. MODEL
class NaiveBayes:
    """Counts how often each word appears in each class, then uses Bayes' rule."""

    def fit(self, examples):
        self.word_counts = defaultdict(Counter)
        self.class_counts = Counter()
        self.vocab = set()
        for text, label in examples:
            self.class_counts[label] += 1
            for tok in tokenize(text):
                self.word_counts[label][tok] += 1
                self.vocab.add(tok)
        for label, words in LEXICON.items():
            for w in words:
                self.word_counts[label][w] += LEXICON_WEIGHT
                self.vocab.add(w)
        self.total_docs = sum(self.class_counts.values())
        return self

    def scores(self, text):
        toks = tokenize(text)
        log_probs = {}
        for label in self.class_counts:
            lp = math.log(self.class_counts[label] / self.total_docs)  # prior
            total = sum(self.word_counts[label].values())
            for tok in toks:
                if tok not in self.vocab:
                    continue  # ignore words never seen in training
                # Laplace smoothing: add 1 so unseen (word, class) pairs aren't zero
                lp += math.log((self.word_counts[label][tok] + 1) / (total + len(self.vocab)))
            log_probs[label] = lp
        # Convert log-scores into probabilities that sum to 1
        m = max(log_probs.values())
        exp = {k: math.exp(v - m) for k, v in log_probs.items()}
        z = sum(exp.values())
        return {k: v / z for k, v in exp.items()}

    def predict(self, text):
        probs = self.scores(text)
        label = max(probs, key=probs.get)
        known = [t for t in tokenize(text) if t in self.vocab]
        return label, probs[label], probs, len(known)


# --------------------------------------------------------------- 4. EVALUATE
def evaluate(folds=5, seed=42):
    """5-fold cross-validation: every sentence is used as unseen test data exactly once."""
    rng = random.Random(seed)
    items = [(t, l) for l, ts in DATA.items() for t in ts]
    rng.shuffle(items)
    labels = list(DATA)
    matrix = {a: Counter() for a in labels}
    correct = 0
    for k in range(folds):
        test = items[k::folds]
        train = [x for i, x in enumerate(items) if i % folds != k]
        model = NaiveBayes().fit(train)
        for text, truth in test:
            pred = model.predict(text)[0]
            matrix[truth][pred] += 1
            correct += pred == truth
    return correct / len(items), matrix, len(items)


def print_report():
    acc, matrix, n = evaluate()
    print(f"{n} labelled sentences, 5-fold cross-validation (always tested on unseen sentences).")
    print(f"Accuracy: {acc:.0%}\n")
    labels = list(DATA)
    print("Confusion matrix (rows = truth, columns = prediction)")
    print(" " * 10 + "".join(f"{l[:8]:>10}" for l in labels))
    for a in labels:
        print(f"{a:<10}" + "".join(f"{matrix[a][b]:>10}" for b in labels))
    print()


# -------------------------------------------------------------------- 5. DEMO
EMOJI = {"positive": ":)", "negative": ":(", "neutral": ":|"}
LOW_CONFIDENCE = 0.55


def describe(model, text):
    label, conf, probs, known = model.predict(text)
    if known == 0:
        return "I don't recognise any of those words, so I can't judge this one."
    note = "  (not very sure - treat with caution)" if conf < LOW_CONFIDENCE else ""
    detail = ", ".join(f"{k} {v:.0%}" for k, v in sorted(probs.items(), key=lambda x: -x[1]))
    return f"{EMOJI[label]} {label.upper()} ({conf:.0%} confident){note}\n   [{detail}]"


def chat(model):
    print("Type a sentence and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if text:
            print(describe(model, text), "\n")


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Mood Meter</title>
<style>body{{font-family:system-ui,sans-serif;max-width:560px;margin:3rem auto;padding:0 1rem}}
input{{width:100%;padding:.7rem;font-size:1rem;box-sizing:border-box}}button{{margin-top:.6rem;padding:.6rem 1.2rem;font-size:1rem}}
pre{{background:#f3f3f3;padding:1rem;border-radius:8px;white-space:pre-wrap}}small{{color:#666}}</style>
<h1>Mood Meter</h1>
<form method=post><input name=text placeholder="Type a sentence..." value="{value}" autofocus>
<button>Analyse</button></form>
<pre>{result}</pre>
<small>A teaching demo trained on 150 sentences. It makes mistakes - do not use it for real decisions.</small>"""


def serve(model, port=8000):
    from html import escape
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs

    class Handler(BaseHTTPRequestHandler):
        def _send(self, value="", result="Result appears here."):
            body = PAGE.format(value=escape(value, quote=True), result=escape(result)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send()

        def do_POST(self):
            size = min(int(self.headers.get("Content-Length", 0)), 2000)
            text = parse_qs(self.rfile.read(size).decode()).get("text", [""])[0].strip()
            self._send(text, describe(model, text) if text else "Please type something.")

        def log_message(self, *a):
            pass

    print(f"Open http://localhost:{port}  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    acc, *_ = evaluate()
    if "--eval" in sys.argv:
        print_report()
    else:
        print(f"Mood Meter ready. Cross-validated accuracy: {acc:.0%}\n")
        model = NaiveBayes().fit([(t, l) for l, ts in DATA.items() for t in ts])  # final model uses all data
        serve(model) if "--web" in sys.argv else chat(model)
