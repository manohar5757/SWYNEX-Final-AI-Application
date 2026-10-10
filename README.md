# Mood Meter

A small, beginner-friendly AI application that reads a short English sentence and labels it **positive**, **negative** or **neutral**, with a confidence score.

It uses **no libraries and no internet**: just Python 3.8+ and about 250 lines of readable code. The goal is to let a beginner see exactly how a machine-learning model learns from examples.

---

## 1. The Problem

People constantly need to sort short text by tone: customer feedback, reviews, messages. Doing it by hand doesn't scale. **Can a program learn to judge the tone of a sentence from labelled examples?**

This project is a teaching demo of that idea. It is **not** a production sentiment tool (see Limitations).

## 2. The Method

**Model: Naive Bayes text classifier, written from scratch.**

| Step | What happens | Where in `app.py` |
|---|---|---|
| 1. Data | 150 hand-written sentences, 50 per class | `DATA` |
| 2. Preprocess | Lowercase, split into words, drop filler words ("the", "is"), mark negation (`not good` becomes `NOT_good`) | `tokenize()` |
| 3. Starter vocabulary | ~130 well-known sentiment words are pre-counted as if seen a few times | `LEXICON` |
| 4. Train | Count how often each word appears in each class | `NaiveBayes.fit()` |
| 5. Predict | Apply Bayes' rule: which class most likely produced these words? Laplace smoothing avoids zero probabilities | `NaiveBayes.scores()` |
| 6. Evaluate | 5-fold cross-validation: every sentence is tested by a model that never saw it | `evaluate()` |

**Why Naive Bayes?** It is fast, needs little data, and every number can be inspected, which makes it ideal for learning. "Naive" means it treats words as independent, which is wrong for real language but works surprisingly well.

### Results (honest numbers)

Run `python app.py --eval` to reproduce.

- **Accuracy: 75%** (5-fold cross-validation, seed 42). Across other random splits: **73% to 82%**.
- Random guessing would score about 33%.

```
            positive  negative   neutral
positive          43         4         3
negative           7        42         1
neutral           16         7        27
```

Main weakness: **neutral sentences are often mislabelled as positive** (16 of 50).

Development note: the first versions scored ~33% to ~47% (chance level), because 150 sentences are too few for the model to learn sentiment words on its own. Adding the starter vocabulary fixed this. That vocabulary was written by me with this dataset in mind, so **75% is probably optimistic** for text from outside this project.

## 3. The Demo

### Run it

```bash
python app.py            # terminal chat
python app.py --web      # web page at http://localhost:8000
python app.py --eval     # accuracy report only
```

### Example session

```
> I love this wonderful app
:) POSITIVE (94% confident)
   [positive 94%, neutral 3%, negative 3%]

> the food was terrible and the staff were rude
:( NEGATIVE (98% confident)

> The train leaves at noon
:| NEUTRAL (90% confident)

> not bad
:) POSITIVE (46% confident)  (not very sure - treat with caution)

> xyzzy
I don't recognise any of those words, so I can't judge this one.
```

The app deliberately **admits uncertainty**: it warns when confidence is below 55%, and refuses when it knows none of the words.

### Try to break it (good learning exercises)

- Sarcasm: `Oh great, another delay.` (it will likely say positive)
- Words it never saw: `The vibe was immaculate`
- Add 10 of your own sentences to `DATA` and re-run `--eval`. Does accuracy change?
- Set `LEXICON_WEIGHT = 0` and watch accuracy collapse. This shows how much data matters.

## 4. Limitations

- **Tiny data.** 150 sentences from one author. Real systems use many thousands from many people.
- **Narrow vocabulary.** Unknown words are ignored, so slang, new words, typos and other languages fail.
- **Word-order blind.** Only simple negation ("not good") is handled. Sarcasm, irony, humour and mixed feelings ("great food, awful service") are mostly beyond it.
- **English only.**
- **Weak on neutral text**, as the confusion matrix shows.
- **Optimistic accuracy** because the starter vocabulary was chosen by the developer (see Results).
- **Confidence is not correctness.** A 90% confident answer can still be wrong.

## 5. Ethics Notes

1. **Don't use it for decisions about people.** Hiring, grading, moderation, credit or mental-health screening need rigorously tested systems and human review. This one is a classroom demo.
2. **Bias.** The model reflects its training data and its author's idea of "positive" and "negative". Dialects, cultures and communities not represented in the data may be systematically misread. Any sentiment tool should be checked for different error rates across groups.
3. **Emotion is not text.** A sentence cannot reveal how a person truly feels. Treat output as a rough signal about wording, never a diagnosis.
4. **Privacy.** Everything runs locally and nothing is stored or sent anywhere. If you extend it to real user messages, get consent and avoid collecting personal data.
5. **Transparency.** The app shows its confidence and its uncertainty instead of hiding them, and this README reports its real error rate.
6. **Keep a human in the loop.** If you build on this, let people review and override the label.
7. **Data provenance.** All training sentences were written for this project. No scraped or copyrighted content was used.

## 6. Project Structure

```
mood-meter/
├── app.py       # the whole application (data, model, evaluation, demo)
└── README.md    # this file
```

## 7. Ideas for Next Steps

- Collect more and more varied data (with consent), at least a few thousand examples.
- Compare against `scikit-learn`'s `MultinomialNB` or logistic regression.
- Add a fourth class (e.g. "mixed").
- Test accuracy separately for different writing styles to look for bias.

## License

Free to use and modify for learning.
