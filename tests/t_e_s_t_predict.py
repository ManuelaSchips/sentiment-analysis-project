# tests/test_predict.py
import sys
import os

# <-- fügt den Ordner "src" zum Python-Pfad hinzu
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from predict import load_model, predict_texts

# Wenn ein Test fehlschlägt, tausche 0/1:
POS_LABEL = 1  # 1 = positiv (Annahme)
NEG_LABEL = 0  # 0 = negativ (Annahme)

MODEL_PATH = "/Users/ela/Documents/sentiment-analysis-project/models/sentiment.joblib"


def test_positive_and_negative_sentences():
    clf = load_model(MODEL_PATH)
    texts = [
        "I love this movie, it was fantastic and inspiring!",  # positiv
        "The service was terrible and the food was awful.",  # negativ
    ]
    preds, _ = predict_texts(clf, texts)

    assert preds[0] == POS_LABEL
    assert preds[1] == NEG_LABEL
