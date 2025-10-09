import pytest
from src.predict import load_model, predict_texts


@pytest.fixture(scope="module")
def model():
    # load the modell once for all tests
    return load_model()


POS_LABEL = 1  # 1 = positiv
NEG_LABEL = 0  # 0 = negativ

MODEL_PATH = "/Users/ela/Documents/sentiment-analysis-project/models/sentiment.joblib"

def test_positive_sentence():
    clf = load_model(MODEL_PATH)
    texts = ["I love this movie, it was fantastic and inspiring!"]
    preds, _ = predict_texts(clf, texts)
    assert preds[0] == POS_LABEL

def test_negative_sentence():
    clf = load_model(MODEL_PATH)
    texts = ["The service was terrible and the food was awful."]
    preds, _ = predict_texts(clf, texts)
    assert preds[0] == NEG_LABEL
