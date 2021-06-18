from itertools import repeat
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class Classifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.75)
        self.model = LogisticRegression()
        self.types_names = []

    def fit(self, training_set: List[List[str]], types_names: List[str]):
        self.types_names = types_names
        types, types_count_multiplied = self.getting_type_counts_multiplied(types_count_multiplied=1, types=[],
                                                                            training_set=training_set,
                                                                            types_counts=dict())
        types_counts = self.getting_type_counts_from_key(types_count_multiplied, types_counts=dict())
        self.model = LogisticRegression(class_weight=types_counts)
        training_x = sum(training_set, [])
        text_features = self.vectorizer.fit_transform(training_x).toarray()
        self.model.fit(text_features, types)

    @staticmethod
    def getting_type_counts_from_key(types_count_multiplied=1, types_counts=None):
        for key, value in types_counts.items():
            types_counts[key] = types_count_multiplied / value
        return types_counts

    @staticmethod
    def getting_type_counts_multiplied(training_set: List[List[str]], types,
                                       types_count_multiplied, types_counts):
        for idx, specific_type_items in enumerate(training_set):
            specific_type_items_count = len(specific_type_items)
            types += list(repeat(idx, specific_type_items_count))
            types_counts[idx] = specific_type_items_count
            types_count_multiplied = types_count_multiplied * specific_type_items_count
        return types, types_count_multiplied

    def predict(self, text: str):
        vectorized_text = self.vectorizer.transform([text]).toarray()
        pred_probas = self.model.predict_proba(vectorized_text)
        pred_probas_with_names = dict(zip(self.types_names, pred_probas[0]))
        print(pred_probas_with_names)
        return pred_probas_with_names
