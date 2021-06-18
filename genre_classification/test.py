from unittest import TestCase

from genre_classification.classifier import Classifier
from genre_classification.read_file_to_text import read_file_text, read_directory


class TestClassification(TestCase):
    def test_reading_files(self):
        expected = ["Ale nie zawsze wychodzi hahah", "Lubie programowac Byle nie w piątki"]
        actual = read_directory('test/resources')
        self.assertEqual(expected, actual)

    def test_artist_classification(self):
        beatles_train = read_directory('test/texts/beatles_train')
        doom_train = read_directory('test/texts/doom_train')

        beatles_test = read_file_text('test/texts/beatles_test/1.rtf')
        doom_test = read_file_text('test/texts/doom_test/1.txt')

        classifier = Classifier()

        classifier.fit(training_set=[beatles_train, doom_train], types_names=['beatles', 'doom'])

        beatles_pred = classifier.predict(beatles_test)
        doom_pred = classifier.predict(doom_test)

        expected_beatles = 0.74
        expected_doom = 0.36

        self.assertAlmostEqual(expected_beatles, beatles_pred['beatles'], places=2)
        self.assertAlmostEqual(expected_doom, doom_pred['beatles'], places=2)
