from EmotionDetection.emotion_detection import emotion_detector
import unittest

class TestEmotionDetector(unittest.TestCase):
    def test_emotion_detector(self):

        #test 1, joy
        test1Result = emotion_detector("I am glad this happened")
        self.assertEqual(test1Result['dominant_emotion'],'joy')
        
        #test 2, anger
        test2Result = emotion_detector("I am really mad about this")
        self.assertEqual(test2Result['dominant_emotion'],'anger')

        #test 3, disgust
        test3Result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(test3Result['dominant_emotion'],'disgust')

        #test 4, sadness
        test4Result = emotion_detector("I am so sad about this")
        self.assertEqual(test4Result['dominant_emotion'],'sadness')

        #test 5, fear
        test5Result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(test5Result['dominant_emotion'],'fear')

unittest.main()