import json
import keras_ocr
from stream import weights_from_s3, coder_from_s3

class CRAFT():
    def __init__(self):
        self.decoder = coder_from_s3('character_label_decoder')
        recognizer_alphabet = ''.join(self.decoder.values())
        try:
            #initialize recognizer
            self.recognizer = keras_ocr.recognition.Recognizer(
                alphabet=recognizer_alphabet,
                weights='kurapan'
            )
            self.recognizer.include_top = True
            self.recognizer.compile()
            for layer in self.recognizer.backbone.layers:
                layer.trainable = False
        except Exception as e:
            print(e)
            # Default model
            self.recognizer = keras_ocr.recognition.Recognizer()
            self.recognizer.compile()

        #initialize detector
        self.detector = keras_ocr.detection.Detector()

        #load weights
        recognizer_weights = weights_from_s3('recognizer-weights')
        try:
            self.recognizer.model.set_weights(recognizer_weights)
        except Exception as e:
            print(e)
        self.pipeline = keras_ocr.pipeline.Pipeline(recognizer=self.recognizer)

    #compete pipeline predictions
    def __call__(self, img):
        prediction_groups = self.pipeline.recognize(img)

    def get_recognizer(self):
        return self.recognizer
        
    def get_pipeline(self):
        return self.pipeline

    def recognize(self, img):
        recognitions = self.recognizer.recognize([img])
        return recognitions

    def detect(self, img):
        detections = self.detector.detect([img])
        return detections

    def predict(self, img):
        prediction_groups = self.pipeline.recognize([img])
        return prediction_groups

def create_craft():
    return CRAFT()

    
