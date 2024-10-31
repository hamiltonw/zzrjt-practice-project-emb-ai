import requests, json
import numpy as np

def emotion_detector(text_to_analyse):
    
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = { "raw_document": { "text": text_to_analyse } }
    
    response = requests.post(url, json = myobj, headers = header)
    predLabels = ['anger', 'disgust','fear','joy','sadness']

    if response.status_code == 400:
        predScoresDict = {label: None for label in predLabels}
        predScoresDict['dominant_emotion'] = None
    else:
        formatted_response = json.loads(response.text)
        predScoresDict = formatted_response['emotionPredictions'][0]['emotion']
        predScores = [predScoresDict[label] for label in predLabels]
        maxScoreArg = np.argmax(predScores)
        predScoresDict['dominant_emotion'] = predLabels[maxScoreArg]

    return predScoresDict