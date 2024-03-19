from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

def analyze_sentiment(message):
    encoded_message = tokenizer(message, return_tensors='pt')
    output = model(**encoded_message)
    
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    positive_score = scores[2]
    negative_score = scores[0]
    neutral_score = scores[1]

    if positive_score > negative_score and positive_score > neutral_score:
        sentiment = "Você ta feliz, que bom!"
    elif negative_score > positive_score and negative_score > neutral_score:
        sentiment = "Você ta triste, que pena!"
    else:
        if positive_score > negative_score:
            sentiment = "Até que ta felizinho"
        else:
            sentiment = "Ta meio borocoxo né"
    
    return sentiment
