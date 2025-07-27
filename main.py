import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

devices = "cuda" if torch.cuda.is_available() else "cpu"
def sentiment_analysis(text):
    classifier = pipeline("sentiment-analysis", 
                         model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0 if torch.cuda.is_available() else -1)
    result = classifier(text)
    return result[0]
    
class SentimentAnalyzer:
    def __init__(self):
        self.model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                           padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
        labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        scores = predictions[0].tolist()
    
        results = [] 
        for label, score in zip(labels, scores):
            results.append({'label': label, 'score': score})
    
        best_result = max(results, key=lambda x: x['score'])
        return best_result, results 
    
def analyze_batch(texts):
    classifier = pipeline("sentiment-analysis", 
                         model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0 if torch.cuda.is_available() else -1)
    
    results = classifier(texts)
    return results

if __name__ == "__main__":
    test_text = [
        "I love this product! It's amazing!",
        "This is terrible, worst experience ever.",
        "The weather is okay today.",
        "I'm so excited about the new movie!",
        "The service was disappointing and slow."
    ]
    print("=== Method 1: Simple Pipeline ===")
    for text in test_text:
        result = sentiment_analysis(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (confidence: {result['score']:.3f})")
        print("-" * 50)

    print("\n=== Method 2: Detailed Analysis ===")
    analyzer = SentimentAnalyzer()
    
    for text in test_text:
        best_result, all_results = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Predicted: {best_result['label']} (confidence: {best_result['score']:.3f})")
        print("All scores:")
        for result in all_results:
            print(f"  {result['label']}: {result['score']:.3f}")
        print("-" * 50)

    print("\n=== Method 3: Batch Processing ===")
    batch_results = analyze_batch(test_text)
    
    for text, result in zip(test_text, batch_results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (confidence: {result['score']:.3f})")
        print("-" * 30)