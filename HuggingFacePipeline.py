from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
text = input("how are you: ")
output = classifier(text)
print(output[0][0]['label'])
