import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, MarianMTModel, MarianTokenizer

# Load the dataset
imdb = load_dataset("imdb")

# Load the sentiment analysis model and tokenizer
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# Load the translation model and tokenizer
translation_model_name = "Helsinki-NLP/opus-mt-en-fr"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
sentiment_model = sentiment_model.to(device)
translation_model = translation_model.to(device)


def classify_sentiment(text):
    # Tokenize and move inputs to the same device
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = sentiment_model(**inputs)

    # Get prediction and move back to CPU for further processing if necessary
    prediction = torch.softmax(outputs.logits, dim=-1).argmax(dim=1).item()

    # Convert prediction to stars
    stars = "★" * (prediction + 1)

    print(f"Sentiment: {prediction}")
    print(f"Stars: {stars}")

    return stars


def translate_to_french(text):
    # Tokenize and move inputs to the same device
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = translation_model.generate(**inputs)

    # Decode the generated translation
    translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


def sentiment_and_translation_pipeline(text):
    # Classify sentiment
    stars_text = classify_sentiment(text)

    # Translate text
    translated = translate_to_french(text)
    print(f"Translated text: {translated}")

    return stars_text, translated


# Test the pipeline with a sample text
test_text = "I love this movie. It was fantastic!"
stars, translated_text = sentiment_and_translation_pipeline(test_text)
