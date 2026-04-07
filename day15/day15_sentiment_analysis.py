from transformers import pipeline

sentiment_pipeline = pipeline('sentiment-analysis')


# Test on movie reviews
reviews = [
    "This movie was amazing! I loved it.",
    "Terrible film. Waste of time.",
    "It was okay, nothing special.",
]

for review in reviews:
    result = sentiment_pipeline(review)
    print(f'Review {review}')
    print(f'sentiment{result}')
    print()

