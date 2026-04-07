from transformers import pipeline

from datasets import load_dataset

dataset = load_dataset("tweet_eval", "sentiment")

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

correct = 0
total = 0

for i in range(100):
    example = dataset["test"][i]
    text = example["text"]
    true_label = example["label"]
    predict = sentiment_pipeline(text)

    # LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
    predicted_label = int(predict[0]['label'].split('_')[1])

    if predicted_label == true_label:
        correct += 1

    total += 1

accuracy = 100*correct/total

print("accuracy: ", accuracy)
