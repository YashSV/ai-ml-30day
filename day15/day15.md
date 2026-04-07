# Day 15: NLP & Sentiment Analysis with BERT

## What I Learned

**NLP Fundamentals:**
- NLU (Natural Language Understanding): extracting meaning from text
- NLG (Natural Language Generation): creating new text
- Tokenization: breaking text into words/tokens
- Embeddings: representing words as numbers

**Transformers & HuggingFace:**
- Transformers use attention mechanisms to understand context
- Pre-trained models (BERT, RoBERTa) learn from billions of words
- HuggingFace provides easy-to-use pipelines

## What I Built

**1. Sentiment Analysis on Movie Reviews**
- Used pre-trained DistilBERT
- Classified reviews as POSITIVE/NEGATIVE
- Works out-of-the-box

**2. Custom Sentiment Analysis on Tweets**
- Loaded tweet_eval dataset (100 samples)
- Used Twitter-RoBERTa model
- Achieved 70% accuracy
- Pre-trained movie model doesn't perfectly transfer to tweets

## Key Insight

Transfer learning works across domains but isn't perfect. A model trained on movies gets 70% on tweets—good, but not great. Fine-tuning the model on tweet data would improve accuracy.

## Next Steps

Learn fine-tuning (next week). Understand transformer architecture deeper through practice, not theory.