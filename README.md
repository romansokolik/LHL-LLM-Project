# LLM Project

## Project Task - Sentiment Analysis Tool

The goal of this project is to build a sentiment analysis tool that can classify the sentiment of a given text as
positive, negative, or neutral.
The tool will be trained on a dataset of movie reviews and will be evaluated based on its accuracy in classifying the
sentiment of a given text.

## Dataset

For the purpose of this project, I will be using the IMDB movie reviews dataset. This dataset contains 100,000 movie
reviews that are labeled as either positive or negative. The dataset is split into 25,000 reviews for training and
25,000 reviews for testing.
Half of the dataset is labeled as unsupervised, meaning that the sentiment of the reviews is not provided. This will
allow me to evaluate the performance of the model on unseen data.
The reviews are preprocessed and stored in a format that is easy to work with.
While pre-processing the data, I will be removing any HTML tags, punctuation, and special characters from the reviews. I
will also be converting the reviews to lowercase, removing stopwords,chat-text, emojis, punctuations, html tags, and
special characters.

## Pre-trained Model

I will be using a pre-trained model called "aychang/roberta-base-imdb" (A simple base roBERTa model trained on the "
imdb" dataset) for this project.
roBERT is a transformer-based model that has been pre-trained on a large corpus of text data.
I will be fine-tuning the DistilBERT model on the IMDB movie reviews dataset to classify the sentiment of the reviews as
positive, negative, or neutral.
I will be using the Hugging Face Transformers library to load the pre-trained DistilBERT model and fine-tune it on the
IMDB movie reviews dataset.

## Performance Metrics

I will be evaluating the performance of the sentiment analysis tool based on the accuracy of the model in classifying
the sentiment of the reviews as positive or negative.
The accuracy of the model is calculated as the number of correct predictions divided by the total number of predictions.
I will also be looking at the confusion matrix to see how well the model is performing in classifying the reviews as
positive or negative.

## Hyperparameters

I will be fine-tuning the roBERT model on the IMDB movie reviews dataset using the following hyperparameters:
- output_dir='./models'
- overwrite_output_dir=False
- num_train_epochs=2
- per_device_train_batch_size=24
- per_device_eval_batch_size=8
- warmup_steps=500
- weight_decay=0.01
- evaluation_strategy="steps"
- logging_dir='./logs'
- fp16=False
- eval_steps=800
- save_steps=300000

I will be fine-tuning the DistilBERT model on the IMDB movie reviews dataset using the following hyperparameters:
- Batch Size: 32
- Learning Rate: 5e-5
- Number of Epochs: 3
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Evaluation Metric: Accuracy
- Dropout: 0.1
- Warmup Steps: 500
- Weight Decay: 0.01
- Max Sequence Length: 128
- Gradient Accumulation Steps: 1
- Scheduler: Linear Scheduler with Warmup
- Early Stopping: True
- Early Stopping Patience: 3
- Early Stopping Delta: 0.01

## Results

## References
#### https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/pipelines#transformers.pipeline
#### look at the pipeline() function documentation from hugging face
#### model selection
#### https://huggingface.co/transformers/v2.9.1/pretrained_models.html


