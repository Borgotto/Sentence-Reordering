# Sentence Reordering using Transformer architecture

This task was part of my Deep Learning examination on june 2024 for the Master’s program in Artificial Intelligence at the University of Bologna. The objective of this project is to create a model capable of reordering the words in a sentence. The input is presented as a sequence of words in a random permutation and the task is to recostruct the original English sentence.

In this study I suggest a Transformer Seq to Seq model capable of generating rearranged sentences that maintain the original sentences’ meaning and grammatical correctness.


## Dataset
The Dataset was taken from HuggingFace and contains a large (3.5M+ sentence) knowledge base of generic sentences. The HuggingFace page of the dataset can be found [here](https://huggingface.co/datasets/community-datasets/generics_kb)


## Evaluation Metric

Given the original string and the prediction, the evaluation function:

- checks for the longest matching sequence between the original and the predicted string ()
- divides the lenght of the longest matching sequence by the longest lenght between the predicted and the original sentence

$$ \frac{ LongestMatchingSequence(original, prediction)}{max(len(original), len(prediction))} $$

#### Example

```python
original = "at first Henry wanted to be friends with the king of france"
generated = "Henry wanted to be friends with king of france at the first"

print("your score is ", score(original, generated))
```
```
$ your score is  0.5423728813559322
```

## Constraints
- No pretrained model can be used
- The neural network model should have less than 20M parameters
- No postprocessing techniques can be used

## Model and hyperparameters
For this task a Transformer Seq to Seq model was used.

This kind of model is composed by 2 main parts:

- Encoder: reads the input sequence (in this case the shuffled words) and produces a fixed-dimensional vector representation.
- Decoder: generates the output sequence (original sentence) from the input given by the Encoder.

This kind of models are well known and largely used in natural language processing (NLP) like translations, summarization and classifications.

### Why this model?
The main reason to choose this kind of architecture is the self-attention mechanism. This property should help the model retain the semantic meaning of the words, helping it achive good performance in reorder the words inside a phrase

## Hyperparameters
List of the hyperparameters used:

**Model (~8M param)**:
- `EMBEDDING_DIM` = 128
- `LATENT_DIM` = 512
- `NUM_HEADS` = 20

\
**Training**:
- `EPOCH` = 30
- `BATCH_SIZE` = 256

## Results
The final model has a score of `~0.49` using the provided evaluation metrics, way above the estimated performance of a random classifier, estimated to be around `~0.19` with a standard deviation of `~0.06`.

## Excecution Enviroment
The notebook has been created using [Kaggle notebook](https://www.kaggle.com/code). A requirement.txt file is provided inside the repository and report all the dependencies found at the end of the excecution of the Kaggle notebook.
The same code can also be excecuted in [Colab](https://colab.google/), but the dependencies may not work.
