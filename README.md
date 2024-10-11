# The Geometry of Tokens in Internal Representations of Large Language Models
Source code for the paper: 'The Geometry of Tokens in Internal Representations of Large Language Models'

## Description
In this project we characterize the geometry of tokens in the the hidden layers of large language models using
cosine similarity, intrinsic dimension and neighborhood overlap. We use the [DADApy](https://github.com/sissa-data-science/DADApy) \
to calculate the intrinsic dimension estimated using [GRIDE](https://www.nature.com/articles/s41598-022-20991-1). 
This is done by first extracting the internal representations of the token embeddings using the \
[hidden states](https://huggingface.co/docs/transformers/v4.45.2/en/internal/generation_utils#generate-outputs) \
variable from the  [Transformers](https://huggingface.co/docs/transformers/index) library on Hugging Face.

## Requirements
