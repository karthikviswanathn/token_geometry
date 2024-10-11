# The Geometry of Tokens in Internal Representations of Large Language Models
Source code for the paper: 'The Geometry of Tokens in Internal Representations of Large Language Models'

## Description

In this project, we analyze the geometry of tokens in the hidden layers of large language models using cosine similarity, intrinsic dimension (ID), and neighborhood overlap (NO). 

We use [DADApy](https://github.com/sissa-data-science/DADApy) to estimate the intrinsic dimension using the [GRIDE method](https://www.nature.com/articles/s41598-022-20991-1).

To do this, we extract the internal representations of token embeddings using the \
[hidden states](https://huggingface.co/docs/transformers/v4.45.2/en/internal/generation_utils#generate-outputs) \
variable from the [Transformers](https://huggingface.co/docs/transformers/index) library on Hugging Face.

## References

- [DADApy](https://github.com/sissa-data-science/DADApy)
- [GRIDE Method](https://www.nature.com/articles/s41598-022-20991-1)
- [Transformers Library](https://huggingface.co/docs/transformers/index)
