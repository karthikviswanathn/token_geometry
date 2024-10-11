# The Geometry of Tokens in Internal Representations of Large Language Models
Source code for the paper: 'The Geometry of Tokens in Internal Representations of Large Language Models'

## Description

In this project, we analyze the geometry of tokens in the hidden layers of large language Models
using cosine similarity, intrinsic dimension (ID), and neighborhood overlap (NO). Given a prompt,
this is done in roughly two steps -

1. Extract the internal representations of the tokens -  We use the
[hidden states](https://huggingface.co/docs/transformers/v4.45.2/en/internal/generation_utils#generate-outputs)
variable from the [Transformers](https://huggingface.co/docs/transformers/index) library on Hugging Face.

2. Calculating the observables - For each hidden layer, we consider the point cloud formed
by the token representation at that layer. On this point cloud, we calculate
the cosine similarity, intrinsic dimension (ID), and neighborhood overlap (NO).
Specifically, we use the
[Generalized Ratio Intrinsic Dimension Estimator (GRIDE)](https://www.nature.com/articles/s41598-022-20991-1)
to estimate the intrinsic dimension implemented using the
[DADApy linrary.](https://github.com/sissa-data-science/DADApy) 


## References

- [DADApy](https://github.com/sissa-data-science/DADApy)
- [GRIDE Method](https://www.nature.com/articles/s41598-022-20991-1)
- [Transformers Library](https://huggingface.co/docs/transformers/index)
