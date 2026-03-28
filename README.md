<div align="center">
<h1>SAKE - Software architectural Knowledge Evaluation Benchmark for Large Language Models</h1>
</div>
<br>

<div align="center">
</div>

## Introduction
We introduce SAKE, a benchmark designed to evaluate language models across a broad set of software architecture knowledge domains.

## Dataset
The dataset has been saved under data/ as ''sake-dataset.json''. An example with only one test query per domain and the entire validation set (for n-shot) has been also saved under data/ as ''sake-dataset-sample.json''.
In case of acceptance, the dataset will be officially made available on Huggingface, making it even more accessible to the scientific community.

## Setup
A python 3.14 environment is needed to run the experiments, together with the dependencies in the requirements.txt file.
Then, to re-run the benchmark a personal API key must be added in the .env file in this repository.
```bash
OPENROUTER_API_KEY=<your_key>
```

## Evaluation scripts
An utility script named "eval_runner.sh" can be run to re-run all the benchmarks on the entire dataset (stored under data/).
Alternatively, a single model can be benchmarked using its specific script available under scripts/.
The lowest level file that actually runs the benchmark can be found under eval/, and it is named "eval_from_openai.py".

The accuracy of the models can be extracted calling the file eval/score.py.

## Prompts
The prompt templates are under prompt_lib/.
nshots.txt is prepended n times to base.txt in case of n>0.

## Results
IMPORTANT: We provide the outputs of the models present in the paper under outputs/.
It is therefore not necessary to run again the benchmarks, but it should be enough to run eval/score.py to obtain again the same results present in the paper.
