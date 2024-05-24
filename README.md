# Text generator

## Create LM arpa

Requirements:

1. A language model ARPA file (e.g. `example/shakespeare.arpa`)

## Generate text

```
generate.py [-h] -lm LM_FILENAME [-nw NUM_WORDS] [-n NUM_SENTENCES] [-s SEED] [-c CONTEXT [CONTEXT ...]] [-p NUM_PROCESSES]

options:
  -h, --help show this help message and exit
  -lm LM_FILENAME, --language_model LM_FILENAME Language model arpa file used for generation
  -nw NUM_WORDS, --num_words NUM_WORDS Number of words to generate
  -n NUM_SENTENCES, --num_sentences NUM_SENTENCES Number of sentences to generate
  -s SEED, --seed SEED  Seed used for randomization
  -c CONTEXT [CONTEXT ...], --context CONTEXT [CONTEXT ...] Context used for words generation
  -p NUM_PROCESSES, --num_processes NUM_PROCESSES Number of processes used for sentence generation
```

### Example generating words without context

In this package run:

```
python generate.py --language_model example/shakespeare.arpa --num_words 100
```

The 100 generated words will be printed to the terminal.

### Example generating words with context

In this package run:

```
python generate.py --language_model example/shakespeare.arpa --num_words 100 --context hello world
```

The 100 generated words will be printed to the terminal.

### Example generating sentences

In this package run:

```
python generate.py --language_model example/shakespeare.arpa --num_sentences 10
```

The 10 generated sentences will be printed to the terminal.
