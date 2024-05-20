import argparse
import os
import random
import time
import normalize
import queue
import threading


class Generator:
    def __init__(self, arpa_filename: str) -> None:
        self.model = {}
        self.unique_ngram_counts = {}
        with open(arpa_filename, "r") as arpa_file:
            content = arpa_file.read()
            sections = content.split("\n\n")

            # data section
            data_section = sections[0]
            for line in data_section.strip().split("\n")[1:]:
                n, count = line.split(" ")[1].split("=")
                self.unique_ngram_counts[int(n)] = int(count)
            self.n = int(n)

            # n-gram sections
            n = 1
            for section in sections[1:]:
                for line in section.strip().split("\n")[1:]:
                    self._parse_ngram(n, line)
                n += 1

    def _is_sentence_start(self, word: str) -> bool:
        return word == "<s>"

    def _is_sentence_end(self, word: str) -> bool:
        return word == "</s>"

    def _is_unknown(self, word: str) -> bool:
        return word == "<unk>"

    def _parse_ngram(self, n: int, line: str):
        parts = line.split("\t")
        prob = 10 ** float(parts[0])
        context = " ".join(parts[1:n])
        word = parts[n]
        if self._is_unknown(word):
            # skip <unk>, the generated text should only contain words from the vocabulary
            return
        if context in self.model:
            self.model[context].append((word, prob))
        else:
            self.model[context] = [(word, prob)]

    def _next(self, context: list[str]) -> str:
        context_string = " ".join(context)
        if context_string not in self.model:
            return self._next(context[1:])
        r = random.random()
        summ = 0
        for word, prob in self.model[context_string]:
            summ += prob
            if summ > r:
                return word
        return self._next(context[1:])

    def generate_words(self, num_words: int, context: list[str] = ["<s>"]):
        generated_words = []
        while len(generated_words) < num_words:
            if len(context) >= self.n:
                # limit context to what the model supports
                context = context[1 - self.n :]
            next_word = self._next(context)
            if self._is_sentence_start(next_word) or self._is_sentence_end(next_word):
                # reset context to new sentence
                context = ["<s>"]
            else:
                context.append(next_word)
                generated_words.append(next_word)
        print(" ".join(generated_words))

    def _generate_sentence(self) -> list[str]:
        context = ["<s>"]
        words = []
        while True:
            if len(context) >= self.n:
                # limit context to what the model supports
                context = context[1 - self.n :]
            next_word = self._next(context)
            if self._is_sentence_start(next_word) or self._is_sentence_end(next_word):
                if len(words) == 0:
                    # skip empty generated sentence
                    continue
                else:
                    return words
            words.append(next_word)
            context.append(next_word)

    def generate_sentences(self, num_sentences: int) -> None:
        threads = []
        num_threads = 1000  # TODO: this is a magic number which can probably be optimized
        sentence_queue = queue.Queue()  # thread-safe queue
        writer_thread = threading.Thread(target=self._writer_thread, args=([sentence_queue]))
        writer_thread.start()

        # Start generator threads
        for _ in range(num_threads):
            thread = threading.Thread(target=self._generate_and_enqueue_sentences, args=([num_sentences // num_threads, sentence_queue]))
            thread.start()
            threads.append(thread)

        # Wait for generator threads to finish
        for thread in threads:
            thread.join()

        # Signal writer thread to stop
        sentence_queue.put(None)
        writer_thread.join()

    def _generate_and_enqueue_sentences(self, num_sentences, sentence_queue: queue.Queue):
        for _ in range(num_sentences):
            sentence = self._generate_sentence()
            sentence_queue.put(sentence)

    def _writer_thread(self, sentence_queue: queue.Queue):
        while True:
            sentence = sentence_queue.get()
            if sentence is None:
                break
            print(" ".join(sentence))


def validate_cli_args(cli_args: argparse.Namespace) -> None:
    if cli_args.num_words is None and cli_args.num_sentences is None:
        raise argparse.ArgumentError("either --num_words or --num_sentences needs to be provided, neither were provided")

    if cli_args.num_words is not None and cli_args.num_sentences is not None:
        raise argparse.ArgumentError("either --num_words or --num_sentences needs to be provided, both were provided")

    if cli_args.num_words is not None:
        if cli_args.num_words <= 0:
            raise argparse.ArgumentError(f"--num_words needs to be an integer greater than 0, but is {cli_args.num_words}")

    if cli_args.num_sentences is not None:
        if cli_args.num_sentences <= 0:
            raise argparse.ArgumentError(f"--num_sentences needs to be an integer greater than 0, but is {cli_args.num_sentences}")

    if not os.path.exists(cli_args.lm_filename):
        raise argparse.ArgumentError(f"language model file {cli_args.lm_filename} does not exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lm", "--language_model", type=str, required=True, help="Language model arpa file used for generation", dest="lm_filename"
    )
    parser.add_argument("-nw", "--num_words", type=int, default=None, help="Number of words to generate", dest="num_words")
    parser.add_argument("-n", "--num_sentences", type=int, default=None, help="Number of sentences to generate", dest="num_sentences")
    parser.add_argument("-s", "--seed", type=int, default=time.time_ns(), help="Seed used for randomization", dest="seed")
    parser.add_argument("-c", "--context", type=str, nargs="+", default=[], help="Context used for words generation", dest="context")
    cli_args = parser.parse_args()
    validate_cli_args(cli_args)
    generator = Generator(cli_args.lm_filename)
    if cli_args.num_words is not None:
        generator.generate_words(cli_args.num_words, [normalize.normalize(word) for word in cli_args.context])
    elif cli_args.num_sentences is not None:
        generator.generate_sentences(cli_args.num_sentences)
