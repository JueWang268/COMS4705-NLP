# %% [markdown]
# # **Assignment 0: Tokenization!**
# 
# # **Code Submission Instructions**
# 1. Set `SUBMISSION_READY = True`
# 2. In Google Colab, please click File > Download > Download .py
# 3. Upload the .py file to Gradescope
# # **Introduction**
# 
# This notebook implements a **byte-level tokenizer** using a **trie-based vocabulary** and an iterative **byte-pair merging algorithm** (similar to BPE).
# 
# The goal is to learn a **compact and efficient vocabulary** from raw text data, which can then be used to tokenize and encode text into integer IDs for NLP applications.
# 
# Key features:
# - Works at the **byte level**, so it can handle any Unicode text without pre-tokenization.
# - Uses a **maximum-length greedy tokenization** heuristic to match the longest token in the trie.
# - Supports **configurable vocabulary size**, maximum token length, and control over merging across spaces.
# - Can **save/load** the vocabulary for reuse in other tasks or models.
# 
# This notebook contains:
# 1. Class definitions for `TokenizerLearner` and `Tokenizer`.
# 2. Dataset loading and preprocessing.
# 3. Vocabulary learning loop with adjacency counting.
# 4. Vocabulary saving and verification.
# 5. Example encoding and decoding to test the tokenizer.
# 
# There is no need for a GPU for this assignment.

# %% [markdown]
# # **Imports and Dependencies**


# %%
import pygtrie
from collections import Counter
import datasets
import json
import itertools

SUBMISSION_READY = False

# %% [markdown]
# # **TokenizerLearner**
# The `TokenizerLearner` class builds a vocabulary using byte-pair merges.

# %%
class TokenizerLearner:
    def __init__(self, data_iterator, vocab_size=65536, docs_per_iter=10000, max_token_length=30, no_subwords_across_space=True):
        print(f"\nInitializing TokenizerLearner:")
        print(f"  - vocab_size: {vocab_size}")
        print(f"  - docs_per_iter: {docs_per_iter}")
        print(f"  - max_token_length: {max_token_length}")
        self.vocab_size = vocab_size
        self.data = data_iterator
        self.no_subwords_across_space = no_subwords_across_space
        self.data_iterator = iter(data_iterator)
        self.max_token_length = max_token_length
        self.docs_per_iter = docs_per_iter
        self.vocab = None

        # Byte-level vs unicode literals to control token boundaries
        self.space_char = list(b' ')[0]
        self.newline_char = list(b'\n')[0]

    def maybe_add(self, most_common_adjacencies):
        for most_common_adjacency in most_common_adjacencies:
            if self.space_char in most_common_adjacency[0][1] and self.no_subwords_across_space:
                continue
            if len(most_common_adjacency[0][0]) + len(most_common_adjacency[0][1]) > self.max_token_length:
                continue
            if (most_common_adjacency[0][0] == self.newline_char) + (most_common_adjacency[0][1] == self.newline_char) == 1:
                continue
            new_token = most_common_adjacency[0][0] + most_common_adjacency[0][1]
            new_string = bytes(new_token).decode('utf-8', errors='ignore')
            print(f"  Most common adjacency: '{new_token}' (count: {most_common_adjacency[1]})")
            print(f"  Most common adjacency: '{new_string}' (count: {most_common_adjacency[1]})")
            return new_token
        return None

    def learn(self):
        print("\nStarting vocabulary learning...")
        iteration = 0

        # Make a tokenizer
        print("  Creating tokenizer...")
        self.tokenizer = Tokenizer(vocab=self.vocab, max_token_length=self.max_token_length)

        # Initialize with all one-length byte strings
        self.tokenizer.update_trie([(x,) for x in range(256)])

        while len(self.tokenizer.trie) < self.vocab_size:
            iteration += 1
            print(f"\nIteration {iteration}:")
            print(f"Current vocab size: {len(self.tokenizer.trie)}")

            text_docs = []
            for i in range(self.docs_per_iter):
                try:
                    doc = next(self.data_iterator)
                except StopIteration:
                    self.data_iterator = iter(self.data)
                    doc = next(self.data_iterator)
                text = doc['text'].encode('utf-8')
                text_docs.append(text)

            # In this section,
            # (1) iterate through a batch of text documents, tokenizing
            # each one and counting token pair adjacences.
            # use self.tokenizer._tokenize(doc) to tokenize (so you'll need to
            # implement that first.)
            # (2) next, go through the sorted token adjacenies pair and
            # use the self.maybe_add function to get which token should be added
            # (3) update the tokenizer's trie with the new token.
            # --------------------------------- BEGIN STUDENT TODO
            for doc_in_iter in text_docs:
                tokens = self.tokenizer._tokenize(doc_in_iter)
                pairs = []
                for i in range(len(tokens)-1):
                    pairs.append((tokens[i], tokens[i+1]))
                pair_counts = Counter(pairs)
            most_common_adjacencies = pair_counts.most_common()
            new_token = self.maybe_add(most_common_adjacencies)
            
            if new_token is None:
                print("No more new tokens to be learned.")
                break
            self.tokenizer.update_trie([new_token]) # have to wrap new token in a list due to typing

            # --------------------------------- END STUDENT TODO

    def save(self, path):
        print(f"\nSaving vocabulary to {path}")
        with open(path, 'w') as f:
            for token in sorted(self.tokenizer.trie):
                f.write(json.dumps([token])+'\n')
        print(f"Saved {len(self.tokenizer.trie)} tokens")


# %% [markdown]
# # **Tokenizer**
# The `Tokenizer` class performs encoding/decoding with the learned vocabulary.
# 

# %%
class Tokenizer:
    def __init__(self, vocab_path=None, vocab=None, max_token_length=30, partial_trie=None):
        print(f"\nInitializing Tokenizer:")
        print(f"  - vocab_path: {vocab_path}")
        print(f"  - vocab size: {len(vocab) if vocab else 'None'}")
        print(f"  - max_token_length: {max_token_length}")
        self.vocab_path = vocab_path
        self.id_to_tok = []
        self.trie = pygtrie.Trie()
        self.trie = self.trie if partial_trie is None else partial_trie
        self.max_token_length = max_token_length

        if vocab_path or vocab:
            self.update_trie(vocab)

    def update_trie(self, new_vocab=None):
        print("\nUpdating trie...")
        if new_vocab is None and self.vocab_path:
            print(f"Loading from vocab file: {self.vocab_path}")
            with open(self.vocab_path, 'r') as f:
                for i, line in enumerate(f):
                    token = tuple(json.loads(line)[0])
                    self.id_to_tok.append(token)
                    self.trie[token] = i
        elif new_vocab:
            for token in new_vocab:
                print(token)
                self.id_to_tok.append(token)
                self.trie[token] = len(self.trie)

    def encode(self, text):
        return self._tokenize(text, return_ids=True)

    def decode(self, tokens):
        tokens = [self.id_to_tok[x] for x in tokens]
        return bytes(itertools.chain.from_iterable(tokens)).decode('utf-8', errors='ignore')

    def _tokenize(self, text, return_ids=False):
        # In this section,
        # (1) encode the text to receive a bytestring using
        #     text.encode('utf-8', errors='ignore')
        # (2) tokenize the string using the trie we're developing
        # As a hint, consider how to use the self.max_token_length to
        # efficiently query the trie, and note that we use the maximum-length
        # greedy tokenization heuristic.
        # (3) if return_ids=True, then return a list of integer ids. Otherwise,
        # return a list of byte lists.
        # --------------------------------- BEGIN STUDENT TODO
        bytes = text.encode('utf-8', errors='ignore') if isinstance(text, str) else text
        tokens = []
        i = 0
        while i < len(bytes):
            tok = None
            for j in range(self.max_token_length, 0, -1):
                if i + j <= len(bytes):
                    sub_bytes = tuple(bytes[i:i+j])
                    if sub_bytes in self.trie:
                        tok = sub_bytes
                        break
            if tok is None:
                tok = (bytes[i],)
                i += 1
            else:
                i += len(tok)
            tokens.append(self.trie[tok] if return_ids else tok)
        return tokens

        # --------------------------------- END STUDENT TODO

# %% [markdown]
# # **Dataset Loading, Tokenizer Training, and Saving Vocabulary**
# The dataset lives [here](https://huggingface.co/datasets/coms4705-hewitt/fineweb-linuxlike/tree/main). It should download automatically.
# 
# Runtime roughly scales with vocab size. Feel free to play around with it. What happens when it is less than 256?

# %%
print("Starting tokenizer test...")
print("Loading dataset...")
dataset = datasets.load_dataset('coms4705-hewitt/fineweb-linuxlike', 'default', streaming=True)['train']
print("Dataset loaded")

print("\nCreating TokenizerLearner...")
# learner = TokenizerLearner(dataset, vocab_size=65536, docs_per_iter=20, no_subwords_across_space=False)
learner = TokenizerLearner(dataset, vocab_size=600, docs_per_iter=20, no_subwords_across_space=True)
print("Starting learning process...")
if not SUBMISSION_READY:
    learner.learn()
print("\nLearning complete!")
learner.save('vocab-65k-fw-byte-sas.txt')

# %% [markdown]
# # **Testing the Tokenizer**
# 
# Once the vocabulary is learned, we can test the tokenizer by encoding some example strings and decoding them back to verify correctness.

# %%
examples = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def tokenize(text): return text.split()",
    "🌟 Unicode characters work too! 🚀",
    "What happens when you decode a \u2603?"
]

print("Encoding and decoding examples:")
for text in examples:
    print("\nOriginal text: ", text)
    encoded = learner.tokenizer.encode(text)
    print("Encoded bytes: ", encoded)
    token_strings = [learner.tokenizer.decode([token]) for token in encoded]
    print("Individual 'decoded' token strings: ", token_strings)
    print(f"Broke {len(text)} characters into {len(encoded)} tokens!")
    decoded = learner.tokenizer.decode(encoded)
    print("Decoded text:", decoded)
    print('Is decoded same as original text?: ', decoded==text)

