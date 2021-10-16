from collections import Counter

from tokenizers import BertWordPieceTokenizer


def ngram(tokens, n):
    l = len(tokens)
    tokens = ['[SEP]'] * (n - 1) + tokens + ['[SEP]'] * (n - 1)
    count = Counter()
    for i in range(n - 1, l):
        key = " ".join(tokens[i:i + n])
        if len(set([',', '.', '-', '\'']) & set(key)) > 0:
            continue
        if count.get(key, None) is None:
            count[key] = 0
        count[key] += 1
    print(f"=== {n}-gram ===")
    for kv in count.most_common(5):
        print(kv)


def main():
    tokenizer = BertWordPieceTokenizer(
        "../bert-base-uncased-vocab/file/bert-base-uncased-vocab.txt",
        lowercase=True)
    output = tokenizer.encode(
        open("titech/CSC.T459.NaturalLanguageProcessing/HW1/1984.txt").read())
    print(f"number of word tokens: {len(output.tokens)}")
    print(f"number of word types: {len(set(output.tokens))}")

    ngram(output.tokens, 1)
    ngram(output.tokens, 2)
    ngram(output.tokens, 3)


if __name__ == "__main__":
    main()
