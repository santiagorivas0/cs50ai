import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatic" | "little" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "holmes" | "watson" | "armchair" | "companion" | "day" | "door" | "hand" | "he" | "him" | "his" | "home" | "i" | "landlady" | "letter" | "night" | "paint" | "palm" | "paper" | "she" | "smile" | "thursday" | "walk"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "chuckled" | "had" | "lit" | "sat" | "smiled" | "took" | "was"
"""

# Define your NONTERMINALS here
NONTERMINALS = """
S -> NP VP | S Conj S

NP -> Det N | Det AdjP N | N | NP PP
AdjP -> Adj | Adj AdjP

VP -> V | V NP | V PP | V Adv | Adv V | VP PP

PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)

def main():
    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            sentence = f.read()

    # Otherwise, get sentence as input
    else:
        sentence = input("Sentence: ")

    # Pre-process sentence
    words = preprocess(sentence)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(words))
    except ValueError as e:
        print(e)
        return

    # If no parse trees, sentence is invalid
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))

def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process the sentence by converting all words to lowercase
    and removing any word without at least one alphabetic character.
    """
    words = nltk.word_tokenize(sentence.lower())
    return [word for word in words if any(c.isalpha() for c in word)]

def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is a subtree labeled "NP" that does not itself
    contain any other NP subtrees.
    """
    chunks = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        if not any(sub.label() == "NP" for sub in subtree.subtrees(lambda t: t != subtree)):
            chunks.append(subtree)
    return chunks

if __name__ == "__main__":
    main()
