import pickle
import pathlib
import random
import re
import unicodedata
import tensorflow as tf
import matplotlib.pyplot as plt


def normalize(line, PATH):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    line = re.sub(r"^([^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(\s[^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(?!\s)([^ \w])$", r" \1", line)
    line = re.sub(r"(?!\s)([^ \w]\s)", r" \1", line)
    
    eng, it = line.split("\t")
    eng = "[start] " + eng + " [end]" # the spaces are important
    return it, eng
 
def normalized_lines(PATH):
    ''' 
    Store in the file the pre-preprocessed lines
    '''
    text_pairs = []
    # normalize each line and separate into English and French
    with open(PATH) as fp:
        for line in fp:
            new_line = "\t".join(line.split("\t")[:2])
            new_line = normalize(new_line,PATH)
            text_pairs.append(new_line)
    
    # print some samples
    for _ in range(5):
        print(random.choice(text_pairs))
    
    with open("text_pairs.pickle", "wb") as fp:
        pickle.dump(text_pairs, fp)
        
    return text_pairs
    

def Stat():
    ''' 
    Compute some important statistics on the preprocessed file
    '''
    with open("text_pairs.pickle", "rb") as fp:
        text_pairs = pickle.load(fp)
    
    # count tokens
    it_tokens, eng_tokens = set(), set()
    
    it_maxlen, eng_maxlen = 0, 0
    for it, eng in text_pairs:
        it_tok, eng_tok = it.split(), eng.split()
        it_maxlen = max(it_maxlen, len(it_tok))
        eng_maxlen = max(eng_maxlen, len(eng_tok))
        it_tokens.update(it_tok)
        eng_tokens.update(eng_tok)
        
    print(f"Total Italian tokens: {len(it_tokens)}")
    print(f"Total English tokens: {len(eng_tokens)}")
    print(f"Max Italian length: {it_maxlen}")
    print(f"Max English length: {eng_maxlen}")
    print(f"{len(text_pairs)} total pairs")
    
    
def hist():
    ''' 
    knowing the maximum length of sentences is not as useful as knowing their distribution
    '''
    with open("text_pairs.pickle", "rb") as fp:
        text_pairs = pickle.load(fp)
    
    # histogram of sentence length in tokens
    it_lengths = [len(eng.split()) for eng, fra in text_pairs]
    eng_lengths = [len(fra.split()) for eng, fra in text_pairs]
    
    plt.hist(it_lengths, label="it", color="red", alpha=0.33)
    plt.hist(eng_lengths, label="eng", color="blue", alpha=0.33)
    plt.yscale("log")     # sentence length fits Benford"s law
    plt.ylim(plt.ylim())  # make y-axis consistent for both plots
    plt.plot([max(it_lengths), max(it_lengths)], plt.ylim(), color="red")
    plt.plot([max(eng_lengths), max(eng_lengths)], plt.ylim(), color="blue")
    plt.legend()
    plt.title("Examples count vs Token length")
    plt.show()