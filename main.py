#CS_2017_001
#G.K.Akalanka
import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

file = open("corpus.txt", "r")
text = file.read()
tokens = nltk.word_tokenize(text)

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

# The default tagger

default_tagger = nltk.DefaultTagger('NN')
print(default_tagger.tag(tokens))

print('*********************')

# Unigram Tagging

unigram_tagger = nltk.UnigramTagger(brown_tagged_sents, backoff=default_tagger)
print(unigram_tagger.tag(tokens))

print('*********************')

# Bigram Tagging

bigram_tagger = nltk.BigramTagger(brown_tagged_sents, backoff=unigram_tagger)
print(bigram_tagger.tag(tokens))


#Create Text Files

# Three text files are created in the output folder for default,bigram, and unigram tagging

file = open("./output/default.txt", "w")
dvalues = ','.join(str(v) for v in default_tagger.tag(tokens))
file.write(dvalues)
file.close()

file = open("./output/bigram.txt", "w")
bvalues = ','.join(str(v) for v in bigram_tagger.tag(tokens))
file.write(bvalues)
file.close()

file = open("./output/unigram.txt", "w")
uvalues = ','.join(str(v) for v in unigram_tagger.tag(tokens))
file.write(uvalues)
file.close()


