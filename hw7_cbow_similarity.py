import sys
import nltk
import gensim
from scipy.stats import spearmanr


def remove_punc(sent):
    """
    Make all words lowercase.
    Remove Punctuation
    """
    punc = {'(', ')', '!', '.', ',', '-', ':', ';', '/', '\\', '?', '"', '\'', '``'}
    new_sent = []

    for word in sent:
        word = word.lower()
        if word in punc:
            continue

        word_len = len(word)
        if word_len >= 2:
            start_index = 1 if word[0] in punc else 0
            end_index = word_len - 1 if word[word_len - 1] in punc else word_len
            new_sent.append(word[start_index:end_index])
        else:
            new_sent.append(word)

    return new_sent


def create_word2vec_model(brown_sents, window_size):
    """
    Build a continuous bag of words model using word2vec
    """
    return gensim.models.Word2Vec(brown_sents, size=100, window=window_size, min_count=1, workers=1)


def cbow_similarity_model(model, judgment_filename, output_file):
    """
    1. Read in a file of human judgments of similarity between pairs of words.
    2. For each word pair in the file:
        Compute the similarity between the two words, using the word2vec model
        Print out the similarity
    """
    with open(judgment_filename) as fp:
        sim_given_list = []
        sim_computed_list = []

        for line in fp:
            w1 = line.split(",")[0]
            w2 = line.split(",")[1]

            sim_given = line.split(",")[2]
            sim_given_list.append(sim_given)

            sim_computed = 0
            if (w1 in model.wv) & (w2 in model.wv):
                sim_computed = model.wv.similarity(w1, w2)
            sim_computed_list.append(sim_computed)

            output_file.write(w1 + "," + w2 + ":" + str(sim_computed) + "\n")
        corr, p_value = spearmanr(sim_given_list, sim_computed_list)
        output_file.write("correlation:" + str(corr) + "\n")


def main():
    # Read in a corpus that will form the basis of the predictive CBOW distributional model
    brown_sents = nltk.corpus.brown.sents()
    cleaned_brown_sents = []
    for sent in brown_sents:
        cleaned_brown_sents.append(remove_punc(sent))

    window_size = sys.argv[1]
    judgment_filename = sys.argv[2]
    output_filename = sys.argv[3]

    model = create_word2vec_model(cleaned_brown_sents, window_size)

    output_file = open(output_filename, "w+")

    cbow_similarity_model(model, judgment_filename, output_file)

    output_file.close()


if __name__ == '__main__':
    main()
