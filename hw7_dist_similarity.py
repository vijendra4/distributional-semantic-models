import sys
import nltk
from scipy.stats import spearmanr
from math import sqrt, log
from CollocationMatrix import CollocationMatrix

word_fw_dict = dict()
word_pw_dict = dict()


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


def lookup_word_fw_dict(f, matrix, total_sum):
    """
    Return f(w) in PMI calculation if already computed,
    otherwise computer and store in dict
    """
    if f in word_fw_dict:
        return word_fw_dict[f]
    else:
        pf = matrix.get_col_sum(f) / total_sum
        word_fw_dict[f] = pf
        return pf


def lookup_word_pw_dict(matrix, w, total_sum):
    """
        Return p(w) in PMI calculation if already computed,
        otherwise computer and store in dict
        """
    if w in word_pw_dict:
        return word_pw_dict[w]
    else:
        pw = matrix.get_row_sum(w) / total_sum
        word_pw_dict[w] = pw
        return pw


def populate_colocate(window_size, brown_sents):
    """
    Read in a corpus that will form the basis of the distributional model and perform basic preprocessing.
    """
    matrix = CollocationMatrix()
    window_size = int(window_size)
    brown_words = nltk.corpus.brown.words()

    sent = remove_punc(brown_words)
    for i, word in enumerate(sent):
        # Increment the count of words we've seen.
        for j in range(-window_size, window_size + 1):
            # Skip counting the word itself.
            if (j == 0) | (j == len(sent) - 1):
                continue

            if len(sent) > i + j > 0:
                word_1 = sent[i].lower()
                word_2 = sent[i + j].lower()

                matrix.add_pair(word_1, word_2)
    return matrix


def cosine_similarity_pmf(matrix, w1, w2, total_sum):
    """
        Compute the similarity between the two words, based on cosine similarity using PMI weighting
        """
    w1_id = matrix.word_id(w1)
    w2_id = matrix.word_id(w2)

    if (w1_id is not None) & (w2_id is not None):
        sum_v_x_w = 0
        sqrt_sum_v_x_v = 0
        sqrt_sum_w_x_w = 0

        for feature_index1, feature_value1 in matrix[w1_id].items():
            feature_value1_pmi = calculate_ppmi(matrix, w1, matrix.get_word(feature_index1), total_sum)
            sqrt_sum_v_x_v = sqrt_sum_v_x_v + (feature_value1_pmi * feature_value1_pmi)

            if feature_index1 in matrix[w2_id]:
                sum_v_x_w = sum_v_x_w + (feature_value1_pmi * calculate_ppmi(matrix, w2, matrix.get_word(feature_index1), total_sum))

        for feature_index2, feature_value2 in matrix[w2_id].items():
            feature_value2_pmi = calculate_ppmi(matrix, w2, matrix.get_word(feature_index2), total_sum)
            sqrt_sum_w_x_w = sqrt_sum_w_x_w + (feature_value2_pmi * feature_value2_pmi)

        sqrt_sum_v_x_v = sqrt(sqrt_sum_v_x_v)
        sqrt_sum_w_x_w = sqrt(sqrt_sum_w_x_w)

        return sum_v_x_w / (sqrt_sum_v_x_v * sqrt_sum_w_x_w)

    else:
        return 0


def cosine_similarity_freq(matrix, w1, w2):
    """
    Compute the similarity between the two words, based on cosine similarity using FREQ weighting
    """
    w1_id = matrix.word_id(w1)
    w2_id = matrix.word_id(w2)

    if (w1_id is not None) & (w2_id is not None):
        sum_v_x_w = 0
        sqrt_sum_v_x_v = 0
        sqrt_sum_w_x_w = 0

        for feature_index1, feature_value1 in matrix[w1_id].items():
            sqrt_sum_v_x_v = sqrt_sum_v_x_v + (feature_value1 * feature_value1)
            if feature_index1 in matrix[w2_id]:
                sum_v_x_w = sum_v_x_w + (feature_value1 * matrix[w2_id][feature_index1])

        for feature_index2, feature_value2 in matrix[w2_id].items():
            sqrt_sum_w_x_w = sqrt_sum_w_x_w + (feature_value2 * feature_value2)

        sqrt_sum_v_x_v = sqrt(sqrt_sum_v_x_v)
        sqrt_sum_w_x_w = sqrt(sqrt_sum_w_x_w)

        return sum_v_x_w / (sqrt_sum_v_x_v * sqrt_sum_w_x_w)

    else:
        return 0


def calculate_ppmi(matrix, w, f, total_sum):
    """
    Code to calculate PPMI for two words, using the
    colocation matrix calculated above.
    """

    pw = lookup_word_pw_dict(matrix, w, total_sum)
    pf = lookup_word_fw_dict(f, matrix, total_sum)
    pwf = matrix[matrix.word_id(w)][matrix.word_id(f)] / total_sum

    return 0 if ((pwf == 0) | (pwf < -1)) else log(pwf / (pw * pf))


def print_ten_highest_features(matrix, w, output_file, weighting, total_sum):
    print("Calculating PMI print_ten_highest_features" + w)
    w_id = matrix.word_id(w)

    if w_id is not None:
        output_file.write(matrix.get_word(w_id))
        feature_weight_dict = dict()
        for feature_index in matrix[w_id]:
            feature = matrix.get_word(feature_index)
            if weighting == "PMI":

                feature_weight_dict[feature] = calculate_ppmi(matrix, w, feature, total_sum)
            elif weighting == "FREQ":
                feature_weight_dict[feature] = matrix.get_pair(w, feature)

        sorted_feature_weight_dict = [(k, feature_weight_dict[k]) for k in
                                      sorted(feature_weight_dict, key=feature_weight_dict.get, reverse=True)]

        for k, v in sorted_feature_weight_dict[0:9]:
            output_file.write(" " + k + ":" + str(v))
        output_file.write("\n")
    else:
        output_file.write(w + " not found in matrix" + "\n")


def distributional_similarity_model(matrix, weighting, judgment_filename, output_file):
    """
    1. Read in a file of human judgments of similarity between pairs of words
    2. For each word in the word pair:
          Print the word and its ten (10) highest weighted features (words) and their weights
    3. Compute the similarity between the two words, based on cosine similarity
    """
    with open(judgment_filename) as fp:
        sim_given_list = []
        sim_computed_list = []
        total_sum = matrix.total_sum

        for line in fp:
            w1 = line.split(",")[0]
            w2 = line.split(",")[1]

            print_ten_highest_features(matrix, w1, output_file, weighting, total_sum)
            print_ten_highest_features(matrix, w2, output_file, weighting, total_sum)

            sim_given = line.split(",")[2]
            sim_given_list.append(sim_given)

            sim_computed = 0
            if weighting == "FREQ":
                sim_computed = cosine_similarity_freq(matrix, w1, w2)
            elif weighting == "PMI":
                sim_computed = cosine_similarity_pmf(matrix, w1, w2, total_sum)
            sim_computed_list.append(sim_computed)

            output_file.write(w1 + "," + w2 + ":" + str(sim_computed) + "\n")

        # Compute and print the Spearman correlation between the similarity scores
        corr, p_value = spearmanr(sim_computed_list, sim_given_list)
        output_file.write("correlation:" + str(corr) + "\n")


def main():
    brown_sents = nltk.corpus.brown.sents()

    window_size = sys.argv[1]
    weighting = sys.argv[2]
    judgment_filename = sys.argv[3]
    output_filename = sys.argv[4]

    matrix = populate_colocate(window_size, brown_sents)

    output_file = open(output_filename, "w+")

    distributional_similarity_model(matrix, weighting, judgment_filename, output_file)

    output_file.close()


if __name__ == '__main__':
    main()
