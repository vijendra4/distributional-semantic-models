from collections import defaultdict


class CollocationMatrix(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._word_to_index_mapping = {}  # Where we'll store string->int mapping.
        self._index_to_word_mapping = {}

    def get_word(self, index):
        if index in self._index_to_word_mapping:
            return self._index_to_word_mapping[index]
        else:
            return None

    def word_id(self, word, store_new=False):
        """
        Return the integer ID for the given vocab item. If we haven't
        seen this vocab item before, give ia a new ID. We can do this just
        as 1, 2, 3... based on how many words we've seen before.
        """
        if word not in self._word_to_index_mapping:
            if store_new:
                self._index_to_word_mapping[len(self._word_to_index_mapping)] = word
                self._word_to_index_mapping[word] = len(self._word_to_index_mapping)
                self[self._word_to_index_mapping[word]] = defaultdict(int)  # Also add a new row for this new word.
            else:
                return None
        return self._word_to_index_mapping[word]

    def add_pair(self, w_1, w_2):
        """
        Add a pair of colocated words into the coocurrence matrix.
        """
        w_id_1 = self.word_id(w_1, store_new=True)
        w_id_2 = self.word_id(w_2, store_new=True)
        self[w_id_1][w_id_2] += 1  # Increment the count for this collocation

    def get_pair(self, w_1, w_2):
        """
        Return the colocation for w_1, w_2
        """
        w_1_id = self.word_id(w_1)
        w_2_id = self.word_id(w_2)
        if w_1_id and w_2_id:
            return self[w_1_id][w_2_id]
        else:
            return 0

    def get_row(self, word):
        word_id = self.word_id(word)
        if word_id is not None:
            return self.get(word_id)
        else:
            return defaultdict(int)

    def get_row_sum(self, word):
        """
        Get the number of total contexts available for a given word        
        """
        return sum(self.get_row(word).values())

    def get_col_sum(self, word):
        """
        Get the number of total contexts a given word occurs in
        """
        f_id = self.word_id(word)
        return sum([self[w][f_id] for w in self.keys()])

    @property
    def total_sum(self):
        return sum([self.get_row_sum(w) for w in self._word_to_index_mapping.keys()])
