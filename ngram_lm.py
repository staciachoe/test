# An elegant weapon, for a more civilized age
from __future__ import division

from collections import defaultdict
import numpy as np

class AddKTrigramLM(object):
    """Trigram LM with add-k smoothing."""
    order_n = 3

    def __eq__(self, other):
        """Do not modify."""
        state_vars = ['k', 'counts', 'context_totals', 'words', 'V']
        return all([getattr(self, v) == getattr(other, v) for v in state_vars])

    def __init__(self, tokens):
        """Build our smoothed trigram model.

        This should be very similar to SimpleTrigramLM.__init__ from the demo
        notebook, with the exception that we _don't_ want to actually normalize
        the probabilities at training time. Instead, we'll compute the corpus
        counts C_abc = C(w_2, w_1, w) and C_ab = C(w_2, w_1), after which we can
        compute the probabilities on the fly for any value of k. (We'll do this
        in the next_word_proba() function.)

        The starter code will fill in:
          self.counts
          self.wordset

        Your code should populate:
          self.context_totals (total count C_ab for context ab)

        Args:
          tokens: (list or np.array) of training tokens

        Returns:
          None
        """
        self.k = 0.0
        # Raw trigram counts over the corpus.
        # c(w | w_1 w_2) = self.counts[(w_2,w_1)][w]
        # Be sure to use tuples (w_2,w_1) as keys, *not* lists [w_2,w_1]
        self.counts = defaultdict(lambda: defaultdict(lambda: 0.0))

        # Map of (w_1, w_2) -> int
        # Entries are c( w_2, w_1 ) = sum_w c(w_2, w_1, w)
        self.context_totals = dict()

        # Track unique words seen, for normalization
        # Use wordset.add(word) to add words
        wordset = set()

        # Iterate through the word stream once
        # Compute trigram counts as in SimpleTrigramLM
        w_1, w_2 = None, None
        for word in tokens:
            wordset.add(word)
            if w_1 is not None and w_2 is not None:
                self.counts[(w_2,w_1)][word] += 1
            # Update context
            w_2 = w_1
            w_1 = word

        #### YOUR CODE HERE ####
        # Compute context counts

	for context, ctr in self.counts.iteritems():
            self.context_totals[context] = sum(ctr.itervalues())

        #### END(YOUR CODE) ####
        # Total vocabulary size, for normalization
        self.words = list(wordset)
        self.V = len(self.words)

    def set_live_params(self, k=0.0, **params):
        self.k = k

    def next_word_proba(self, word, seq):
        """Next word probability for smoothed n-gram.

        Your code should implement the corresponding equation from the
        notebook, using self.counts and self.context_totals as defined in
        __init__(), above.

        Args:
          word: (string) w in P(w | w_1 w_2 )
          seq: (list of string) [w_1, w_2, w_3, ...]

        Returns:
          (float) P_k(w | w_1 w_2), according to the model
        """
        context = tuple(seq[-2:])  # (w_2, w_1)
        k = self.k
        #### YOUR CODE HERE ####
        # Hint: self.counts.get(...) and self.context_totals.get(...) may be
        # useful here. See note in defaultdict.md about how this works.

	if self.counts.get(context,0) == 0 : 
            return k / (k * len(self.words))
	else :
            return (self.counts.get(context, 0)[word] + k) / (k * len(self.words) + sum(self.counts.get(context).itervalues()))

        #### END(YOUR CODE) ####



class KNTrigramLM(object):
    """Trigram LM with Kneser-Ney smoothing."""
    order_n = 3

    def __eq__(self, other):
        """Do not modify."""
        state_vars = ['delta', 'counts', 'type_contexts',
                      'context_totals', 'context_nnz', 'type_fertility',
                      'z_tf', 'words']
        return all([getattr(self, v) == getattr(other, v) for v in state_vars])

    def __init__(self, tokens):
        """Build our smoothed trigram model.

        This should be similar to the AddKTrigramLM.__init__ function, above,
        but will compute a number of additional quantities that we need for the
        more sophisticated KN model.

        See the documentation in the notebook for the KN backoff model
        definition and equations, and be sure to read the in-line comments
        carefully to understand what each data structure represents.

        Note the usual identification of variables:
          w : c : current word
          w_1 : w_{i-1} : b : previous word
          w_2 : w_{i-2} : a : previous-previous word

        There are two blocks of code to fill here. In the first one, you should
        fill in the inner loop to compute:
          self.counts         (unigram, bigram, and trigram)
          self.type_contexts  (set of preceding words for each word (type))

        In the second one, you should compute:
          self.context_totals  (as in AddKTrigramLM)
          self.context_nnz     (number of nonzero elements for each context)
          self.type_fertility  (number of unique preceding words for each word
                                      (type))

        The starter code will fill in:
          self.z_tf
          self.words

        Args:
          tokens: (list or np.array) of training tokens

        Returns:
          None
        """
        self.delta = 0.75
        # Raw counts over the corpus.
        # Keys are context (N-1)-grams, values are dicts of word -> count.
        # You can access C(w | w_{i-1}, ...) as:
        # unigram: self.counts[()][w]
        # bigram:  self.counts[(w_1,)][w]
        # trigram: self.counts[(w_2,w_1)][w]
        self.counts = defaultdict(lambda: defaultdict(lambda: 0))
        # As in AddKTrigramLM, but also store the unigram and bigram counts
        # self.context_totals[()] = (total word count)
        # self.context_totals[(w_1,)] = c(w_1)
        # self.context_totals[(w_2, w_1)] = c(w_2, w_1)
        self.context_totals = dict()
        # Also store in self.context_nnz the number of nonzero entries for each
        # context; as long as \delta < 1 this is equal to nnz(context) as
        # defined in the notebook.
        self.context_nnz = dict()

        # Context types: store the set of preceding words for each word
        # map word -> {preceding_types}
        self.type_contexts = defaultdict(lambda: set())
        # Type fertility is the size of the set above
        # map word -> |preceding_types|
        self.type_fertility = dict()
        # z_tf is the sum of type fertilities
        self.z_tf = 0.0


        # Iterate through the word stream once
        # Compute unigram, bigram, trigram counts and type fertilities
        w_1, w_2 = None, None
        for word in tokens:
            #### YOUR CODE HERE ####
            self.counts[()][word] += 1
            if w_1 is not None :
                self.type_contexts[word].add(w_1)
                self.counts[(w_1,)][word] += 1
                if w_2 is not None :
                    self.counts[(w_2,w_1)][word] += 1
#            pass

            #### END(YOUR CODE) ####
            # Update context
            w_2 = w_1
            w_1 = word

        ##
        # We'll compute type fertilities and normalization constants now,
        # but not actually store the normalized probabilities. That way, we can compute
        # them (efficiently) on the fly.

        #### YOUR CODE HERE ####
        # Count the total for each context.
        self.context_totals = {ctx:sum(self.counts.get(ctx).values()) for ctx in self.counts}
#        self.context_totals = {ctx:len(self.counts.get(ctx).keys()) for ctx in self.counts}

        # Count the number of nonzero entries for each context.
        self.context_nnz = {ctx:len([ct for ct in self.counts.get(ctx).keys() if ct is not 0]) for ctx in self.counts}
      

        # Compute type fertilities, and the sum z_tf.
        self.type_fertility = {wd:len(self.type_contexts[wd]) for wd in self.type_contexts}

        self.z_tf = float(sum(self.type_fertility.values()))
        #### END(YOUR CODE) ####

        # Total vocabulary size, for normalization
        self.words = self.counts[()].keys()
        self.V = len(self.words)

    def set_live_params(self, delta = 0.75, **params):
        self.delta = delta

    def kn_interp(self, word, context, delta, pw):
        """Compute KN estimate P_kn(w | context) given a backoff probability

        Your code should implement the absolute discounting equation from the
        notebook, using the counts computed in __init__(). Note that you don't
        need to deal with type fertilities here; this is handled in the
        next_word_proba() function in the starter code, below.

        Be sure you correctly handle the case where c(context) = 0, so as to not
        divide by zero later on. You should just return the backoff probability
        directly, since we have no information to decide otherwise.

        Args:
          word: (string) w in P(w | context )
          context: (tuple of string)
          delta: (float) discounting term
          pw: (float) backoff P_kn(w | less_context), precomputed

        Returns:
          (float) P_kn(w | context)
        """
#        pass
        #### YOUR CODE HERE ####
        # Hint: self.counts.get(...) and self.context_totals.get(...) may be
        # useful here. See note in defaultdict.md about how this works.
	if self.counts.get(context,0) == 0 :
            return pw
        else :
#            print "COR", self.context_totals.get(context)
#            #alpha = delta / sum(self.counts.get(context).values())
#            #ppp = (max(0, self.counts.get(context,0).get(word,0) - delta)/sum(self.counts.get(context).values())) + alpha*pw
#	    print "DELTA"
#       	    return (max(0, self.counts.get(context).get(word,0)- delta)/sum(self.counts.get(context).values())) + delta * pw/sum(self.counts.get(context).values())
             alpha = delta * self.context_nnz.get(context) / self.context_totals.get(context)
             return (max(0, self.counts.get(context,0).get(word,0) - delta)/self.context_totals.get(context) + alpha*pw)


        #### END(YOUR CODE) ####


    def next_word_proba(self, word, seq):
        """Compute next word probability with KN backoff smoothing.

        Args:
          word: (string) w in P(w | w_1 w_2 )
          seq: (list of string) [w_1, w_2, w_3, ...]
          delta: (float) discounting term

        Returns:
          (float) P_kn(w | w_1 w_2)
        """
        delta = delta = self.delta
#        # KN unigram, then recursively compute bigram, trigram
        pw1 = self.type_fertility.get(word) / self.z_tf
        pw2 = self.kn_interp(word, tuple(seq[-1:]), delta, pw1)
        pw3 = self.kn_interp(word, tuple(seq[-2:]), delta, pw2)
        return pw3
