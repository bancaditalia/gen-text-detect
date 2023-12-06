import re
from collections.abc import Collection

from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import FunctionTransformer
from nltk.stem.snowball import SnowballStemmer

# Constants for default word tokenizer regular expression
RGX_CHARS = r'A-Za-zÀ-ÖØ-öø-ÿ'
TK_WORDS_REGEXP = f'[{RGX_CHARS}\d]*[{RGX_CHARS}]+[{RGX_CHARS}\d]*|(?:[\d]+(?:[\.,]\d)?)+|[^\w\s]'


def _apply_nested_function(fn, data, raw_type) -> list:
    """
    Function that applies `fn` to the nested elements of `data` of type `raw_type`

    Function that recursively drills-down in nested lists applying a function `fn` to all sub-lists with
    elements of type `raw_type`

    Parameters
    ----------
    fn : function
        Function to apply to the nested `data`
    data : Collection[Any]
        List of elements (possibly nested) on which `fn` is applied
    raw_type : type
        Element type on which `fn` has to be applied

    Raises
    ------
    TypeError
        Type mismatch in the input sentences

    Returns
    -------
    fn_out : Collection[Any]
        Resulting data structure after the application of `fn` to its elements of type `raw_type`
    """
    if isinstance(data, list):
        if len(data) > 0:
            if all(isinstance(x, raw_type) for x in data):
                fn_out = []
                for x in data:
                    fx = fn(x)
                    if fx is not None:
                        fn_out.append(fx)
                return fn_out
            elif all(isinstance(x, list) for x in data):
                fn_out = []
                for x in data:
                    fx = _apply_nested_function(fn, x, raw_type)
                    fn_out.append(fx)
                return fn_out
            raise TypeError(
                f"Type mismatch while parsing nested structure: expected 'list' or '{raw_type}', "
                f"got {set([type(data[i]) for i in data])} instead.")
        else:
            return data
    elif isinstance(data, raw_type):
        #TODO: refactor function to either make it fully recursive or fully iterative
        return fn(data)
    raise TypeError(f"Type mismatch while parsing nested structure: expected 'list', got {type(data)} instead.\n{data}")


def _replace_tokens(input_sentence, mapping) -> list:
    """
    Function that replaces sequences of tokens in a sentence according to a mapping

    Function that transforms `input_sentence` by scanning for sequences of tokens matching a key in the `mapping`
    dictionary and replacing them with their associated value (prioritizing longer sequences over shorter ones).

    Parameters
    ----------
    input_sentence : Collection[str]
       Input sequence of words
    mapping : dict
       Dictionary mapping whitespace-separated sequence of words to replacement words

    Raises
    ------
    TypeError
       Type mismatch in the input sentences

    Returns
    -------
    replaced_sentence : Collection[str]
       Resulting sentence after replacement of words according to the given `mapping`
    """
    max_query_length = max([len(x.split()) for x in mapping.keys()])
    sentence_length = len(input_sentence)
    replaced_sentence = []
    i = 0
    while i < sentence_length:
        match = False
        for l in range(max_query_length, 0, -1):
            if i + l < sentence_length:
                tks = " ".join(input_sentence[i:i + l])
                if tks in mapping:
                    replaced_sentence.append(mapping[tks])
                    i += l
                    match = True
                    break
        if not match:
            replaced_sentence.append(input_sentence[i])
            i += 1
    return replaced_sentence


class SentenceTokenizer(FunctionTransformer):
    """
    NLTK-based sentence tokenizer

    Class implementing a scikit-learn `FunctionTransformer` that tokenizes a document into a list of sentences

    Parameters
    ----------
    language : str
        Language to use for the NLTK sentence tokenizer
    """

    def __init__(self, language='italian') -> None:
        self.language = language
        super().__init__(self.__transform, validate=False)

    def __transform(self, X) -> list:
        """
        Function that tokenizes documents into lists of sentences

        Parameters
        ----------
        X : list
            List of documents (strings) to tokenize

        Raises
        ------
        TypeError
            Type mismatch in the input sentences

        Returns
        -------
        tokenized_documents : list
            The list of sentences
        """
        return [sent_tokenize(x, self.language) for x in X]


class WordTokenizer(FunctionTransformer):
    """
    Wrapper for `nltk.tokenize.RegexpTokenizer`

    Class implementing a scikit-learn `FunctionTransformer` that tokenizes a list of sentences into lists of words
    using as base tokenizer `nltk.tokenize.RegexpTokenizer`

    Parameters
    ----------
    regex : str
        Custom regex used to tokenize sentences
    to_lowercase : bool
        Flag to transform all tokenized words to lowercase
    """

    def __init__(self,
                 regex=TK_WORDS_REGEXP,
                 to_lowercase=True) -> None:
        self.regex = regex
        self.to_lowercase = to_lowercase
        super().__init__(self.__transform, validate=False)

    def __transform(self, X) -> list:
        """
        Function that tokenizes sentences into lists of words

        Parameters
        ----------
        X : list
            Sentences (strings or lists of strings) to tokenize

        Returns
        -------
        tokenized_sentences : list
            The list of tokenized sentences
        """
        tokenizer = RegexpTokenizer(self.regex, discard_empty=True)
        fn = None
        if self.to_lowercase:
            fn = lambda x: tokenizer.tokenize(x.lower())
        else:
            fn = tokenizer.tokenize
        return _apply_nested_function(fn, X, str)


class WordsFilter(FunctionTransformer):
    """
    Custom word filtering data transformer

    Class implementing a scikit-learn `FunctionTransformer` that filters stopwords and blacklisted tokens

    Parameters
    ----------
    drop_symbols : bool
        Flag instructing the tokenizer to drop non-characters [A-Za-z]
    drop_digits : bool
        Flag instructing the tokenizer to drop digits [0-9]
    stopwords_language : str
        Language to use for stopwords filtering (default='italian')
    blacklist : Collection or None
        Additional list of words to filter
    whitelist : Collection or None
        Collection of whitelisted words
    """

    def __init__(self,
                 drop_symbols=False,
                 drop_digits=False,
                 stopwords_language=None,
                 blacklist=None,
                 whitelist=None) -> None:
        self.drop_symbols = drop_symbols
        self.drop_digits = drop_digits
        self.stopwords_language = stopwords_language
        self.blacklist = blacklist
        self.whitelist = whitelist
        super().__init__(self.__transform, validate=False)

    def __check_whitelist(self, token):
        """
        Utility function that checks if input `token` is in `self.whitelist`, if one is provided

        Parameters
        ----------
        token : str
            String token to be checked against `self.whitelist`

        Returns
        -------
        check : bool
            True if `token` is in the whitelist or if no `self.whitelist` is provided, False otherwise
        """
        if self.whitelist is None:
            return True
        else:
            return token in self.whitelist

    def __transform(self, X):
        """
        Function that filters-out words from sentences on the basis of a blacklist and/or blacklist

        Parameters
        ----------
        X : list
            List of tokenized sentences (lists of words) to filter

        Returns
        -------
        filtered_sentences : list
            The list of filtered tokenized sentences
        """
        # prepare filtering support regular expressions and blacklists
        filtered_words = set()
        if self.blacklist is not None:
            filtered_words = filtered_words.union(set(self.blacklist))
        if self.stopwords_language is not None:
            filtered_words = filtered_words.union(set(stopwords.words(self.stopwords_language)))
        regexs = []
        if self.drop_symbols:
            regexs.append(r"[^\wèéàòùì]|_")
        if self.drop_digits:
            regexs.append(r"\d")
        if len(regexs) > 0:
            regex = "|".join(regexs)
        else:
            # match anything
            regex = r"[^\s\S]"
        regex = "(" + regex + ")+"
        return _apply_nested_function(
            lambda x: x if x not in filtered_words and re.match(regex, x) is None and self.__check_whitelist(x) else None,
            X,
            str)


class WordsStemmer(FunctionTransformer):
    """
    Wrapper for `nltk.stem.snowball.SnowballStemmer`

    Class implementing a scikit-learn `FunctionTransformer` that extracts the stem from words in sentences

    Parameters
    ----------
    language : str
        Language to use for word stemming (default='italian')
    """

    def __init__(self, language='italian') -> None:
        self.language = language
        super().__init__(self.__transform, validate=False)

    def __transform(self, X):
        """
        Function that performs word stemming on input tokenized sentences

        Parameters
        ----------
        X : list
            List of tokenized sentences (lists of words) to stem

        Raises
        ------
        TypeError
            Type mismatch in the input tokenized sentences

        Returns
        -------
        stemmed_sentences : list
            The list of stemmed tokenized sentences
        """
        stemmer = SnowballStemmer(self.language)
        return _apply_nested_function(stemmer.stem, X, str)


class TokensReplacer(FunctionTransformer):
    """
    Class that replaces tokens (or sets of tokens) with a given mapping

    Class implementing a scikit-learn `FunctionTransformer` that replaces input tokens according to a mapping
    stored in a dictionary given as a parameter to the constructor.
    The mapping dictionary has key-value pairs of the form <query string> : <replace string>, where "query string"
    is a whitespace-separated sequence of tokens that will be replaced with "replace string".

    Parameters
    ----------
    mapping : dict
        Mapping of tokens to be replaced with their matching replace string
    """

    def __init__(self, mapping) -> None:
        self.mapping = mapping
        super().__init__(self.__transform, validate=False)

    def __transform(self, X):
        """
        Function that replaces sequences of tokens in the input sentences X according to the given mapping

        Parameters
        ----------
        X : list
            List of tokenized sentences (lists of words) to scan for replacement

        Raises
        ------
        TypeError
            Type mismatch in the input tokenized sentences

        Returns
        -------
        replaced_sentences : list
            The list of sentences with matching tokens replaced
        """
        # NOTE: using _apply_nested_function with raw_type = list can lead to problems when using sentence tokenization
        return _apply_nested_function(lambda x: _replace_tokens(x, self.mapping), X, list)


def get_vocabulary(corpus) -> set:
    """
    Simple function that extracts the set of tokens from a tokenized corpus

    Parameters
    ----------
    corpus : Collection
        Collection of tokenized documents (either flat list, or nested)

    Raises
    ------
    TypeError
        Type mismatch in the corpus

    Returns
    -------
    vocabulary : set[str]
        Set of tokens in the corpus
    """
    vocabulary = set()
    if isinstance(corpus, Collection) and not isinstance(corpus, (str, bytes, bytearray)):
        if all(isinstance(x, str) for x in corpus):
            # flat corpus
            vocabulary = set(corpus)
        else:
            for document in corpus:
                if isinstance(document, list):
                    # nested corpus "document -> words"
                    if all(isinstance(word, str) for word in document):
                        vocabulary.update(set(document))
                    elif all(isinstance(sentence, list) for sentence in document):
                        for sentence in document:
                            if all(isinstance(word, str) for word in sentence):
                                vocabulary.update(set(sentence))
                            else:
                                raise TypeError(
                                    f"Type mismatch (expected str) while parsing tokens in sentence: \n{sentence}"
                                )
                    else:
                        raise TypeError(
                            f"Type mismatch (expected list or str) while parsing tokens in document: \n{document}"
                        )
                # nested corpus "document -> sentence -> words"
                else:
                    raise TypeError(
                        f"Type mismatch (expected list, got {type(document)}) while parsing document type in corpus "
                        f"for:\n{document}"
                    )
    else:
        raise TypeError(
            f"Type mismatch while parsing corpus: expected Collection, got {type(corpus)}"
        )
    return vocabulary
