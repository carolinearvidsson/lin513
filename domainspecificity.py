from nltk import word_tokenize

class DomainSpecificity:


    def __init__(self, ws):
        self.ws = ws
        self.domains = {}
        self.domain_specific = set()
        self.__get_domains()
        self.__get_domain_specific()

    def is_domain_specific(self, wordobj):
        is_specific = 0
        if wordobj.token.lower() in self.domain_specific:
            is_specific = 1
        return [is_specific]

    def __get_domain_specific(self):
        '''Populates a set with word types that only occur 
        in one of the given subcorpuses.
        '''
        for domain in self.domains:
            self.domain_specific = self.domain_specific ^ self.domains[domain]

    def __get_domains(self):
        for entry in self.ws.single_word:
            domain = entry.subcorpus
            sentence = word_tokenize(entry.sentence)
            for word in sentence:
                self.domains.setdefault(domain, set()).add(word.lower())
