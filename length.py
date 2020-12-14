from nltk.corpus import cmudict

class Length():

    def __init__(self):
        self.syll_dict = cmudict.dict()
        self.observed_tokens = {}
    
    def word_length(self, wordobject):
        token = wordobject.token.lower()
        if token not in self.observed_tokens:
            length = len(token)
            syll_count = self.__syll_count(token)
            self.observed_tokens[token] = [length, syll_count]
            return [length, syll_count]
        else:
            return self.observed_tokens[token]
    
    def __syll_count(self, token):
        syll_count = 0
        try:
            phon_token = self.syll_dict[token]
            for phoneme in phon_token[0]:
                if any(char.isdigit() for char in phoneme) == True:
                    syll_count += 1
        except:
            vowels = ['a','e','o','u','i']
            for i, character in enumerate(token):
                if character in vowels:
                    if i == 0:
                        syll_count += 1
                    else:
                        if token[i-1] not in vowels:
                            syll_count += 1

        return syll_count
