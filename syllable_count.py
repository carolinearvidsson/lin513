from nltk.corpus import cmudict
from wordspace import WS
# doesnt count correctly the cmudict isdigit

class SyllCount():

    def __init__(self):
        self.syll_dict = cmudict.dict()
    
    def get_syll_count(self, wordobject):
        token = wordobject.token
        syll_count = 0
        try:
            phon_token = self.syll_dict[token]
            print(phon_token)
            for phoneme in phon_token[0]:
                if any(char.isdigit() for char in phoneme) == True:
                    syll_count += 1
                    print(token, phoneme, syll_count)
        except:
            vowels = ['a','e','o','u','i']
            for i, character in enumerate(token):
                if character in vowels:
                    if i == 0:
                        syll_count += 1
                    else:
                        if token[i-1] not in vowels:
                            syll_count += 1

        return [syll_count]

# if __name__ == "__main__":
#     ws = WS(['homemade_test.tsv'])
#     s = SyllCount()
#     for word in ws.single_word:
#         s.get_syll_count(word)
