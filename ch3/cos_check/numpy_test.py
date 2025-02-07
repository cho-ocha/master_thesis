from numpy import dot
from numpy.linalg import norm

def grab_cosine_similarity(self, word1: str, word2: str) -> float:

    vec1 = self.vec[word1].tolist()
    vec2 = self.vec[word2].tolist()

    return dot(vec1, vec2) / (norm(vec1) * norm(vec2)) 

word1 = "類似度出せるかな"
word2 = "類似度出したい"

print(grab_cosine_similarity(word1,word2))