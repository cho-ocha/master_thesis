documents = ["低く大きな吠え声はよく響く", "吠え声は低くよく響く", "吠え声が大きくよく響く", "運動量は多く低い吠え声はよく響く", "低く力強い音が出る", "吠え声はよく響く"]

import numpy as np
import MeCab
from sklearn.feature_extraction.text import CountVectorizer

def parse_text(text):
    tagger = MeCab.Tagger()
    words = []
    for c in tagger.parse(text).splitlines()[:-1]:
        surface, feature = c.split('\t')
        pos = feature.split(',')[0]
        if pos == '名詞' or pos == '動詞':
            words.append(surface)
    return words

parsed_documents =  [" ".join(parse_text(line)) for line in documents]
doc_array = np.array(parsed_documents)
cv = CountVectorizer()
bow = cv.fit_transform(doc_array)
features = cv.get_feature_names()
print('次元数:', len(features))
print('元の文書:', documents[0])
print('BOW変換後:',bow.toarray()[0] )


from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
tf_idf = tfidf.fit_transform(bow)

#IDFを確認する
print('元の文書:', documents[0])
print('TF-IDF 変換後:', tf_idf.toarray()[0])


from sklearn.metrics.pairwise import cosine_similarity

print('cos類似度 BOW')
sim_bow = cosine_similarity(bow[0], bow)[0] #1行目との比較
bow_sim_top5 = np.argsort(sim_bow)[::-1][:5] #1行目との類似度上位5計算
for i in bow_sim_top5:
    print("score: {:.2f} {}".format(sim_bow[i], documents[i]))
print()
print('cos類似度 TF-IDF')
sim_tfidf = cosine_similarity(tf_idf[0], tf_idf)[0]
tfidf_sim_top5 = np.argsort(sim_tfidf)[::-1][:6] 
for i in tfidf_sim_top5:
    print("score: {:.2f} {}".format(sim_tfidf[i], documents[i]))