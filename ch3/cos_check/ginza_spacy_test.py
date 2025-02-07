import spacy
nlp = spacy.load('ja_ginza')
doc1 = nlp('商隊はすぐに襲われる')
doc2 = nlp('一部の参加者が暴徒化し店舗での略奪などが起きた')
doc3 = nlp('大量仕入多売のスーパーマーケット商法が全国に波及した')

print(doc1.similarity(doc1))
print(doc1.similarity(doc2))
print(doc1.similarity(doc3))