import numpy as np
import spacy

# Ginzaモデルをロード
nlp = spacy.load('ja_ginza')

# tsvファイルの一列目を読み取ってリストにする
def read_tsv_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip().split('\t')[0] for line in lines]

# textファイルを読み取ってリストにする
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def calculate_similarity(texts1, texts2):
    doc1 = [nlp(text) for text in texts1]
    doc2 = [nlp(text) for text in texts2]
    similarities = np.zeros((len(texts1), len(texts2)))
    for i, d1 in enumerate(doc1):
        for j, d2 in enumerate(doc2):
            similarities[i, j] = d1.similarity(d2)
    return similarities

def get_top_similar_lines(lines1, lines2, top_n=5):
    similarities = calculate_similarity(lines1, lines2)
    top_indices = np.argsort(similarities, axis=0)[-top_n:][::-1]
    top_similar_lines = []
    for j in range(similarities.shape[1]):
        top_similar_lines.append([lines1[i] for i in top_indices[:, j]])
    return top_similar_lines

file_path1 = 'train.tsv'
file_path2 = 'input.txt'

lines1 = read_tsv_file(file_path1)
lines2 = read_text_file(file_path2)
print("line done")

top_similar_lines = get_top_similar_lines(lines1, lines2, top_n=5)
print("top done")

f = open("log_spacy.txt","w")
for j, line2 in enumerate(lines2):
    print(f"Target Line: {line2}")
    print(f"Target Line: {line2}",file = f)
    print("Similar Lines:")
    print("Similar Lines:",file=f)
    for i, similar_line in enumerate(top_similar_lines[j]):
        similarity = calculate_similarity([similar_line], [line2])[0, 0]
        print(f"  - {similar_line} (Similarity: {similarity})")
        print(f"  - {similar_line} (Similarity: {similarity})",file=f)
    print()

f.close()