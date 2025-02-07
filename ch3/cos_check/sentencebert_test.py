from transformers import BertJapaneseTokenizer, BertModel
import torch

class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)

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

model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")

#sentences = ["お辞儀をしている男性会社員", "笑い袋", "テクニカルエバンジェリスト（女性）", "戦うAI", "笑う男性（5段階）", "お金を見つめてニヤけている男性", "「ありがとう」と言っている人", "定年（女性）", "テクニカルエバンジェリスト（男性）", "スタンディングオベーション"]
sentences = read_tsv_file("train.tsv")
print("setences done")
sentence_vectors = model.encode(sentences)
print("encoded done")

import scipy.spatial

#queries = ['暴走したAI', '暴走した人工知能', 'いらすとやさんに感謝', 'つづく']
queries = read_text_file("input.txt")
print("quaries done")
query_embeddings = model.encode(queries).numpy()


closest_n = 5
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_vectors, metric="cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")


    for idx, distance in results[0:closest_n]:
        print(sentences[idx].strip(), "(Score: %.4f)" % (distance / 2))
