import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer
import numpy as np
import re
import pandas as pd
from tqdm import tqdm

# トークナイザーとモデルの準備
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
model_q_k = GPT2LMHeadModel.from_pretrained("q_k_iter1", device_map="auto")
model_qk_a = GPT2LMHeadModel.from_pretrained("qk_a_iter1_tinamini", device_map="auto")
# eval.tsv ファイルのパス
train_file_path = 'eval_under6.tsv'
# knowledgeを加えたファイル
knowledge_path = 'answerprompt/iter1_withk_tinamini_answer_6.tsv'
#bertscore出力ファイル
output_file_path_bs = '/home/cho/evaluation/bertscore/iter1_withk_tinamini_answer_6_bs.tsv'
output_path_bleu = '/home/cho/evaluation/bleuscore/iter1_withk_tinamini_answer_6_bleu.tsv'

# プロンプト
prompt = """
入力の概念に関する知識を生成する。
例：

入力： 黒酢を毎日摂りすぎると、どうなるのですか？
知識： 過分な水分は尿として排出される。

入力： ファブリーズで手を洗ったら危ないですか？消毒液が高くて買えません
知識： ファブリーズは中和剤である。

入力：今中3で176ぐらいあるんですが、これって185～190ぐらいは行く可能性はあるんですかね？　まだ伸びてます。
知識：身長は遺伝が強い。

入力： ずっと体調が悪いです。病院に行ってもいつも原因不明って言われます。　　最近体臭が薬みたいな、においがします。　粉薬みたいな甘いような苦いような。　強烈に臭いってにおいではないです。　　何かわかる人いますか？
知識： 甘い体臭は糖尿病の可能性がある。

入力： 皆さんは今日クーラーをつけましたか？
知識： クーラーをつけると涼しくなる。

入力： {question}
知識：
"""

# 回答ようプロンプト
a_prompt = """
質問： {question}
ちなみに {knowledg}
回答：
"""

# 質問と知識を受け取ったら回答を生成
def mkknowledge(prompt, qanda, a_prompt):
    #質問と回答に分ける
    question = qanda.split("\t")[0]

    # promptの質問部分に質問を挿入
    text = prompt.replace('{question}', question)

    # テキストのエンコード
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    token_ids = token_ids.to(model_q_k.device)

    # 文生成
    input_length = len(token_ids[0])
    generate_length = 32
    with torch.no_grad():
        output_ids = model_q_k.generate(
            token_ids,
            do_sample=True,
            max_length=input_length+generate_length,
            min_length=input_length+generate_length,
            num_return_sequences=1,
            top_k=500,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    knowledge = tokenizer.batch_decode([tokens[input_length:] for tokens in output_ids.tolist()])


    # 知識を含めた入力
    text = a_prompt.replace('{question}', question)
    text = text.replace('{knowledg}', knowledge[0])

    # テキストのエンコード
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    token_ids = token_ids.to(model_qk_a.device)
    #token_ids = token_ids.to(model.module.device)

    # 文生成
    input_length = len(token_ids[0])
    generate_length = 128
    with torch.no_grad():
        output_ids = model_qk_a.generate(
        #output_ids = model.module.generate(
            token_ids,
            do_sample=True,
            max_length=input_length+generate_length,
            min_length=input_length+generate_length,
            num_return_sequences=1,
            top_k=500,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    sentence = tokenizer.batch_decode([tokens[input_length:] for tokens in output_ids.tolist()])

    return sentence[0],knowledge[0]

# 結果を保存するための空のDataFrameを作成
result_data = pd.DataFrame(columns=['correct', 'answer', 'knowledge'])

with open(train_file_path) as f:
    # tqdmで進捗バーを表示するためのラップ
    # ファイルの行数を取得
    total_lines = sum(1 for _ in f)
    # ファイルを再度開いて処理を始める
    f.seek(0)
    for line in tqdm(f, total=total_lines, desc="Processing"):
        result_sentence,knowledge = mkknowledge(prompt,line,a_prompt)
        answer = line.split("\t")[1]
        answer = answer.split("\n")[0]
        result_data = result_data.append({'correct': answer, 'answer': result_sentence, 'knowledge':knowledge}, ignore_index=True)
        # 結果をファイルに追記
        result_data.to_csv(knowledge_path, sep='\t', mode='a', header=False, index=False)
        # 追記後にDataFrameをクリア
        result_data = pd.DataFrame(columns=['correct', 'answer', 'knowledge'])


#bertscore
from bert_score import score

refs = []
cands = []

# TSVファイルのパス(上に)


# TSVファイルを読み込む
with open(knowledge_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 各行をタブで分割
        columns = line.strip().split('\t')

        # 1列目をrefsに追加
        refs.append(columns[0])

        # 2列目をcandsに追加
        cands.append(columns[1])


P, R, F1 = score(cands, refs, lang="ja", verbose=True)
with open(output_file_path_bs, 'w', encoding='utf-8') as output_file:
    # ヘッダー行を書き込む
    output_file.write("cands\trefs\tP\tR\tF1\n")

    # 各ペアの結果を書き込む
    for i in range(len(cands)):
        output_file.write(f"{cands[i]}\t{refs[i]}\t{P[i].item():.4f}\t{R[i].item():.4f}\t{F1[i].item():.4f}\n")


print(f"System level Precision score: {P.mean():.3f}")
print(f"System level Recall score: {R.mean():.3f}")
print(f"System level F1 score: {F1.mean():.3f}")

# bleuscore

import MeCab
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
cc = SmoothingFunction()

mecab = MeCab.Tagger('-Owakati')
# TSVファイルを読み込む(上に)


refs = []
cands = []
with open(knowledge_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 各行をタブで分割
        columns = line.strip().split('\t')

        # 1列目をrefsに追加
        refs.append(columns[0])
        # 2列目をcandsに追加
        cands.append(columns[1])

l = len(refs)
ow_ref = [0] * l
ow_can = [0] * l

for i in range(l):
    ow_ref[i] = mecab.parse(refs[i]).strip()
    ow_can[i] = mecab.parse(cands[i]).strip()

l = len(ow_ref)
b = [1] * l
score = []

for i in range(l):
    ref = word_tokenize(ow_ref[i])
    hyp = word_tokenize(ow_can[i])
    b[i] = str(bleu_score.sentence_bleu([ref], hyp, smoothing_function=cc.method7))
    score.append(bleu_score.sentence_bleu([ref], hyp, smoothing_function=cc.method7))


with open(output_path_bleu, 'w', encoding='utf-8') as output_file:
    # ヘッダー行を書き込む
    output_file.write("cands\trefs\tscore\n")

    # 各ペアの結果を書き込む
    for i in range(len(cands)):
        output_file.write(f"{cands[i]}\t{refs[i]}\t{b[i]}\n")

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t           
    avg = sum_num / len(num)
    return avg
print(cal_average(score))
