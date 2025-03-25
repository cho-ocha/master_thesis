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
model = GPT2LMHeadModel.from_pretrained("qk_a_iter1/")
# model = nn.DataParallel(model)

# train.tsv ファイルのパス
train_file_path = 'iter1_data/eval_knowledge.tsv'
# knowledgeを加えたファイル
knowledge_path = 'iter1_data/eval_answer.tsv'

if torch.cuda.is_available():
    model = model.to("cuda")


# 質問と知識を受け取ったら回答を生成
def mkknowledge(qanda):
    #質問と回答に分ける
    question = qanda.split("\t")[0]
    knowledge = qanda.split("\t")[2]
    # promptの質問部分に質問を挿入
    text = question + knowledge

    # テキストのエンコード
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    token_ids = token_ids.to(model.device)

    # 文生成
    input_length = len(token_ids[0])
    generate_length = 128
    with torch.no_grad():
        output_ids = model.generate(
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

    return sentence[0]

# 結果を保存するための空のDataFrameを作成
result_data = pd.DataFrame(columns=['correct', 'answer'])

with open(train_file_path) as f:
    # tqdmで進捗バーを表示するためのラップ
    # ファイルの行数を取得
    total_lines = sum(1 for _ in f)
    # ファイルを再度開いて処理を始める
    f.seek(0)
    for line in tqdm(f, total=total_lines, desc="Processing"):
        result_sentence = mkknowledge(line)
        answer = line.split("\t")[1]
        answer = answer.split("\n")[0]
        result_data = result_data.append({'correct': answer, 'answer': result_sentence}, ignore_index=True)
        # 結果をファイルに追記
        result_data.to_csv(knowledge_path, sep='\t', mode='a', header=False, index=False)
        # 追記後にDataFrameをクリア
        result_data = pd.DataFrame(columns=['correct', 'answer'])
