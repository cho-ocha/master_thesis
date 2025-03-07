import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer
import numpy as np
import re
import pandas as pd
from tqdm import tqdm

# トークナイザーとモデルの準備
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
model_q_k = GPT2LMHeadModel.from_pretrained("q_k_iter1_rm", device_map="auto")
model_qk_a = GPT2LMHeadModel.from_pretrained("qk_a_iter1_rm", device_map="auto")

# train.tsv ファイルのパス
train_file_path = 'train_half1.tsv'
# knowledgeを加えたファイル
knowledge_path = 'iter2_rm_data/train_knowledge_half1.tsv'
# 10個の生成知識を収容
onlyknowledge_path = 'iter2_rm_data/knowledge_half1.txt'

#if torch.cuda.is_available():
#    model = model.to("cuda")

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

# softmax関数
def softmax(x):
	exps = np.exp(x)
	return np.divide(exps, np.sum(exps))

# "¥t"区切りの文を与えたときに、一列目から二列目が出る生成対数確率を出力
def cloze_prob(text, model):
	whole_text_encoding = tokenizer.encode(text)
	#tokens = tokenizer.convert_ids_to_tokens(whole_text_encoding)
	# Parse out the stem of the whole sentence (i.e., the part leading up to but not including the critical word)
	text_list = text.split("\t")
	stem = ' '.join(text_list[:-1])
	stem_encoding = tokenizer.encode(stem)
	# Run the entire sentence through the model. Then go "back in time" to look at what the model predicted for each token, starting at the stem.
	cw_encoding = whole_text_encoding[len(stem_encoding):]
	#cw_tokens = tokenizer.convert_ids_to_tokens(cw_encoding)
 
	# Put the whole text encoding into a tensor, and get the model's comprehensive output
	tokens_tensor = torch.tensor([whole_text_encoding]).to(model.device)
	
	with torch.no_grad():
		outputs = model(tokens_tensor)
		predictions = outputs[0]   
          
	logprobs = [np.log(softmax(predictions[-1][j].cpu().numpy())) for j in range(-1-len(cw_encoding), -1, 1)]

	conditional_probs = [logprobs[j][cw] for j, cw in enumerate(cw_encoding)]

	# now that you have all the relevant probabilities, return their product.
	# This is the probability of the critical word given the context before it.
	return np.sum(conditional_probs)/len(cw_encoding)

# 質疑とpromptを受け取ったら知識を10個生成し、スコアを出し、最も高いスコアの知識を含むデータを返す
def mkknowledge(prompt, qanda):
    #質問と回答に分ける
    question = qanda.split("\t")[0]
    answer = qanda.split("\t")[1]
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
            num_return_sequences=10,
            top_k=500,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    sentences = tokenizer.batch_decode([tokens[input_length:] for tokens in output_ids.tolist()])

    best_score = float('-inf')  # 最初は最低のスコアを設定
    best_sentence = None

    k = open(onlyknowledge_path, 'a')

    for sentence in sentences:
        q_k = cloze_prob(question + "\t" + sentence, model_q_k)
        qk_a = cloze_prob(question + sentence + "\t" + answer, model_qk_a)
        score = q_k + qk_a
        
        k.write(sentence)
        k.write(str(score))
        k.write("\n")

        if score > best_score:
            best_score = score
            best_sentence = sentence
    k.write("\n")

    return best_sentence, best_score

# 結果を保存するための空のDataFrameを作成
result_data = pd.DataFrame(columns=['query', 'answer', 'sentence', 'score'])

with open(train_file_path) as f:
    # tqdmで進捗バーを表示するためのラップ
    # ファイルの行数を取得
    total_lines = sum(1 for _ in f)
    # ファイルを再度開いて処理を始める
    f.seek(0)
    for line in tqdm(f, total=total_lines, desc="Processing"):
        for _ in range(3):
            result_sentence, result_score = mkknowledge(prompt, line)
            question = line.split("\t")[0]
            answer = line.split("\t")[1]
            answer = answer.split("\n")[0]
            result_data = result_data.append({'query': question, 'answer': answer, 'sentence': result_sentence, 'score': result_score}, ignore_index=True)
            # 結果をファイルに追記
            result_data.to_csv(knowledge_path, sep='\t', mode='a', header=False, index=False)
            # 追記後にDataFrameをクリア
            result_data = pd.DataFrame(columns=['query', 'answer', 'sentence', 'score'])
