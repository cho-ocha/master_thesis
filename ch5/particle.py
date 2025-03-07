from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
from tqdm import tqdm
import gc

#モデルのロード(llama3)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto")

# dev.tsv ファイルのパス
#dev_file_path = '3hop1_paragraph.tsv'
dev_file_path = '3hop1_paragraph_2.tsv'
# knowledgeを加えたファイル
knowledge_path = 'data_para/q_a_k1_k2_k3.tsv'
# 10個の生成知識を収容
knowledges_1 = 'data_para/q_a_k1_s.tsv'
knowledges_2 = 'data_para/q_a_k1_k2_s.tsv'
knowledges_3 = 'data_para/q_a_k1_k2_k3_s.tsv'


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
    
	del tokens_tensor
	torch.cuda.empty_cache()
	gc.collect()

	logprobs = [np.log(softmax(predictions[-1][j].cpu().numpy())) for j in range(-1-len(cw_encoding), -1, 1)]

	conditional_probs = [logprobs[j][cw] for j, cw in enumerate(cw_encoding)]

	# now that you have all the relevant probabilities, return their product.
	# This is the probability of the critical word given the context before it.
	return np.sum(conditional_probs)/len(cw_encoding)

#promptを受け取ったらn個の知識を生成
def mk_knowledges(prompt,n):
    text = prompt

    token_ids = tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    )
    with torch.no_grad():
        #デフォルト
        #output_ids = model.generate(
        #    token_ids.to(model.device),
        #    min_new_tokens=10,
        #    max_new_tokens=60,
        #    do_sample=True,
        #    temperature=1.0,
        #    top_p=0.95,
        #    top_k=500,
        #    num_return_sequences=n,
        #    #pad_token_id=eos_token_id,
        #)
        output_ids = model.generate(
            token_ids.to(model.device),
            min_new_tokens=10,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=500,
            num_return_sequences=n,
            #pad_token_id=eos_token_id,
        )
    input_length = len(token_ids[0])
    sentences = tokenizer.batch_decode(
        [tokens[input_length:] for tokens in output_ids.tolist()]
    )
    # 新しいシーケンスリストの作成
    # 改行で終わらせるリスト
    new_sentences = []

    for i, sequence in enumerate(sentences):
        # 改行文字でシーケンスを分割し、最初の部分のみを保持
        truncated_sequence = sequence.split('\n', 1)[0]
        new_sentences.append(truncated_sequence)

    return new_sentences

def normalize(scores):
    # logを戻す
    scores = np.exp(scores)

        # NaNをチェック
    if np.isnan(scores).any():
        print("Warning: NaN detected in scores, replacing NaN with 0.")
        scores = np.nan_to_num(scores, nan=0.0)

    # 正規化
    min_score = np.min(scores)
    max_score = np.max(scores)
    normalized_scores = (scores - min_score) / (max_score - min_score)

    # ステップ2: 正規化されたスコアの合計
    total_normalized_scores = np.sum(normalized_scores)

    # ステップ3: ご褒美の分配
    total_rewards = 10
    reward_distribution = (normalized_scores / total_normalized_scores) * total_rewards

    # 各人に割り当てるご褒美の数を整数にする
    integer_distribution = np.floor(reward_distribution)
    remaining_rewards = total_rewards - np.sum(integer_distribution)

    # 割り当てられる報酬が足りない場合は、大きい方から順に1個ずつ追加
    if remaining_rewards > 0:
        remainder_indices = np.argsort(reward_distribution - integer_distribution)[-int(remaining_rewards):]
        for i in remainder_indices:
            integer_distribution[i] += 1

    return_n = [int(x) for x in integer_distribution]

    return return_n

with open(dev_file_path) as f:
    # tqdmで進捗バーを表示するためのラップ
    # ファイルの行数を取得
    total_lines = sum(1 for _ in f)
    # ファイルを再度開いて処理を始める
    f.seek(0)
    for line in tqdm(f, total=total_lines, desc="Processing"):# line = question \t answer
        #質問と回答に分ける
        question = line.split("\t")[0]
        answer = line.split("\t")[1]
        paragraphs = line.split("\t")[2]
        answer = answer.split("\n")[0]

        #hop1
        hop1_knowledge = []
        #hop1_n = []

        #プロンプトにquestionを挿入
        with open("hop1.txt") as h1:
            prompt = h1.read().strip('\n')
        prompt = prompt.replace('{question}', question)
        prompt = prompt.replace('{paragraphs}', paragraphs)

        #10この知識1を生成
        hop1_knowledge = mk_knowledges(prompt, 10)

        #知識に対するスコア
        k1 = open(knowledges_1, 'a')
        qk_a_s = []
        for sentence in hop1_knowledge:
            qk_a = cloze_prob(question + sentence + "\t" + answer, model)
            qk_a_s.append(qk_a)

            k1.write(question + "\t" + answer + "\t" + sentence + "\t" + str(qk_a) + "\n")
        k1.write("\n")
        k1.close()

        hop1_n = normalize(qk_a_s)

        #hop2
        k1_k2 = []
        qk_a_s = []

        for knowledge1, n in zip(hop1_knowledge, hop1_n):
            if n == 0:
                continue
            #プロンプトにquestion,knowledge1を挿入
            with open("hop2.txt") as h2:
                prompt = h2.read().strip('\n')
            prompt = prompt.replace('{question}', question)
            prompt = prompt.replace('{knowledge1}', knowledge1)
            prompt = prompt.replace('{paragraphs}', paragraphs)
            hop2_knowledge = mk_knowledges(prompt,n)

            k2 = open(knowledges_2, 'a')
            
            for sentence in hop2_knowledge:
                qk_a = cloze_prob(question + sentence + "\t" + answer, model)
                qk_a_s.append(qk_a)
                k1_k2.append(knowledge1 + "\t" + sentence)

                k2.write(question + "\t" + answer + "\t" + knowledge1 + "\t" + sentence + "\t" + str(qk_a) + "\n")
            k2.write("\n")
            k2.close()
        hop2_n = normalize(qk_a_s)
        
        #hop3
        k1_k2_k3 = []
        qk_a_s = []
        for knowledge1_knowledge2, n in zip(k1_k2, hop2_n):
            if n == 0:
                continue
            #プロンプトにquestion,knowledge1,knowledge2を挿入
            with open("hop3.txt") as h3:
                prompt = h3.read().strip('\n')
            knowledge1 = knowledge1_knowledge2.split("\t")[0]
            knowledge2 = knowledge1_knowledge2.split("\t")[1]
            prompt = prompt.replace('{question}', question)
            prompt = prompt.replace('{knowledge1}', knowledge1)
            prompt = prompt.replace('{knowledge2}', knowledge2)
            prompt = prompt.replace('{paragraphs}', paragraphs)
            hop3_knowledge = mk_knowledges(prompt,n)

            k3 = open(knowledges_3, 'a')
            
            for sentence in hop3_knowledge:
                qk_a = cloze_prob(question + sentence + "\t" + answer, model)
                qk_a_s.append(qk_a)
                k1_k2_k3.append(knowledge1 + "\t" + knowledge2 + "\t" + sentence)

                k3.write(question + "\t" + answer + "\t" + knowledge1 + "\t" + knowledge2 + "\t" + sentence + "\t" + str(qk_a) + "\n")
            k3.write("\n")
            k3.close()

        # 最もスコアのいいものを評価
        best_score = float('-inf')
        best_sentence = None
        k = open(knowledge_path,'a')

        for k1_k2_k3, score in zip(k1_k2_k3, qk_a_s):
            if score > best_score:
                best_score = score
                best_sentence = k1_k2_k3
        k.write(question + "\t" + answer + "\t" + best_sentence + "\t" + str(best_score) + "\n")
