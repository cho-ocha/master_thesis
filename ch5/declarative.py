import csv
from openai import OpenAI

# OpenAI API clientを設定
client = OpenAI()

# TSVファイルの読み込みとAPIの応答を追加する処理
#input_tsv = '3hop_1_eval.tsv'  # 入力TSVファイルのパス
#output_tsv = '3hop_1_declarative.tsv'  # 出力TSVファイルのパス
input_tsv = '/home/cho/particle/train_3hop_1_rm.tsv'  # 入力TSVファイルのパス
output_tsv = '/home/cho/particle/train_3hop_1_rm_declarative.tsv'  # 出力TSVファイルのパス

with open(input_tsv, 'r', encoding='utf-8') as infile, open(output_tsv, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter='\t')

    for row in reader:
        # 3, 4, 5列目をOpenAI APIに送信
        responses = []
        for col in row[2:5]:  # 3, 4, 5列目を対象
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Rewrite the sentences I gave you in declarative sentence."
                    },
                    {
                        "role": "user",
                        "content": col
                    }
                ],
                temperature=0.7,
                max_tokens=64,
                top_p=1
            )
            print(response.choices[0].message.content)
            responses.append(response.choices[0].message.content)  # 応答を格納

        # 元の行にAPI応答を新しい3列として追加
        new_row = row[0:2] + responses
        writer.writerow(new_row)
