import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

def load_tsv(file_path, num_columns):
    data = pd.read_csv(file_path, sep='\t', header=None)
    if data.shape[1] != num_columns:
        raise ValueError(f"ファイル {file_path} は {num_columns} 列ではありません。")
    return data

def calculate_bleu(reference, candidate):
    # BLEUスコアを計算（デフォルトのパラメータを使用）
    return sentence_bleu([str(reference).split()], str(candidate).split())

def calculate_and_save_bleu_scores(file1_path, file2_path, output_file3, output_file4, output_file5):
    # TSVファイルを読み込み
    #file1_data = load_tsv(file1_path, 6)  # 6列のファイル
    file1_data = load_tsv(file1_path, 5)  # 5列のファイル
    file2_data = load_tsv(file2_path, 5)  # 5列のファイル

    # 対応する列を抽出
    file1_col3 = file1_data[2].tolist()  # 6列ファイルの3列目
    file1_col4 = file1_data[3].tolist()  # 6列ファイルの4列目
    file1_col5 = file1_data[4].tolist()  # 6列ファイルの5列目

    file2_col3 = file2_data[2].tolist()  # 5列ファイルの3列目
    file2_col4 = file2_data[3].tolist()  # 5列ファイルの4列目
    file2_col5 = file2_data[4].tolist()  # 5列ファイルの5列目

    bleu_scores_col3 = []
    bleu_scores_col4 = []
    bleu_scores_col5 = []

    # 3列目のBLEUスコアを計算して保存
    with open(output_file3, 'w', encoding='utf-8') as f3:
        for ref, cand in zip(file1_col3, file2_col3):
            bleu_score = calculate_bleu(ref, cand)
            bleu_scores_col3.append(bleu_score)
            f3.write(f"{ref}\t{cand}\t{bleu_score:.4f}\n")
    
    # 4列目のBLEUスコアを計算して保存
    with open(output_file4, 'w', encoding='utf-8') as f4:
        for ref, cand in zip(file1_col4, file2_col4):
            bleu_score = calculate_bleu(ref, cand)
            bleu_scores_col4.append(bleu_score)
            f4.write(f"{ref}\t{cand}\t{bleu_score:.4f}\n")

    # 5列目のBLEUスコアを計算して保存
    with open(output_file5, 'w', encoding='utf-8') as f5:
        for ref, cand in zip(file1_col5, file2_col5):
            bleu_score = calculate_bleu(ref, cand)
            bleu_scores_col5.append(bleu_score)
            f5.write(f"{ref}\t{cand}\t{bleu_score:.4f}\n")

    # BLEUスコアの平均を計算
    avg_bleu_col3 = sum(bleu_scores_col3) / len(bleu_scores_col3)
    avg_bleu_col4 = sum(bleu_scores_col4) / len(bleu_scores_col4)
    avg_bleu_col5 = sum(bleu_scores_col5) / len(bleu_scores_col5)

    # 平均を出力
    print(f"3列目のBLEUスコアの平均: {avg_bleu_col3:.4f}")
    print(f"4列目のBLEUスコアの平均: {avg_bleu_col4:.4f}")
    print(f"5列目のBLEUスコアの平均: {avg_bleu_col5:.4f}")

# ファイルパスを指定して実行
#file1_path = '/home/cho/particle/data/blank_rm_best.tsv'  # 6列のTSVファイル
#file1_path = '/home/cho/particle/sonota/prompts_para/data_para/q_a_k1_k2_k3.tsv'  # 6列のTSVファイル
file1_path = '/home/cho/particle/ramdom_para/data_random/q_a_k1_k2_k3.tsv'  # 5列のTSVファイル
#file1_path = '/home/cho/particle/data_temprature09/q_a_k1_k2_k3.tsv'  # 6列のTSVファイル
file2_path = '/home/cho/particle/eval/3hop_1_eval.tsv'  # 5列のTSVファイル
#file2_path = '/home/cho/particle/eval/3hop_1_declarative.tsv'  # 5列のTSVファイル

# 保存するファイル名
output_file3 = 'ramdom_para/bleu_scores_col3.tsv'
output_file4 = 'ramdom_para/bleu_scores_col4.tsv'
output_file5 = 'ramdom_para/bleu_scores_col5.tsv'


calculate_and_save_bleu_scores(file1_path, file2_path, output_file3, output_file4, output_file5)

print("BLEUスコアがそれぞれのファイルに保存されました。")
