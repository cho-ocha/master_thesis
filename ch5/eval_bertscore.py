from bert_score import score
import pandas as pd

def load_tsv(file_path, num_columns):
    data = pd.read_csv(file_path, sep='\t', header=None)
    if data.shape[1] != num_columns:
        raise ValueError(f"ファイル {file_path} は {num_columns} 列ではありません。")
    return data

def clean_column(column):
    # NaN値を空文字列に置き換える
    return column.fillna('').astype(str).tolist()

def calculate_and_save_bert_scores(file1_path, file2_path, output_file3, output_file4, output_file5):
    # TSVファイルを読み込み
    #file1_data = load_tsv(file1_path, 6)  # 6列のファイル
    file1_data = load_tsv(file1_path, 5)  # 6列のファイル
    file2_data = load_tsv(file2_path, 5)  # 5列のファイル

    # 対応する列を抽出し、NaN値を処理
    file1_col3 = clean_column(file1_data[2])  # 6列ファイルの3列目
    file1_col4 = clean_column(file1_data[3])  # 6列ファイルの4列目
    file1_col5 = clean_column(file1_data[4])  # 6列ファイルの5列目

    file2_col3 = clean_column(file2_data[2])  # 5列ファイルの3列目
    file2_col4 = clean_column(file2_data[3])  # 5列ファイルの4列目
    file2_col5 = clean_column(file2_data[4])  # 5列ファイルの5列目

    # 各列のBERTスコアを計算
    precision_3, recall_3, f1_3 = score(file1_col3, file2_col3, rescale_with_baseline=True, lang="en", verbose=True)
    precision_4, recall_4, f1_4 = score(file1_col4, file2_col4, rescale_with_baseline=True, lang="en", verbose=True)
    precision_5, recall_5, f1_5 = score(file1_col5, file2_col5, rescale_with_baseline=True, lang="en", verbose=True)

    # 3列目の結果を保存
    with open(output_file3, 'w', encoding='utf-8') as f3:
        for i in range(len(file1_col3)):
            f3.write(f"{file1_col3[i]}\t{file2_col3[i]}\t{precision_3[i].item():.4f}\t{recall_3[i].item():.4f}\t{f1_3[i].item():.4f}\n")
    
    # 4列目の結果を保存
    with open(output_file4, 'w', encoding='utf-8') as f4:
        for i in range(len(file1_col4)):
            f4.write(f"{file1_col4[i]}\t{file2_col4[i]}\t{precision_4[i].item():.4f}\t{recall_4[i].item():.4f}\t{f1_4[i].item():.4f}\n")

    # 5列目の結果を保存
    with open(output_file5, 'w', encoding='utf-8') as f5:
        for i in range(len(file1_col5)):
            f5.write(f"{file1_col5[i]}\t{file2_col5[i]}\t{precision_5[i].item():.4f}\t{recall_5[i].item():.4f}\t{f1_5[i].item():.4f}\n")

    return {
        "Column 3 BERT Scores": {
            "Precision": precision_3.mean().item(),
            "Recall": recall_3.mean().item(),
            "F1 Score": f1_3.mean().item()
        },
        "Column 4 BERT Scores": {
            "Precision": precision_4.mean().item(),
            "Recall": recall_4.mean().item(),
            "F1 Score": f1_4.mean().item()
        },
        "Column 5 BERT Scores": {
            "Precision": precision_5.mean().item(),
            "Recall": recall_5.mean().item(),
            "F1 Score": f1_5.mean().item()
        }
    }

# ファイルパスを指定して実行
#file1_path = '/home/cho/particle/sonota/prompts_para/data_para/q_a_k1_k2_k3.tsv'  # 6列のTSVファイル
file1_path = '/home/cho/particle/finetune/q_a_k1_k2_k3.tsv'  # 5列のTSVファイル
#file1_path = '/home/cho/particle/data/blank_rm_best.tsv'  # 6列のTSVファイル
file2_path = '/home/cho/particle/eval/3hop_1_declarative.tsv'  # 5列のTSVファイル
#file2_path = '/home/cho/particle/eval/3hop_1_eval.tsv'  # 5列のTSVファイル

# 保存するファイル名
output_file3 = '/home/cho/particle/eval/bertscore/rescale/finetune/bert_scores_col3.tsv'
output_file4 = '/home/cho/particle/eval/bertscore/rescale/finetune/bert_scores_col4.tsv'
output_file5 = '/home/cho/particle/eval/bertscore/rescale/finetune/bert_scores_col5.tsv'

bert_scores = calculate_and_save_bert_scores(file1_path, file2_path, output_file3, output_file4, output_file5)

# 結果を表示
for column, scores in bert_scores.items():
    print(f"{column}:")
    print(f"  Precision: {scores['Precision']:.4f}")
    print(f"  Recall: {scores['Recall']:.4f}")
    print(f"  F1 Score: {scores['F1 Score']:.4f}")

print("BERTスコアがそれぞれのファイルに保存されました。")
