import pandas as pd
import os

def csv_to_latex_table(csv_path: str, output_tex_path: str = None):
    """
    CSVファイルを読み込み、最高値を太字にしたLaTeXの表コードを生成して保存します。
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: ファイルが見つかりません: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        
        # 数値列（モデルの列）を特定 (1列目は 'class' なので除外)
        model_cols = df.columns[1:]
        
        # --- LaTeXコードの構築開始 ---
        lines = []
        
        # 1. 表の定義 (左揃え + モデル数分の中央揃え)
        # 例: \begin{tabular}{l c c c}
        col_format = "l" + " c" * len(model_cols)
        lines.append(f"\\begin{{tabular}}{{{col_format}}}")
        lines.append("\\toprule")
        
        # 2. ヘッダー行
        # カラム名のアンダースコアをエスケープ (例: CMT_scale -> CMT\_scale)
        header_names = [col.replace('_', '\\_') for col in df.columns]
        lines.append(" & ".join(header_names) + " \\\\")
        lines.append("\\midrule")
        
        # 3. データ行の処理
        for _, row in df.iterrows():
            class_name = str(row['class']).replace('_', '\\_')
            
            # その行の数値データを取得
            numeric_values = row[model_cols].values
            # 行内の最大値を探す（太字にするため）
            row_max = max(numeric_values)
            
            row_str_parts = [class_name]
            
            for col in model_cols:
                val = row[col]
                # 小数点3桁でフォーマット
                val_str = f"{val:.3f}"
                
                # 最大値なら太字にする
                if val == row_max:
                    val_str = f"\\textbf{{{val_str}}}"
                
                row_str_parts.append(val_str)
            
            # 行を結合
            lines.append(" & ".join(row_str_parts) + " \\\\")
            
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        
        # --- 保存処理 ---
        latex_code = "\n".join(lines)
        
        # 出力パスが指定されていない場合は、元のファイル名の拡張子を .tex に変える
        if output_tex_path is None:
            output_tex_path = os.path.splitext(csv_path)[0] + ".tex"
            
        with open(output_tex_path, "w", encoding='utf-8') as f:
            f.write(latex_code)
            
        print(f"✅ LaTeX表を保存しました: {output_tex_path}")
        print("💡 以下のコードが生成されました:\n")
        print(latex_code)
        print("-" * 40)

    except Exception as e:
        print(f"❌ Error: LaTeX変換中にエラーが発生しました: {e}")

# --- 実行部分 ---
if __name__ == "__main__":
    # 変換したいCSVファイルのリスト
    target_csvs = [
        'comparison_maestro_mpauc.csv',
        'comparison_psds_results.csv'
    ]

    for csv_file in target_csvs:
        csv_to_latex_table(csv_file)