import pandas as pd
import os
import glob
from typing import List

# 設定: CSVファイルがあるディレクトリ
SCORES_DIR = 'scores'

def process_and_pivot_data(file_paths: List[str], metric_name: str) -> pd.DataFrame:
    """
    指定されたファイルリストを読み込み、以下の処理を行います。
    1. 縦持ち形式で結合 (model, class, score)
    2. 横持ち形式に変換 (行: class, 列: model, 値: score)
    """
    all_data = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        
        # --- モデル名の抽出ロジック ---
        # ファイル名: "CMT-scale_per_class_..." -> "_" で分割 -> 先頭の "CMT-scale" を取得
        model_name = file_name.split('_')[0] if '_' in file_name else os.path.splitext(file_name)[0]

        try:
            df = pd.read_csv(file_path)

            # スコア列の判定 (mpauc または psds)
            target_col = None
            if 'mpauc' in df.columns:
                target_col = 'mpauc'
            elif 'psds' in df.columns:
                target_col = 'psds'
            
            # 必要な列がない場合はスキップ
            if 'class' not in df.columns or target_col is None:
                print(f"⚠️ Warning: '{file_name}' に必要な列('class' と 'mpauc'/'psds')がありません。スキップします。")
                continue

            # 必要なデータのみ抽出し、モデル名を追加
            df_subset = df[['class', target_col]].copy()
            df_subset.rename(columns={target_col: 'score'}, inplace=True) # 一時的に 'score' に統一
            df_subset['model'] = model_name
            
            all_data.append(df_subset)

        except Exception as e:
            print(f"❌ Error: '{file_name}' の読み込み中にエラーが発生しました: {e}")

    if not all_data:
        return pd.DataFrame()

    # 1. 全データを結合 (縦持ち)
    df_long = pd.concat(all_data, ignore_index=True)

    # 2. ピボットテーブルを作成 (横持ちに変換)
    # index=クラス, columns=モデル, values=スコア
    df_pivot = df_long.pivot_table(index='class', columns='model', values='score')
    
    # クラス名(index)を列に戻す
    df_pivot.reset_index(inplace=True)
    
    return df_pivot

def main():
    if not os.path.exists(SCORES_DIR):
        print(f"❌ Error: ディレクトリ '{SCORES_DIR}' が見つかりません。")
        return

    # 全CSVファイルを取得
    all_files = glob.glob(os.path.join(SCORES_DIR, '*.csv'))
    
    # ファイル名でグループ分け
    maestro_files = [f for f in all_files if 'maestro' in os.path.basename(f).lower()]
    psds_files = [f for f in all_files if 'psds' in os.path.basename(f).lower()]

    print(f"検出: Maestro系 {len(maestro_files)} ファイル, PSDS系 {len(psds_files)} ファイル")

    # --- 1. Maestro (mpauc) の処理 ---
    if maestro_files:
        print("\n--- Processing Maestro (mpauc) ---")
        df_maestro = process_and_pivot_data(maestro_files, 'mpauc')
        
        if not df_maestro.empty:
            output_file = 'comparison_maestro_mpauc.csv'
            # float_format='%.3f' で小数点以下3桁に固定して保存
            df_maestro.to_csv(output_file, index=False, float_format='%.3f')
            print(f"✅ 保存完了: {output_file}")
            print(df_maestro.head())
        else:
            print("⚠️ Maestroの有効なデータがありませんでした。")

    # --- 2. PSDS の処理 ---
    if psds_files:
        print("\n--- Processing PSDS ---")
        df_psds = process_and_pivot_data(psds_files, 'psds')
        
        if not df_psds.empty:
            output_file = 'comparison_psds_results.csv'
            # float_format='%.3f' で小数点以下3桁に固定して保存
            df_psds.to_csv(output_file, index=False, float_format='%.3f')
            print(f"✅ 保存完了: {output_file}")
            print(df_psds.head())
        else:
            print("⚠️ PSDSの有効なデータがありませんでした。")

if __name__ == "__main__":
    main()