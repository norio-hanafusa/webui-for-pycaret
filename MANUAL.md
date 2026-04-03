# WebUI for PyCaret User Manual / WebUI for PyCaret 取扱説明書

---

# 日本語マニュアル

---

## 1. 概要

WebUI for PyCaret は、PyCaret ライブラリを基盤としたノーコード機械学習プラットフォームです。Streamlit ベースの Web インターフェースにより、ブラウザ上でデータの読み込み、前処理、モデル学習、ハイパーパラメータチューニング、予測、結果の可視化、SHAP/LIME 解析までの一連の機械学習ワークフローをコードを書くことなく実行できます。

Docker コンテナとして動作し、`pycaret/full` イメージをベースに構築されているため、PyCaret が対応する全てのアルゴリズムと機能を利用できます。

### 主な機能

- CSV / Excel ファイルのアップロードとデータ編集
- 分類、回帰、クラスタリング、異常検知の 4 つのタスクに対応
- モデル比較（プログレスバー付き）
- ハイパーパラメータチューニング（ベースライン vs チューニング後の比較表付き）
- SHAP 解析（11 種類のプロット）
- LIME 解析（特徴量貢献度の可視化）
- 新規データに対する予測
- 学習済みモデルの保存・読み込み・管理
- 日本語 / 英語の多言語対応

---

## 2. 動作環境

### 必須要件

- Docker および Docker Compose がインストールされていること
- 推奨メモリ: 8GB 以上
- 推奨ディスク空き容量: 10GB 以上（Docker イメージのサイズが大きいため）

### ベースイメージ

本アプリケーションは `pycaret/full:latest` Docker イメージをベースとしています。このイメージには PyCaret とその全ての依存パッケージが含まれています。

### 追加パッケージ

以下のパッケージが Dockerfile により追加インストールされます。

| パッケージ | バージョン | 用途 |
|---|---|---|
| streamlit | >= 1.30.0 | Web UI フレームワーク |
| streamlit-aggrid | >= 0.3.4 | 高機能データエディタ |
| openpyxl | >= 3.1.0 | Excel ファイル読み込み |
| xlsxwriter | >= 3.1.0 | Excel ファイル書き出し |
| shap | >= 0.43.0 | SHAP 解析 |
| lime | >= 0.2.0.1 | LIME 解析 |
| matplotlib | >= 3.7.0 | グラフ描画 |

---

## 3. インストールと起動

### 手順

1. プロジェクトのディレクトリに移動します。

2. Docker Compose でコンテナをビルド・起動します。

```bash
docker compose up --build -d
```

3. ブラウザで以下の URL にアクセスします。

```
http://localhost:8501
```

### 停止

```bash
docker compose down
```

### ディレクトリ構成

```
webui-for-pycaret/
  app.py              -- メインアプリケーション
  i18n.py             -- 多言語対応 (日本語/英語)
  requirements.txt    -- 追加 Python パッケージ
  Dockerfile          -- コンテナビルド定義
  docker-compose.yml  -- Docker Compose 設定
  data/               -- データファイル用ボリュームマウント先
  models/             -- 保存済みモデル用ボリュームマウント先
  exports/            -- エクスポートファイル用ボリュームマウント先
```

---

## 4. ページ別操作ガイド

WebUI for PyCaret は画面上部の水平タブで各ページに切り替えます。全 7 ページで構成されています。

画面右上の「言語 / Language」セレクタで日本語と英語を切り替えられます。

### 4.1 データ読み込み

データ読み込みページでは、分析に使用するデータファイルをアップロードします。

**対応ファイル形式:**
- CSV (.csv)
- Excel (.xls, .xlsx)

**操作手順:**

1. 「ファイルを選択」ボタンをクリックしてファイルを選択します。
2. ファイルが読み込まれると、行数と列数が表示されます。
3. 読み込まれたデータは以下の 3 つのタブで確認できます。
   - **プレビュー**: 先頭 100 行のデータ表示
   - **基本統計量**: 全列の記述統計量（平均、標準偏差、最小値、最大値、四分位数等）
   - **データ型**: 各列のデータ型、非 Null 数、Null 数、ユニーク数

**注意:** ファイルの最大アップロードサイズは 500MB です（Docker 環境変数 STREAMLIT_SERVER_MAX_UPLOAD_SIZE で設定）。

### 4.2 データ表示・編集

データの閲覧と編集を行うページです。

**データ編集機能:**

- セルをダブルクリックして値を直接編集できます。
- AgGrid がインストールされている場合は高機能エディタが使用されます（ページネーション、フィルタリング、ソート、複数行選択に対応）。
- AgGrid が利用できない場合は Streamlit 標準のデータエディタが使用されます。

**操作ボタン:**

- **編集を確定**: 編集した内容を確定し、以降の機械学習で使用するデータとして反映します。
- **元に戻す**: 編集をキャンセルし、確定済みのデータに戻します。

**列操作:**

- **列の削除**: 削除する列を複数選択して一括削除できます。
- **欠損値の補完**: 指定した列の欠損値（NA/NaN）を以下の方法で補完できます。
  - 平均値
  - 中央値
  - 最頻値
  - 0

### 4.3 機械学習

機械学習の実行ページです。以下の 4 つのセクションで構成されています。

#### 4.3.1 タスク選択

以下の 4 つのタスクから選択します。

| タスク | 説明 |
|---|---|
| 分類 (Classification) | カテゴリの予測 |
| 回帰 (Regression) | 数値の予測 |
| クラスタリング (Clustering) | データのグループ分け（教師なし） |
| 異常検知 (Anomaly Detection) | 異常値の検出（教師なし） |

分類・回帰（教師あり学習）の場合は、ターゲット列（予測対象の列）を選択する必要があります。

#### 4.3.2 Setup（前処理）

PyCaret の Setup を実行し、データの前処理を行います。

**基本設定:**
- **ランダムシード**: 再現性のためのシード値（デフォルト: 42）

**前処理オプション（詳細設定）:**

前処理オプションは展開可能なセクション内に配置されています。

- **不均衡データ補正 (SMOTE)** -- 分類タスクのみ
  - 少数クラスをオーバーサンプリングして、クラス間のバランスを改善します。
  - 選択可能な手法: デフォルト (SMOTE), BorderlineSMOTE, SVMSMOTE, ADASYN, RandomOverSampler
  - クラス分布の表示: 各クラスの件数と割合が自動表示されます。
  - 不均衡比率が 3:1 を超える場合、SMOTE の使用が推奨されます。

- **正規化 (Normalize)**
  - 特徴量のスケーリングを行います。
  - 手法: zscore, minmax, maxabs, robust

- **外れ値除去**
  - 外れ値を自動的に除去します。
  - 閾値: 0.01 から 0.10 の範囲で指定（デフォルト: 0.05）

- **学習データ割合** -- 教師あり学習のみ
  - 学習用とテスト用のデータ分割比率（デフォルト: 0.7 = 70%）

「Setup 実行」ボタンを押すと前処理が開始されます。

#### 4.3.3 モデル比較

Setup 完了後に利用可能になります。

**教師あり学習（分類・回帰）の場合:**

- 比較上位モデル数を指定します（1 から 10）。
- 比較するモデルをリストから選択できます（空欄の場合は全モデルが対象）。
- 「モデル比較実行」ボタンで比較が開始されます。
- 各モデルは個別に `create_model` で学習され、プログレスバーで進捗が表示されます。
- 結果テーブルには各モデルの平均スコア（Mean 行のみ）が表示されます。
- 分類の場合は Accuracy、回帰の場合は R2 でソートされます。
- 最も成績の良いモデルが自動的にベストモデルとして保存されます。

**教師なし学習の場合:**

- クラスタリング: アルゴリズム（kmeans, ap, meanshift, sc, hclust, dbscan, optics, birch）とクラスタ数を指定してモデルを作成します。
- 異常検知: アルゴリズム（iforest, knn, lof, svm, pca, mcd, sod, histogram）と異常の割合を指定してモデルを作成します。

#### 4.3.4 ハイパーパラメータチューニング

教師あり学習でベストモデルが選択された後に利用可能です。

**設定項目:**

- **最適化指標**: チューニングで最大化（または最小化）する指標
  - 分類: Accuracy, AUC, Recall, Precision, F1, Kappa, MCC
  - 回帰: MAE, MSE, RMSE, R2, RMSLE, MAPE
- **探索回数 (n_iter)**: ハイパーパラメータの探索回数（5 から 100）
- **探索アルゴリズム**: scikit-learn, optuna, scikit-optimize

**実行フロー:**

1. ベースラインスコアの取得: チューニング前のモデルの交差検証スコアを取得
2. ハイパーパラメータ探索: 指定された探索回数でチューニングを実行
3. 結果表示:
   - **ベースライン vs チューニング後の比較表**: 全指標のベースラインスコア、チューニング後スコア、差分を表示
   - **交差検証結果（全 fold）**: チューニング後モデルの詳細な交差検証結果（展開可能）
   - **チューニング後のパラメータ**: 最適化されたハイパーパラメータの一覧（展開可能）

#### 4.3.5 学習データでの予測

教師あり学習でモデルが作成された後に利用可能です。

- チューニング済みモデルがある場合はそれを使用し、ない場合はベストモデルを使用します。
- 「予測実行」ボタンで学習データに対する予測を実行します。
- 予測結果の先頭 50 行が表示されます。

### 4.4 結果・可視化・SHAP/LIME

モデルの学習結果を様々な方法で可視化するページです。

#### 4.4.1 PyCaret 可視化

PyCaret の組み込みプロット機能を使用します。タスクの種類に応じて異なるプロットが選択可能です。

**分類タスクのプロット:**
- AUC 曲線
- 混同行列
- 特徴量重要度
- 学習曲線
- Precision-Recall
- 分類レポート
- 境界線

**回帰タスクのプロット:**
- 残差プロット
- 予測誤差
- 特徴量重要度
- 学習曲線
- Cook's Distance

**クラスタリングのプロット:**
- クラスタ分布
- エルボー法
- シルエット
- 分布

**異常検知のプロット:**
- t-SNE
- UMAP

#### 4.4.2 SHAP 解析

教師あり学習のモデルに対して、SHAP (SHapley Additive exPlanations) による特徴量の貢献度解析を実行します。

**Step 1: SHAP 値の一括計算（キャッシュ方式）**

SHAP 値は一括で計算し、セッションステートにキャッシュします。一度計算すれば、プロットの種類を切り替えても再計算は不要です。

- **サンプル数スライダー**: SHAP 計算に使用するサンプル数を 50 から訓練データ実数（X_train のサイズ）まで選択できます。
  - TreeExplainer: 全件でも高速に計算可能
  - KernelExplainer: 200 以下を推奨（バッチ処理でプログレスバー表示）
- **「SHAP 値を計算（一括）」ボタン**: SHAP 値を一括計算してキャッシュに保存します。
- **「SHAP キャッシュをリセット」ボタン**: キャッシュをクリアして再計算可能にします（サンプル数変更時などに使用）。

**SHAP 値の計算順序（Explainer 優先順位）:**
1. TreeExplainer（ツリーベースモデル向け、高速、全データ対応）
2. KernelExplainer（モデルに依存しない汎用手法、バッチ処理でプログレスバー表示）
3. PermutationExplainer（最後の手段として使用）

計算完了後、使用した Explainer 名・計算時間・サンプル数が表示されます。

**Step 2: キャッシュ済み SHAP 値を使ったプロット生成**

キャッシュされた SHAP 値を使用して、以下の 11 種類のプロットを生成します。プロットの切り替えやパラメータ変更のたびに SHAP 値を再計算する必要はありません。

**利用可能なプロットタイプ（全 11 種類）:**

| プロット | 説明 |
|---|---|
| Summary Plot (Bar) | 特徴量の重要度を棒グラフで表示 |
| Summary Plot (Dot) | 特徴量ごとの SHAP 値分布をドットプロットで表示 |
| Waterfall Plot | 個別サンプルの予測に対する各特徴量の貢献度を段階的に表示 |
| Beeswarm Plot | 全サンプルの SHAP 値をバイオリン風に表示 |
| Bar Plot (shap.plots.bar) | shap.plots.bar による棒グラフ表示 |
| Scatter Plot | 特定の特徴量の値と SHAP 値の関係を散布図で表示 |
| Dependence Plot | 特徴量間の依存関係を SHAP 値で可視化 |
| Force Plot | 個別またはデータセット全体の予測力を矢印で可視化 |
| Decision Plot | 各特徴量が予測にどう影響するかを決定パスとして表示 |
| Violin Plot | 特徴量ごとの SHAP 値分布をバイオリンプロットで表示 |
| Heatmap Plot | SHAP 値のヒートマップ表示 |

**パラメータ:**
- Waterfall Plot, Force Plot: 対象サンプルのインデックスを指定（Force Plot は -1 で全体表示）
- Scatter Plot: 注目する特徴量を選択（キャッシュ済み特徴量の一覧から選択可能。自動選択も可）

**プロット履歴:**
生成したプロットはセッションステートに履歴として保持され、タブを切り替えても結果が消えません。「SHAP 履歴をクリア」ボタンでプロット履歴を削除できます（SHAP 値のキャッシュは保持されます）。

#### 4.4.3 LIME 解析

LIME (Local Interpretable Model-agnostic Explanations) は、個々の予測を局所的に解釈可能なモデルで近似し、各特徴量の貢献度を可視化します。

**パラメータ:**
- **解析対象サンプルのインデックス**: 解析する個別サンプルを指定
- **表示する特徴量の数**: 結果に表示する上位特徴量の数（5 から 30、デフォルト: 10）
- **LIME 近傍サンプル数**: 局所近似に使用するサンプル数（100 から 10000、デフォルト: 1000）。多いほど結果は安定しますが計算時間が増加します。

**出力内容:**
- 特徴量貢献度の棒グラフ
- 特徴量の貢献度テーブル（特徴量条件、貢献度、方向、絶対値でソート）
- 予測確率（分類タスクの場合）
- ローカルモデルの切片 (Intercept)
- R2 スコア（局所近似の信頼性指標。1.0 に近いほど信頼性が高い）

**キャッシュ機能:**
LIME の解析結果も SHAP と同様にセッションステートにキャッシュされます。「LIME 履歴をクリア」ボタンで全履歴を削除できます。

#### 4.4.4 予測結果

学習データに対する予測結果がある場合、このセクションに先頭 100 行が表示されます。

### 4.5 新規データ予測

学習済みモデルを使用して、新しいデータに対する予測を行うページです。

**前提条件:**
- 機械学習ページでモデルを作成済みであること、またはモデル読み込みページで保存済みモデルをロード済みであること。

**操作手順:**

1. 新規データファイル（CSV または Excel）をアップロードします。
2. アップロードされたデータのプレビューとサイズが表示されます。
3. 「予測実行」ボタンで予測を実行します。
4. 予測結果は CSV または Excel でダウンロードできます。

**予測の動作:**
- PyCaret の Setup が完了している場合は `predict_model` が使用されます。
- それ以外の場合はモデルの `predict` メソッドが直接呼び出されます。
- `predict_proba` が利用可能な場合は予測確率も出力に含まれます。

### 4.6 モデル読み込み

以前に保存した学習済みモデルを読み込むページです。

**読み込み方法:**

1. **ファイルをアップロード (.pkl)**: ローカルマシンから pickle ファイルをアップロード
2. **サーバー上のファイルを指定**: コンテナ内の `/app/models/` ディレクトリに保存されたモデルから選択、またはフルパスを直接入力

**操作手順:**

1. 読み込み方法を選択します。
2. このモデルのタスク種別（分類、回帰、クラスタリング、異常検知）を指定します。
3. ファイルを選択またはパスを入力します。
4. 「モデルを読み込む」ボタンで読み込みを実行します。

**読み込み済みモデル情報:**

読み込みが成功すると、以下の情報が表示されます。
- ファイル名
- タスク種別
- モデル型
- パイプラインステップ（Pipeline オブジェクトの場合）
- パラメータ一覧（get_params 対応モデルの場合）

読み込んだモデルは「新規データ予測」ページで使用できます。

### 4.7 保存・エクスポート

データ、予測結果、モデルの保存とエクスポートを行うページです。

#### データのエクスポート

現在のデータ（編集後を含む）を以下の形式でダウンロードできます。
- CSV
- Excel

#### 予測結果のエクスポート

学習データに対する予測結果を以下の形式でダウンロードできます。
- CSV
- Excel

#### 学習済みモデルの保存

- モデルファイル名を指定して、コンテナ内の `/app/models/` に保存します。
- 「ベストモデルを保存」: モデル比較で選ばれたベストモデルを保存
- 「チューニング済みモデルを保存」: ハイパーパラメータチューニング後のモデルを保存（ファイル名に `_tuned` が付加されます）

#### モデルのダウンロード

現在の学習済みモデル（チューニング済みがあればそれを優先）を pickle 形式でブラウザからダウンロードできます。

#### 保存済みモデルの管理

`/app/models/` 内のファイル一覧が表示されます。各ファイルに対して以下の操作が可能です。

- **名前を変更**: ファイル名を変更（拡張子は自動的に保持されます）
- **削除**: 確認ダイアログ付きでファイルを削除
- **ダウンロード**: ファイルをブラウザからダウンロード

---

## 5. 高度な機能

### 5.1 不均衡データ補正 (SMOTE)

分類タスクでクラス間のデータ数に偏りがある場合、SMOTE を使用して少数クラスのオーバーサンプリングを行えます。

**選択可能な手法:**
- デフォルト (SMOTE): 標準的な SMOTE アルゴリズム
- BorderlineSMOTE: 境界付近のサンプルに重点をおいたオーバーサンプリング
- SVMSMOTE: SVM を利用した境界に基づくオーバーサンプリング
- ADASYN: 学習が困難なサンプル近傍に重点をおいたオーバーサンプリング
- RandomOverSampler: ランダムな複製によるオーバーサンプリング

不均衡比率が 3:1 を超える場合、アプリケーションが自動的に SMOTE の使用を推奨します。

### 5.2 正規化

特徴量のスケーリングに対応しています。

- **zscore**: 標準化（平均 0、標準偏差 1）
- **minmax**: 最小-最大スケーリング（0 から 1）
- **maxabs**: 最大絶対値スケーリング
- **robust**: ロバストスケーリング（中央値とIQRを使用）

### 5.3 外れ値除去

外れ値を自動的に検出して除去します。閾値（0.01 から 0.10）で検出の厳しさを調整できます。値が小さいほど多くのデータが外れ値として除去されます。

### 5.4 Pickle モデルの読み込み

モデル読み込みページでは 2 つの方法で pickle 形式のモデルを読み込めます。

1. **PyCaret 形式**: PyCaret の `save_model` で保存されたモデル（パイプライン込み）。`load_model` で読み込まれます。
2. **直接 pickle**: 標準的な pickle ファイル。PyCaret 形式での読み込みに失敗した場合、`pickle.load` でフォールバック読み込みが行われます。

### 5.5 多言語対応

画面右上の言語セレクタで日本語 (ja) と英語 (en) を切り替えられます。言語を切り替えると画面が自動的にリロードされ、全ての UI テキストが選択した言語で表示されます。デフォルト言語は日本語です。

---

## 6. トラブルシューティング

### データの読み込みに失敗する

- ファイル形式が CSV または Excel (.xls, .xlsx) であることを確認してください。
- ファイルサイズが 500MB 以下であることを確認してください。
- Excel ファイルの場合、openpyxl が対応する形式であることを確認してください。

### Setup でエラーが発生する

- ターゲット列が正しく選択されているか確認してください。
- データに機械学習に適さない列（ID 列など）が含まれている場合は、データ表示・編集ページで事前に削除してください。
- エラーメッセージとスタックトレースが表示されるので、詳細を確認してください。

### モデル比較が非常に遅い

- 比較するモデルを選択して、対象を絞ることで高速化できます。
- データサイズが大きい場合、特に SVM 系のモデルは非常に時間がかかることがあります。

### SHAP 解析でエラーが発生する

- 一部のモデル（特にアンサンブルモデルやカスタムモデル）では SHAP 値の計算に失敗することがあります。
- TreeExplainer が使用できないモデルの場合、KernelExplainer や PermutationExplainer が自動的に試行されますが、計算時間が大幅に増加する場合があります。
- サンプル数はスライダーで調整可能です（50 から X_train 実数まで）。KernelExplainer 使用時はサンプル数を 200 以下に設定することを推奨します。
- 「SHAP キャッシュをリセット」ボタンでキャッシュをクリアしてから、サンプル数やモデルを変更して再計算してください。

### LIME 解析でエラーが発生する

- LIME はモデルの predict または predict_proba メソッドを必要とします。
- 近傍サンプル数を増やすと安定しますが、計算時間も増加します。

### モデルの保存に失敗する

- コンテナ内の `/app/models/` ディレクトリに書き込み権限があることを確認してください。
- docker-compose.yml でボリュームマウントが正しく設定されていることを確認してください。

### コンテナが起動しない

- Docker Desktop が起動していることを確認してください。
- ポート 8501 が他のプロセスで使用されていないことを確認してください。
- `docker compose logs` でエラーログを確認してください。

---

## 7. Docker 設定の詳細

### docker-compose.yml

```yaml
services:
  webui-for-pycaret:
    build: .
    container_name: webui-for-pycaret
    ports:
      - "127.0.0.1:8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./exports:/app/exports
    environment:
      - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
    restart: unless-stopped
```

ポートバインドに `127.0.0.1` を指定しているため、ローカルホスト (自分のPC) からのみアクセス可能です。LAN 上の他のマシンからはアクセスできません。

### ボリュームマウント

| ホスト側 | コンテナ内 | 用途 |
|---|---|---|
| ./data | /app/data | データファイルの永続化 |
| ./models | /app/models | 学習済みモデルの永続化 |
| ./exports | /app/exports | エクスポートファイルの永続化 |

ボリュームマウントにより、コンテナを再起動してもデータ、モデル、エクスポートファイルが保持されます。モデルの保存先 `/app/models/` はホスト側の `./models/` ディレクトリに対応しているため、ホストマシンから直接ファイルを確認・管理することも可能です。

### 環境変数

| 変数名 | 値 | 説明 |
|---|---|---|
| STREAMLIT_SERVER_MAX_UPLOAD_SIZE | 500 | アップロード可能なファイルの最大サイズ (MB) |

### Dockerfile

- ベースイメージ: `pycaret/full:latest`
- 作業ディレクトリ: `/app`
- 公開ポート: 8501 (127.0.0.1 のみ)
- 起動コマンド: `streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false`
- コンテナ内では 0.0.0.0 でリッスンしますが、docker-compose のポートバインドにより外部からのアクセスは遮断されます。

---
---

# English Manual

---

## 1. Overview

WebUI for PyCaret is a no-code machine learning platform built on top of the PyCaret library. Through a Streamlit-based web interface, users can execute the entire machine learning workflow -- from data loading, preprocessing, model training, hyperparameter tuning, prediction, result visualization, to SHAP/LIME analysis -- all without writing any code.

The application runs as a Docker container based on the `pycaret/full` image, providing access to all algorithms and features supported by PyCaret.

### Key Features

- CSV / Excel file upload with interactive data editing
- Four task types: Classification, Regression, Clustering, Anomaly Detection
- Model comparison with progress bars
- Hyperparameter tuning with baseline vs tuned comparison table
- SHAP analysis (11 plot types)
- LIME analysis (feature contribution visualization)
- Prediction on new data
- Model save / load / management
- Bilingual support (Japanese and English)

---

## 2. System Requirements

### Prerequisites

- Docker and Docker Compose installed
- Recommended RAM: 8GB or more
- Recommended disk space: 10GB or more (the Docker image is large)

### Base Image

This application is built on the `pycaret/full:latest` Docker image, which includes PyCaret and all its dependencies.

### Additional Packages

The following packages are installed via the Dockerfile:

| Package | Version | Purpose |
|---|---|---|
| streamlit | >= 1.30.0 | Web UI framework |
| streamlit-aggrid | >= 0.3.4 | Advanced data editor |
| openpyxl | >= 3.1.0 | Excel file reading |
| xlsxwriter | >= 3.1.0 | Excel file writing |
| shap | >= 0.43.0 | SHAP analysis |
| lime | >= 0.2.0.1 | LIME analysis |
| matplotlib | >= 3.7.0 | Chart rendering |

---

## 3. Installation and Startup

### Steps

1. Navigate to the project directory.

2. Build and start the container with Docker Compose:

```bash
docker compose up --build -d
```

3. Open your browser and go to:

```
http://localhost:8501
```

### Stopping

```bash
docker compose down
```

### Directory Structure

```
webui-for-pycaret/
  app.py              -- Main application
  i18n.py             -- Internationalization (Japanese/English)
  requirements.txt    -- Additional Python packages
  Dockerfile          -- Container build definition
  docker-compose.yml  -- Docker Compose configuration
  data/               -- Volume mount for data files
  models/             -- Volume mount for saved models
  exports/            -- Volume mount for exported files
```

---

## 4. User Guide by Page

WebUI for PyCaret uses horizontal tabs at the top of the screen for navigation. The application consists of 7 pages.

The language can be switched between Japanese and English using the selector in the upper right corner of the screen.

### 4.1 Data Loading

Upload data files for analysis on this page.

**Supported file formats:**
- CSV (.csv)
- Excel (.xls, .xlsx)

**Steps:**

1. Click the "Select file" button to choose a file.
2. Once loaded, the number of rows and columns is displayed.
3. The loaded data can be examined through three tabs:
   - **Preview**: First 100 rows of data
   - **Descriptive Statistics**: Summary statistics for all columns (mean, std, min, max, quartiles, etc.)
   - **Data Types**: Data type, non-null count, null count, and unique count for each column

**Note:** Maximum upload file size is 500MB (configured via the STREAMLIT_SERVER_MAX_UPLOAD_SIZE environment variable).

### 4.2 Data View / Edit

View and edit data on this page.

**Data Editing:**

- Double-click a cell to edit its value directly.
- When AgGrid is installed, an advanced editor is used (with pagination, filtering, sorting, and multi-row selection).
- When AgGrid is unavailable, the standard Streamlit data editor is used.

**Action Buttons:**

- **Apply Changes**: Confirm edits and update the data used for subsequent machine learning.
- **Undo**: Discard edits and revert to the last confirmed data.

**Column Operations:**

- **Drop Columns**: Select multiple columns for batch deletion.
- **Fill Missing Values**: Fill NA/NaN values in a selected column using:
  - Mean
  - Median
  - Mode
  - 0

### 4.3 Machine Learning

The main machine learning execution page, consisting of four sections.

#### 4.3.1 Task Selection

Choose from four task types:

| Task | Description |
|---|---|
| Classification | Predicting categories |
| Regression | Predicting numerical values |
| Clustering | Grouping data (unsupervised) |
| Anomaly Detection | Detecting anomalies (unsupervised) |

For Classification and Regression (supervised learning), you must select a target column.

#### 4.3.2 Setup (Preprocessing)

Executes PyCaret Setup for data preprocessing.

**Basic settings:**
- **Random seed**: Seed value for reproducibility (default: 42)

**Preprocessing options (Advanced):**

Located in an expandable section.

- **Imbalanced Data Correction (SMOTE)** -- Classification only
  - Oversamples the minority class to improve class balance.
  - Available methods: Default (SMOTE), BorderlineSMOTE, SVMSMOTE, ADASYN, RandomOverSampler
  - Class distribution display: Count and percentage for each class are shown automatically.
  - When the imbalance ratio exceeds 3:1, SMOTE is recommended.

- **Normalize**
  - Performs feature scaling.
  - Methods: zscore, minmax, maxabs, robust

- **Remove Outliers**
  - Automatically detects and removes outliers.
  - Threshold: 0.01 to 0.10 (default: 0.05)

- **Training Data Ratio** -- Supervised learning only
  - Train/test split ratio (default: 0.7 = 70%)

Click "Run Setup" to start preprocessing.

#### 4.3.3 Model Comparison

Available after Setup is complete.

**Supervised learning (Classification / Regression):**

- Specify the number of top models to select (1 to 10).
- Optionally select specific models to compare (empty = all models).
- Click "Run Model Comparison" to start.
- Each model is trained individually via `create_model`, with a progress bar showing progress.
- The results table displays the Mean row for each model (average cross-validation scores).
- Results are sorted by Accuracy (classification) or R2 (regression).
- The best-performing model is automatically saved as the best model.

**Unsupervised learning:**

- Clustering: Select an algorithm (kmeans, ap, meanshift, sc, hclust, dbscan, optics, birch) and number of clusters.
- Anomaly Detection: Select an algorithm (iforest, knn, lof, svm, pca, mcd, sod, histogram) and anomaly fraction.

#### 4.3.4 Hyperparameter Tuning

Available after a best model has been selected in supervised learning.

**Settings:**

- **Optimization Metric**: The metric to maximize (or minimize)
  - Classification: Accuracy, AUC, Recall, Precision, F1, Kappa, MCC
  - Regression: MAE, MSE, RMSE, R2, RMSLE, MAPE
- **Search iterations (n_iter)**: Number of hyperparameter search iterations (5 to 100)
- **Search Algorithm**: scikit-learn, optuna, scikit-optimize

**Execution flow:**

1. Baseline score retrieval: Cross-validation score of the model before tuning
2. Hyperparameter search: Tuning with the specified number of iterations
3. Results display:
   - **Baseline vs Tuned comparison table**: Baseline score, tuned score, and difference for all metrics
   - **CV results (all folds)**: Detailed cross-validation results for the tuned model (expandable)
   - **Tuned model parameters**: List of optimized hyperparameters (expandable)

#### 4.3.5 Prediction on Training Data

Available after a model has been created in supervised learning.

- If a tuned model exists, it is used; otherwise, the best model is used.
- Click "Run Prediction" to predict on the training data.
- The first 50 rows of predictions are displayed.

### 4.4 Results / Visualization / SHAP / LIME

A page for visualizing model results using various methods.

#### 4.4.1 PyCaret Visualization

Uses PyCaret built-in plotting functions. Available plots depend on the task type.

**Classification plots:**
- AUC Curve
- Confusion Matrix
- Feature Importance
- Learning Curve
- Precision-Recall
- Classification Report
- Decision Boundary

**Regression plots:**
- Residuals Plot
- Prediction Error
- Feature Importance
- Learning Curve
- Cook's Distance

**Clustering plots:**
- Cluster Distribution
- Elbow Method
- Silhouette
- Distribution

**Anomaly Detection plots:**
- t-SNE
- UMAP

#### 4.4.2 SHAP Analysis

Performs SHAP (SHapley Additive exPlanations) feature contribution analysis for supervised learning models.

**Step 1: Bulk SHAP Computation (Cached)**

SHAP values are computed once in bulk and cached in session state. Once computed, switching between plot types does not require recomputation.

- **Sample count slider**: Select the number of samples for SHAP computation from 50 up to the actual X_train size.
  - TreeExplainer: Handles full datasets efficiently
  - KernelExplainer: 200 or fewer samples recommended (batch processing with progress bar)
- **"Compute SHAP values (bulk)" button**: Computes SHAP values and stores them in cache.
- **"Reset SHAP cache" button**: Clears the cache to allow recomputation (e.g., after changing sample count).

**Explainer priority order:**
1. TreeExplainer (for tree-based models, fast, supports full data)
2. KernelExplainer (model-agnostic, batch processing with progress bar)
3. PermutationExplainer (used as a last resort)

After computation, the explainer name, computation time, and sample count are displayed.

**Step 2: Plot Generation from Cached SHAP Values**

Generate plots from cached SHAP values. No recomputation is needed when switching plot types or changing parameters.

**Available plot types (all 11):**

| Plot | Description |
|---|---|
| Summary Plot (Bar) | Feature importance as bar chart |
| Summary Plot (Dot) | SHAP value distribution per feature as dot plot |
| Waterfall Plot | Step-by-step contribution of each feature to an individual prediction |
| Beeswarm Plot | SHAP values for all samples in violin-style display |
| Bar Plot (shap.plots.bar) | Bar chart using shap.plots.bar |
| Scatter Plot | Scatter plot of feature values vs SHAP values |
| Dependence Plot | Visualize feature dependencies through SHAP values |
| Force Plot | Visualize prediction forces for individual samples or entire dataset |
| Decision Plot | Show how each feature affects predictions as decision paths |
| Violin Plot | SHAP value distribution per feature as violin plot |
| Heatmap Plot | Heatmap display of SHAP values |

**Parameters:**
- Waterfall Plot, Force Plot: Specify the target sample index (Force Plot uses -1 for all samples)
- Scatter Plot: Select the feature of interest from cached feature list (auto-selection also available)

**Plot history:**
Generated plots are retained as a history in session state and persist across tab switches. Click "Clear SHAP History" to delete plot history (SHAP value cache is preserved).

#### 4.4.3 LIME Analysis

LIME (Local Interpretable Model-agnostic Explanations) approximates individual predictions with a locally interpretable model and visualizes each feature's contribution.

**Parameters:**
- **Target sample index**: The individual sample to analyze
- **Number of features to display**: Top features shown in results (5 to 30, default: 10)
- **LIME neighborhood samples**: Number of samples for local approximation (100 to 10000, default: 1000). More samples yield stable results but increase computation time.

**Output:**
- Feature contribution bar chart
- Feature contribution table (feature condition, contribution, direction, sorted by absolute value)
- Prediction probabilities (classification only)
- Local model intercept
- R2 score (reliability indicator for local approximation; closer to 1.0 means higher reliability)

**Caching:**
LIME analysis results are also cached in session state, just like SHAP. Click "Clear LIME History" to delete all history.

#### 4.4.4 Prediction Results

When prediction results on training data exist, the first 100 rows are displayed in this section.

### 4.5 New Data Prediction

Predict on new data using a trained model.

**Prerequisites:**
- A model must be created on the Machine Learning page, or a saved model must be loaded on the Load Model page.

**Steps:**

1. Upload a new data file (CSV or Excel).
2. A preview and size of the uploaded data is displayed.
3. Click "Run Prediction" to execute predictions.
4. Prediction results can be downloaded as CSV or Excel.

**Prediction behavior:**
- When PyCaret Setup has been completed, `predict_model` is used.
- Otherwise, the model's `predict` method is called directly.
- When `predict_proba` is available, prediction probabilities are included in the output.

### 4.6 Load Model

Load a previously saved trained model on this page.

**Loading methods:**

1. **Upload file (.pkl)**: Upload a pickle file from your local machine
2. **Select file on server**: Choose from models saved in `/app/models/` inside the container, or enter a full path directly

**Steps:**

1. Select the loading method.
2. Specify the task type for this model (Classification, Regression, Clustering, Anomaly Detection).
3. Select or specify the file.
4. Click "Load Model" to execute.

**Loaded model information:**

Upon successful loading, the following information is displayed:
- Filename
- Task type
- Model type
- Pipeline steps (for Pipeline objects)
- Parameters list (for models supporting get_params)

The loaded model can be used on the "New Data Prediction" page.

### 4.7 Save / Export

Save and export data, predictions, and models on this page.

#### Export Data

Download the current data (including edits) in the following formats:
- CSV
- Excel

#### Export Predictions

Download prediction results on training data in the following formats:
- CSV
- Excel

#### Save Trained Model

- Specify a model filename to save to `/app/models/` inside the container.
- "Save Best Model": Saves the best model selected from model comparison
- "Save Tuned Model": Saves the hyperparameter-tuned model (filename appended with `_tuned`)

#### Download Model

Download the current trained model (tuned model takes priority if available) as a pickle file directly from the browser.

#### Saved Model Management

A file listing of `/app/models/` is displayed. The following operations are available for each file:

- **Rename**: Change the filename (extension is preserved automatically)
- **Delete**: Delete the file with a confirmation dialog
- **Download**: Download the file from the browser

---

## 5. Advanced Features

### 5.1 SMOTE for Imbalanced Data

For classification tasks where class sizes are imbalanced, SMOTE can be used to oversample the minority class.

**Available methods:**
- Default (SMOTE): Standard SMOTE algorithm
- BorderlineSMOTE: Oversampling focused on borderline samples
- SVMSMOTE: SVM-based borderline oversampling
- ADASYN: Oversampling focused on difficult-to-learn sample neighborhoods
- RandomOverSampler: Oversampling by random duplication

When the imbalance ratio exceeds 3:1, the application automatically recommends using SMOTE.

### 5.2 Normalization

Feature scaling is supported with the following methods:

- **zscore**: Standardization (mean 0, standard deviation 1)
- **minmax**: Min-max scaling (0 to 1)
- **maxabs**: Max absolute value scaling
- **robust**: Robust scaling (using median and IQR)

### 5.3 Outlier Removal

Automatically detects and removes outliers. The threshold (0.01 to 0.10) controls detection sensitivity. Smaller values result in more data being flagged as outliers.

### 5.4 Pickle Model Loading

The Load Model page supports two methods for loading pickle-format models:

1. **PyCaret format**: Models saved with PyCaret's `save_model` (including pipeline). Loaded using `load_model`.
2. **Direct pickle**: Standard pickle files. If PyCaret-format loading fails, a fallback using `pickle.load` is attempted.

### 5.5 Multi-language Support

Switch between Japanese (ja) and English (en) using the language selector in the upper right corner. Switching the language automatically reloads the page with all UI text displayed in the selected language. The default language is Japanese.

---

## 6. Troubleshooting

### Data loading fails

- Verify the file format is CSV or Excel (.xls, .xlsx).
- Verify the file size is 500MB or less.
- For Excel files, ensure the format is supported by openpyxl.

### Setup errors

- Verify the target column is correctly selected.
- If the data contains columns unsuitable for machine learning (such as ID columns), remove them on the Data View / Edit page beforehand.
- Error messages and stack traces are displayed for detailed diagnosis.

### Model comparison is very slow

- Select specific models to compare to reduce the scope and speed up execution.
- With large datasets, SVM-based models in particular can take a very long time.

### SHAP analysis errors

- Some models (especially ensemble or custom models) may fail SHAP value computation.
- When TreeExplainer cannot be used, KernelExplainer and PermutationExplainer are automatically tried, but computation time may increase significantly.
- The sample count is adjustable via the slider (50 to actual X_train size). When using KernelExplainer, setting the sample count to 200 or fewer is recommended.
- Use the "Reset SHAP cache" button to clear the cache, then adjust the sample count or model and recompute.

### LIME analysis errors

- LIME requires the model to have a predict or predict_proba method.
- Increasing the number of neighborhood samples improves stability but increases computation time.

### Model save fails

- Verify write permissions exist for the `/app/models/` directory inside the container.
- Verify the volume mount is correctly configured in docker-compose.yml.

### Container does not start

- Verify Docker Desktop is running.
- Verify port 8501 is not in use by another process.
- Check error logs with `docker compose logs`.

---

## 7. Docker Configuration Details

### docker-compose.yml

```yaml
services:
  webui-for-pycaret:
    build: .
    container_name: webui-for-pycaret
    ports:
      - "127.0.0.1:8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./exports:/app/exports
    environment:
      - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
    restart: unless-stopped
```

The port binding uses `127.0.0.1`, so the application is accessible only from localhost (the host machine). Other machines on the LAN cannot access it.

### Volume Mounts

| Host | Container | Purpose |
|---|---|---|
| ./data | /app/data | Data file persistence |
| ./models | /app/models | Saved model persistence |
| ./exports | /app/exports | Export file persistence |

Volume mounts ensure data, models, and export files are preserved across container restarts. The model save directory `/app/models/` corresponds to `./models/` on the host, so files can also be managed directly from the host machine.

### Environment Variables

| Variable | Value | Description |
|---|---|---|
| STREAMLIT_SERVER_MAX_UPLOAD_SIZE | 500 | Maximum upload file size (MB) |

### Dockerfile

- Base image: `pycaret/full:latest`
- Working directory: `/app`
- Exposed port: 8501 (127.0.0.1 only)
- Startup command: `streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false`
- The container listens on 0.0.0.0 internally, but the docker-compose port binding restricts access to localhost only.
