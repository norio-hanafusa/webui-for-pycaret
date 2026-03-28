"""
PyCaret WebUI - No-Code Machine Learning Platform
Based on pycaret/full Docker image
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import pickle
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PyCaret WebUI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
DEFAULTS = {
    "df": None,
    "df_edited": None,
    "pycaret_setup_done": False,
    "pycaret_task": None,
    "best_model": None,
    "tuned_model": None,
    "comparison_df": None,
    "predictions": None,
    "new_predictions": None,
    "setup_params": None,
    "loaded_model": None,
    "loaded_model_name": None,
    # SHAP / LIME cached results
    "shap_results": [],   # list of dicts: {type, fig_bytes, html, params}
    "lime_results": [],   # list of dicts: {fig_bytes, table_df, proba_df, intercept, score, sample_idx}
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🧪 PyCaret WebUI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "ナビゲーション",
    [
        "📂 データ読み込み",
        "📊 データ表示・編集",
        "🤖 機械学習",
        "📈 結果・可視化・SHAP/LIME",
        "🔮 新規データ予測",
        "📦 モデル読み込み",
        "💾 保存・エクスポート",
    ],
)

# Show loaded model info in sidebar
if st.session_state.loaded_model is not None:
    st.sidebar.markdown("---")
    st.sidebar.success(f"📦 読込済: {st.session_state.loaded_model_name}")

# ===================================================================
# Helper functions
# ===================================================================

def load_file(uploaded_file) -> pd.DataFrame:
    """CSV or Excel file to DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        st.error("サポートされていないファイル形式です。CSV または Excel をアップロードしてください。")
        return None


def get_module(task: str):
    """Return the correct PyCaret module for the given task."""
    if task == "分類 (Classification)":
        from pycaret import classification as mod
    elif task == "回帰 (Regression)":
        from pycaret import regression as mod
    elif task == "クラスタリング (Clustering)":
        from pycaret import clustering as mod
    elif task == "異常検知 (Anomaly Detection)":
        from pycaret import anomaly as mod
    else:
        mod = None
    return mod


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buf.getvalue()


# ===================================================================
# Page 1 : Data Loading
# ===================================================================
if page == "📂 データ読み込み":
    st.header("📂 データ読み込み")
    st.markdown("CSV または Excel ファイルをアップロードしてください。")

    uploaded = st.file_uploader(
        "ファイルを選択", type=["csv", "xls", "xlsx"], key="uploader_main"
    )

    if uploaded is not None:
        df = load_file(uploaded)
        if df is not None:
            st.session_state.df = df
            st.session_state.df_edited = df.copy()
            st.success(f"✅ 読み込み完了: {uploaded.name}  ({df.shape[0]} 行 × {df.shape[1]} 列)")

    if st.session_state.df is not None:
        df = st.session_state.df
        tab1, tab2, tab3 = st.tabs(["プレビュー", "基本統計量", "データ型"])

        with tab1:
            st.dataframe(df.head(100), use_container_width=True)

        with tab2:
            st.dataframe(df.describe(include="all").T, use_container_width=True)

        with tab3:
            dtype_df = pd.DataFrame({
                "列名": df.columns,
                "データ型": [str(d) for d in df.dtypes],
                "非Null数": [df[c].notna().sum() for c in df.columns],
                "Null数": [df[c].isna().sum() for c in df.columns],
                "ユニーク数": [df[c].nunique() for c in df.columns],
            })
            st.dataframe(dtype_df, use_container_width=True)

# ===================================================================
# Page 2 : Data Viewer / Editor
# ===================================================================
elif page == "📊 データ表示・編集":
    st.header("📊 データ表示・編集")

    if st.session_state.df is None:
        st.warning("⚠️ まずデータを読み込んでください。")
    else:
        df = st.session_state.df_edited.copy()

        st.subheader("データ編集")
        st.markdown("セルをダブルクリックして編集できます。")

        try:
            from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(editable=True, filterable=True, sortable=True, resizable=True)
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
            gb.configure_selection("multiple", use_checkbox=True)
            grid_options = gb.build()

            grid_response = AgGrid(
                df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                fit_columns_on_grid_load=True,
                height=500,
                allow_unsafe_jscode=True,
            )
            edited_df = grid_response["data"]
            if edited_df is not None:
                st.session_state.df_edited = pd.DataFrame(edited_df)

        except Exception:
            st.info("AgGrid が利用できないため、標準エディタを使用します。")
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                height=500,
            )
            st.session_state.df_edited = edited_df

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ 編集を確定"):
                st.session_state.df = st.session_state.df_edited.copy()
                st.success("データを更新しました。")
        with col2:
            if st.button("↩️ 元に戻す"):
                st.session_state.df_edited = st.session_state.df.copy()
                st.rerun()
        with col3:
            st.write(f"現在のデータ: {st.session_state.df_edited.shape[0]} 行 × {st.session_state.df_edited.shape[1]} 列")

        # Column operations
        st.subheader("列操作")
        col_a, col_b = st.columns(2)
        with col_a:
            drop_cols = st.multiselect("削除する列を選択", st.session_state.df_edited.columns.tolist())
            if st.button("選択した列を削除") and drop_cols:
                st.session_state.df_edited = st.session_state.df_edited.drop(columns=drop_cols)
                st.session_state.df = st.session_state.df_edited.copy()
                st.rerun()
        with col_b:
            fill_col = st.selectbox("欠損値を埋める列", ["---"] + st.session_state.df_edited.columns.tolist())
            fill_method = st.selectbox("埋め方", ["平均値", "中央値", "最頻値", "0"])
            if st.button("欠損値を補完") and fill_col != "---":
                col_data = st.session_state.df_edited[fill_col]
                if fill_method == "平均値":
                    val = col_data.mean()
                elif fill_method == "中央値":
                    val = col_data.median()
                elif fill_method == "最頻値":
                    val = col_data.mode()[0] if not col_data.mode().empty else 0
                else:
                    val = 0
                st.session_state.df_edited[fill_col] = col_data.fillna(val)
                st.session_state.df = st.session_state.df_edited.copy()
                st.success(f"{fill_col} の欠損値を {val} で補完しました。")
                st.rerun()

# ===================================================================
# Page 3 : Machine Learning
# ===================================================================
elif page == "🤖 機械学習":
    st.header("🤖 機械学習 (PyCaret)")

    if st.session_state.df is None:
        st.warning("⚠️ まずデータを読み込んでください。")
    else:
        df = st.session_state.df.copy()

        # --- Task selection ---
        task = st.selectbox(
            "タスクを選択",
            ["分類 (Classification)", "回帰 (Regression)", "クラスタリング (Clustering)", "異常検知 (Anomaly Detection)"],
        )
        st.session_state.pycaret_task = task
        mod = get_module(task)

        is_supervised = task in ["分類 (Classification)", "回帰 (Regression)"]

        target_col = None
        if is_supervised:
            target_col = st.selectbox("ターゲット列", df.columns.tolist())

        # --- Setup ---
        st.subheader("1. Setup (前処理)")
        session_id = st.number_input("ランダムシード", value=42, step=1)

        # --- Advanced setup options ---
        with st.expander("⚙️ 前処理オプション（詳細設定）", expanded=False):
            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                # --- Imbalanced data (classification only) ---
                if task == "分類 (Classification)":
                    fix_imbalance = st.checkbox(
                        "不均衡データ補正 (SMOTE)", value=False,
                        help="SMOTE を使用して少数クラスをオーバーサンプリングします。"
                    )
                    fix_imbalance_method = None
                    if fix_imbalance:
                        smote_method = st.selectbox(
                            "SMOTE 手法",
                            ["デフォルト (SMOTE)", "BorderlineSMOTE", "SVMSMOTE", "ADASYN", "RandomOverSampler"],
                            help="デフォルトは imblearn の SMOTE。他の手法も選べます。",
                        )
                        if smote_method != "デフォルト (SMOTE)":
                            try:
                                from imblearn.over_sampling import (
                                    BorderlineSMOTE, SVMSMOTE, ADASYN, RandomOverSampler,
                                )
                                _method_map = {
                                    "BorderlineSMOTE": BorderlineSMOTE(random_state=int(session_id)),
                                    "SVMSMOTE": SVMSMOTE(random_state=int(session_id)),
                                    "ADASYN": ADASYN(random_state=int(session_id)),
                                    "RandomOverSampler": RandomOverSampler(random_state=int(session_id)),
                                }
                                fix_imbalance_method = _method_map[smote_method]
                            except ImportError:
                                st.warning("imblearn が利用できません。デフォルト SMOTE を使用します。")

                    # Show class distribution
                    if target_col and target_col in df.columns:
                        st.markdown("**クラス分布:**")
                        class_counts = df[target_col].value_counts()
                        st.dataframe(
                            pd.DataFrame({"クラス": class_counts.index, "件数": class_counts.values,
                                          "割合 (%)": (class_counts.values / len(df) * 100).round(2)}),
                            use_container_width=True, hide_index=True,
                        )
                        imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float("inf")
                        if imbalance_ratio > 3:
                            st.warning(f"⚠️ 不均衡比率: {imbalance_ratio:.1f}:1 — SMOTE の使用を推奨します。")
                        else:
                            st.info(f"不均衡比率: {imbalance_ratio:.1f}:1")
                else:
                    fix_imbalance = False
                    fix_imbalance_method = None

            with col_opt2:
                normalize = st.checkbox("正規化 (Normalize)", value=False)
                normalize_method = "zscore"
                if normalize:
                    normalize_method = st.selectbox(
                        "正規化手法", ["zscore", "minmax", "maxabs", "robust"],
                    )

                remove_outliers = st.checkbox("外れ値除去", value=False)
                outliers_threshold = 0.05
                if remove_outliers:
                    outliers_threshold = st.slider("外れ値の閾値", 0.01, 0.1, 0.05, 0.01)

                if is_supervised:
                    train_size = st.slider("学習データ割合", 0.5, 0.9, 0.7, 0.05)
                else:
                    train_size = 0.7

        if st.button("🚀 Setup 実行"):
            with st.spinner("PyCaret Setup 実行中..."):
                try:
                    setup_kwargs = dict(
                        data=df,
                        session_id=int(session_id),
                        verbose=False,
                        html=False,
                    )

                    if is_supervised:
                        setup_kwargs["target"] = target_col
                        setup_kwargs["train_size"] = train_size

                        # Normalize
                        if normalize:
                            setup_kwargs["normalize"] = True
                            setup_kwargs["normalize_method"] = normalize_method

                        # Outlier removal
                        if remove_outliers:
                            setup_kwargs["remove_outliers"] = True
                            setup_kwargs["outliers_threshold"] = outliers_threshold

                        # Imbalanced data (classification only)
                        if task == "分類 (Classification)" and fix_imbalance:
                            setup_kwargs["fix_imbalance"] = True
                            if fix_imbalance_method is not None:
                                setup_kwargs["fix_imbalance_method"] = fix_imbalance_method

                    setup_result = mod.setup(**setup_kwargs)

                    st.session_state.pycaret_setup_done = True
                    st.session_state.setup_params = {"task": task, "target": target_col}
                    st.success("✅ Setup 完了")

                    # Show setup summary
                    if fix_imbalance and task == "分類 (Classification)":
                        method_name = type(fix_imbalance_method).__name__ if fix_imbalance_method else "SMOTE"
                        st.info(f"📊 不均衡データ補正: **{method_name}** を適用しました。")

                except Exception as e:
                    st.error(f"Setup エラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if st.session_state.pycaret_setup_done:
            # --- Compare models ---
            st.subheader("2. モデル比較")

            if is_supervised:
                n_select = st.slider("比較上位モデル数", 1, 10, 3)

                # Let user choose which models to compare
                all_model_ids = list(mod.models().index)
                all_model_names = [f"{mid} ({mod.models().loc[mid, 'Name']})" for mid in all_model_ids]
                include_models = st.multiselect(
                    "比較するモデルを選択（空欄＝全モデル）",
                    options=all_model_ids,
                    format_func=lambda mid: f"{mid} ({mod.models().loc[mid, 'Name']})",
                    default=[],
                )

                if st.button("🔍 モデル比較実行"):
                    try:
                        model_list = include_models if include_models else all_model_ids
                        total = len(model_list)

                        progress_bar = st.progress(0, text="モデル比較を開始します...")
                        status_text = st.empty()
                        results_placeholder = st.empty()

                        trained_models = []
                        summary_rows = []  # Mean row only (1 per model)

                        for i, mid in enumerate(model_list):
                            model_display = mod.models().loc[mid, "Name"] if mid in mod.models().index else mid
                            status_text.markdown(f"**学習中**: `{model_display}` ({i+1}/{total})")
                            progress_bar.progress((i) / total, text=f"{model_display} を学習中... ({i+1}/{total})")

                            try:
                                m = mod.create_model(mid, verbose=False)
                                score_df = mod.pull()
                                # pull() returns 10 CV folds + Mean + Std (12 rows)
                                # Extract the Mean row (second to last)
                                mean_row = score_df.iloc[-2:].head(1).copy()
                                mean_row.insert(0, "Model", model_display)
                                summary_rows.append(mean_row)
                                trained_models.append((mid, m, mean_row))
                            except Exception as model_err:
                                status_text.warning(f"⚠️ {model_display}: スキップ ({model_err})")

                            # Show intermediate summary table
                            if summary_rows:
                                interim = pd.concat(summary_rows, ignore_index=True)
                                results_placeholder.dataframe(interim, use_container_width=True)

                        progress_bar.progress(1.0, text="完了!")
                        status_text.empty()

                        if trained_models:
                            if task == "分類 (Classification)":
                                sort_col = "Accuracy"
                            else:
                                sort_col = "R2"

                            comparison = pd.concat(summary_rows, ignore_index=True)

                            # Sort by metric
                            if sort_col in comparison.columns:
                                comparison = comparison.sort_values(sort_col, ascending=False).reset_index(drop=True)

                            trained_models.sort(
                                key=lambda x: x[2][sort_col].values[0] if sort_col in x[2].columns else 0,
                                reverse=True,
                            )
                            best_list = [tm[1] for tm in trained_models[:n_select]]
                            st.session_state.best_model = best_list[0]
                            st.session_state.comparison_df = comparison
                            st.success(f"✅ モデル比較完了 ({len(trained_models)} モデル学習)")
                        else:
                            st.error("学習に成功したモデルがありません。")

                    except Exception as e:
                        st.error(f"モデル比較エラー: {e}")

                if st.session_state.comparison_df is not None:
                    st.dataframe(st.session_state.comparison_df, use_container_width=True)

            else:
                # Unsupervised - create model
                if task == "クラスタリング (Clustering)":
                    model_name = st.selectbox("クラスタリングアルゴリズム", ["kmeans", "ap", "meanshift", "sc", "hclust", "dbscan", "optics", "birch"])
                    n_clusters = st.slider("クラスタ数", 2, 20, 4)
                    if st.button("🔍 モデル作成"):
                        with st.spinner("クラスタリング実行中..."):
                            try:
                                model = mod.create_model(model_name, num_clusters=n_clusters, verbose=False)
                                st.session_state.best_model = model
                                result = mod.assign_model(model)
                                st.session_state.predictions = result
                                st.success("✅ クラスタリング完了")
                                st.dataframe(result.head(50), use_container_width=True)
                            except Exception as e:
                                st.error(f"エラー: {e}")

                elif task == "異常検知 (Anomaly Detection)":
                    model_name = st.selectbox("異常検知アルゴリズム", ["iforest", "knn", "lof", "svm", "pca", "mcd", "sod", "histogram"])
                    fraction = st.slider("異常の割合", 0.01, 0.5, 0.05, 0.01)
                    if st.button("🔍 モデル作成"):
                        with st.spinner("異常検知実行中..."):
                            try:
                                model = mod.create_model(model_name, fraction=fraction, verbose=False)
                                st.session_state.best_model = model
                                result = mod.assign_model(model)
                                st.session_state.predictions = result
                                st.success("✅ 異常検知完了")
                                st.dataframe(result.head(50), use_container_width=True)
                            except Exception as e:
                                st.error(f"エラー: {e}")

            # --- Tuning (supervised only) ---
            if is_supervised and st.session_state.best_model is not None:
                st.subheader("3. ハイパーパラメータチューニング")

                if task == "分類 (Classification)":
                    optimize_options = ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]
                else:
                    optimize_options = ["MAE", "MSE", "RMSE", "R2", "RMSLE", "MAPE"]

                optimize = st.selectbox("最適化指標", optimize_options)
                n_iter = st.slider("探索回数 (n_iter)", 5, 100, 10, 5)
                search_library = st.selectbox("探索アルゴリズム", ["scikit-learn", "optuna", "scikit-optimize"])

                if st.button("⚙️ チューニング実行"):
                    try:
                        status_text = st.empty()
                        progress_bar = st.progress(0, text="チューニングを開始します...")
                        result_placeholder = st.empty()

                        # --- Step 1: Show baseline score from best model ---
                        status_text.markdown(f"**Step 1/3**: ベースラインスコアを取得中...")
                        progress_bar.progress(0.1, text="ベースラインスコア取得中...")

                        baseline_model = mod.create_model(
                            st.session_state.best_model, verbose=False
                        )
                        baseline_pull = mod.pull()
                        # pull() returns CV folds + Mean + Std; Mean is second-to-last row
                        if "Mean" in baseline_pull.index:
                            baseline_mean = baseline_pull.loc["Mean"]
                        else:
                            baseline_mean = baseline_pull.iloc[-2]

                        baseline_score = baseline_mean[optimize] if optimize in baseline_mean.index else None
                        model_name = type(baseline_model).__name__

                        result_placeholder.info(
                            f"📊 ベースライン ({model_name}): {optimize} = "
                            f"**{baseline_score:.4f}**" if baseline_score is not None else "取得不可"
                        )

                        # --- Step 2: Run tune_model with full n_iter ---
                        status_text.markdown(
                            f"**Step 2/3**: ハイパーパラメータ探索中 "
                            f"({search_library}, {n_iter} 回)..."
                        )
                        progress_bar.progress(0.3, text=f"チューニング実行中 ({n_iter} イテレーション)...")

                        tuned = mod.tune_model(
                            st.session_state.best_model,
                            optimize=optimize,
                            n_iter=n_iter,
                            search_library=search_library,
                            verbose=False,
                            return_tuner=False,
                        )
                        tuned_pull = mod.pull()

                        # --- Step 3: Display results ---
                        status_text.markdown("**Step 3/3**: 結果を整理中...")
                        progress_bar.progress(0.9, text="結果表示中...")

                        # Extract Mean row from tuned results
                        if "Mean" in tuned_pull.index:
                            tuned_mean = tuned_pull.loc["Mean"]
                        else:
                            tuned_mean = tuned_pull.iloc[-2]

                        tuned_score = tuned_mean[optimize] if optimize in tuned_mean.index else None

                        progress_bar.progress(1.0, text="完了!")
                        status_text.empty()

                        # Show comparison table: Baseline vs Tuned
                        compare_rows = []
                        for col in tuned_pull.columns:
                            try:
                                b_val = float(baseline_mean[col]) if col in baseline_mean.index else None
                                t_val = float(tuned_mean[col]) if col in tuned_mean.index else None
                                diff = (t_val - b_val) if (b_val is not None and t_val is not None) else None
                                compare_rows.append({
                                    "指標": col,
                                    "ベースライン": f"{b_val:.4f}" if b_val is not None else "-",
                                    "チューニング後": f"{t_val:.4f}" if t_val is not None else "-",
                                    "差分": f"{diff:+.4f}" if diff is not None else "-",
                                })
                            except (ValueError, TypeError):
                                pass

                        if compare_rows:
                            result_placeholder.empty()
                            st.markdown("#### ベースライン vs チューニング後")
                            st.dataframe(
                                pd.DataFrame(compare_rows),
                                use_container_width=True,
                            )

                        # Show full CV results of tuned model
                        with st.expander("📋 チューニング後モデルの交差検証結果 (全fold)", expanded=False):
                            st.dataframe(tuned_pull, use_container_width=True)

                        # Show tuned model parameters
                        with st.expander("🔧 チューニング後のパラメータ", expanded=False):
                            st.json(tuned.get_params())

                        if tuned_score is not None:
                            st.session_state.tuned_model = tuned
                            improvement = ""
                            if baseline_score is not None:
                                diff = tuned_score - baseline_score
                                improvement = f"（差分: {diff:+.4f}）"
                            st.success(
                                f"✅ チューニング完了 — {optimize}: "
                                f"**{baseline_score:.4f}** → **{tuned_score:.4f}** {improvement}"
                            )
                        else:
                            st.session_state.tuned_model = tuned
                            st.warning("チューニングは完了しましたが、スコアの取得に失敗しました。")

                    except Exception as e:
                        import traceback
                        st.error(f"チューニングエラー: {e}")
                        st.code(traceback.format_exc())

            # --- Predict on train data (supervised) ---
            if is_supervised and (st.session_state.tuned_model is not None or st.session_state.best_model is not None):
                st.subheader("4. 学習データでの予測")
                if st.button("📊 予測実行"):
                    with st.spinner("予測中..."):
                        try:
                            use_model = st.session_state.tuned_model or st.session_state.best_model
                            preds = mod.predict_model(use_model)
                            st.session_state.predictions = preds
                            st.success("✅ 予測完了")
                            st.dataframe(preds.head(50), use_container_width=True)
                        except Exception as e:
                            st.error(f"予測エラー: {e}")

# ===================================================================
# Page 4 : Results / Visualization / SHAP / LIME
# ===================================================================
elif page == "📈 結果・可視化・SHAP/LIME":
    st.header("📈 結果・可視化・SHAP / LIME")

    if st.session_state.best_model is None:
        st.warning("⚠️ まず機械学習ページでモデルを作成してください。")
    else:
        task = st.session_state.pycaret_task
        mod = get_module(task)
        use_model = st.session_state.tuned_model or st.session_state.best_model
        is_supervised = task in ["分類 (Classification)", "回帰 (Regression)"]

        # --- PyCaret built-in plots ---
        st.subheader("PyCaret 可視化")

        if is_supervised:
            if task == "分類 (Classification)":
                plot_options = {
                    "AUC曲線": "auc",
                    "混同行列": "confusion_matrix",
                    "特徴量重要度": "feature",
                    "学習曲線": "learning",
                    "Precision-Recall": "pr",
                    "分類レポート": "class_report",
                    "境界線": "boundary",
                }
            else:
                plot_options = {
                    "残差プロット": "residuals",
                    "予測誤差": "error",
                    "特徴量重要度": "feature",
                    "学習曲線": "learning",
                    "Cook's Distance": "cooks",
                }
        else:
            if task == "クラスタリング (Clustering)":
                plot_options = {
                    "クラスタ分布": "cluster",
                    "エルボー法": "elbow",
                    "シルエット": "silhouette",
                    "分布": "distribution",
                }
            else:
                plot_options = {
                    "t-SNE": "tsne",
                    "UMAP": "umap",
                }

        selected_plot = st.selectbox("プロットを選択", list(plot_options.keys()))

        if st.button("📊 プロット生成"):
            with st.spinner("プロット生成中..."):
                try:
                    fig_path = mod.plot_model(use_model, plot=plot_options[selected_plot], save=True, verbose=False)
                    # PyCaret saves plot as .png in current directory
                    if isinstance(fig_path, str) and os.path.exists(fig_path):
                        st.image(fig_path)
                    else:
                        # Fallback: look for recently created png files
                        png_files = sorted(
                            [f for f in os.listdir(".") if f.endswith(".png")],
                            key=os.path.getmtime,
                            reverse=True,
                        )
                        if png_files:
                            st.image(png_files[0])
                        else:
                            st.info("プロットの画像ファイルが見つかりません。")
                except Exception as e:
                    st.error(f"プロット生成エラー: {e}")

        # ---------------------------------------------------------------
        # Helper: save matplotlib figure to PNG bytes
        # ---------------------------------------------------------------
        def _fig_to_bytes(fig=None):
            """Return current (or given) matplotlib figure as PNG bytes."""
            _buf = io.BytesIO()
            (fig or plt.gcf()).savefig(_buf, format="png", bbox_inches="tight", dpi=120)
            _buf.seek(0)
            return _buf.getvalue()

        # --- SHAP ---
        if is_supervised:
            st.subheader("SHAP 解析")
            st.markdown("モデルの予測に対する各特徴量の貢献度を可視化します。")

            shap_type = st.selectbox(
                "SHAP プロットタイプ",
                [
                    "Summary Plot (Bar)",
                    "Summary Plot (Dot)",
                    "Waterfall Plot",
                    "Beeswarm Plot",
                    "Bar Plot (shap.plots.bar)",
                    "Scatter Plot",
                    "Dependence Plot",
                    "Force Plot",
                    "Decision Plot",
                    "Violin Plot",
                    "Heatmap Plot",
                ],
            )

            # Waterfall / Force need a sample index
            shap_sample_idx = None
            if shap_type in ["Waterfall Plot", "Scatter Plot", "Force Plot"]:
                max_idx = min(499, st.session_state.df.shape[0] - 1)
                label_map = {
                    "Waterfall Plot": "対象サンプルのインデックス",
                    "Force Plot": "対象サンプルのインデックス (全体表示は -1)",
                    "Scatter Plot": "Scatter で注目する特徴量のインデックス (自動=最重要)",
                }
                shap_sample_idx = st.number_input(
                    label_map.get(shap_type, "インデックス"),
                    min_value=-1 if shap_type == "Force Plot" else 0,
                    max_value=max_idx,
                    value=0,
                    step=1,
                )

            # Scatter: choose feature
            scatter_feature = None
            if shap_type == "Scatter Plot":
                if st.session_state.df is not None:
                    scatter_feature = st.selectbox(
                        "Scatter で表示する特徴量（空欄＝最重要特徴量）",
                        ["(自動)"] + st.session_state.df.columns.tolist(),
                        key="shap_scatter_feat",
                    )

            col_shap_btn1, col_shap_btn2 = st.columns([1, 1])
            with col_shap_btn1:
                run_shap = st.button("🔬 SHAP 解析実行")
            with col_shap_btn2:
                if st.button("🗑️ SHAP 履歴をクリア"):
                    st.session_state.shap_results = []
                    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

            if run_shap:
                with st.spinner("SHAP 値を計算中... (少し時間がかかります)"):
                    try:
                        import shap

                        # Get transformed data from PyCaret pipeline
                        pipeline = mod.get_config("pipeline")
                        X_train = mod.get_config("X_train")
                        y_train = mod.get_config("y_train")

                        if hasattr(pipeline, "transform"):
                            X_transformed = pipeline.transform(X_train)
                        else:
                            X_transformed = X_train

                        # Unwrap model — get the fitted estimator
                        final_model = use_model
                        if hasattr(use_model, "steps"):
                            final_model = use_model.steps[-1][1]
                        if hasattr(final_model, "estimator"):
                            final_model = final_model.estimator

                        from sklearn.utils.validation import check_is_fitted as _cif
                        try:
                            _cif(final_model)
                        except Exception:
                            final_model = use_model

                        # Limit samples
                        max_shap_samples = 500
                        if isinstance(X_transformed, pd.DataFrame):
                            X_shap = X_transformed.iloc[:max_shap_samples]
                        else:
                            X_shap = X_transformed[:max_shap_samples]

                        feature_names = list(X_shap.columns) if isinstance(X_shap, pd.DataFrame) else None

                        # --- Compute SHAP values ---
                        shap_explanation = None
                        sv_array = None

                        # 1) TreeExplainer
                        try:
                            exp = shap.TreeExplainer(final_model)
                            raw = exp.shap_values(X_shap)
                            base = exp.expected_value
                            if isinstance(raw, list):
                                sv_arr = raw[1] if len(raw) == 2 else raw[0]
                                bv = base[1] if isinstance(base, (list, np.ndarray)) and len(base) > 1 else base
                            else:
                                sv_arr = raw
                                bv = base
                            if hasattr(bv, "__len__") and not isinstance(bv, (float, int, np.floating)):
                                bv = float(bv[0]) if len(bv) == 1 else float(np.mean(bv))
                            shap_explanation = shap.Explanation(
                                values=np.array(sv_arr),
                                base_values=np.full(len(X_shap), float(bv)),
                                data=np.array(X_shap),
                                feature_names=feature_names,
                            )
                            sv_array = np.array(sv_arr)
                        except Exception:
                            pass

                        # 2) KernelExplainer
                        if shap_explanation is None:
                            predict_fn = getattr(final_model, "predict_proba", None) or getattr(final_model, "predict", None)
                            if predict_fn is not None:
                                try:
                                    bg = shap.sample(X_shap, min(50, len(X_shap)))
                                    exp = shap.KernelExplainer(predict_fn, bg)
                                    raw = exp.shap_values(X_shap)
                                    base = exp.expected_value
                                    if isinstance(raw, list):
                                        sv_arr = raw[1] if len(raw) == 2 else raw[0]
                                        bv = base[1] if isinstance(base, (list, np.ndarray)) and len(base) > 1 else base
                                    else:
                                        sv_arr = raw
                                        bv = base
                                    if hasattr(bv, "__len__") and not isinstance(bv, (float, int, np.floating)):
                                        bv = float(bv[0]) if len(bv) == 1 else float(np.mean(bv))
                                    shap_explanation = shap.Explanation(
                                        values=np.array(sv_arr),
                                        base_values=np.full(len(X_shap), float(bv)),
                                        data=np.array(X_shap),
                                        feature_names=feature_names,
                                    )
                                    sv_array = np.array(sv_arr)
                                except Exception:
                                    pass

                        # 3) PermutationExplainer
                        if shap_explanation is None:
                            predict_fn = getattr(final_model, "predict_proba", None) or final_model.predict
                            exp = shap.PermutationExplainer(predict_fn, X_shap)
                            shap_obj = exp(X_shap)
                            sv_arr = shap_obj.values
                            if sv_arr.ndim == 3:
                                sv_arr = sv_arr[:, :, 1] if sv_arr.shape[2] == 2 else sv_arr[:, :, 0]
                            bv = shap_obj.base_values
                            if hasattr(bv, "ndim") and bv.ndim > 1:
                                bv = bv[:, 1] if bv.shape[1] == 2 else bv[:, 0]
                            shap_explanation = shap.Explanation(
                                values=np.array(sv_arr),
                                base_values=np.array(bv).flatten(),
                                data=np.array(X_shap),
                                feature_names=feature_names,
                            )
                            sv_array = np.array(sv_arr)

                        # If 3D, reduce
                        if sv_array is not None and sv_array.ndim == 3:
                            sv_array = sv_array[:, :, 1] if sv_array.shape[2] == 2 else sv_array[:, :, 0]
                            shap_explanation = shap.Explanation(
                                values=sv_array,
                                base_values=shap_explanation.base_values,
                                data=shap_explanation.data,
                                feature_names=feature_names,
                            )

                        # --- Generate plot & save to session_state ---
                        plt.close("all")
                        result_entry = {"type": shap_type, "fig_bytes": None, "html": None, "params": {}}

                        if shap_type == "Summary Plot (Bar)":
                            shap.summary_plot(sv_array, X_shap, plot_type="bar", show=False, max_display=20)
                            result_entry["fig_bytes"] = _fig_to_bytes()

                        elif shap_type == "Summary Plot (Dot)":
                            shap.summary_plot(sv_array, X_shap, show=False, max_display=20)
                            result_entry["fig_bytes"] = _fig_to_bytes()

                        elif shap_type == "Waterfall Plot":
                            idx = int(shap_sample_idx) if shap_sample_idx is not None else 0
                            plt.figure(figsize=(12, 8))
                            shap.plots.waterfall(shap_explanation[idx], show=False, max_display=20)
                            result_entry["fig_bytes"] = _fig_to_bytes()
                            result_entry["params"]["sample_idx"] = idx

                        elif shap_type == "Beeswarm Plot":
                            plt.figure(figsize=(12, 8))
                            shap.plots.beeswarm(shap_explanation, show=False, max_display=20)
                            result_entry["fig_bytes"] = _fig_to_bytes()

                        elif shap_type == "Bar Plot (shap.plots.bar)":
                            plt.figure(figsize=(12, 8))
                            shap.plots.bar(shap_explanation, show=False, max_display=20)
                            result_entry["fig_bytes"] = _fig_to_bytes()

                        elif shap_type == "Scatter Plot":
                            if scatter_feature and scatter_feature != "(自動)" and feature_names and scatter_feature in feature_names:
                                feat_idx = feature_names.index(scatter_feature)
                            else:
                                feat_idx = int(np.abs(sv_array).mean(axis=0).argmax())
                            plt.figure(figsize=(12, 8))
                            shap.plots.scatter(shap_explanation[:, feat_idx], show=False)
                            result_entry["fig_bytes"] = _fig_to_bytes()
                            result_entry["params"]["feature"] = feature_names[feat_idx] if feature_names else feat_idx

                        elif shap_type == "Dependence Plot":
                            if isinstance(X_shap, pd.DataFrame):
                                top_feat = X_shap.columns[np.abs(sv_array).mean(axis=0).argmax()]
                            else:
                                top_feat = int(np.abs(sv_array).mean(axis=0).argmax())
                            shap.dependence_plot(top_feat, sv_array, X_shap, show=False)
                            result_entry["fig_bytes"] = _fig_to_bytes()
                            result_entry["params"]["feature"] = str(top_feat)

                        elif shap_type == "Force Plot":
                            shap.initjs()
                            idx = int(shap_sample_idx) if shap_sample_idx is not None else 0
                            if idx < 0:
                                force_html = shap.force_plot(
                                    shap_explanation.base_values[0], sv_array, X_shap,
                                    feature_names=feature_names, show=False,
                                )
                            else:
                                force_html = shap.force_plot(
                                    shap_explanation.base_values[idx], sv_array[idx],
                                    X_shap.iloc[idx] if isinstance(X_shap, pd.DataFrame) else X_shap[idx],
                                    feature_names=feature_names, show=False,
                                )
                            import shap as _shap_mod
                            result_entry["html"] = f"<head>{_shap_mod.getjs()}</head><body>{force_html.html()}</body>"
                            result_entry["params"]["sample_idx"] = idx

                        elif shap_type == "Decision Plot":
                            plt.figure(figsize=(12, 8))
                            shap.decision_plot(
                                shap_explanation.base_values[0], sv_array,
                                feature_names=feature_names, show=False,
                            )
                            result_entry["fig_bytes"] = _fig_to_bytes()

                        elif shap_type == "Violin Plot":
                            plt.figure(figsize=(12, 8))
                            shap.summary_plot(sv_array, X_shap, plot_type="violin", show=False, max_display=20)
                            result_entry["fig_bytes"] = _fig_to_bytes()

                        elif shap_type == "Heatmap Plot":
                            plt.figure(figsize=(14, 8))
                            shap.plots.heatmap(shap_explanation, show=False, max_display=20)
                            result_entry["fig_bytes"] = _fig_to_bytes()

                        plt.close("all")

                        # Append to session history
                        st.session_state.shap_results.append(result_entry)
                        st.success(f"✅ SHAP 解析完了: {shap_type}")

                    except Exception as e:
                        st.error(f"SHAP 解析エラー: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            # --- Display all cached SHAP results ---
            if st.session_state.shap_results:
                st.markdown("---")
                st.markdown("#### 📂 SHAP 解析履歴")
                for i, res in enumerate(st.session_state.shap_results):
                    params_str = ""
                    if res.get("params"):
                        params_str = " | ".join(f"{k}={v}" for k, v in res["params"].items())
                        params_str = f" ({params_str})"
                    with st.expander(f"#{i+1}  {res['type']}{params_str}", expanded=(i == len(st.session_state.shap_results) - 1)):
                        if res.get("fig_bytes"):
                            st.image(res["fig_bytes"], use_container_width=True)
                        elif res.get("html"):
                            import streamlit.components.v1 as components
                            components.html(res["html"], height=350, scrolling=True)

        # --- LIME ---
        if is_supervised:
            st.subheader("LIME 解析")
            st.markdown(
                "LIME (Local Interpretable Model-agnostic Explanations) は、"
                "個々の予測を局所的に解釈可能なモデルで近似し、各特徴量の貢献度を可視化します。"
            )

            lime_sample_idx = st.number_input(
                "解析対象サンプルのインデックス",
                min_value=0,
                max_value=max(0, st.session_state.df.shape[0] - 1) if st.session_state.df is not None else 0,
                value=0,
                step=1,
                key="lime_sample_idx",
            )

            lime_num_features = st.slider(
                "表示する特徴量の数", 5, 30, 10, key="lime_num_features"
            )

            lime_num_samples = st.slider(
                "LIME 近傍サンプル数 (多いほど安定・遅い)", 100, 10000, 1000, 100,
                key="lime_num_samples",
            )

            col_lime_btn1, col_lime_btn2 = st.columns([1, 1])
            with col_lime_btn1:
                run_lime = st.button("🔬 LIME 解析実行")
            with col_lime_btn2:
                if st.button("🗑️ LIME 履歴をクリア"):
                    st.session_state.lime_results = []
                    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

            if run_lime:
                with st.spinner("LIME 解析を計算中... (少し時間がかかります)"):
                    try:
                        import lime
                        import lime.lime_tabular

                        # Get transformed data
                        pipeline = mod.get_config("pipeline")
                        X_train = mod.get_config("X_train")
                        y_train = mod.get_config("y_train")

                        if hasattr(pipeline, "transform"):
                            X_transformed = pipeline.transform(X_train)
                        else:
                            X_transformed = X_train

                        # Unwrap fitted model
                        final_model = use_model
                        if hasattr(use_model, "steps"):
                            final_model = use_model.steps[-1][1]
                        if hasattr(final_model, "estimator"):
                            final_model = final_model.estimator

                        from sklearn.utils.validation import check_is_fitted as _cif
                        try:
                            _cif(final_model)
                        except Exception:
                            final_model = use_model

                        # Prepare data as numpy for LIME
                        if isinstance(X_transformed, pd.DataFrame):
                            feature_names_lime = list(X_transformed.columns)
                            X_np = X_transformed.values
                        else:
                            feature_names_lime = [f"feature_{i}" for i in range(X_transformed.shape[1])]
                            X_np = np.array(X_transformed)

                        is_classification = task == "分類 (Classification)"

                        if is_classification:
                            y_vals = y_train.unique() if hasattr(y_train, "unique") else np.unique(y_train)
                            class_names = [str(c) for c in sorted(y_vals)]

                            cat_feat_indices = []
                            if isinstance(X_transformed, pd.DataFrame):
                                for ci, col in enumerate(X_transformed.columns):
                                    if X_transformed[col].dtype == "object" or X_transformed[col].dtype.name == "category":
                                        cat_feat_indices.append(ci)

                            explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                                training_data=X_np,
                                feature_names=feature_names_lime,
                                class_names=class_names,
                                mode="classification",
                                categorical_features=cat_feat_indices if cat_feat_indices else None,
                                random_state=42,
                            )

                            if hasattr(final_model, "predict_proba"):
                                predict_fn_lime = final_model.predict_proba
                            else:
                                def predict_fn_lime(X):
                                    preds = final_model.predict(X)
                                    n_classes = len(class_names)
                                    proba = np.zeros((len(preds), n_classes))
                                    for pi, p in enumerate(preds):
                                        pidx = list(sorted(y_vals)).index(p) if p in y_vals else 0
                                        proba[pi, pidx] = 1.0
                                    return proba

                            sample = X_np[lime_sample_idx]
                            explanation = explainer_lime.explain_instance(
                                sample, predict_fn_lime,
                                num_features=lime_num_features,
                                num_samples=lime_num_samples,
                            )
                        else:
                            class_names = None
                            explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                                training_data=X_np,
                                feature_names=feature_names_lime,
                                mode="regression",
                                random_state=42,
                            )
                            predict_fn_lime = final_model.predict
                            sample = X_np[lime_sample_idx]
                            explanation = explainer_lime.explain_instance(
                                sample, predict_fn_lime,
                                num_features=lime_num_features,
                                num_samples=lime_num_samples,
                            )

                        # --- Build result entry for session cache ---
                        lime_entry = {"sample_idx": int(lime_sample_idx)}

                        # Figure bytes
                        fig_lime = explanation.as_pyplot_figure()
                        fig_lime.set_size_inches(12, max(6, lime_num_features * 0.4))
                        fig_lime.tight_layout()
                        lime_entry["fig_bytes"] = _fig_to_bytes(fig_lime)
                        plt.close(fig_lime)

                        # Feature weight table
                        lime_weights = explanation.as_list()
                        lime_df = pd.DataFrame(lime_weights, columns=["特徴量 (条件)", "貢献度"])
                        lime_df["方向"] = lime_df["貢献度"].apply(
                            lambda x: "🟢 正 (予測を押し上げ)" if x > 0 else "🔴 負 (予測を押し下げ)"
                        )
                        lime_df["|貢献度|"] = lime_df["貢献度"].abs()
                        lime_df = lime_df.sort_values("|貢献度|", ascending=False).reset_index(drop=True)
                        lime_entry["table_df"] = lime_df

                        # Prediction probabilities (classification only)
                        if is_classification and explanation.predict_proba is not None:
                            lime_entry["proba_df"] = pd.DataFrame({
                                "クラス": class_names[:len(explanation.predict_proba)],
                                "確率": explanation.predict_proba,
                            })
                        else:
                            lime_entry["proba_df"] = None

                        # Intercept
                        try:
                            _intercept = explanation.intercept
                            if isinstance(_intercept, dict):
                                _intercept_val = list(_intercept.values())[0]
                            elif hasattr(_intercept, "__len__") and not isinstance(_intercept, (float, int)):
                                _intercept_val = _intercept[1] if is_classification and len(_intercept) > 1 else _intercept[0]
                            else:
                                _intercept_val = float(_intercept)
                            lime_entry["intercept"] = f"{_intercept_val:.6f}"
                        except Exception:
                            lime_entry["intercept"] = str(explanation.intercept)

                        # R² score
                        lime_entry["score"] = explanation.score if hasattr(explanation, "score") else None

                        # Append to session history
                        st.session_state.lime_results.append(lime_entry)
                        st.success("✅ LIME 解析完了")

                    except Exception as e:
                        st.error(f"LIME 解析エラー: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            # --- Display all cached LIME results ---
            if st.session_state.lime_results:
                st.markdown("---")
                st.markdown("#### 📂 LIME 解析履歴")
                for i, res in enumerate(st.session_state.lime_results):
                    with st.expander(
                        f"#{i+1}  サンプル #{res['sample_idx']}",
                        expanded=(i == len(st.session_state.lime_results) - 1),
                    ):
                        if res.get("fig_bytes"):
                            st.image(res["fig_bytes"], use_container_width=True)
                        if res.get("table_df") is not None:
                            st.markdown("##### 特徴量の貢献度テーブル")
                            st.dataframe(res["table_df"], use_container_width=True)
                        if res.get("proba_df") is not None:
                            st.markdown("##### 予測確率 (ローカルモデル)")
                            st.dataframe(res["proba_df"], use_container_width=True)
                        if res.get("intercept"):
                            st.markdown(f"##### ローカルモデルの切片 (Intercept): `{res['intercept']}`")
                        if res.get("score") is not None:
                            st.info(f"ローカルモデルの R² スコア: **{res['score']:.4f}** (1.0に近いほど局所近似の信頼性が高い)")

        # --- Prediction results ---
        if st.session_state.predictions is not None:
            st.subheader("予測結果")
            st.dataframe(st.session_state.predictions.head(100), use_container_width=True)

# ===================================================================
# Page 5 : Predict on New Data
# ===================================================================
elif page == "🔮 新規データ予測":
    st.header("🔮 新規データ予測")

    if st.session_state.best_model is None and st.session_state.loaded_model is None:
        st.warning("⚠️ まず機械学習ページでモデルを作成するか、📦 モデル読み込みページで保存済みモデルをロードしてください。")
    else:
        task = st.session_state.pycaret_task
        mod = get_module(task) if task else None
        use_model = st.session_state.tuned_model or st.session_state.best_model or st.session_state.loaded_model
        is_supervised = task in ["分類 (Classification)", "回帰 (Regression)"] if task else True

        # Show which model is being used
        if st.session_state.loaded_model is not None and st.session_state.loaded_model_name:
            st.info(f"📦 使用モデル: **{st.session_state.loaded_model_name}**")

        st.markdown("学習済みモデルで新しいデータに対する予測を行います。")
        st.markdown("CSV または Excel ファイルをアップロードしてください。")

        new_file = st.file_uploader(
            "新規データファイル",
            type=["csv", "xls", "xlsx"],
            key="uploader_new",
        )

        if new_file is not None:
            new_df = load_file(new_file)
            if new_df is not None:
                st.subheader("アップロードされたデータ")
                st.dataframe(new_df.head(50), use_container_width=True)
                st.write(f"サイズ: {new_df.shape[0]} 行 × {new_df.shape[1]} 列")

                if st.button("🔮 予測実行"):
                    with st.spinner("新規データの予測中..."):
                        try:
                            # Try PyCaret predict_model first
                            if mod is not None and st.session_state.pycaret_setup_done:
                                preds = mod.predict_model(use_model, data=new_df)
                            elif hasattr(use_model, "predict"):
                                # Direct sklearn-compatible predict
                                predictions = use_model.predict(new_df)
                                preds = new_df.copy()
                                preds["prediction_label"] = predictions
                                # Also add probabilities if available
                                if hasattr(use_model, "predict_proba"):
                                    try:
                                        proba = use_model.predict_proba(new_df)
                                        for i in range(proba.shape[1]):
                                            preds[f"prediction_score_{i}"] = proba[:, i]
                                    except Exception:
                                        pass
                            else:
                                st.error("このモデルは predict メソッドを持っていません。")
                                preds = None

                            if preds is not None:
                                st.session_state.new_predictions = preds
                                st.success("✅ 予測完了")
                                st.dataframe(preds, use_container_width=True)
                        except Exception as e:
                            st.error(f"予測エラー: {e}")

        if st.session_state.new_predictions is not None:
            st.subheader("予測結果ダウンロード")
            col1, col2 = st.columns(2)
            with col1:
                csv_data = st.session_state.new_predictions.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 CSV でダウンロード",
                    csv_data,
                    file_name="new_predictions.csv",
                    mime="text/csv",
                )
            with col2:
                xlsx_data = to_excel_bytes(st.session_state.new_predictions)
                st.download_button(
                    "📥 Excel でダウンロード",
                    xlsx_data,
                    file_name="new_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

# ===================================================================
# Page 6 : Load Model
# ===================================================================
elif page == "📦 モデル読み込み":
    st.header("📦 モデル読み込み")
    st.markdown("以前に保存した学習済みモデル（pickle / PyCaret形式）を読み込んで、予測や可視化に使用できます。")

    # --- Method selection ---
    load_method = st.radio(
        "読み込み方法を選択",
        ["ファイルをアップロード (.pkl)", "サーバー上のファイルを指定"],
        horizontal=True,
    )

    # --- Task selection (needed to use the model with PyCaret) ---
    load_task = st.selectbox(
        "このモデルのタスク種別",
        ["分類 (Classification)", "回帰 (Regression)", "クラスタリング (Clustering)", "異常検知 (Anomaly Detection)"],
        key="load_task_select",
    )

    loaded = None
    model_label = None

    if load_method == "ファイルをアップロード (.pkl)":
        uploaded_model = st.file_uploader(
            "pickle ファイルを選択",
            type=["pkl"],
            key="uploader_model",
        )
        if uploaded_model is not None:
            if st.button("📦 モデルを読み込む", key="load_upload"):
                with st.spinner("モデルを読み込み中..."):
                    try:
                        loaded = pickle.load(uploaded_model)
                        model_label = uploaded_model.name
                    except Exception as e:
                        st.error(f"読み込みエラー: {e}")

    else:  # サーバー上のファイル
        st.markdown("コンテナ内の `/app/models/` に保存されたモデルを読み込めます。")

        # List available model files
        model_dir = "/app/models"
        available_models = []
        if os.path.isdir(model_dir):
            available_models = sorted(
                [f for f in os.listdir(model_dir) if f.endswith(".pkl")],
            )

        if available_models:
            selected_file = st.selectbox("保存済みモデルを選択", available_models)
            model_path = os.path.join(model_dir, selected_file)
        else:
            st.info(f"`{model_dir}` にモデルファイルがありません。")
            model_path = None

        custom_path = st.text_input("または、フルパスを直接入力", placeholder="/app/models/my_model.pkl")
        if custom_path:
            model_path = custom_path

        if model_path and st.button("📦 モデルを読み込む", key="load_server"):
            with st.spinner("モデルを読み込み中..."):
                try:
                    # Try PyCaret load_model first (handles pipeline + model)
                    try:
                        mod = get_module(load_task)
                        # PyCaret save_model appends .pkl, so strip it for load_model
                        path_no_ext = model_path
                        if path_no_ext.endswith(".pkl"):
                            path_no_ext = path_no_ext[:-4]
                        loaded = mod.load_model(path_no_ext)
                        model_label = os.path.basename(model_path)
                    except Exception:
                        # Fallback: raw pickle
                        with open(model_path, "rb") as f:
                            loaded = pickle.load(f)
                        model_label = os.path.basename(model_path)
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")

    # --- Apply loaded model to session ---
    if loaded is not None:
        st.session_state.loaded_model = loaded
        st.session_state.loaded_model_name = model_label
        st.session_state.best_model = loaded
        st.session_state.tuned_model = None
        st.session_state.pycaret_task = load_task
        st.success(f"✅ モデル「{model_label}」を読み込みました。")

    # --- Show currently loaded model info ---
    if st.session_state.loaded_model is not None:
        st.subheader("読み込み済みモデル情報")
        model_obj = st.session_state.loaded_model
        st.markdown(f"- **ファイル名**: `{st.session_state.loaded_model_name}`")
        st.markdown(f"- **タスク種別**: {st.session_state.pycaret_task}")
        st.markdown(f"- **モデル型**: `{type(model_obj).__name__}`")

        # Show pipeline steps if it's a Pipeline
        if hasattr(model_obj, "steps"):
            st.markdown("**パイプラインステップ:**")
            for i, (name, step) in enumerate(model_obj.steps):
                st.markdown(f"  {i+1}. `{name}` → `{type(step).__name__}`")
        elif hasattr(model_obj, "get_params"):
            st.markdown("**パラメータ:**")
            params = model_obj.get_params()
            params_df = pd.DataFrame(
                [{"パラメータ": k, "値": str(v)} for k, v in params.items()]
            )
            st.dataframe(params_df, use_container_width=True, height=300)

        st.info("💡 「🔮 新規データ予測」ページで、このモデルを使って予測を実行できます。")

# ===================================================================
# Page 7 : Save / Export
# ===================================================================
elif page == "💾 保存・エクスポート":
    st.header("💾 保存・エクスポート")

    # --- Edited data export ---
    st.subheader("データのエクスポート")
    if st.session_state.df is not None:
        col1, col2 = st.columns(2)
        with col1:
            csv_data = st.session_state.df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 データを CSV でダウンロード",
                csv_data,
                file_name="data_export.csv",
                mime="text/csv",
            )
        with col2:
            xlsx_data = to_excel_bytes(st.session_state.df)
            st.download_button(
                "📥 データを Excel でダウンロード",
                xlsx_data,
                file_name="data_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("データが読み込まれていません。")

    # --- Prediction results export ---
    st.subheader("予測結果のエクスポート")
    if st.session_state.predictions is not None:
        col1, col2 = st.columns(2)
        with col1:
            csv_data = st.session_state.predictions.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 予測結果を CSV でダウンロード",
                csv_data,
                file_name="predictions.csv",
                mime="text/csv",
            )
        with col2:
            xlsx_data = to_excel_bytes(st.session_state.predictions)
            st.download_button(
                "📥 予測結果を Excel でダウンロード",
                xlsx_data,
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("予測結果がありません。先に機械学習ページで予測を実行してください。")

    # --- Model export ---
    st.subheader("学習済みモデルの保存")
    if st.session_state.best_model is not None:
        model_name = st.text_input("モデルファイル名", value="best_model")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 ベストモデルを保存"):
                try:
                    mod = get_module(st.session_state.pycaret_task)
                    mod.save_model(st.session_state.best_model, f"/app/models/{model_name}")
                    st.success(f"✅ モデルを /app/models/{model_name}.pkl に保存しました。")
                except Exception as e:
                    st.error(f"保存エラー: {e}")

        with col2:
            if st.session_state.tuned_model is not None:
                if st.button("💾 チューニング済みモデルを保存"):
                    try:
                        mod = get_module(st.session_state.pycaret_task)
                        mod.save_model(st.session_state.tuned_model, f"/app/models/{model_name}_tuned")
                        st.success(f"✅ チューニング済みモデルを /app/models/{model_name}_tuned.pkl に保存しました。")
                    except Exception as e:
                        st.error(f"保存エラー: {e}")

        # Download model as pickle
        st.subheader("モデルのダウンロード")
        use_model = st.session_state.tuned_model or st.session_state.best_model
        buf = io.BytesIO()
        pickle.dump(use_model, buf)
        buf.seek(0)
        st.download_button(
            "📥 モデルを pickle でダウンロード",
            buf.getvalue(),
            file_name=f"{model_name}.pkl",
            mime="application/octet-stream",
        )
    else:
        st.info("学習済みモデルがありません。先に機械学習ページでモデルを作成してください。")

    # --- Model file management ---
    st.markdown("---")
    st.subheader("📁 保存済みモデルの管理")
    st.markdown("`/app/models/` 内のファイル一覧です。名前の変更・削除ができます。")

    _model_dir = "/app/models"
    if os.path.isdir(_model_dir):
        _model_files = sorted(
            [f for f in os.listdir(_model_dir) if os.path.isfile(os.path.join(_model_dir, f))],
        )
    else:
        _model_files = []

    if not _model_files:
        st.info("保存済みモデルはありません。")
    else:
        # Initialise per-file session keys once
        if "model_mgmt_confirm_delete" not in st.session_state:
            st.session_state.model_mgmt_confirm_delete = {}

        st.markdown(f"**{len(_model_files)} 件のファイル**")

        for _mf in _model_files:
            _full = os.path.join(_model_dir, _mf)
            _size_kb = os.path.getsize(_full) / 1024
            _size_str = f"{_size_kb:.1f} KB" if _size_kb < 1024 else f"{_size_kb/1024:.2f} MB"

            with st.expander(f"📄 {_mf}　({_size_str})", expanded=False):
                col_r1, col_r2 = st.columns(2)

                # --- Rename ---
                with col_r1:
                    _base, _ext = os.path.splitext(_mf)
                    new_name = st.text_input(
                        "新しいファイル名",
                        value=_base,
                        key=f"rename_{_mf}",
                    )
                    if st.button("✏️ 名前を変更", key=f"rename_btn_{_mf}"):
                        new_full_name = new_name + _ext
                        new_full_path = os.path.join(_model_dir, new_full_name)
                        if new_full_name == _mf:
                            st.warning("同じ名前です。変更はありません。")
                        elif os.path.exists(new_full_path):
                            st.error(f"⚠️ `{new_full_name}` は既に存在します。別の名前を指定してください。")
                        elif not new_name.strip():
                            st.error("ファイル名を入力してください。")
                        else:
                            try:
                                os.rename(_full, new_full_path)
                                st.success(f"✅ `{_mf}` → `{new_full_name}` にリネームしました。")
                                st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
                            except Exception as _e:
                                st.error(f"リネームエラー: {_e}")

                # --- Delete ---
                with col_r2:
                    _del_key = f"del_{_mf}"
                    if st.session_state.model_mgmt_confirm_delete.get(_mf, False):
                        st.warning(f"⚠️ `{_mf}` を本当に削除しますか？")
                        col_y, col_n = st.columns(2)
                        with col_y:
                            if st.button("🗑️ はい、削除する", key=f"del_yes_{_mf}"):
                                try:
                                    os.remove(_full)
                                    st.session_state.model_mgmt_confirm_delete[_mf] = False
                                    st.success(f"✅ `{_mf}` を削除しました。")
                                    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
                                except Exception as _e:
                                    st.error(f"削除エラー: {_e}")
                        with col_n:
                            if st.button("キャンセル", key=f"del_no_{_mf}"):
                                st.session_state.model_mgmt_confirm_delete[_mf] = False
                                st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
                    else:
                        if st.button("🗑️ 削除", key=_del_key):
                            st.session_state.model_mgmt_confirm_delete[_mf] = True
                            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

                # --- Download ---
                with open(_full, "rb") as _fp:
                    st.download_button(
                        f"📥 ダウンロード",
                        _fp.read(),
                        file_name=_mf,
                        mime="application/octet-stream",
                        key=f"dl_{_mf}",
                    )

# ===================================================================
# Footer
# ===================================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **PyCaret WebUI**
    Built on `pycaret/full`
    Powered by Streamlit
    """
)
