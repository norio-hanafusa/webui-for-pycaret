"""
WebUI for PyCaret - No-Code Machine Learning Platform
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

from i18n import t

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WebUI for PyCaret",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide sidebar completely since we now use tabs
st.markdown(
    """<style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-size: 0.95rem;
    }
    </style>""",
    unsafe_allow_html=True,
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
    "lang": "ja",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Language selector
# ---------------------------------------------------------------------------
lang = st.session_state.lang

_lang_col1, _lang_col2 = st.columns([6, 1])
with _lang_col2:
    _lang_choice = st.selectbox(
        t("lang_label", lang),
        ["ja", "en"],
        index=0 if st.session_state.lang == "ja" else 1,
        key="lang_selector",
    )
    if _lang_choice != st.session_state.lang:
        st.session_state.lang = _lang_choice
        st.rerun()

lang = st.session_state.lang

# ---------------------------------------------------------------------------
# Header & loaded-model indicator
# ---------------------------------------------------------------------------
_header_cols = st.columns([6, 2])
with _header_cols[0]:
    st.title(t("page_title", lang))
with _header_cols[1]:
    if st.session_state.loaded_model is not None:
        st.success(t("msg_loaded_short", lang).format(name=st.session_state.loaded_model_name))

# ---------------------------------------------------------------------------
# Tab-style navigation using horizontal radio buttons
# ---------------------------------------------------------------------------
_PAGES = [
    t("nav_data_load", lang),
    t("nav_data_edit", lang),
    t("nav_ml", lang),
    t("nav_results", lang),
    t("nav_predict", lang),
    t("nav_load_model", lang),
    t("nav_save_export", lang),
]
page = st.radio(
    t("nav_label", lang),
    _PAGES,
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("---")

# ===================================================================
# Helper functions
# ===================================================================

_TASK_LABELS = {
    "classification": "opt_classification",
    "regression": "opt_regression",
    "clustering": "opt_clustering",
    "anomaly": "opt_anomaly",
}


def get_module(task: str):
    """Return the correct PyCaret module for the given task key."""
    if task == "classification":
        from pycaret import classification as mod
    elif task == "regression":
        from pycaret import regression as mod
    elif task == "clustering":
        from pycaret import clustering as mod
    elif task == "anomaly":
        from pycaret import anomaly as mod
    else:
        mod = None
    return mod


def load_file(uploaded_file) -> pd.DataFrame:
    """CSV or Excel file to DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        st.error(t("msg_unsupported_format", lang))
        return None


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buf.getvalue()


# ===================================================================
# Page 1 : Data Loading
# ===================================================================
if page == t("nav_data_load", lang):
    st.header(t("nav_data_load", lang))
    st.markdown(t("msg_upload_csv_excel", lang))

    uploaded = st.file_uploader(
        t("btn_select_file", lang), type=["csv", "xls", "xlsx"], key="uploader_main"
    )

    if uploaded is not None:
        df = load_file(uploaded)
        if df is not None:
            st.session_state.df = df
            st.session_state.df_edited = df.copy()
            st.success(t("msg_load_complete", lang).format(
                name=uploaded.name, rows=df.shape[0], cols=df.shape[1]
            ))

    if st.session_state.df is not None:
        df = st.session_state.df
        tab1, tab2, tab3 = st.tabs([
            t("label_preview", lang),
            t("label_descriptive_stats", lang),
            t("label_data_types", lang),
        ])

        with tab1:
            st.dataframe(df.head(100), use_container_width=True)

        with tab2:
            st.dataframe(df.describe(include="all").T, use_container_width=True)

        with tab3:
            dtype_df = pd.DataFrame({
                t("label_column", lang): df.columns,
                t("label_type", lang): [str(d) for d in df.dtypes],
                t("label_non_null_count", lang): [df[c].notna().sum() for c in df.columns],
                t("label_null_count", lang): [df[c].isna().sum() for c in df.columns],
                t("label_unique_count", lang): [df[c].nunique() for c in df.columns],
            })
            st.dataframe(dtype_df, use_container_width=True)

# ===================================================================
# Page 2 : Data Viewer / Editor
# ===================================================================
elif page == t("nav_data_edit", lang):
    st.header(t("nav_data_edit", lang))

    if st.session_state.df is None:
        st.warning(t("msg_load_first", lang))
    else:
        df = st.session_state.df_edited.copy()

        st.subheader(t("label_data_editor", lang))
        st.markdown(t("msg_double_click_edit", lang))

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
            st.info(t("msg_aggrid_unavailable", lang))
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                height=500,
            )
            st.session_state.df_edited = edited_df

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(t("btn_apply_changes", lang)):
                st.session_state.df = st.session_state.df_edited.copy()
                st.success(t("msg_data_updated", lang))
        with col2:
            if st.button(t("btn_undo", lang)):
                st.session_state.df_edited = st.session_state.df.copy()
                st.rerun()
        with col3:
            st.write(t("msg_current_data", lang).format(
                rows=st.session_state.df_edited.shape[0],
                cols=st.session_state.df_edited.shape[1],
            ))

        # Column operations
        st.subheader(t("label_column_ops", lang))
        col_a, col_b = st.columns(2)
        with col_a:
            drop_cols = st.multiselect(t("label_select_cols_drop", lang), st.session_state.df_edited.columns.tolist())
            if st.button(t("btn_drop_columns", lang)) and drop_cols:
                st.session_state.df_edited = st.session_state.df_edited.drop(columns=drop_cols)
                st.session_state.df = st.session_state.df_edited.copy()
                st.rerun()
        with col_b:
            fill_col = st.selectbox(t("label_fill_na_col", lang), ["---"] + st.session_state.df_edited.columns.tolist())
            fill_method = st.selectbox(t("label_fill_method", lang), [
                t("opt_mean", lang), t("opt_median", lang), t("opt_mode", lang), "0",
            ])
            if st.button(t("btn_fill_na", lang)) and fill_col != "---":
                col_data = st.session_state.df_edited[fill_col]
                if fill_method == t("opt_mean", lang):
                    val = col_data.mean()
                elif fill_method == t("opt_median", lang):
                    val = col_data.median()
                elif fill_method == t("opt_mode", lang):
                    val = col_data.mode()[0] if not col_data.mode().empty else 0
                else:
                    val = 0
                st.session_state.df_edited[fill_col] = col_data.fillna(val)
                st.session_state.df = st.session_state.df_edited.copy()
                st.success(t("msg_filled_na", lang).format(col=fill_col, val=val))
                st.rerun()

# ===================================================================
# Page 3 : Machine Learning
# ===================================================================
elif page == t("nav_ml", lang):
    st.header(t("nav_ml", lang))

    if st.session_state.df is None:
        st.warning(t("msg_load_first", lang))
    else:
        df = st.session_state.df.copy()

        # --- Task selection ---
        task = st.selectbox(
            t("label_select_task", lang),
            list(_TASK_LABELS.keys()),
            format_func=lambda k: t(_TASK_LABELS[k], lang),
        )
        st.session_state.pycaret_task = task
        mod = get_module(task)

        is_supervised = task in ["classification", "regression"]

        target_col = None
        if is_supervised:
            target_col = st.selectbox(t("label_target_col", lang), df.columns.tolist())

        # --- Setup ---
        st.subheader(t("heading_setup", lang))
        session_id = st.number_input(t("label_random_seed", lang), value=42, step=1)

        # --- Advanced setup options ---
        with st.expander(t("label_preprocess_options", lang), expanded=False):
            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                # --- Imbalanced data (classification only) ---
                if task == "classification":
                    fix_imbalance = st.checkbox(
                        t("label_smote", lang), value=False,
                        help=t("msg_smote_desc", lang),
                    )
                    fix_imbalance_method = None
                    if fix_imbalance:
                        smote_method = st.selectbox(
                            t("label_smote_method", lang),
                            [t("opt_smote_default", lang), "BorderlineSMOTE", "SVMSMOTE", "ADASYN", "RandomOverSampler"],
                            help=t("msg_smote_default_desc", lang),
                        )
                        if smote_method != t("opt_smote_default", lang):
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
                                st.warning(t("msg_imblearn_unavailable", lang))

                    # Show class distribution
                    if target_col and target_col in df.columns:
                        st.markdown(f"**{t('label_class_distribution', lang)}**")
                        class_counts = df[target_col].value_counts()
                        st.dataframe(
                            pd.DataFrame({
                                t("label_class", lang): class_counts.index,
                                t("label_count", lang): class_counts.values,
                                t("label_ratio_pct", lang): (class_counts.values / len(df) * 100).round(2),
                            }),
                            use_container_width=True, hide_index=True,
                        )
                        imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float("inf")
                        if imbalance_ratio > 3:
                            st.warning(t("msg_imbalance_recommend", lang).format(ratio=f"{imbalance_ratio:.1f}"))
                        else:
                            st.info(t("msg_imbalance_ratio", lang).format(ratio=f"{imbalance_ratio:.1f}"))
                else:
                    fix_imbalance = False
                    fix_imbalance_method = None

            with col_opt2:
                normalize = st.checkbox(t("label_normalize", lang), value=False)
                normalize_method = "zscore"
                if normalize:
                    normalize_method = st.selectbox(
                        t("label_normalize_method", lang), ["zscore", "minmax", "maxabs", "robust"],
                    )

                remove_outliers = st.checkbox(t("label_remove_outliers", lang), value=False)
                outliers_threshold = 0.05
                if remove_outliers:
                    outliers_threshold = st.slider(t("label_outlier_threshold", lang), 0.01, 0.1, 0.05, 0.01)

                if is_supervised:
                    train_size = st.slider(t("label_train_ratio", lang), 0.5, 0.9, 0.7, 0.05)
                else:
                    train_size = 0.7

        if st.button(t("btn_run_setup", lang)):
            with st.spinner(t("msg_setup_running", lang)):
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
                        if task == "classification" and fix_imbalance:
                            setup_kwargs["fix_imbalance"] = True
                            if fix_imbalance_method is not None:
                                setup_kwargs["fix_imbalance_method"] = fix_imbalance_method

                    setup_result = mod.setup(**setup_kwargs)

                    st.session_state.pycaret_setup_done = True
                    st.session_state.setup_params = {"task": task, "target": target_col}
                    st.success(t("msg_setup_complete", lang))

                    # Show setup summary
                    if fix_imbalance and task == "classification":
                        method_name = type(fix_imbalance_method).__name__ if fix_imbalance_method else "SMOTE"
                        st.info(t("msg_smote_applied", lang).format(method=method_name))

                except Exception as e:
                    st.error(t("msg_setup_error", lang).format(e=e))
                    import traceback
                    st.code(traceback.format_exc())

        if st.session_state.pycaret_setup_done:
            # --- Compare models ---
            st.subheader(t("heading_compare", lang))

            if is_supervised:
                n_select = st.slider(t("label_top_n", lang), 1, 10, 3)

                # Let user choose which models to compare
                all_model_ids = list(mod.models().index)
                all_model_names = [f"{mid} ({mod.models().loc[mid, 'Name']})" for mid in all_model_ids]
                include_models = st.multiselect(
                    t("label_select_models", lang),
                    options=all_model_ids,
                    format_func=lambda mid: f"{mid} ({mod.models().loc[mid, 'Name']})",
                    default=[],
                )

                if st.button(t("btn_run_compare", lang)):
                    try:
                        model_list = include_models if include_models else all_model_ids
                        total = len(model_list)

                        progress_bar = st.progress(0, text=t("msg_compare_start", lang))
                        status_text = st.empty()
                        results_placeholder = st.empty()

                        trained_models = []
                        summary_rows = []  # Mean row only (1 per model)

                        for i, mid in enumerate(model_list):
                            model_display = mod.models().loc[mid, "Name"] if mid in mod.models().index else mid
                            status_text.markdown(f"**{t('msg_training', lang)}**: `{model_display}` ({i+1}/{total})")
                            progress_bar.progress((i) / total, text=f"{model_display} {t('msg_training_model', lang)} ({i+1}/{total})")

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
                                status_text.warning(f"{model_display}: {t('msg_skipped', lang)} ({model_err})")

                            # Show intermediate summary table
                            if summary_rows:
                                interim = pd.concat(summary_rows, ignore_index=True)
                                results_placeholder.dataframe(interim, use_container_width=True)

                        progress_bar.progress(1.0, text=t("msg_done", lang))
                        status_text.empty()

                        if trained_models:
                            if task == "classification":
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
                            st.success(t("msg_compare_complete", lang).format(n=len(trained_models)))
                        else:
                            st.error(t("msg_no_models_trained", lang))

                    except Exception as e:
                        st.error(t("msg_compare_error", lang).format(e=e))

                if st.session_state.comparison_df is not None:
                    st.dataframe(st.session_state.comparison_df, use_container_width=True)

            else:
                # Unsupervised - create model
                if task == "clustering":
                    model_name = st.selectbox(t("label_clustering_algo", lang), ["kmeans", "ap", "meanshift", "sc", "hclust", "dbscan", "optics", "birch"])
                    n_clusters = st.slider(t("label_num_clusters", lang), 2, 20, 4)
                    if st.button(t("btn_create_model", lang)):
                        with st.spinner(t("msg_clustering_running", lang)):
                            try:
                                model = mod.create_model(model_name, num_clusters=n_clusters, verbose=False)
                                st.session_state.best_model = model
                                result = mod.assign_model(model)
                                st.session_state.predictions = result
                                st.success(t("msg_clustering_complete", lang))
                                st.dataframe(result.head(50), use_container_width=True)
                            except Exception as e:
                                st.error(t("msg_error", lang).format(e=e))

                elif task == "anomaly":
                    model_name = st.selectbox(t("label_anomaly_algo", lang), ["iforest", "knn", "lof", "svm", "pca", "mcd", "sod", "histogram"])
                    fraction = st.slider(t("label_anomaly_fraction", lang), 0.01, 0.5, 0.05, 0.01)
                    if st.button(t("btn_create_model", lang)):
                        with st.spinner(t("msg_anomaly_running", lang)):
                            try:
                                model = mod.create_model(model_name, fraction=fraction, verbose=False)
                                st.session_state.best_model = model
                                result = mod.assign_model(model)
                                st.session_state.predictions = result
                                st.success(t("msg_anomaly_complete", lang))
                                st.dataframe(result.head(50), use_container_width=True)
                            except Exception as e:
                                st.error(t("msg_error", lang).format(e=e))

            # --- Tuning (supervised only) ---
            if is_supervised and st.session_state.best_model is not None:
                st.subheader(t("heading_tuning", lang))

                if task == "classification":
                    optimize_options = ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]
                else:
                    optimize_options = ["MAE", "MSE", "RMSE", "R2", "RMSLE", "MAPE"]

                optimize = st.selectbox(t("label_opt_metric", lang), optimize_options)
                n_iter = st.slider(t("label_n_iter", lang), 5, 100, 10, 5)
                search_library = st.selectbox(t("label_search_algo", lang), ["scikit-learn", "optuna", "scikit-optimize"])

                if st.button(t("btn_run_tuning", lang)):
                    try:
                        status_text = st.empty()
                        progress_bar = st.progress(0, text=t("msg_tuning_start", lang))
                        result_placeholder = st.empty()

                        # --- Step 1: Show baseline score from best model ---
                        status_text.markdown(f"**Step 1/3**: {t('msg_getting_baseline', lang)}")
                        progress_bar.progress(0.1, text=t("msg_getting_baseline_alt", lang))

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
                            f"{t('label_baseline', lang)} ({model_name}): {optimize} = "
                            f"**{baseline_score:.4f}**" if baseline_score is not None else "-"
                        )

                        # --- Step 2: Run tune_model with full n_iter ---
                        status_text.markdown(
                            f"**Step 2/3**: {t('msg_searching_hp', lang).format(lib=search_library, n=n_iter)}"
                        )
                        progress_bar.progress(0.3, text=t("msg_tuning_running", lang).format(n=n_iter))

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
                        status_text.markdown(f"**Step 3/3**: {t('msg_preparing_results', lang)}")
                        progress_bar.progress(0.9, text=t("msg_displaying_results", lang))

                        # Extract Mean row from tuned results
                        if "Mean" in tuned_pull.index:
                            tuned_mean = tuned_pull.loc["Mean"]
                        else:
                            tuned_mean = tuned_pull.iloc[-2]

                        tuned_score = tuned_mean[optimize] if optimize in tuned_mean.index else None

                        progress_bar.progress(1.0, text=t("msg_done", lang))
                        status_text.empty()

                        # Show comparison table: Baseline vs Tuned
                        compare_rows = []
                        for col in tuned_pull.columns:
                            try:
                                b_val = float(baseline_mean[col]) if col in baseline_mean.index else None
                                t_val = float(tuned_mean[col]) if col in tuned_mean.index else None
                                diff = (t_val - b_val) if (b_val is not None and t_val is not None) else None
                                compare_rows.append({
                                    t("label_metric", lang): col,
                                    t("label_baseline", lang): f"{b_val:.4f}" if b_val is not None else "-",
                                    t("label_after_tuning", lang): f"{t_val:.4f}" if t_val is not None else "-",
                                    t("label_diff", lang): f"{diff:+.4f}" if diff is not None else "-",
                                })
                            except (ValueError, TypeError):
                                pass

                        if compare_rows:
                            result_placeholder.empty()
                            st.markdown(f"#### {t('label_baseline_vs_tuned', lang)}")
                            st.dataframe(
                                pd.DataFrame(compare_rows),
                                use_container_width=True,
                            )

                        # Show full CV results of tuned model
                        with st.expander(t("label_tuned_cv_results", lang), expanded=False):
                            st.dataframe(tuned_pull, use_container_width=True)

                        # Show tuned model parameters
                        with st.expander(t("label_tuned_params", lang), expanded=False):
                            st.json(tuned.get_params())

                        if tuned_score is not None:
                            st.session_state.tuned_model = tuned
                            improvement = ""
                            if baseline_score is not None:
                                diff = tuned_score - baseline_score
                                improvement = f"({diff:+.4f})"
                            st.success(
                                t("msg_tuning_complete", lang).format(
                                    metric=optimize,
                                    before=f"{baseline_score:.4f}",
                                    after=f"{tuned_score:.4f}",
                                    diff=improvement,
                                )
                            )
                        else:
                            st.session_state.tuned_model = tuned
                            st.warning(t("msg_tuning_score_fail", lang))

                    except Exception as e:
                        import traceback
                        st.error(t("msg_tuning_error", lang).format(e=e))
                        st.code(traceback.format_exc())

            # --- Predict on train data (supervised) ---
            if is_supervised and (st.session_state.tuned_model is not None or st.session_state.best_model is not None):
                st.subheader(t("heading_predict_train", lang))
                if st.button(t("btn_run_predict", lang)):
                    with st.spinner(t("msg_predicting", lang)):
                        try:
                            use_model = st.session_state.tuned_model or st.session_state.best_model
                            preds = mod.predict_model(use_model)
                            st.session_state.predictions = preds
                            st.success(t("msg_predict_complete", lang))
                            st.dataframe(preds.head(50), use_container_width=True)
                        except Exception as e:
                            st.error(t("msg_predict_error", lang).format(e=e))

# ===================================================================
# Page 4 : Results / Visualization / SHAP / LIME
# ===================================================================
elif page == t("nav_results", lang):
    st.header(t("nav_results", lang))

    if st.session_state.best_model is None:
        st.warning(t("msg_create_model_first", lang))
    else:
        task = st.session_state.pycaret_task
        mod = get_module(task)
        use_model = st.session_state.tuned_model or st.session_state.best_model
        is_supervised = task in ["classification", "regression"]

        # --- PyCaret built-in plots ---
        st.subheader(t("heading_pycaret_viz", lang))

        if is_supervised:
            if task == "classification":
                plot_options = {
                    t("plot_auc", lang): "auc",
                    t("plot_confusion_matrix", lang): "confusion_matrix",
                    t("plot_feature_importance", lang): "feature",
                    t("plot_learning_curve", lang): "learning",
                    "Precision-Recall": "pr",
                    t("plot_class_report", lang): "class_report",
                    t("plot_boundary", lang): "boundary",
                }
            else:
                plot_options = {
                    t("plot_residuals", lang): "residuals",
                    t("plot_prediction_error", lang): "error",
                    t("plot_feature_importance", lang): "feature",
                    t("plot_learning_curve", lang): "learning",
                    "Cook's Distance": "cooks",
                }
        else:
            if task == "clustering":
                plot_options = {
                    t("plot_cluster_distribution", lang): "cluster",
                    t("plot_elbow", lang): "elbow",
                    t("plot_silhouette", lang): "silhouette",
                    t("plot_distribution", lang): "distribution",
                }
            else:
                plot_options = {
                    "t-SNE": "tsne",
                    "UMAP": "umap",
                }

        selected_plot = st.selectbox(t("label_select_plot", lang), list(plot_options.keys()))

        if st.button(t("btn_generate_plot", lang)):
            with st.spinner(t("msg_plot_generating", lang)):
                try:
                    import glob as _glob
                    plot_key = plot_options[selected_plot]
                    displayed = False

                    def _call_plot_model(save=False, suppress_show=False):
                        """Call mod.plot_model with appropriate args."""
                        _restore = None
                        if suppress_show:
                            _restore = plt.show
                            plt.show = lambda *a, **k: None
                        try:
                            kwargs = {"plot": plot_key}
                            if save:
                                kwargs["save"] = True
                            if st.session_state.pycaret_task not in ["clustering", "anomaly"]:
                                kwargs["verbose"] = False
                            return mod.plot_model(use_model, **kwargs)
                        finally:
                            if _restore:
                                plt.show = _restore

                    def _is_plotly_fig(obj):
                        """Check if obj is a plotly Figure."""
                        try:
                            import plotly.graph_objects as go
                            return isinstance(obj, go.Figure)
                        except ImportError:
                            return False

                    # --- Strategy A: save=True ---
                    try:
                        _before = set(_glob.glob("*.png")) | set(_glob.glob("*.html"))
                        ret = _call_plot_model(save=True)

                        # A1) Plotly figure returned
                        if not displayed and _is_plotly_fig(ret):
                            st.plotly_chart(ret, use_container_width=True)
                            displayed = True

                        # A2) File path returned
                        if not displayed and isinstance(ret, str):
                            for candidate in [ret, os.path.basename(ret)]:
                                if os.path.exists(candidate):
                                    if candidate.endswith(".html"):
                                        import plotly.io as pio
                                        pfig = pio.read_html(candidate)
                                        st.plotly_chart(pfig, use_container_width=True)
                                    else:
                                        st.image(candidate, use_container_width=True)
                                    displayed = True
                                    break

                        # A3) Matplotlib figure returned
                        if not displayed and ret is not None and not isinstance(ret, str) and hasattr(ret, "savefig"):
                            buf = io.BytesIO()
                            ret.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                            buf.seek(0)
                            st.image(buf.getvalue(), use_container_width=True)
                            plt.close(ret)
                            displayed = True

                        # A4) New files created
                        if not displayed:
                            _after = set(_glob.glob("*.png")) | set(_glob.glob("*.html"))
                            _new_files = _after - _before
                            if _new_files:
                                newest = max(_new_files, key=os.path.getmtime)
                                if newest.endswith(".html"):
                                    with open(newest, "r", encoding="utf-8") as f:
                                        st.components.v1.html(f.read(), height=600, scrolling=True)
                                else:
                                    st.image(newest, use_container_width=True)
                                displayed = True

                    except Exception:
                        pass

                    # --- Strategy B: no save, suppress plt.show, capture ---
                    if not displayed:
                        try:
                            plt.close("all")
                            ret = _call_plot_model(save=False, suppress_show=True)

                            # B1) Plotly figure
                            if _is_plotly_fig(ret):
                                st.plotly_chart(ret, use_container_width=True)
                                displayed = True

                            # B2) Matplotlib figures
                            if not displayed:
                                fig_nums = plt.get_fignums()
                                if fig_nums:
                                    for fn in fig_nums:
                                        fig = plt.figure(fn)
                                        if fig.get_axes():
                                            buf = io.BytesIO()
                                            fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                                            buf.seek(0)
                                            st.image(buf.getvalue(), use_container_width=True)
                                            displayed = True
                                    plt.close("all")
                        except Exception:
                            pass

                    # --- Strategy C: st.pyplot / st.plotly_chart fallback ---
                    if not displayed:
                        try:
                            plt.close("all")
                            ret = _call_plot_model(save=False, suppress_show=True)
                            if _is_plotly_fig(ret):
                                st.plotly_chart(ret, use_container_width=True)
                                displayed = True
                            else:
                                fig_nums = plt.get_fignums()
                                if fig_nums:
                                    for fn in fig_nums:
                                        st.pyplot(plt.figure(fn))
                                    plt.close("all")
                                    displayed = True
                        except Exception:
                            pass

                    if not displayed:
                        st.info(t("msg_plot_not_found", lang))

                except Exception as e:
                    st.error(t("msg_plot_error", lang).format(e=e))

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
            st.subheader(t("heading_shap", lang))
            st.markdown(t("msg_shap_desc", lang))

            shap_type = st.selectbox(
                t("label_shap_plot_type", lang),
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
                    "Waterfall Plot": t("label_sample_index", lang),
                    "Force Plot": t("label_sample_index_all", lang),
                    "Scatter Plot": t("label_scatter_feature_idx", lang),
                }
                shap_sample_idx = st.number_input(
                    label_map.get(shap_type, t("label_index", lang)),
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
                        t("label_scatter_feature", lang),
                        [t("label_auto", lang)] + st.session_state.df.columns.tolist(),
                        key="shap_scatter_feat",
                    )

            col_shap_btn1, col_shap_btn2 = st.columns([1, 1])
            with col_shap_btn1:
                run_shap = st.button(t("btn_run_shap", lang))
            with col_shap_btn2:
                if st.button(t("btn_clear_shap", lang)):
                    st.session_state.shap_results = []
                    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

            if run_shap:
                with st.spinner(t("msg_shap_computing", lang)):
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

                        # Unwrap model -- get the fitted estimator
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
                            if scatter_feature and scatter_feature != t("label_auto", lang) and feature_names and scatter_feature in feature_names:
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
                        st.success(t("msg_shap_complete", lang).format(type=shap_type))

                    except Exception as e:
                        st.error(t("msg_shap_error", lang).format(e=e))
                        import traceback
                        st.code(traceback.format_exc())

            # --- Display all cached SHAP results ---
            if st.session_state.shap_results:
                st.markdown("---")
                st.markdown(f"#### {t('heading_shap_history', lang)}")
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
            st.subheader(t("heading_lime", lang))
            st.markdown(t("msg_lime_desc", lang))

            lime_sample_idx = st.number_input(
                t("label_lime_sample_index", lang),
                min_value=0,
                max_value=max(0, st.session_state.df.shape[0] - 1) if st.session_state.df is not None else 0,
                value=0,
                step=1,
                key="lime_sample_idx",
            )

            lime_num_features = st.slider(
                t("label_lime_num_features", lang), 5, 30, 10, key="lime_num_features"
            )

            lime_num_samples = st.slider(
                t("label_lime_num_samples", lang), 100, 10000, 1000, 100,
                key="lime_num_samples",
            )

            col_lime_btn1, col_lime_btn2 = st.columns([1, 1])
            with col_lime_btn1:
                run_lime = st.button(t("btn_run_lime", lang))
            with col_lime_btn2:
                if st.button(t("btn_clear_lime", lang)):
                    st.session_state.lime_results = []
                    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

            if run_lime:
                with st.spinner(t("msg_lime_computing", lang)):
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

                        is_classification = task == "classification"

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
                        lime_df = pd.DataFrame(lime_weights, columns=[
                            t("label_feature_condition", lang),
                            t("label_contribution", lang),
                        ])
                        lime_df[t("label_direction", lang)] = lime_df[t("label_contribution", lang)].apply(
                            lambda x: t("label_positive_direction", lang) if x > 0 else t("label_negative_direction", lang)
                        )
                        lime_df[t("label_abs_contribution", lang)] = lime_df[t("label_contribution", lang)].abs()
                        lime_df = lime_df.sort_values(t("label_abs_contribution", lang), ascending=False).reset_index(drop=True)
                        lime_entry["table_df"] = lime_df

                        # Prediction probabilities (classification only)
                        if is_classification and explanation.predict_proba is not None:
                            lime_entry["proba_df"] = pd.DataFrame({
                                t("label_class", lang): class_names[:len(explanation.predict_proba)],
                                t("label_probability", lang): explanation.predict_proba,
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

                        # R2 score
                        lime_entry["score"] = explanation.score if hasattr(explanation, "score") else None

                        # Append to session history
                        st.session_state.lime_results.append(lime_entry)
                        st.success(t("msg_lime_complete", lang))

                    except Exception as e:
                        st.error(t("msg_lime_error", lang).format(e=e))
                        import traceback
                        st.code(traceback.format_exc())

            # --- Display all cached LIME results ---
            if st.session_state.lime_results:
                st.markdown("---")
                st.markdown(f"#### {t('heading_lime_history', lang)}")
                for i, res in enumerate(st.session_state.lime_results):
                    with st.expander(
                        f"#{i+1}  {t('label_sample', lang)} #{res['sample_idx']}",
                        expanded=(i == len(st.session_state.lime_results) - 1),
                    ):
                        if res.get("fig_bytes"):
                            st.image(res["fig_bytes"], use_container_width=True)
                        if res.get("table_df") is not None:
                            st.markdown(f"##### {t('label_feature_contribution_table', lang)}")
                            st.dataframe(res["table_df"], use_container_width=True)
                        if res.get("proba_df") is not None:
                            st.markdown(f"##### {t('label_pred_prob_local', lang)}")
                            st.dataframe(res["proba_df"], use_container_width=True)
                        if res.get("intercept"):
                            st.markdown(f"##### {t('label_intercept', lang)}: `{res['intercept']}`")
                        if res.get("score") is not None:
                            st.info(t("msg_lime_r2", lang).format(score=f"{res['score']:.4f}"))

        # --- Prediction results ---
        if st.session_state.predictions is not None:
            st.subheader(t("heading_prediction_results", lang))
            st.dataframe(st.session_state.predictions.head(100), use_container_width=True)

# ===================================================================
# Page 5 : Predict on New Data
# ===================================================================
elif page == t("nav_predict", lang):
    st.header(t("nav_predict", lang))

    if st.session_state.best_model is None and st.session_state.loaded_model is None:
        st.warning(t("msg_create_or_load_model", lang))
    else:
        task = st.session_state.pycaret_task
        mod = get_module(task) if task else None
        use_model = st.session_state.tuned_model or st.session_state.best_model or st.session_state.loaded_model
        is_supervised = task in ["classification", "regression"] if task else True

        # Show which model is being used
        if st.session_state.loaded_model is not None and st.session_state.loaded_model_name:
            st.info(t("msg_using_model", lang).format(name=st.session_state.loaded_model_name))

        st.markdown(t("msg_predict_new_desc", lang))
        st.markdown(t("msg_upload_csv_excel", lang))

        new_file = st.file_uploader(
            t("label_new_data_file", lang),
            type=["csv", "xls", "xlsx"],
            key="uploader_new",
        )

        if new_file is not None:
            new_df = load_file(new_file)
            if new_df is not None:
                st.subheader(t("label_uploaded_data", lang))
                st.dataframe(new_df.head(50), use_container_width=True)
                st.write(t("msg_data_size", lang).format(rows=new_df.shape[0], cols=new_df.shape[1]))

                if st.button(t("btn_run_predict", lang)):
                    with st.spinner(t("msg_predicting_new", lang)):
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
                                st.error(t("msg_no_predict_method", lang))
                                preds = None

                            if preds is not None:
                                st.session_state.new_predictions = preds
                                st.success(t("msg_predict_complete", lang))
                                st.dataframe(preds, use_container_width=True)
                        except Exception as e:
                            st.error(t("msg_predict_error", lang).format(e=e))

        if st.session_state.new_predictions is not None:
            st.subheader(t("label_download_predictions", lang))
            col1, col2 = st.columns(2)
            with col1:
                csv_data = st.session_state.new_predictions.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    t("btn_download_csv", lang),
                    csv_data,
                    file_name="new_predictions.csv",
                    mime="text/csv",
                )
            with col2:
                xlsx_data = to_excel_bytes(st.session_state.new_predictions)
                st.download_button(
                    t("btn_download_excel", lang),
                    xlsx_data,
                    file_name="new_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

# ===================================================================
# Page 6 : Load Model
# ===================================================================
elif page == t("nav_load_model", lang):
    st.header(t("nav_load_model", lang))
    st.markdown(t("msg_load_model_desc", lang))

    # --- Method selection ---
    load_method = st.radio(
        t("label_load_method", lang),
        [t("opt_upload_pkl", lang), t("opt_server_file", lang)],
        horizontal=True,
    )

    # --- Task selection (needed to use the model with PyCaret) ---
    load_task = st.selectbox(
        t("label_task_type", lang),
        ["classification", "regression", "clustering", "anomaly"],
        format_func=lambda k: t(_TASK_LABELS[k], lang),
        key="load_task_select",
    )

    loaded = None
    model_label = None

    if load_method == t("opt_upload_pkl", lang):
        uploaded_model = st.file_uploader(
            t("label_select_pkl", lang),
            type=["pkl"],
            key="uploader_model",
        )
        if uploaded_model is not None:
            if st.button(t("btn_load_model", lang), key="load_upload"):
                with st.spinner(t("msg_loading_model", lang)):
                    try:
                        loaded = pickle.load(uploaded_model)
                        model_label = uploaded_model.name
                    except Exception as e:
                        st.error(t("msg_load_error", lang).format(e=e))

    else:  # Server file
        st.markdown(t("msg_server_models_desc", lang))

        # List available model files
        model_dir = "/app/models"
        available_models = []
        if os.path.isdir(model_dir):
            available_models = sorted(
                [f for f in os.listdir(model_dir) if f.endswith(".pkl")],
            )

        if available_models:
            selected_file = st.selectbox(t("label_select_saved_model", lang), available_models)
            model_path = os.path.join(model_dir, selected_file)
        else:
            st.info(f"`{model_dir}` {t('msg_no_model_files', lang)}")
            model_path = None

        custom_path = st.text_input(t("label_enter_path", lang), placeholder="/app/models/my_model.pkl")
        if custom_path:
            model_path = custom_path

        if model_path and st.button(t("btn_load_model", lang), key="load_server"):
            with st.spinner(t("msg_loading_model", lang)):
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
                    st.error(t("msg_load_error", lang).format(e=e))

    # --- Apply loaded model to session ---
    if loaded is not None:
        st.session_state.loaded_model = loaded
        st.session_state.loaded_model_name = model_label
        st.session_state.best_model = loaded
        st.session_state.tuned_model = None
        st.session_state.pycaret_task = load_task
        st.success(t("msg_model_loaded", lang).format(name=model_label))

    # --- Show currently loaded model info ---
    if st.session_state.loaded_model is not None:
        st.subheader(t("heading_loaded_model_info", lang))
        model_obj = st.session_state.loaded_model
        st.markdown(f"- **{t('label_filename', lang)}**: `{st.session_state.loaded_model_name}`")
        st.markdown(f"- **{t('label_task_type_info', lang)}**: {st.session_state.pycaret_task}")
        st.markdown(f"- **{t('label_model_type', lang)}**: `{type(model_obj).__name__}`")

        # Show pipeline steps if it's a Pipeline
        if hasattr(model_obj, "steps"):
            st.markdown(f"**{t('label_pipeline_steps', lang)}**")
            for i, (name, step) in enumerate(model_obj.steps):
                st.markdown(f"  {i+1}. `{name}` -> `{type(step).__name__}`")
        elif hasattr(model_obj, "get_params"):
            st.markdown(f"**{t('label_parameters', lang)}**")
            params = model_obj.get_params()
            params_df = pd.DataFrame(
                [{t("label_parameter", lang): k, t("label_value", lang): str(v)} for k, v in params.items()]
            )
            st.dataframe(params_df, use_container_width=True, height=300)

        st.info(t("msg_use_model_predict", lang))

# ===================================================================
# Page 7 : Save / Export
# ===================================================================
elif page == t("nav_save_export", lang):
    st.header(t("nav_save_export", lang))

    # --- Edited data export ---
    st.subheader(t("heading_export_data", lang))
    if st.session_state.df is not None:
        col1, col2 = st.columns(2)
        with col1:
            csv_data = st.session_state.df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                t("btn_download_data_csv", lang),
                csv_data,
                file_name="data_export.csv",
                mime="text/csv",
            )
        with col2:
            xlsx_data = to_excel_bytes(st.session_state.df)
            st.download_button(
                t("btn_download_data_excel", lang),
                xlsx_data,
                file_name="data_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info(t("msg_no_data_loaded", lang))

    # --- Prediction results export ---
    st.subheader(t("heading_export_predictions", lang))
    if st.session_state.predictions is not None:
        col1, col2 = st.columns(2)
        with col1:
            csv_data = st.session_state.predictions.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                t("btn_download_pred_csv", lang),
                csv_data,
                file_name="predictions.csv",
                mime="text/csv",
            )
        with col2:
            xlsx_data = to_excel_bytes(st.session_state.predictions)
            st.download_button(
                t("btn_download_pred_excel", lang),
                xlsx_data,
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info(t("msg_no_predictions", lang))

    # --- Model export ---
    st.subheader(t("heading_save_model", lang))
    if st.session_state.best_model is not None:
        model_name = st.text_input(t("label_model_filename", lang), value="best_model")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("btn_save_best", lang)):
                try:
                    mod = get_module(st.session_state.pycaret_task)
                    mod.save_model(st.session_state.best_model, f"/app/models/{model_name}")
                    st.success(t("msg_model_saved", lang).format(name=model_name))
                except Exception as e:
                    st.error(t("msg_save_error", lang).format(e=e))

        with col2:
            if st.session_state.tuned_model is not None:
                if st.button(t("btn_save_tuned", lang)):
                    try:
                        mod = get_module(st.session_state.pycaret_task)
                        mod.save_model(st.session_state.tuned_model, f"/app/models/{model_name}_tuned")
                        st.success(t("msg_tuned_model_saved", lang).format(name=model_name))
                    except Exception as e:
                        st.error(t("msg_save_error", lang).format(e=e))

        # Download model as pickle
        st.subheader(t("heading_download_model", lang))
        use_model = st.session_state.tuned_model or st.session_state.best_model
        buf = io.BytesIO()
        pickle.dump(use_model, buf)
        buf.seek(0)
        st.download_button(
            t("btn_download_model_pkl", lang),
            buf.getvalue(),
            file_name=f"{model_name}.pkl",
            mime="application/octet-stream",
        )
    else:
        st.info(t("msg_no_trained_model", lang))

    # --- Model file management ---
    st.markdown("---")
    st.subheader(t("heading_model_management", lang))
    st.markdown(t("msg_model_dir_desc", lang))

    _model_dir = "/app/models"
    if os.path.isdir(_model_dir):
        _model_files = sorted(
            [f for f in os.listdir(_model_dir) if os.path.isfile(os.path.join(_model_dir, f))],
        )
    else:
        _model_files = []

    if not _model_files:
        st.info(t("msg_no_saved_models", lang))
    else:
        # Initialise per-file session keys once
        if "model_mgmt_confirm_delete" not in st.session_state:
            st.session_state.model_mgmt_confirm_delete = {}

        st.markdown(f"**{t('msg_n_files', lang).format(n=len(_model_files))}**")

        for _mf in _model_files:
            _full = os.path.join(_model_dir, _mf)
            _size_kb = os.path.getsize(_full) / 1024
            _size_str = f"{_size_kb:.1f} KB" if _size_kb < 1024 else f"{_size_kb/1024:.2f} MB"

            with st.expander(f"{_mf} ({_size_str})", expanded=False):
                col_r1, col_r2 = st.columns(2)

                # --- Rename ---
                with col_r1:
                    _base, _ext = os.path.splitext(_mf)
                    new_name = st.text_input(
                        t("label_new_filename", lang),
                        value=_base,
                        key=f"rename_{_mf}",
                    )
                    if st.button(t("btn_rename", lang), key=f"rename_btn_{_mf}"):
                        new_full_name = new_name + _ext
                        new_full_path = os.path.join(_model_dir, new_full_name)
                        if new_full_name == _mf:
                            st.warning(t("msg_same_name", lang))
                        elif os.path.exists(new_full_path):
                            st.error(f"`{new_full_name}` {t('msg_already_exists', lang)}")
                        elif not new_name.strip():
                            st.error(t("msg_enter_filename", lang))
                        else:
                            try:
                                os.rename(_full, new_full_path)
                                st.success(f"`{_mf}` -> `{new_full_name}` {t('msg_renamed', lang)}")
                                st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
                            except Exception as _e:
                                st.error(t("msg_rename_error", lang).format(e=_e))

                # --- Delete ---
                with col_r2:
                    _del_key = f"del_{_mf}"
                    if st.session_state.model_mgmt_confirm_delete.get(_mf, False):
                        st.warning(f"`{_mf}` {t('msg_confirm_delete', lang)}")
                        col_y, col_n = st.columns(2)
                        with col_y:
                            if st.button(t("btn_yes_delete", lang), key=f"del_yes_{_mf}"):
                                try:
                                    os.remove(_full)
                                    st.session_state.model_mgmt_confirm_delete[_mf] = False
                                    st.success(f"`{_mf}` {t('msg_deleted', lang)}")
                                    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
                                except Exception as _e:
                                    st.error(t("msg_delete_error", lang).format(e=_e))
                        with col_n:
                            if st.button(t("btn_cancel", lang), key=f"del_no_{_mf}"):
                                st.session_state.model_mgmt_confirm_delete[_mf] = False
                                st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
                    else:
                        if st.button(t("btn_delete", lang), key=_del_key):
                            st.session_state.model_mgmt_confirm_delete[_mf] = True
                            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

                # --- Download ---
                with open(_full, "rb") as _fp:
                    st.download_button(
                        t("btn_download", lang),
                        _fp.read(),
                        file_name=_mf,
                        mime="application/octet-stream",
                        key=f"dl_{_mf}",
                    )

# ===================================================================
# Footer
# ===================================================================
st.markdown("---")
st.markdown(
    "**WebUI for PyCaret** -- Built on `pycaret/full` -- Powered by Streamlit",
    unsafe_allow_html=False,
)
