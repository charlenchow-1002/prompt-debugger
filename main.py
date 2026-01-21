import io
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st

# 页面配置
st.set_page_config(
    page_title="Charlene的模型调试平台",
    page_icon="🧪",
    layout="wide"
)

# 初始化 session state
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "prompt_results" not in st.session_state:
    st.session_state.prompt_results = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# 模型选项（基础）
BASE_MODEL_KEYS = {
    "doubao": "doubao_openai_compat",
    "gemini": "gemini"
}

# 可选任务（用于输出展示）
TASKS = {
    "新answer": "new_answer",
    "长度(字符数)": "length",
    "与原始answer相似度": "similarity",
    "是否包含RAG关键词": "rag_hit"
}


def safe_text(value: str) -> str:
    return "" if value is None else str(value)


def parse_excel(uploaded_file: io.BytesIO) -> pd.DataFrame:
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as exc:
        st.error(f"Excel 解析失败: {exc}")
        return pd.DataFrame()


def normalize_inputs(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["session_id", "message_id", "query", "rag", "raw_answer"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    return df[required_cols].fillna("")


def build_prompt(row: Dict[str, str], template: str) -> str:
    return template.format(
        session_id=safe_text(row.get("session_id")),
        message_id=safe_text(row.get("message_id")),
        query=safe_text(row.get("query")),
        rag=safe_text(row.get("rag")),
        raw_answer=safe_text(row.get("raw_answer"))
    )


def call_doubao_openai_compat(
    prompt: str,
    api_key: str,
    base_url: str,
    model_name: str,
    timeout_s: int
) -> str:
    if not api_key:
        return "❌ 缺少豆包 API Key"
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"❌ 豆包调用失败: {exc}"


def call_gemini(
    prompt: str,
    api_key: str,
    model_name: str,
    base_url: str,
    timeout_s: int
) -> str:
    if not api_key:
        return "❌ 缺少 Gemini API Key"
    api_base = base_url.strip() or "https://generativelanguage.googleapis.com"
    endpoint = (
        f"{api_base.rstrip('/')}/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        return f"❌ Gemini 调用失败: {exc}"


def mock_call(model_name: str, prompt: str) -> str:
    time.sleep(0.4)
    return f"模拟 {model_name} 返回：{prompt[:80]}..."


def run_model_call(
    model_key: str,
    prompt: str,
    configs: Dict[str, str],
    use_mock: bool,
    timeout_s: int
) -> str:
    if use_mock:
        return mock_call(model_key, prompt)
    if model_key == "doubao_openai_compat":
        return call_doubao_openai_compat(
            prompt=prompt,
            api_key=configs.get("doubao_api_key", ""),
            base_url=configs.get("doubao_base_url", ""),
            model_name=configs.get("doubao_model", ""),
            timeout_s=timeout_s
        )
    if model_key == "custom_openai_compat":
        return call_doubao_openai_compat(
            prompt=prompt,
            api_key=configs.get("custom_api_key", ""),
            base_url=configs.get("custom_base_url", ""),
            model_name=configs.get("custom_model", ""),
            timeout_s=timeout_s
        )
    if model_key == "gemini":
        return call_gemini(
            prompt=prompt,
            api_key=configs.get("gemini_api_key", ""),
            model_name=configs.get("gemini_model", ""),
            base_url=configs.get("gemini_base_url", ""),
            timeout_s=timeout_s
        )
    return "❌ 未知模型"


def get_model_key_and_configs(
    model_display: str,
    base_configs: Dict[str, str],
    available_models: Dict[str, str],
    custom_configs: Dict[str, Dict[str, str]]
) -> (str, Dict[str, str]):
    model_key = available_models.get(model_display, "custom_openai_compat")
    run_configs = base_configs
    if model_key == "custom_openai_compat":
        custom_cfg = custom_configs.get(model_display, {})
        run_configs = {
            **base_configs,
            "custom_api_key": custom_cfg.get("api_key", ""),
            "custom_base_url": custom_cfg.get("base_url", ""),
            "custom_model": custom_cfg.get("model", "")
        }
    return model_key, run_configs


def calc_tasks(new_answer: str, row: Dict[str, str], task_keys: List[str]) -> Dict[str, str]:
    results = {}
    if "length" in task_keys:
        results["长度(字符数)"] = str(len(new_answer))
    if "similarity" in task_keys:
        raw_answer = safe_text(row.get("raw_answer"))
        if not raw_answer:
            results["与原始answer相似度"] = "N/A"
        else:
            ratio = SequenceMatcher(None, raw_answer, new_answer).ratio()
            results["与原始answer相似度"] = f"{ratio:.2f}"
    if "rag_hit" in task_keys:
        rag = safe_text(row.get("rag"))
        if not rag:
            results["是否包含RAG关键词"] = "N/A"
        else:
            keywords = [w for w in re.split(r"[\s,，;；]+", rag) if w.strip()]
            keywords = keywords[:5]
            hit = any(k in new_answer for k in keywords)
            results["是否包含RAG关键词"] = "是" if hit else "否"
    return results


def pick_random_eval_session() -> None:
    sessions = st.session_state.get("eval_session_options", [])
    if sessions:
        st.session_state.eval_session_pick = random.choice(sessions)


# 侧边栏：模型配置与步骤提示
with st.sidebar:
    st.header("🔧 模型配置")
    use_mock = st.checkbox("没有 Key 时使用模拟模式", value=True)
    timeout_s = st.number_input("请求超时(秒)", min_value=5, max_value=120, value=30, step=5)
    concurrency = st.number_input("并发数", min_value=1, max_value=20, value=1, step=1)

    st.subheader("豆包")
    if "doubao_display_name" not in st.session_state:
        st.session_state.doubao_display_name = "豆包"
    st.session_state.doubao_display_name = st.text_input(
        "豆包 显示名",
        value=st.session_state.doubao_display_name
    )
    doubao_api_key = st.text_input("豆包 API Key", type="password")
    doubao_base_url = st.text_input(
        "豆包 Base URL",
        value="https://ark.cn-beijing.volces.com/api",
        help="示例: https://ark.cn-beijing.volces.com/api"
    )
    doubao_model = st.text_input(
        "豆包模型名",
        value="doubao-seed-1-6-250615",
        help="示例: doubao-seed-1-6-250615"
    )

    st.subheader("Gemini")
    if "gemini_display_name" not in st.session_state:
        st.session_state.gemini_display_name = "Gemini"
    st.session_state.gemini_display_name = st.text_input(
        "Gemini 显示名",
        value=st.session_state.gemini_display_name
    )
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    gemini_base_url = st.text_input(
        "Gemini Base URL",
        value="https://generativelanguage.googleapis.com",
        help="示例: https://generativelanguage.googleapis.com"
    )
    gemini_model = st.text_input(
        "Gemini 模型名",
        value="gemini-1.5-pro",
        help="示例: gemini-1.5-pro"
    )

    st.subheader("更多模型")
    if "custom_models" not in st.session_state:
        st.session_state.custom_models = []
    if st.button("➕ 添加模型"):
        st.session_state.custom_models.append(
            {
                "name": "",
                "api_key": "",
                "base_url": "",
                "model": ""
            }
        )
    custom_configs = {}
    custom_model_labels = []
    for idx, cfg in enumerate(st.session_state.custom_models):
        st.markdown(f"**自定义模型 {idx + 1}**")
        name = st.text_input("模型显示名", value=cfg.get("name", ""), key=f"custom_name_{idx}")
        api_key = st.text_input("API Key", value=cfg.get("api_key", ""), type="password", key=f"custom_key_{idx}")
        base_url = st.text_input("Base URL", value=cfg.get("base_url", ""), key=f"custom_base_{idx}")
        model = st.text_input("模型名", value=cfg.get("model", ""), key=f"custom_model_{idx}")
        if st.button("删除该模型", key=f"custom_delete_{idx}"):
            st.session_state.custom_models.pop(idx)
            st.rerun()
        st.divider()
        if name.strip():
            label = f"{name.strip()}（自定义）"
            custom_model_labels.append(label)
            custom_configs[label] = {
                "api_key": api_key,
                "base_url": base_url,
                "model": model
            }

    with st.expander("评测 Prompt 模板（高级设置）", expanded=False):
        prompt_template = st.text_area(
            "Prompt 模板（支持 {query} {rag} {raw_answer} 等变量）",
            value=(
                "你是一个专业助手。\n"
                "用户问题：{query}\n"
                "检索信息：{rag}\n"
                "原始回答：{raw_answer}\n"
                "请输出更好的回答。"
            ),
            height=160
        )
        st.caption("可用变量：{session_id}、{message_id}、{query}、{rag}、{raw_answer}")

    st.divider()
    st.markdown(
        "### ✅ 模型调用步骤\n"
        "1. 在平台/控制台创建 API Key\n"
        "2. 把 Key 填入上方对应输入框\n"
        "3. 填好 Base URL 与模型名\n"
        "4. 关闭“模拟模式”，开始调用\n"
    )


st.title("🧪 Charlene的模型调试平台")
st.caption("支持多模型、多任务、Excel 导入与导出，并包含 Prompt 调试区。")
st.markdown("---")


tab_input, tab_output = st.tabs(["评测输入与调试", "输出结果"])


with tab_input:
    available_models = {
        st.session_state.doubao_display_name: BASE_MODEL_KEYS["doubao"],
        st.session_state.gemini_display_name: BASE_MODEL_KEYS["gemini"]
    }
    model_options = list(available_models.keys()) + custom_model_labels
    st.subheader("① 筛选与任务选择")
    col_a, col_b = st.columns([2, 2])
    with col_a:
        selected_models = st.multiselect(
            "选择模型（支持多选）",
            options=model_options,
            default=list(available_models.keys()),
            key="selected_models"
        )
    with col_b:
        selected_tasks = st.multiselect(
            "选择任务（展示哪些结果字段）",
            options=list(TASKS.keys()),
            default=["新answer"],
            key="selected_tasks"
        )
        if "新answer" not in selected_tasks:
            selected_tasks = ["新answer"] + selected_tasks
            st.info("已自动包含“新answer”，避免输出结果为空。")

    st.markdown("---")
    st.subheader("② 输入数据（手动 / Excel）")

    input_mode = st.radio(
        "选择输入方式",
        options=["手动输入", "上传 Excel", "合并(手动 + Excel)"],
        horizontal=True
    )

    with st.expander("📌 字段说明"):
        st.markdown(
            "- `session_id`：会话编号\n"
            "- `message_id`：消息编号\n"
            "- `query`：用户问题（必填）\n"
            "- `rag`：检索/背景信息（可选）\n"
            "- `raw_answer`：原始回答（可选）"
        )

    manual_df = pd.DataFrame(
        [
            {"session_id": "", "message_id": "", "query": "", "rag": "", "raw_answer": ""}
            for _ in range(3)
        ]
    )
    st.markdown("**手动输入表**")
    manual_df = st.data_editor(
        manual_df,
        num_rows="dynamic",
        use_container_width=True,
        key="manual_input_table"
    )

    uploaded_excel = st.file_uploader(
        "上传 Excel（字段：session_id, message_id, query, rag, raw_answer）",
        type=["xlsx", "xls"]
    )
    upload_df = pd.DataFrame()
    if uploaded_excel is not None:
        upload_df = normalize_inputs(parse_excel(uploaded_excel))
        if not upload_df.empty:
            st.success(f"✅ 解析到 {len(upload_df)} 行数据")
            st.dataframe(upload_df, use_container_width=True, height=200)

    if input_mode == "手动输入":
        input_df = normalize_inputs(manual_df)
    elif input_mode == "上传 Excel":
        input_df = upload_df
    else:
        combined = pd.concat([manual_df, upload_df], ignore_index=True)
        input_df = normalize_inputs(combined)

    input_df = input_df[(input_df["query"].astype(str).str.strip() != "")]

    st.markdown("---")
    st.subheader("③ Prompt 调试（单个 / 多个）")
    st.markdown("可以手动添加多条 Prompt（点击“+”新增），并选择多个模型对比输出。")

    if "prompt_inputs" not in st.session_state:
        st.session_state.prompt_inputs = ["请简要介绍你自己。", "请总结一下 AI 的优缺点。"]

    add_col, _ = st.columns([1, 5])
    with add_col:
        if st.button("➕ 添加一个 Prompt"):
            st.session_state.prompt_inputs.append("")

    prompt_inputs: List[str] = []
    for idx in range(len(st.session_state.prompt_inputs)):
        row_col, del_col = st.columns([6, 1])
        with row_col:
            prompt_value = st.text_area(
                f"Prompt {idx + 1}",
                value=st.session_state.prompt_inputs[idx],
                height=90,
                key=f"prompt_input_{idx}"
            )
        with del_col:
            if st.button("删除", key=f"delete_prompt_{idx}"):
                st.session_state.prompt_inputs.pop(idx)
                st.rerun()

        prompt_inputs.append(prompt_value)

    st.session_state.prompt_inputs = prompt_inputs

    prompt_file = st.file_uploader(
        "上传 Prompt 文件（TXT/Excel）",
        type=["txt", "xlsx", "xls"],
        key="prompt_file_uploader"
    )

    file_prompts: List[Dict[str, str]] = []
    if prompt_file is not None:
        if prompt_file.name.endswith(".txt"):
            content = prompt_file.read().decode("utf-8")
            file_prompts = [
                {"prompt": line.strip(), "session_id": ""}
                for line in content.splitlines()
                if line.strip()
            ]
        else:
            df_prompts = parse_excel(prompt_file)
            if {"session_id", "prompt"}.issubset(set(df_prompts.columns)):
                for _, row in df_prompts.iterrows():
                    prompt_value = str(row.get("prompt", "")).strip()
                    if prompt_value:
                        file_prompts.append(
                            {"prompt": prompt_value, "session_id": str(row.get("session_id", "")).strip()}
                        )
            else:
                for col in df_prompts.columns:
                    file_prompts.extend(
                        [
                            {"prompt": str(p).strip(), "session_id": ""}
                            for p in df_prompts[col].dropna().tolist()
                            if str(p).strip()
                        ]
                    )

    manual_prompts = [
        {"prompt": p.strip(), "session_id": ""}
        for p in prompt_inputs
        if p.strip()
    ]
    seen = set()
    all_prompts: List[Dict[str, str]] = []
    for item in file_prompts + manual_prompts:
        key = (item.get("session_id", ""), item.get("prompt", ""))
        if key not in seen:
            seen.add(key)
            all_prompts.append(item)

    st.markdown("---")
    can_run_prompt = len(all_prompts) > 0
    can_run_eval = not input_df.empty

    if not can_run_eval:
        st.info("评测输入为空，将只运行 Prompt 调试。")
    if not can_run_prompt:
        st.info("Prompt 为空，将只运行评测。")

    run_all = st.button(
        "🚀 开始运行",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_running
        or len(selected_models) == 0
        or (not can_run_prompt and not can_run_eval)
    )

    if run_all:
        st.session_state.is_running = True
        configs = {
            "doubao_api_key": doubao_api_key,
            "doubao_base_url": doubao_base_url,
            "doubao_model": doubao_model,
            "gemini_api_key": gemini_api_key,
            "gemini_base_url": gemini_base_url,
            "gemini_model": gemini_model
        }

        if can_run_prompt:
            st.session_state.prompt_results = None
            total = max(1, len(all_prompts) * len(selected_models))
            current = 0
            progress = st.progress(0)
            status = st.empty()
            prompt_rows = []
            tasks = []
            for item in all_prompts:
                for model_display in selected_models:
                    tasks.append((item, model_display))

            def run_prompt_task(task):
                item, model_display = task
                model_key, run_configs = get_model_key_and_configs(
                    model_display, configs, available_models, custom_configs
                )
                answer = run_model_call(
                    model_key=model_key,
                    prompt=item["prompt"],
                    configs=run_configs,
                    use_mock=use_mock,
                    timeout_s=int(timeout_s)
                )
                return {
                    "session_id": item.get("session_id", ""),
                    "prompt": item["prompt"],
                    "model": model_display,
                    "answer": answer
                }

            if int(concurrency) <= 1:
                for task in tasks:
                    current += 1
                    status.text(f"🔄 Prompt 调试：{task[1]}（{current}/{total}）")
                    prompt_rows.append(run_prompt_task(task))
                    progress.progress(min(current / total, 1.0))
            else:
                with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
                    future_map = {executor.submit(run_prompt_task, task): task for task in tasks}
                    for future in as_completed(future_map):
                        current += 1
                        task = future_map[future]
                        status.text(f"🔄 Prompt 调试：{task[1]}（{current}/{total}）")
                        prompt_rows.append(future.result())
                        progress.progress(min(current / total, 1.0))
            st.session_state.prompt_results = pd.DataFrame(prompt_rows)

        if can_run_eval:
            st.session_state.eval_results = None
            task_keys = [TASKS[t] for t in selected_tasks]
            rows = input_df.to_dict(orient="records")
            total = max(1, len(rows) * len(selected_models))
            current = 0
            progress = st.progress(0)
            status = st.empty()
            output_rows = []
            tasks = []
            for row in rows:
                for model_display in selected_models:
                    tasks.append((row, model_display))

            def run_eval_task(task):
                row, model_display = task
                model_key, run_configs = get_model_key_and_configs(
                    model_display, configs, available_models, custom_configs
                )
                prompt = build_prompt(row, prompt_template)
                new_answer = run_model_call(
                    model_key=model_key,
                    prompt=prompt,
                    configs=run_configs,
                    use_mock=use_mock,
                    timeout_s=int(timeout_s)
                )
                task_result = calc_tasks(new_answer, row, task_keys)
                return {
                    "session_id": row.get("session_id", ""),
                    "message_id": row.get("message_id", ""),
                    "query": row.get("query", ""),
                    "rag": row.get("rag", ""),
                    "raw_answer": row.get("raw_answer", ""),
                    "model": model_display,
                    "new_answer": new_answer,
                    **task_result
                }

            if int(concurrency) <= 1:
                for task in tasks:
                    current += 1
                    status.text(f"🔄 评测：{task[1]}（{current}/{total}）")
                    output_rows.append(run_eval_task(task))
                    progress.progress(min(current / total, 1.0))
            else:
                with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
                    future_map = {executor.submit(run_eval_task, task): task for task in tasks}
                    for future in as_completed(future_map):
                        current += 1
                        task = future_map[future]
                        status.text(f"🔄 评测：{task[1]}（{current}/{total}）")
                        output_rows.append(future.result())
                        progress.progress(min(current / total, 1.0))
            st.session_state.eval_results = pd.DataFrame(output_rows)

        st.session_state.is_running = False
        st.success("✅ 运行完成")
        st.rerun()

with tab_output:
    st.subheader("评测输出结果")
    if st.session_state.eval_results is not None:
        eval_sessions = (
            st.session_state.eval_results["session_id"].astype(str).tolist()
            if "session_id" in st.session_state.eval_results.columns
            else []
        )
        eval_sessions = [s for s in eval_sessions if str(s).strip()]
        eval_unique = sorted(set(eval_sessions))
        st.session_state.eval_session_options = eval_unique
        st.markdown("### Sessions")
        eval_col_a, eval_col_b = st.columns([4, 1])
        with eval_col_a:
            eval_options = eval_unique if eval_unique else ["全部"]
            if "eval_session_pick" not in st.session_state or st.session_state.eval_session_pick not in eval_options:
                st.session_state.eval_session_pick = random.choice(eval_unique) if eval_unique else "全部"
            current_label = st.session_state.eval_session_pick
            st.selectbox(
                f"Select a session（当前：{current_label}）",
                options=eval_options,
                key="eval_session_pick"
            )
        with eval_col_b:
            st.button("Random session", disabled=len(eval_unique) == 0, on_click=pick_random_eval_session)

        base_cols = ["session_id", "message_id", "query", "rag", "raw_answer", "model"]
        selected_tasks = st.session_state.get("selected_tasks", ["新answer"])
        metric_cols = [t for t in selected_tasks if t != "新answer"]
        display_cols = base_cols + ["new_answer"] + metric_cols
        available_cols = [c for c in display_cols if c in st.session_state.eval_results.columns]
        display_df = st.session_state.eval_results.loc[:, available_cols]
        if st.session_state.eval_session_pick != "全部" and "session_id" in display_df.columns:
            display_df = display_df[display_df["session_id"].astype(str) == st.session_state.eval_session_pick]
        if display_df.empty:
            st.info("该 session 暂无评测结果。")
        else:
            group_keys = ["session_id", "message_id", "query", "raw_answer", "rag"]
            group_keys = [k for k in group_keys if k in display_df.columns]
            model_col = "model" if "model" in display_df.columns else None
            answer_col = "new_answer" if "new_answer" in display_df.columns else None
            metric_cols = [c for c in display_df.columns if c in metric_cols]

            if model_col and answer_col:
                if st.session_state.eval_session_pick != "全部":
                    session_fields = ["session_id", "message_id", "query", "rag", "raw_answer"]
                    session_fields = [c for c in session_fields if c in display_df.columns]
                    session_rows = display_df[session_fields].drop_duplicates(subset=session_fields)
                    st.subheader("原始输入数据")
                    st.dataframe(
                        session_rows,
                        use_container_width=True,
                        height=200
                    )

                for (session_id, message_id, query, raw_answer, rag), group in display_df.fillna("").groupby(
                    [c for c in group_keys]
                ):
                    st.markdown(f"### Conversation\nmessage_id {message_id}")
                    st.markdown("**Query**")
                    st.write(query)
                    st.markdown("**Original answer**")
                    st.write(raw_answer)

                    models = group[model_col].astype(str).tolist()
                    cols = st.columns(len(models)) if models else []
                    for idx, (_, row) in enumerate(group.iterrows()):
                        with cols[idx]:
                            st.markdown(f"**New answer: {row.get(model_col, '')}**")
                            st.write(row.get(answer_col, ""))
                            for m in metric_cols:
                                if m in row and row[m] != "":
                                    st.caption(f"{m}: {row[m]}")

                    with st.expander("RAG", expanded=False):
                        st.write(rag)
                    st.markdown("---")

            else:
                st.dataframe(display_df, use_container_width=True, height=400)

        output_buffer = io.BytesIO()
        display_df.to_excel(output_buffer, index=False)
        st.download_button(
            "📥 导出评测结果为 Excel",
            data=output_buffer.getvalue(),
            file_name="eval_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
            st.info("还没有评测结果，先去“评测输入与调试”里点击开始评测。")

    st.markdown("---")
    st.subheader("Prompt 调试输出结果")
    if st.session_state.prompt_results is not None:
        prompt_sessions = (
            st.session_state.prompt_results["session_id"].astype(str).tolist()
            if "session_id" in st.session_state.prompt_results.columns
            else []
        )
        prompt_sessions = [s for s in prompt_sessions if str(s).strip()]
        prompt_unique = sorted(set(prompt_sessions))

        if prompt_unique:
            prompt_col_a, prompt_col_b = st.columns([4, 1])
            with prompt_col_a:
                prompt_options = ["全部"] + prompt_unique
                if "prompt_session_pick" not in st.session_state:
                    st.session_state.prompt_session_pick = "全部"
                st.selectbox(
                    "选择 session_id",
                    options=prompt_options,
                    key="prompt_session_pick"
                )
            with prompt_col_b:
                if st.button("🎲 随机 session"):
                    st.session_state.prompt_session_pick = random.choice(prompt_unique)
                    st.rerun()

        prompt_df = st.session_state.prompt_results
        prompt_pick = st.session_state.get("prompt_session_pick", "全部")
        if prompt_pick != "全部" and "session_id" in prompt_df.columns:
            prompt_df = prompt_df[prompt_df["session_id"].astype(str) == prompt_pick]

        with st.expander("Prompt 调试输出结果", expanded=bool(prompt_unique)):
            st.dataframe(prompt_df, use_container_width=True, height=400)

            prompt_buffer = io.BytesIO()
            st.session_state.prompt_results.to_excel(prompt_buffer, index=False)
            st.download_button(
                "📥 导出 Prompt 结果为 Excel",
                data=prompt_buffer.getvalue(),
                file_name="prompt_debug_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("还没有 Prompt 调试结果，先去“评测输入与调试”里点击开始调试。")

