import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_treatment_questions(cmid_path, output_path):
    with open(cmid_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted = []
    for item in data:
        label_36 = str(item.get("label_36class", []))
        label_4 = str(item.get("label_4class", []))
        if "治疗方法" in label_36 or "治疗方案" in label_4:
            extracted.append(
                {
                    "question": item.get("originalText", ""),
                    "generated_answer": "",
                    "label": "",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)
    print(f"已提取 {len(extracted)} 条治疗相关问题到 {output_path}")
    return extracted


def truncate_to_3_sentences(text):
    sentences = []
    buffer = ""
    for char in text.strip():
        buffer += char
        if char in "。！？!?":
            sentences.append(buffer.strip())
            buffer = ""
            if len(sentences) >= 3:
                break
    if buffer and len(sentences) < 3:
        sentences.append(buffer.strip())
    return " ".join(sentences[:3]).strip()


def siliconflow_answer(question, api_key, model):
    system_prompt = (
        "你是一名专业的医学健康顾问，任务是根据用户问题提供谨慎、通用、非诊断性的健康建议。"
        "规则：不做诊断；不推荐具体药物剂量或用法；不承诺治愈或疗效；"
        "若症状持续或加重，必须建议及时就医或咨询专业医生。"
        "若涉及胸痛、大出血、意识不清等急症，建议立即拨打急救电话。"
        "回答控制在2到3句话内。"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 256,
        "stream": False,
    }
    req = urllib.request.Request(
        url="https://api.siliconflow.cn/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    content = obj.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not content:
        raise ValueError("empty model response")
    return truncate_to_3_sentences(content)


def generate_answers(extracted_path, output_path, model, max_items, sleep_seconds):
    with open(extracted_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        print("未检测到环境变量 SILICONFLOW_API_KEY，跳过回答生成。")
        return []

    limit = max_items if max_items and max_items > 0 else len(items)
    results = []
    for idx, item in enumerate(items[:limit], start=1):
        question = item.get("question", "").strip()
        if not question:
            continue
        try:
            answer = siliconflow_answer(question, api_key=api_key, model=model)
        except Exception:
            answer = "建议根据一般健康原则观察与调整。若症状持续或加重，请及时就医或咨询专业医生。"
        results.append({"question": question, "answer": answer})
        print(f"[{idx}/{limit}] 已生成: {question}")
        time.sleep(sleep_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已生成 {len(results)} 条回答到 {output_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Build the treatment QA knowledge base.")
    parser.add_argument("--cmid-path", default=str(PROJECT_ROOT / "data" / "raw" / "CMID.json"))
    parser.add_argument("--extracted-path", default=str(PROJECT_ROOT / "data" / "processed" / "extracted_treatment_questions.json"))
    parser.add_argument("--output-path", default=str(PROJECT_ROOT / "data" / "kb" / "generated_answers.json"))
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--generate", action="store_true", help="Call the LLM API to generate answers.")
    return parser.parse_args()


def main():
    args = parse_args()
    extracted_path = Path(args.extracted_path)
    extract_treatment_questions(Path(args.cmid_path), extracted_path)
    if args.generate:
        generate_answers(
            extracted_path=extracted_path,
            output_path=Path(args.output_path),
            model=args.model,
            max_items=args.max_items,
            sleep_seconds=args.sleep_seconds,
        )


if __name__ == "__main__":
    main()
