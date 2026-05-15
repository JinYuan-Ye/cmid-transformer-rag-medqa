import json

import torch
from torch.utils.data import Dataset


LABEL_4_MAP = {
    "病症": 0,
    "药物": 1,
    "治疗方案": 2,
    "其他": 3,
}


LABEL_ALIASES = {
    "临床表现(病症表现)": "临床表现",
    "病症禁忌": "禁忌",
    "诱因": "病因",
    "手术时间": "治疗时间",
    "功效": "作用",
}


INTENT_TEMPLATES = {
    "定义": "查询 {entity} 的定义及医学概述。",
    "病因": "查询导致 {entity} 的病因及发病机制。",
    "临床表现": "查询 {entity} 的典型临床症状及表现。",
    "相关病症": "查询与 {entity} 相关的并发症或关联疾病。",
    "治疗方法": "查询 {entity} 的标准治疗方案及手段。",
    "推荐医院": "查询治疗 {entity} 的推荐医院或权威科室。",
    "预防": "查询 {entity} 的日常预防措施及护理建议。",
    "所属科室": "查询确诊或治疗 {entity} 应挂号的科室。",
    "禁忌": "查询患有 {entity} 时的饮食及行为禁忌。",
    "传染性": "查询 {entity} 的传播途径及传染性说明。",
    "治愈率": "查询 {entity} 的治愈率及预后情况。",
    "严重性": "查询 {entity} 的严重程度及对身体的危害。",
    "作用": "查询药物 {entity} 的药理作用及主要功效。",
    "适用症": "查询药物 {entity} 的适应症及适用人群。",
    "价钱": "查询购买药物 {entity} 的市场参考价格。",
    "药物禁忌": "查询服用药物 {entity} 的禁忌症及注意事项。",
    "用法": "查询药物 {entity} 的标准用法用量说明。",
    "副作用": "查询服用药物 {entity} 可能产生的不良反应及副作用。",
    "成分": "查询药物 {entity} 的主要成分及化学构成。",
    "方法": "查询 {entity} 的具体操作步骤及实施方法。",
    "费用": "查询进行 {entity} 所需的医疗费用标准。",
    "有效时间": "查询 {entity} 治疗后的有效维持时间。",
    "临床意义/检查目的": "查询进行 {entity} 检查的临床意义及目的。",
    "治疗时间": "查询进行 {entity} 治疗所需的疗程时长。",
    "疗效": "查询 {entity} 的预期治疗效果及成功率。",
    "恢复时间": "查询 {entity} 治疗后的恢复期时长。",
    "正常指标": "查询 {entity} 检查项目的正常参考值范围。",
    "化验/体检方案": "查询 {entity} 相关的化验流程及体检项目。",
    "恢复": "查询 {entity} 治疗后的康复指导及注意事项。",
    "设备用法": "查询医疗设备 {entity} 的正确操作及使用方法。",
    "多问": "查询关于 {entity} 的综合医疗信息解答。",
    "养生": "查询与 {entity} 相关的养生保健知识。",
    "整容": "查询与 {entity} 相关的整形美容建议及风险。",
    "两性": "查询与 {entity} 相关的性健康及生殖知识。",
    "对比": "查询 {entity} 与其他同类方案的对比分析。",
    "无法确定": "查询 {entity} 的相关医疗信息。",
}


class CMIDDataset(Dataset):
    """CMID dataset with classification labels and a template-based rewrite target."""

    def __init__(self, json_file, tokenizer, max_len=128):
        self.data = self.load_data(json_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_4_map = LABEL_4_MAP
        self.label_36_map = self.get_label_36_map()

    @staticmethod
    def load_data(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_label_36_map(self):
        labels = sorted({item["label_36class"][0].strip("'") for item in self.data})
        return {label: i for i, label in enumerate(labels)}

    @staticmethod
    def _main_entity(text, entities):
        if not entities:
            return "该问题"
        entity = entities[0]
        start, end = entity["start_pos"], entity["end_pos"]
        return text[start:end] or "该问题"

    def generate_target(self, text, entities, label_36):
        entity_text = self._main_entity(text, entities)
        template_key = LABEL_ALIASES.get(label_36, label_36)
        template = INTENT_TEMPLATES.get(template_key, INTENT_TEMPLATES["无法确定"])
        return template.format(entity=entity_text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["originalText"]
        label_4 = item["label_4class"][0].strip("'")
        label_36 = item["label_36class"][0].strip("'")
        target_text = self.generate_target(text, item["entities"], label_36)

        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels_4": torch.tensor(self.label_4_map.get(label_4, 0), dtype=torch.long),
            "labels_36": torch.tensor(self.label_36_map.get(label_36, 0), dtype=torch.long),
            "target_ids": targets["input_ids"].squeeze(0),
            "target_text": target_text,
            "original_text": text,
        }
