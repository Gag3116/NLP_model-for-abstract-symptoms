from flask import Flask, jsonify, request
from flask_cors import CORS  # 导入 CORS 支持
import os
import spacy
from spacy.matcher import PhraseMatcher

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 确保 spaCy 模型已安装
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 症状关键词列表
symptom_keywords = ["headache", "cough", "fever", "sore throat", "runny nose", "muscle pain"]

# 使用 PhraseMatcher 来匹配症状短语
matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp.make_doc(symptom) for symptom in symptom_keywords]
matcher.add("SYMPTOMS", patterns)


# 否定检测函数
def is_negated(token):
    negation_words = {"no", "not", "n't", "don't", "doesn't", "never", "without", "lack"}
    for ancestor in token.ancestors:
        if ancestor.lower_ in negation_words:
            return True
    for child in token.head.children:
        if child.dep_ == "neg" or child.lower_ in negation_words:
            return True
    return False


# 检查对比连词后的状态
def check_contrast_and_status(doc):
    contrast_words = {"but", "however"}
    fine_words = {"fine", "better", "well", "okay", "recovered"}
    for token in doc:
        if token.text.lower() in contrast_words:
            for descendant in token.subtree:
                if descendant.text.lower() in fine_words:
                    return True  # 对比句中状态良好，症状已消失
    return False


# 时态检测函数
def get_tense(token):
    verb = token.head
    if verb.tag_ in {"VBD", "VBN"}:
        return "past"
    elif verb.tag_ in {"VBZ", "VBP"}:
        return "present"
    elif verb.text in {"will", "going"}:
        return "future"
    return "unknown"


# 解析用户输入
def parse_input_function(user_input):
    doc = nlp(user_input.lower())
    detected_symptoms = []
    processed_tokens = set()
    resolved_to_fine = check_contrast_and_status(doc)  # 检查对比关系的状态

    # 使用 PhraseMatcher 匹配症状
    matches = matcher(doc)
    for match_id, start, end in matches:
        symptom = doc[start:end].text
        symptom_root = doc[start:end].root

        if symptom_root in processed_tokens:
            continue

        negated = is_negated(symptom_root)
        tense = get_tense(symptom_root)
        current = tense != "past"  # 默认逻辑

        # 如果对比句中说明症状已恢复，则覆盖当前状态
        if resolved_to_fine:
            current = False

        # 只添加当前且未否定的症状
        if current and not negated:
            detected_symptoms.append(symptom)

        processed_tokens.add(symptom_root)

    return detected_symptoms


# 健康检查接口
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


# 症状解析接口
@app.route('/parse_input', methods=['POST'])
def parse_input():
    # 验证请求是否包含 JSON 数据
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    user_input = data.get("input", "")

    if not user_input:
        return jsonify({"error": "Input field is empty"}), 400

    # 调用解析功能，仅返回症状列表
    symptoms = parse_input_function(user_input)
    return jsonify({"symptoms": symptoms}), 200


if __name__ == "__main__":
    # 使用环境变量 PORT 或默认端口 5001
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
