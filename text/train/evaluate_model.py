import json
import os
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def load_test_data(file_path):
    """加载测试数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_hidden_ad_model(model_path):
    """加载隐性广告检测模型"""
    import paddle
    import paddle.nn as nn
    from paddlenlp.transformers import ErnieModel, ErnieTokenizer

    class HiddenAdClassifier(nn.Layer):
        def __init__(self, pretrained_name: str, num_classes: int = 3, dropout_prob: float = 0.1):
            super().__init__()
            self.ernie = ErnieModel.from_pretrained(pretrained_name)
            self.dropout = nn.Dropout(dropout_prob)
            hidden_size = 768
            self.label_classifier = nn.Linear(hidden_size, num_classes)
            self.confidence_regressor = nn.Linear(hidden_size, 1)
            self.weight_predictor = nn.Linear(hidden_size, 6)
            self.activation = nn.Sigmoid()

        def forward(self, input_ids, token_type_ids):
            outputs = self.ernie(input_ids=input_ids, token_type_ids=token_type_ids)
            pooled = outputs[1]
            pooled = self.dropout(pooled)
            label_logits = self.label_classifier(pooled)
            confidence = self.activation(self.confidence_regressor(pooled))
            keyword_weights = self.activation(self.weight_predictor(pooled))
            return label_logits, confidence, keyword_weights

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    tokenizer = ErnieTokenizer.from_pretrained(model_path)
    model = HiddenAdClassifier(config_data["pretrained_model"], num_classes=3)
    model_state = paddle.load(os.path.join(model_path, "model_state.pdparams"))
    model.set_state_dict(model_state)
    model.eval()

    return model, tokenizer, config_data


def predict_hidden_ad(text, model, tokenizer, max_seq_length=256):
    """对单条文本进行预测"""
    import paddle
    
    device = paddle.get_device()
    if paddle.is_compiled_with_cuda():
        device = 'gpu:0'
    else:
        device = 'cpu'
    paddle.set_device(device)

    encoded = tokenizer(
        text,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='pd'
    )

    input_ids = encoded['input_ids'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device)

    with paddle.no_grad():
        label_logits, confidence, keyword_weights = model(input_ids, token_type_ids)

    predicted_label = int(paddle.argmax(label_logits, axis=1).item())
    return predicted_label


def evaluate_hidden_ad_model(model_path, test_data_path):
    """评测隐性广告检测模型"""
    print(f"正在加载测试数据: {test_data_path}")
    test_data = load_test_data(test_data_path)
    print(f"测试样本数量: {len(test_data)}")

    print(f"\n正在加载模型: {model_path}")
    model, tokenizer, config_data = load_hidden_ad_model(model_path)
    print(f"模型配置: {config_data['pretrained_model']}")
    print(f"最大序列长度: {config_data['max_seq_length']}")

    print("\n正在进行预测...")
    y_true = []
    y_pred = []
    category_counts = {'normal': 0, 'ad': 0, 'hidden_ad': 0}
    correct_counts = {'normal': 0, 'ad': 0, 'hidden_ad': 0}

    for idx, sample in enumerate(test_data):
        if idx % 100 == 0:
            print(f"已处理 {idx}/{len(test_data)} 条...")

        true_label = sample.get('label', sample.get('original_label'))
        if true_label is None:
            continue
            
        text = sample['text']
        category = sample['category']
        
        pred_label = predict_hidden_ad(text, model, tokenizer, config_data['max_seq_length'])
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        category_counts[category] += 1
        
        if pred_label == true_label:
            correct_counts[category] += 1

    print("\n" + "=" * 70)
    print("模型评测报告")
    print("=" * 70)

    print("\n1. 整体指标:")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"准确率 (Accuracy): {accuracy * 100:.2f}%")
    print(f"精确率 (Precision): {precision * 100:.2f}%")
    print(f"召回率 (Recall): {recall * 100:.2f}%")
    print(f"F1值 (F1-Score): {f1 * 100:.2f}%")

    print("\n2. 每类指标:")
    labels = ['normal', 'ad', 'hidden_ad']
    label_names = {'normal': '正常文本', 'ad': '广告', 'hidden_ad': '隐性广告'}
    
    for i, label in enumerate(labels):
        mask = np.array(y_true) == i
        if np.sum(mask) == 0:
            continue
        pred_for_label = np.array(y_pred)[mask]
        correct = np.sum(pred_for_label == i)
        total = np.sum(mask)
        
        print(f"\n{label_names[label]} ({label}):")
        print(f"  样本数: {total}")
        print(f"  正确数: {correct}")
        print(f"  正确率: {correct / total * 100:.2f}%")

    print("\n3. 详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=[label_names[l] for l in labels]))

    print("\n4. 混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"            预测标签")
    print(f"            {label_names['normal']}  {label_names['ad']}  {label_names['hidden_ad']}")
    print(f"真实标签")
    for i, label in enumerate(labels):
        print(f"  {label_names[label]}    {cm[i][0]:>6}    {cm[i][1]:>6}    {cm[i][2]:>10}")

    print("\n5. 各类别统计:")
    total_samples = sum(category_counts.values())
    for category, count in category_counts.items():
        accuracy = correct_counts[category] / count * 100 if count > 0 else 0
        print(f"  {label_names[category]}: {count} 样本 ({count/total_samples*100:.1f}%) - 正确率: {accuracy:.2f}%")

    print("\n" + "=" * 70)

    results = {
        'model_path': model_path,
        'test_data_path': test_data_path,
        'total_samples': len(y_true),
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'class_metrics': classification_report(y_true, y_pred, target_names=[label_names[l] for l in labels], output_dict=True),
        'confusion_matrix': cm.tolist(),
        'category_counts': category_counts,
        'correct_counts': correct_counts
    }

    return results


def load_echo_model(model_path):
    """加载回声指数模型"""
    import paddle
    import paddle.nn as nn
    from paddlenlp.transformers import ErnieModel, ErnieTokenizer

    class EchoRegressor(nn.Layer):
        def __init__(self, pretrained_name: str, dropout_prob: float = 0.1):
            super().__init__()
            self.ernie = ErnieModel.from_pretrained(pretrained_name)
            self.dropout = nn.Dropout(dropout_prob)
            self.regressor = nn.Linear(768, 5)
            self.activation = nn.Sigmoid()

        def forward(self, input_ids, token_type_ids):
            outputs = self.ernie(input_ids=input_ids, token_type_ids=token_type_ids)
            pooled = outputs[1]
            pooled = self.dropout(pooled)
            return self.activation(self.regressor(pooled))

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    tokenizer = ErnieTokenizer.from_pretrained(model_path)
    model = EchoRegressor(config_data["pretrained_model"])
    model_state = paddle.load(os.path.join(model_path, "model_state.pdparams"))
    model.set_state_dict(model_state)
    model.eval()

    return model, tokenizer, config_data


def predict_echo(text, model, tokenizer, max_seq_length=128):
    """预测回声指数相关特征"""
    import paddle
    
    encoded = tokenizer(text, max_seq_len=max_seq_length)
    input_ids = paddle.to_tensor([encoded["input_ids"]])
    token_type_ids = paddle.to_tensor([encoded["token_type_ids"]])

    with paddle.no_grad():
        output = model(input_ids, token_type_ids)

    predictions = output.numpy()[0].tolist()
    return predictions


def evaluate_echo_model(model_path, test_data_path):
    """评测回声指数模型"""
    print(f"正在加载测试数据: {test_data_path}")
    test_data = load_test_data(test_data_path)
    print(f"测试样本数量: {len(test_data)}")

    print(f"\n正在加载模型: {model_path}")
    model, tokenizer, config_data = load_echo_model(model_path)
    print(f"模型配置: {config_data['pretrained_model']}")
    print(f"最大序列长度: {config_data['max_seq_length']}")
    print(f"输出标签: {config_data['label_names']}")

    print("\n正在进行预测...")
    all_preds = []
    all_labels = config_data['label_names']

    for idx, sample in enumerate(test_data):
        if idx % 100 == 0:
            print(f"已处理 {idx}/{len(test_data)} 条...")

        text = sample['text']
        preds = predict_echo(text, model, tokenizer, config_data['max_seq_length'])
        all_preds.append(preds)

    print("\n" + "=" * 70)
    print("回声指数模型评测报告")
    print("=" * 70)

    print("\n1. 预测结果统计:")
    preds_array = np.array(all_preds)
    for i, label in enumerate(all_labels):
        mean_val = np.mean(preds_array[:, i])
        std_val = np.std(preds_array[:, i])
        min_val = np.min(preds_array[:, i])
        max_val = np.max(preds_array[:, i])
        
        print(f"\n{label}:")
        print(f"  均值: {mean_val:.4f}")
        print(f"  标准差: {std_val:.4f}")
        print(f"  最小值: {min_val:.4f}")
        print(f"  最大值: {max_val:.4f}")

    print("\n2. 特征相关性矩阵:")
    corr_matrix = np.corrcoef(preds_array.T)
    print(" " * 15 + " ".join([f"{l:>12}" for l in all_labels]))
    for i, label in enumerate(all_labels):
        row = [f"{corr_matrix[i][j]:.4f}" for j in range(len(all_labels))]
        print(f"{label:>12}  " + "  ".join(row))

    print("\n" + "=" * 70)

    results = {
        'model_path': model_path,
        'test_data_path': test_data_path,
        'total_samples': len(test_data),
        'label_names': all_labels,
        'statistics': {
            label: {
                'mean': float(np.mean(preds_array[:, i])),
                'std': float(np.std(preds_array[:, i])),
                'min': float(np.min(preds_array[:, i])),
                'max': float(np.max(preds_array[:, i]))
            } for i, label in enumerate(all_labels)
        },
        'correlation_matrix': corr_matrix.tolist()
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='模型评测脚本')
    parser.add_argument('--model_type', type=str, choices=['hidden_ad', 'echo'], 
                        default='hidden_ad', help='模型类型')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径')
    parser.add_argument('--test_data', type=str, default=None, help='测试数据路径')
    parser.add_argument('--output', type=str, default=None, help='输出结果文件路径')
    
    args = parser.parse_args()

    if args.model_type == 'hidden_ad':
        model_path = args.model_path or './hidden_ad_model'
        test_data_path = args.test_data or './processed_data/test_data.json'
        
        if not os.path.exists(model_path):
            model_path = './hidden_ad_model'
        
        if not os.path.exists(test_data_path):
            test_data_path = './processed_data/ernie_test_data.json'
            
        results = evaluate_hidden_ad_model(model_path, test_data_path)
        
    elif args.model_type == 'echo':
        model_path = args.model_path or './erine_model/best_model'
        test_data_path = args.test_data or './processed_data/ernie_test_data.json'
        
        if not os.path.exists(test_data_path):
            test_data_path = './processed_data/test_data.json'
            
        results = evaluate_echo_model(model_path, test_data_path)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n评测结果已保存到: {args.output}")


if __name__ == "__main__":
    main()