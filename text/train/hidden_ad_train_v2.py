"""
ERNIE3.0-ZH-medium 隐性广告检测训练程序
支持三大类别分类：正常文本(0)、广告(1)、隐性广告(2)
输出：文本类型 + 置信度 + 词语权重
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKLDNN_THREAD_COUNT'] = '1'

import json
from dataclasses import dataclass
from typing import List, Tuple, Dict
import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from env_config import get_config


config = get_config()
model_config = config.get_model_config()

model_name = model_config['model_name']
max_seq_length = model_config['max_seq_length']
batch_size = model_config['batch_size']
epochs = model_config['epochs']
learning_rate = model_config['learning_rate']
weight_decay = 0.01
early_stopping_patience = model_config.get('early_stopping_patience', 5)
num_workers = model_config.get('num_workers', os.cpu_count() // 2)


@dataclass
class ClassifiedSample:
    text: str
    label: int
    confidence: float
    keyword_weights: Dict[str, float]


class ClassifiedDataset(Dataset):
    def __init__(self, file_path: str):
        self.samples: List[ClassifiedSample] = []
        self._load_json(file_path)

    def _load_json(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        category_to_label = {'normal': 0, 'ad': 1, 'hidden_ad': 2}
        
        if isinstance(data, list):
            for obj in data:
                text = (obj.get("text") or "").strip()
                if not text or len(text) < 5:
                    continue
                
                label = obj.get("label")
                if label is None:
                    category = obj.get("category", "normal")
                    label = category_to_label.get(category, 0)
                else:
                    label = int(label)
                
                if label not in [0, 1, 2]:
                    label = 0
                
                confidence = float(obj.get("confidence", 0.5))
                keyword_weights = obj.get("keyword_weights", {})
                
                self.samples.append(ClassifiedSample(
                    text=text,
                    label=label,
                    confidence=confidence,
                    keyword_weights=keyword_weights
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return (
            sample.text,
            sample.label,
            sample.confidence,
            sample.keyword_weights
        )


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


class MultiTaskLoss(nn.Layer):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.label_loss_fn = nn.CrossEntropyLoss()
        self.confidence_loss_fn = nn.MSELoss()
        self.weight_loss_fn = nn.MSELoss()
        self.num_classes = num_classes

    def forward(self, label_logits, pred_confidence, pred_weights,
                true_labels, true_confidence, true_weights):
        
        label_loss = self.label_loss_fn(label_logits, true_labels)
        
        confidence_loss = self.confidence_loss_fn(pred_confidence.squeeze(-1), true_confidence)
        
        weight_loss = self.weight_loss_fn(pred_weights, true_weights)
        
        total_loss = label_loss * 2.0 + confidence_loss * 0.5 + weight_loss * 0.3
        
        return total_loss


def convert_example(example: Tuple, tokenizer: ErnieTokenizer, max_length: int) -> Dict:
    text, label, confidence, keyword_weights = example
    
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pd'
    )
    
    input_ids = encoded['input_ids'].squeeze(0)
    token_type_ids = encoded['token_type_ids'].squeeze(0)
    
    true_weights = [
        keyword_weights.get('promotion_words', 0.0),
        keyword_weights.get('price_mentions', 0.0),
        keyword_weights.get('urgency_expressions', 0.0),
        keyword_weights.get('brand_mentions', 0.0),
        keyword_weights.get('action_words', 0.0),
        keyword_weights.get('natural_expression', 0.0)
    ]
    
    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'label': paddle.to_tensor([label], dtype='int64'),
        'confidence': paddle.to_tensor([confidence], dtype='float32'),
        'keyword_weights': paddle.to_tensor(true_weights, dtype='float32')
    }


def create_dataloader(dataset: ClassifiedDataset, tokenizer: ErnieTokenizer, 
                      shuffle: bool = True) -> DataLoader:
    
    def collate_fn(samples):
        input_ids_list = []
        token_type_ids_list = []
        labels_list = []
        confidences_list = []
        weights_list = []
        
        for sample in samples:
            converted = convert_example(sample, tokenizer, max_seq_length)
            input_ids_list.append(converted['input_ids'])
            token_type_ids_list.append(converted['token_type_ids'])
            labels_list.append(converted['label'])
            confidences_list.append(converted['confidence'])
            weights_list.append(converted['keyword_weights'])
        
        batch_input_ids = paddle.stack(input_ids_list, axis=0)
        batch_token_type_ids = paddle.stack(token_type_ids_list, axis=0)
        batch_labels = paddle.stack(labels_list, axis=0).squeeze(-1)
        batch_confidences = paddle.stack(confidences_list, axis=0).squeeze(-1)
        batch_weights = paddle.stack(weights_list, axis=0)
        
        return (batch_input_ids, batch_token_type_ids, batch_labels, 
                batch_confidences, batch_weights)
    
    import platform
    effective_workers = 0 if platform.system() == 'Windows' else num_workers
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=effective_workers,
        use_shared_memory=effective_workers > 0
    )


def stratified_sample(data: List[dict], target_ratio: List[float]) -> List[dict]:
    import random
    
    category_to_label = {'normal': 0, 'ad': 1, 'hidden_ad': 2}
    by_label = {0: [], 1: [], 2: []}
    
    for item in data:
        label = item.get('label')
        if label is None:
            category = item.get('category', 'normal')
            label = category_to_label.get(category, 0)
        
        if label in by_label:
            by_label[label].append(item)
    
    total_samples = len(data)
    counts = [int(total_samples * r) for r in target_ratio]
    
    for label in [0, 1, 2]:
        if not by_label[label]:
            print(f"警告: 标签 {label} 没有数据样本")
            continue
        
        samples_needed = counts[label] - len(by_label[label])
        if samples_needed > 0:
            by_label[label] += random.choices(by_label[label], k=samples_needed)
        elif samples_needed < 0:
            by_label[label] = random.sample(by_label[label], counts[label])
    
    result = by_label[0] + by_label[1] + by_label[2]
    random.shuffle(result)
    
    return result


def save_model(model: HiddenAdClassifier, tokenizer: ErnieTokenizer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    paddle.save(model.state_dict(), os.path.join(save_dir, "model_state.pdparams"))
    
    config_data = {
        "pretrained_model": model_name,
        "max_seq_length": max_seq_length,
        "num_classes": 3,
        "output_dim": 3,
        "model_type": "hidden_ad_classifier",
        "training_config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "num_workers": num_workers
        },
        "features": {
            "text_type": {
                "labels": ["normal", "ad", "hidden_ad"],
                "label_ids": [0, 1, 2]
            },
            "confidence": {"min": 0.0, "max": 1.0},
            "keyword_weights": {
                "promotion_words": {"min": 0.0, "max": 1.0},
                "price_mentions": {"min": 0.0, "max": 1.0},
                "urgency_expressions": {"min": 0.0, "max": 1.0},
                "brand_mentions": {"min": 0.0, "max": 1.0},
                "action_words": {"min": 0.0, "max": 1.0},
                "natural_expression": {"min": 0.0, "max": 1.0}
            }
        }
    }
    
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    tokenizer.save_pretrained(save_dir)


def save_checkpoint(model: HiddenAdClassifier, optimizer, epoch: int, 
                    best_val_loss: float, save_dir: str, is_best: bool = False):
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }
    
    checkpoint_path = os.path.join(save_dir, "checkpoint_epoch.pdparams")
    paddle.save(checkpoint, checkpoint_path)
    print(f"  [Checkpoint] 已保存检查点: {checkpoint_path}")
    
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pdparams")
        paddle.save(checkpoint, best_path)
        print(f"  [Checkpoint] 最佳模型已保存: {best_path}")


def load_checkpoint(model: HiddenAdClassifier, optimizer, save_dir: str) -> Tuple[int, float]:
    checkpoint_path = os.path.join(save_dir, "checkpoint_epoch.pdparams")
    
    if not os.path.exists(checkpoint_path):
        return 0, float('inf')
    
    checkpoint = paddle.load(checkpoint_path)
    model.set_state_dict(checkpoint['model_state_dict'])
    optimizer.set_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"  [Checkpoint] 从第 {start_epoch} 个 epoch 恢复训练，最佳验证损失: {best_val_loss:.4f}")
    
    return start_epoch, best_val_loss


def get_device():
    """获取可用的训练设备，优先使用GPU"""
    if paddle.is_compiled_with_cuda():
        try:
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count > 0:
                print(f"  检测到 {gpu_count} 个 GPU:")
                for i in range(gpu_count):
                    props = paddle.device.cuda.get_device_properties(i)
                    print(f"    GPU {i}: {props.name if hasattr(props, 'name') else 'Unknown'}")
                return "gpu:0"
        except Exception as e:
            print(f"  GPU 检测时出错: {e}")
    
    print("  未检测到可用 GPU，将使用 CPU 进行训练")
    return "cpu"


def train():
    print("=" * 70)
    print("ERNIE3.0-ZH-medium 隐性广告检测训练程序")
    print("=" * 70)
    print(f"模型: {model_name}")
    print(f"最大序列长度: {max_seq_length}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"训练轮数: {epochs}")
    print(f"早停耐心值: {early_stopping_patience}")
    print(f"数据加载线程数: {num_workers}")
    print(f"分类数量: 3 (正常文本 | 广告 | 隐性广告)")
    print("=" * 70)
    
    device = get_device()
    paddle.set_device(device)
    
    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    
    # 使用平衡数据集进行训练
    data_path =  r'G:\projects\agent\text\processed_data\balanced_dataset.json'
    
    if not os.path.exists(data_path):
        print(f"错误: 训练数据文件不存在 {data_path}")
        print("请先运行 build_balanced_dataset.py 生成平衡训练数据")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    train_data_raw, val_data = train_test_split(full_data, test_size=0.2, random_state=42)
    
    train_data = stratified_sample(train_data_raw, target_ratio=[0.5,  0.1,0.4])
    
    train_data_path = os.path.join(config.get_output_dir(), 'train_split.json')
    val_data_path = os.path.join(config.get_output_dir(), 'val_split.json')
    
    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_data_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据加载完成:")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    
    label_stats = {0: 0, 1: 0, 2: 0}
    category_to_label = {'normal': 0, 'ad': 1, 'hidden_ad': 2}
    
    for item in train_data:
        label = item.get('label')
        if label is None:
            category = item.get('category', 'normal')
            label = category_to_label.get(category, 0)
        
        if label in label_stats:
            label_stats[label] += 1
        else:
            label_stats[0] += 1
    print(f"训练集类别分布:")
    print(f"  正常文本(0): {label_stats[0]} 条")
    print(f"  广告(1): {label_stats[1]} 条")
    print(f"  隐性广告(2): {label_stats[2]} 条")
    
    train_dataset = ClassifiedDataset(train_data_path)
    val_dataset = ClassifiedDataset(val_data_path)
    
    print(f"\n有效训练样本: {len(train_dataset)}")
    print(f"有效验证样本: {len(val_dataset)}")
    
    train_dataloader = create_dataloader(train_dataset, tokenizer, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, tokenizer, shuffle=False)
    
    model = HiddenAdClassifier(model_name, num_classes=3)
    model = model.to(device)
    model.train()
    
    optimizer = paddle.optimizer.AdamW(
        learning_rate=learning_rate,
        parameters=model.parameters(),
        weight_decay=weight_decay
    )
    
    criterion = MultiTaskLoss(num_classes=3)
    
    save_dir = config.get_model_path()
    
    start_epoch = 1
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    loaded_epoch, loaded_best_loss = load_checkpoint(model, optimizer, save_dir)
    if loaded_epoch > 0:
        start_epoch = loaded_epoch + 1
        best_val_loss = loaded_best_loss
        epochs_without_improvement = 0
    
    label_names = ['normal', 'ad', 'hidden_ad']
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0
        train_steps = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs} [训练]", unit="batch", ncols=120)
        for step, (input_ids, token_type_ids, labels, confidences, weights) in enumerate(pbar, start=1):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            confidences = confidences.to(device)
            weights = weights.to(device)
            
            label_logits, pred_confidence, pred_weights = model(input_ids, token_type_ids)
            
            loss = criterion(label_logits, pred_confidence, pred_weights,
                           labels, confidences, weights)
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            total_loss += loss.item()
            train_steps += 1
            
            pbar.set_postfix({"loss": f"{total_loss / train_steps:.4f}"})
        
        avg_train_loss = total_loss / train_steps
        pbar.close()
        
        model.eval()
        total_val_loss = 0
        val_steps = 0
        all_preds = []
        all_labels = []
        
        with paddle.no_grad():
            pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch}/{epochs} [验证]", unit="batch", ncols=120)
            for input_ids, token_type_ids, labels, confidences, weights in pbar_val:
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                labels = labels.to(device)
                confidences = confidences.to(device)
                weights = weights.to(device)
                
                label_logits, pred_confidence, pred_weights = model(input_ids, token_type_ids)
                
                loss = criterion(label_logits, pred_confidence, pred_weights,
                               labels, confidences, weights)
                
                total_val_loss += loss.item()
                val_steps += 1
                
                preds = paddle.argmax(label_logits, axis=1).numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
            
            pbar_val.close()
        
        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else 0
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        print(f"\nEpoch {epoch} 完成:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        
        print(f"\n  分类报告:")
        print(classification_report(all_labels, all_preds, target_names=label_names))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            save_model(model, tokenizer, save_dir)
            save_checkpoint(model, optimizer, epoch, best_val_loss, save_dir, is_best=True)
            print(f"  ✓ 最佳模型已保存 (val_loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            save_checkpoint(model, optimizer, epoch, best_val_loss, save_dir, is_best=False)
            print(f"  验证损失未改善 ({epochs_without_improvement}/{early_stopping_patience})")
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n早停: 验证损失连续 {early_stopping_patience} 个 epoch 未改善，停止训练")
                break
        
        print("-" * 70)
    
    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存位置: {save_dir}")
    
    return model


def predict(text: str, model_path: str = None):
    if model_path is None:
        model_path = config.get_model_path()
    
    if not os.path.exists(model_path):
        print(f"错误: 模型目录不存在 {model_path}")
        return None
    
    device = get_device()
    paddle.set_device(device)
    
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    tokenizer = ErnieTokenizer.from_pretrained(model_path)
    model = HiddenAdClassifier(config_data["pretrained_model"], num_classes=3)
    model_state = paddle.load(os.path.join(model_path, "model_state.pdparams"))
    model.set_state_dict(model_state)
    model.eval()
    
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
    confidence_score = float(confidence.item())
    weights_list = keyword_weights.numpy()[0].tolist()
    
    label_map = {0: 'normal', 1: 'ad', 2: 'hidden_ad'}
    label_names = {'normal': '正常文本', 'ad': '广告', 'hidden_ad': '隐性广告'}
    
    weight_names = [
        'promotion_words', 'price_mentions', 'urgency_expressions',
        'brand_mentions', 'action_words', 'natural_expression'
    ]
    weights_dict = {name: round(weight, 4) for name, weight in zip(weight_names, weights_list)}
    
    result = {
        'text': text,
        'text_type': label_map[predicted_label],
        'text_type_name': label_names[label_map[predicted_label]],
        'confidence': round(confidence_score, 4),
        'keyword_weights': weights_dict,
        'raw_logits': label_logits.numpy()[0].tolist(),
        'all_probabilities': {
            'normal': round(float(paddle.nn.functional.softmax(label_logits, axis=1)[0][0].item()), 4),
            'ad': round(float(paddle.nn.functional.softmax(label_logits, axis=1)[0][1].item()), 4),
            'hidden_ad': round(float(paddle.nn.functional.softmax(label_logits, axis=1)[0][2].item()), 4)
        }
    }
    
    return result


if __name__ == "__main__":
    train()
