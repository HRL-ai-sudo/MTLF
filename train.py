import argparse
import logging
import os
import pickle
import random
import numpy as np
import torch
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import pearsonr
from tqdm import tqdm
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, get_linear_schedule_with_warmup
from bert import MAG_BertForSequenceClassification
from xlnet import MAG_XLNetForSequenceClassification
from roberta import MAG_RobertaForSequenceClassification
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import pandas as pd
import secrets
from sklearn.manifold import TSNE
from GCAM&MFM import GCAA
from thop import profile, clever_format

# 设备设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multimodal Sentiment Analysis")
    # 新增多任务参数
    parser.add_argument('--loss_plot', type=str, default='multitask_loss.png',
                        help='Path to save the loss curves visualization')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for knowledge distillation')
    parser.add_argument('--kd_weight', type=float, default=0.1, help='Weight for knowledge distillation loss')
    parser.add_argument('--m_loss_weight', type=float, default=0.8, help='Weight for multimodal task loss')
    parser.add_argument('--cos_loss_weight', type=float, default=0.1, help='Weight for cosine similarity loss')

    parser.add_argument('--output_file', type=str, default='./results.txt',
                        help='Path to the file where best metrics will be saved.')
    parser.add_argument("--model_name_or_path", default='./pretrained_model/xlnet/', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--dataset", type=str, choices=["mosi", "mosei", "sims"], default="mosi", help="选择数据集")
    parser.add_argument("--max_seq_length", type=int, default=50, help="最大序列长度")
    parser.add_argument("--train_batch_size", type=int, default=32, help="训练批次大小")
    parser.add_argument("--dev_batch_size", type=int, default=128, help="开发批次大小")
    parser.add_argument("--test_batch_size", type=int, default=128, help="测试批次大小")
    parser.add_argument("--model", type=str, choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base"],
                        default="xlnet-base-cased", help="选择模型")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--gradient_accumulation_step", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--warmup_proportion", type=float, default=0.2, help="预热比例")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")

    # 两阶段训练参数
    parser.add_argument('--n_epochs_stage_1', type=int, default=1, help='Number of epochs for stage 1')
    parser.add_argument('--lr_stage_1', type=float, default=1e-5, help='Learning rate for stage 1')
    parser.add_argument('--n_epochs_stage_2', type=int, default=1, help='Number of epochs for stage 2')
    parser.add_argument('--lr_stage_2', type=float, default=1e-5, help='Learning rate for stage 2')

    return parser.parse_args()


# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids=None, visual=None, acoustic=None, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.visual = visual
        self.acoustic = acoustic
        self.label_id = label_id


def prepare_input(tokens, visual, acoustic, tokenizer, model_type, max_seq_length):
    if model_type == "bert-base-uncased":
        return prepare_bert_input(tokens, visual, acoustic, tokenizer, max_seq_length)
    elif model_type == "xlnet-base-cased":
        return prepare_xlnet_input(tokens, visual, acoustic, tokenizer, max_seq_length)
    elif model_type == "roberta-base":
        return prepare_roberta_input(tokens, visual, acoustic, tokenizer, max_seq_length)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _pad_features(features, pad_value, max_seq_length):
    """Helper function to pad features."""
    padded_features = []
    for feature in features:
        pad_length = max_seq_length - len(feature)
        if pad_length > 0:
            padded_feature = feature + [pad_value] * pad_length
        elif pad_length < 0:
            padded_feature = feature[:max_seq_length]
        else:
            padded_feature = feature
        padded_features.append(padded_feature)
    return padded_features


def prepare_common_input(tokens, visual, acoustic, tokenizer, bos_token, eos_token, max_seq_length):
    tokens = [bos_token] + tokens + [eos_token]

    # Pad zero vectors for acoustic / visual to account for BOS/EOS tokens
    acoustic = np.pad(acoustic, ((1, 1), (0, 0)), mode='constant')
    visual = np.pad(visual, ((1, 1), (0, 0)), mode='constant')

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    input_ids, attention_mask = _pad_features(
        [input_ids, attention_mask], tokenizer.pad_token_id, max_seq_length)
    acoustic = np.pad(acoustic, ((0, max_seq_length - acoustic.shape[0]), (0, 0)), mode='constant')
    visual = np.pad(visual, ((0, max_seq_length - visual.shape[0]), (0, 0)), mode='constant')

    return input_ids, visual, acoustic, attention_mask


def prepare_bert_input(tokens, visual, acoustic, tokenizer, max_seq_length):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    input_ids, visual, acoustic, input_mask = prepare_common_input(tokens, visual, acoustic, tokenizer, CLS, SEP,
                                                                   max_seq_length)
    segment_ids = [0] * len(input_ids)
    return input_ids, visual, acoustic, input_mask, segment_ids


def prepare_xlnet_input(tokens, visual, acoustic, tokenizer, max_seq_length):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    PAD_ID = tokenizer.pad_token_id

    tokens = tokens + [SEP] + [CLS]
    acoustic = np.pad(acoustic, ((0, 2), (0, 0)), mode='constant')
    visual = np.pad(visual, ((0, 2), (0, 0)), mode='constant')

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(tokens) - 1) + [2]

    input_ids, input_mask, segment_ids = _pad_features(
        [input_ids, input_mask, segment_ids], PAD_ID, max_seq_length)
    acoustic = np.pad(acoustic, ((max_seq_length - acoustic.shape[0], 0), (0, 0)), mode='constant')
    visual = np.pad(visual, ((max_seq_length - visual.shape[0], 0), (0, 0)), mode='constant')

    return input_ids, visual, acoustic, input_mask, segment_ids


def prepare_roberta_input(tokens, visual, acoustic, tokenizer, max_seq_length):
    input_ids, visual, acoustic, attention_mask = prepare_common_input(tokens, visual, acoustic, tokenizer,
                                                                       tokenizer.bos_token, tokenizer.eos_token,
                                                                       max_seq_length)
    return input_ids, visual, acoustic, attention_mask, None  # RoBERTa does not use segment_ids


def convert_to_features(examples, max_seq_length, tokenizer, model_type):
    features = []
    valid_example_count = 0
    for ex_index, example in enumerate(examples):
        if isinstance(example, tuple) and len(example) == 3:
            (words, visual, acoustic), label_id, _ = example
            tokens, inversions = [], []
            for idx, word in enumerate(words):
                tokenized = tokenizer.tokenize(word)
                tokens.extend(tokenized)
                inversions.extend([idx] * len(tokenized))

            assert len(tokens) == len(inversions)

            aligned_visual = [visual[idx] for idx in inversions]
            aligned_acoustic = [acoustic[idx] for idx in inversions]

            visual = np.array(aligned_visual)
            acoustic = np.array(aligned_acoustic)

            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:max_seq_length - 2]
                acoustic = acoustic[:max_seq_length - 2]
                visual = visual[:max_seq_length - 2]

            input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
                tokens, visual, acoustic, tokenizer, model_type, max_seq_length
            )

            # Ensure all arrays/lists have the correct length
            assert len(
                input_ids) == max_seq_length, f"Input IDs length ({len(input_ids)}) does not match max_seq_length ({max_seq_length})."
            assert len(
                input_mask) == max_seq_length, f"Input mask length ({len(input_mask)}) does not match max_seq_length ({max_seq_length})."
            if segment_ids is not None:
                assert len(
                    segment_ids) == max_seq_length, f"Segment IDs length ({len(segment_ids)}) does not match max_seq_length ({max_seq_length})."
            assert acoustic.shape[
                       0] == max_seq_length, f"Acoustic features length ({acoustic.shape[0]}) does not match max_seq_length ({max_seq_length})."
            assert visual.shape[
                       0] == max_seq_length, f"Visual features length ({visual.shape[0]}) does not match max_seq_length ({max_seq_length})."

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,  # This can be None for RoBERTa
                    visual=visual,
                    acoustic=acoustic,
                    label_id=label_id,
                )
            )
            valid_example_count += 1
        else:
            logger.error(f"Example {ex_index} has incorrect structure: {example}")

    if valid_example_count == 0:
        logger.error("No valid examples were found in the input data.")
    return features


def get_tokenizer(model_name, local_dir):
    """
    获取与模型类型相匹配的分词器，并强制使用本地文件。
    """
    if model_name.startswith('bert'):
        logger.info(f"Loading BERT tokenizer from {local_dir}")
        tokenizer = BertTokenizer.from_pretrained(local_dir, local_files_only=True)
    elif model_name.startswith('xlnet'):
        logger.info(f"Loading XLNet tokenizer from {local_dir}")
        tokenizer = XLNetTokenizer.from_pretrained(local_dir, local_files_only=True)
    elif model_name.startswith('roberta'):
        logger.info(f"Loading RoBERTa tokenizer from {local_dir}")
        tokenizer = RobertaTokenizer.from_pretrained(local_dir, local_files_only=True)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    return tokenizer


def get_model(model_name, num_labels, local_dir):
    """
    根据提供的模型名称获取相应的预训练模型，并强制使用本地文件。
    """
    if model_name == "bert-base-uncased":
        logger.info(f"Loading BERT model from {local_dir}")
        model = MAG_BertForSequenceClassification.from_pretrained(
            local_dir,
            local_files_only=True,
            num_labels=num_labels
        )
    elif model_name == "xlnet-base-cased":
        logger.info(f"Loading XLNet model from {local_dir}")
        model = MAG_XLNetForSequenceClassification.from_pretrained(
            local_dir,
            local_files_only=True,
            num_labels=num_labels
        )
    elif model_name == "roberta-base":
        logger.info(f"Loading RoBERTa model from {local_dir}")
        model = MAG_RobertaForSequenceClassification.from_pretrained(
            local_dir,
            local_files_only=True,
            num_labels=num_labels
        )
    else:
        raise ValueError("Unsupported model name: {}".format(model_name))

    return model


def get_appropriate_dataset(data, model_type, max_seq_length):
    """
    根据提供的数据和模型类型生成适当的数据集。
    """
    # 确保本地目录存在并且包含所需文件
    local_model_dir = os.path.join('./pre_trained_model', model_type)
    if not os.path.exists(local_model_dir):
        raise FileNotFoundError(f"The specified local directory does not exist: {local_model_dir}")

    logger.info("Initializing tokenizer for %s...", model_type)
    tokenizer = get_tokenizer(model_type, local_dir=local_model_dir)

    logger.info("Converting data to features...")
    features = convert_to_features(data, max_seq_length, tokenizer, model_type)

    # 将特征转换为张量
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)

    # Only create segment_ids tensor if it's not None and model requires it
    if features[0].segment_ids is not None:
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    else:
        all_segment_ids = None

    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    # Ensure that all tensors have the correct shape
    assert all_input_ids.size(0) == len(features), "Mismatch in input IDs length"
    assert all_input_mask.size(0) == len(features), "Mismatch in input mask length"
    if all_segment_ids is not None:
        assert all_segment_ids.size(0) == len(features), "Mismatch in segment IDs length"
    assert all_visual.size(0) == len(features), "Mismatch in visual features length"
    assert all_acoustic.size(0) == len(features), "Mismatch in acoustic features length"
    assert all_label_ids.size(0) == len(features), "Mismatch in label IDs length"

    # Create dataset with or without segment_ids based on model requirements
    if all_segment_ids is not None:
        dataset = TensorDataset(
            all_input_ids,
            all_visual,
            all_acoustic,
            all_input_mask,
            all_segment_ids,
            all_label_ids
        )
    else:
        dataset = TensorDataset(
            all_input_ids,
            all_visual,
            all_acoustic,
            all_input_mask,
            all_label_ids
        )

    logger.info("Dataset created successfully.")

    return dataset


def set_up_data_loader():
    """
    加载数据并设置数据加载器。

    返回:
    tuple: 包含训练、验证、测试数据加载器以及优化步数和分词器的元组。
    """
    args = parse_arguments()
    data_path = f"datasets/{args.dataset}.pkl"
    logger.info(f"Loading dataset from {data_path}")
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    # 获取训练、验证和测试数据
    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    return train_data, dev_data, test_data


def set_random_seed(seed: int):
    print(f"Seed: {seed}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_optimizer_and_scheduler(model, num_train_optimization_steps, learning_rate, warmup_proportion):
    """
    初始化优化器和学习率调度器。
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )

    return optimizer, scheduler


def prep_for_training(num_train_optimization_steps_stage_1, num_train_optimization_steps_stage_2):
    args = parse_arguments()
    local_model_dir = os.path.join('./pre_trained_model', args.model)
    tokenizer = get_tokenizer(args.model, local_dir=local_model_dir)
    model = get_model(args.model, num_labels=1, local_dir=local_model_dir)
    model.to(DEVICE)

    # 初始化两阶段优化器和调度器
    optimizer_1, scheduler_1 = prepare_optimizer_and_scheduler(
        model,
        num_train_optimization_steps_stage_1,
        args.lr_stage_1,
        args.warmup_proportion
    )

    optimizer_2, scheduler_2 = prepare_optimizer_and_scheduler(
        model,
        num_train_optimization_steps_stage_2,
        args.lr_stage_2,
        args.warmup_proportion
    )

    return model, (optimizer_1, optimizer_2), (scheduler_1, scheduler_2), tokenizer


def process_batch(batch):
    """
    Process and unpack the batch data.
    """
    batch = tuple(t.to(DEVICE) for t in batch)
    if len(batch) == 6:
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
    elif len(batch) == 5:
        input_ids, visual, acoustic, input_mask, label_ids = batch
        segment_ids = None
    else:
        raise ValueError("Unexpected number of items in batch")

    visual = torch.squeeze(visual, 1)
    acoustic = torch.squeeze(acoustic, 1)

    return input_ids, visual, acoustic, input_mask, segment_ids, label_ids


def compute_loss(logits, labels):
    """
    Compute the MSE loss.
    """
    mse_loss = torch.nn.MSELoss()(logits.view(-1), labels.view(-1))
    return mse_loss


class CosineLoss(torch.nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, feature1, feature2):
        # Compute cosine similarity between two tensors of shape (batch_size, dimension)
        cos_sim = torch.nn.functional.cosine_similarity(feature1, feature2, dim=1)  # Shape: (batch_size,)

        # The loss is the mean of (1 - cos_sim) across the batch
        loss = (1 - cos_sim).mean()

        return loss


def train_epoch(model, dataloader, optimizer, scheduler,
                gradient_accumulation_steps=1, temperature=2.0, kd_weight=0.1, stage=1):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0

    BASE_WEIGHTS = {
        'm_loss': 0.5,
        'cos_loss': 0.3,
        'kd_loss': 0.2
    }

    # 新增损失记录列表
    mse_losses = []
    cos_losses = []
    kd_losses = []

    # 新增记录容器
    cos_sim_text = []
    cos_sim_vision = []
    cos_sim_audio = []

    # 新增收集容器
    kd_t_losses = []
    kd_v_losses = []
    kd_a_losses = []

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)

        # 前向传播获取新增的相似度值
        (
            logits, logits_text, logits_audio, logits_vision,
            kd_loss, cos_loss, kd_loss_t, kd_loss_v, kd_loss_a,
            cos_sim_t, cos_sim_v, cos_sim_a,
            text_features,  # 文本特征 [batch, hidden]
            vision_features,  # 视觉特征
            audio_features,  # 音频特征
            multimodal_features  # 多模态融合特征
        ) = model(
            input_ids,
            visual,
            acoustic,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None
        )

        # 记录batch的相似度
        cos_sim_text.append(cos_sim_t.item())
        cos_sim_vision.append(cos_sim_v.item())
        cos_sim_audio.append(cos_sim_a.item())

        # 记录各模态损失
        kd_t_losses.append(kd_loss_t.item() * (temperature ** 2))
        kd_v_losses.append(kd_loss_v.item() * (temperature ** 2))
        kd_a_losses.append(kd_loss_a.item() * (temperature ** 2))

        # 计算多模态任务损失
        total_loss_m = compute_loss(logits, label_ids)

        if stage == 1:
            total_loss = total_loss_m
            mse_losses.append(total_loss.item())
        else:
            # 组合损失
            total_base = sum(BASE_WEIGHTS.values())
            total_loss = (
                    (BASE_WEIGHTS['m_loss'] / total_base) * total_loss_m +
                    (BASE_WEIGHTS['cos_loss'] / total_base) * cos_loss +
                    (BASE_WEIGHTS['kd_loss'] / total_base) * kd_loss
            )

            # 记录各子损失
            mse_losses.append(total_loss_m.item())
            cos_losses.append(cos_loss.item())
            kd_losses.append(kd_loss.item())


        if gradient_accumulation_steps > 1:
            total_loss = total_loss / gradient_accumulation_steps

        total_loss.backward()

        tr_loss += total_loss.item()
        nb_tr_steps += 1

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # 计算epoch平均值
    avg_mse = np.mean(mse_losses) if mse_losses else 0
    avg_cos = np.mean(cos_losses) if cos_losses else 0
    avg_kd = np.mean(kd_losses) if kd_losses else 0

    # 计算epoch平均相似度
    avg_cos_text = np.mean(cos_sim_text) if cos_sim_text else 0.0
    avg_cos_vision = np.mean(cos_sim_vision) if cos_sim_vision else 0.0
    avg_cos_audio = np.mean(cos_sim_audio) if cos_sim_audio else 0.0

    return (total_loss / nb_tr_steps,
            avg_mse,
            avg_cos,
            avg_kd,
            avg_cos_text,
            avg_cos_vision,
            avg_cos_audio,
            np.mean(kd_t_losses),
            np.mean(kd_v_losses),
            np.mean(kd_a_losses)
            )


def eval_epoch(model, dataloader, temperature=2.0, kd_weight=0.1, stage=1):
    """Modified: 返回细分损失"""
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    # 初始化所有累积变量
    total_loss_m_accum = 0.0
    total_kd_loss_accum = 0.0
    total_cos_loss_accum = 0.0

    BASE_WEIGHTS = {
        'm_loss': 0.5,
        'cos_loss': 0.3,
        'kd_loss': 0.2
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)

            # 前向传播获取新增的相似度值
            (
                logits, logits_text, logits_audio, logits_vision,
                kd_loss, cos_loss, kd_loss_t, kd_loss_v, kd_loss_a,
                cos_sim_t, cos_sim_v, cos_sim_a,
                text_features,  # 文本特征 [batch, hidden]
                vision_features,  # 视觉特征
                audio_features,  # 音频特征
                multimodal_features  # 多模态融合特征
            ) = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None
            )

            # 计算多模态任务损失
            total_loss_m = compute_loss(logits, label_ids)

            if stage == 1:
                total_loss = total_loss_m
                # 修正累加逻辑
                total_loss_m_accum += total_loss_m.item()  # 直接累加标量值
            else:
                # 组合损失
                total_base = sum(BASE_WEIGHTS.values())
                total_loss = (
                        (BASE_WEIGHTS['m_loss'] / total_base) * total_loss_m +
                        (BASE_WEIGHTS['cos_loss'] / total_base) * cos_loss +
                        (BASE_WEIGHTS['kd_loss'] / total_base) * kd_loss
                )

                batch_loss_m = total_loss_m.item()
                batch_kd_loss = kd_loss.item()
                batch_cos_loss = cos_loss.item()

                # 累加标量值
                total_loss_m_accum += batch_loss_m
                total_kd_loss_accum += batch_kd_loss
                total_cos_loss_accum += batch_cos_loss


            nb_eval_steps += 1
            eval_loss += total_loss.item()

    return (eval_loss / nb_eval_steps,
            total_loss_m_accum / nb_eval_steps,
            total_kd_loss_accum / nb_eval_steps if stage != 1 else 0.0,
            total_cos_loss_accum / nb_eval_steps if stage != 1 else 0.0)


def test_epoch(model: torch.nn.Module, dataloader: DataLoader):
    """
    Test model and collect predictions and labels.
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)
            logits, *_ = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask,
                                labels=None)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            preds.extend(np.squeeze(logits).tolist())
            labels.extend(np.squeeze(label_ids).tolist())

    return np.array(preds), np.array(labels)

def calculate_metrics(preds, y_test, use_zero=False):
    """
    Calculate metrics for test scores.
    """
    # 确保 labels 是一维数组
    y_test = np.squeeze(y_test)

    non_zeros = [i for i, e in enumerate(y_test) if e != 0 or use_zero]
    preds, y_test = preds[non_zeros], y_test[non_zeros]

    mae = np.mean(np.abs(preds - y_test))
    corr = pearsonr(preds, y_test)[0]

    binary_preds = preds > 0
    binary_labels = y_test > 0

    f_score = f1_score(binary_labels, binary_preds, average='weighted')
    acc = accuracy_score(binary_labels, binary_preds)

    return acc, mae, corr, f_score


def multiclass_accuracy(preds, y_test):
    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(y_test, a_min=-2., a_max=2.)

    mult_a7 = np.sum(np.round(test_preds_a7) == np.round(test_truth_a7)) / float(len(test_truth_a7))
    mult_a5 = np.sum(np.round(test_preds_a5) == np.round(test_truth_a5)) / float(len(test_truth_a5))

    return mult_a7, mult_a5


# 在训练开始前计算（示例）
def calculate_model_complexity(model, device):
    # 构造与真实数据一致的输入示例
    input_ids = torch.randint(1, 30000, (1, 50)).to(device).transpose(0, 1)  # (50, 1)
    visual = torch.randn(1, 50, 47).to(device)
    acoustic = torch.randn(1, 50, 74).to(device)
    token_type_ids = torch.randint(0, 2, (1, 50)).to(device).transpose(0, 1)
    attention_mask = torch.randint(0, 2, (1, 50)).to(device).transpose(0, 1)

    flops, params = profile(model, inputs=(input_ids, visual, acoustic, token_type_ids, attention_mask, None))
    return flops, params


def train_loop(
        model,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        optimizers,
        schedulers,
        stage_configs,
        output_file='results.txt',
        gradient_accumulation_steps=1,
        temperature=2.0,
        kd_weight=0.1
):
    best_valid_loss = float('inf')
    best_metrics = None
    best_model_state_dict = None  # 用于保存最佳模型的状态字典

    # 初始化存储结构（在循环外）
    train_loss_records = {
        'mse': [],
        'cos': [],
        'kd': []
    }

    val_loss_records = {
        'mse': [],
        'cos': [],
        'kd': []
    }

    # 修改对齐度记录结构
    alignment_records = {
        'text': [],
        'visual': [],
        'acoustic': [],
        'stage_markers': []  # 新增阶段标记
    }

    # 新增KL记录
    kl_records = {
        'text': [],
        'vision': [],
        'audio': []
    }

    for stage_i, stage_config in enumerate(stage_configs, start=1):
        print(f"\nStarting Stage {stage_i}/{len(stage_configs)}")
        optimizer = optimizers[stage_i - 1]
        scheduler = schedulers[stage_i - 1]
        n_epochs = stage_config['n_epochs']

        # 在每个阶段开始时记录阶段标记
        alignment_records['stage_markers'].append(len(alignment_records['text']))

        for epoch_i in range(n_epochs):
            print(f"Stage {stage_i}, Epoch {epoch_i + 1}/{n_epochs}", end='\r')

            # 获取训练返回的相似度数据
            (train_loss, epoch_mse, epoch_cos, epoch_kd,
             epoch_cos_text, epoch_cos_vision, epoch_cos_audio, avg_kd_t, avg_kd_v, avg_kd_a) = train_epoch(  # 接收新返回值
                model,
                train_dataloader,
                optimizer,
                scheduler,
                gradient_accumulation_steps,
                temperature,
                kd_weight,
                stage=stage_i
            )

            # 记录对齐度
            alignment_records['text'].append(epoch_cos_text)
            alignment_records['visual'].append(epoch_cos_vision)
            alignment_records['acoustic'].append(epoch_cos_audio)

            # 记录各模态KL
            kl_records['text'].append(avg_kd_t)
            kl_records['vision'].append(avg_kd_v)
            kl_records['audio'].append(avg_kd_a)

            # 记录训练损失
            train_loss_records['mse'].append(epoch_mse)
            train_loss_records['cos'].append(epoch_cos)
            train_loss_records['kd'].append(epoch_kd)

            # 验证阶段
            valid_loss, *val_details = eval_epoch(
                model,
                validation_dataloader,
                temperature,
                kd_weight,
                stage=stage_i
            )
            (epoch_mse, epoch_kd, epoch_cos) = val_details[:3]

            # 记录验证损失
            val_loss_records['mse'].append(float(epoch_mse))
            val_loss_records['cos'].append(float(epoch_cos))
            val_loss_records['kd'].append(float(epoch_kd))

            preds, y_test = test_epoch(model, test_dataloader)
            test_acc, test_mae, test_corr, test_f_score = calculate_metrics(preds, y_test)
            test_acc7, test_acc5 = multiclass_accuracy(preds, y_test)

            # 更新最佳指标
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_metrics = {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "test_acc": test_acc,
                    "test_mae": test_mae,
                    "test_corr": test_corr,
                    "test_f_score": test_f_score,
                    "test_acc7": test_acc7,
                    "test_acc5": test_acc5,
                    "epoch": epoch_i + 1,
                    "stage": stage_i
                }

                best_model_state_dict = model.state_dict()  # 保存最佳模型的状态字典

    # 保存最佳模型
    if best_model_state_dict is not None:
        model_path = 'best_model.pth'
        torch.save(best_model_state_dict, model_path)
        print(f"Best model saved to {model_path}")

        return best_metrics, train_loss_records, val_loss_records, alignment_records


def main():
    args = parse_arguments()
    if args.seed is None:
        # 使用 secrets 模块生成一个安全的随机整数作为种子
        args.seed = secrets.randbits(32)
    set_random_seed(args.seed)

    # 加载数据集
    train_data, dev_data, test_data = set_up_data_loader()

    # 创建数据集
    train_dataset = get_appropriate_dataset(train_data, args.model, args.max_seq_length)
    val_dataset = get_appropriate_dataset(dev_data, args.model, args.max_seq_length)
    test_dataset = get_appropriate_dataset(test_data, args.model, args.max_seq_length)

    # 计算训练步数
    num_train_steps_stage1 = (
                                     len(train_dataset) // args.train_batch_size // args.gradient_accumulation_step
                             ) * args.n_epochs_stage_1
    num_train_steps_stage2 = (
                                     len(train_dataset) // args.train_batch_size // args.gradient_accumulation_step
                             ) * args.n_epochs_stage_2

    # 初始化模型和优化
    model, optimizers, schedulers, _ = prep_for_training(num_train_steps_stage1, num_train_steps_stage2)

    # 初始化模型后
    flops, params = calculate_model_complexity(model, DEVICE)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Model Params: {params}, FLOPs: {flops}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.dev_batch_size
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.test_batch_size
    )

    # 定义阶段配置
    stage_configs = [
        {'n_epochs': args.n_epochs_stage_1, 'lr': args.lr_stage_1},
        {'n_epochs': args.n_epochs_stage_2, 'lr': args.lr_stage_2}
    ]

    # 训练循环返回对齐度记录
    best_metrics, train_loss_records, val_loss_records, alignment_records = train_loop(
        model=model,
        train_dataloader=train_loader,
        validation_dataloader=val_loader,
        test_dataloader=test_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        stage_configs=stage_configs,
        output_file=f"{args.output_file}",
        gradient_accumulation_steps=args.gradient_accumulation_step,
        temperature=args.temperature,
        kd_weight=args.kd_weight
    )

    # 输出最终结果
    logger.info("\nFinal best metrics:")
    with open(args.output_file, 'a') as f:
        f.write("\n=== Final Best Metrics ===\n")
        for key, value in best_metrics.items():
            logger.info(f"{key}: {value}")
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
