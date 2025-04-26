# from __future__ import absolute_import, division, print_function
# import argparse
# import copy
# import os
# import random
# import numpy as np
# from pytorch_transformers import AdamW
# from sklearn.metrics import accuracy_score, f1_score
# import torch
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from tqdm import tqdm
# from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, get_linear_schedule_with_warmup
# from bert import MAG_BertForSequenceClassification
# from xlnet import MAG_XLNetForSequenceClassification
# from roberta import MAG_RobertaForSequenceClassification
# from argparse_utils import seed
# from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE
# from thop import profile
# import warnings
# import logging
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# import pickle
# from scipy.stats import pearsonr
# from datetime import datetime
# import torch.nn.functional as F
# import pandas as pd
# from sklearn.model_selection import KFold
#
#
# warnings.filterwarnings('ignore')
#
# # 全局设备设置
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Multimodal Sentiment Analysis")
#     # 知识蒸馏相关参数
#     parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for knowledge distillation')
#     parser.add_argument('--kd_weight', type=float, default=0.1, help='Weight for knowledge distillation loss')
#     # 新增参数：多模态任务损失权重
#     parser.add_argument('--m_loss_weight', type=float, default=0.8, help='Weight for multimodal task loss')
#     # 新增参数：余弦相似度损失权重
#     parser.add_argument('--cos_loss_weight', type=float, default=0.1, help='Weight for cosine similarity loss')
#     parser.add_argument('--output_file', type=str, default='./results.txt',
#                         help='Path to the file where best metrics will be saved.')
#     parser.add_argument("--model_name_or_path", default='./pretrained_model/roberta/', type=str,
#                         help="Path to pre-trained model or shortcut name")
#     parser.add_argument("--dataset", type=str, choices=["mosi", "mosei", "sims"], default="mosi", help="选择数据集")
#     parser.add_argument("--max_seq_length", type=int, default=50, help="最大序列长度")
#     parser.add_argument("--train_batch_size", type=int, default=32, help="训练批次大小")
#     parser.add_argument("--dev_batch_size", type=int, default=128, help="开发批次大小")
#     parser.add_argument("--test_batch_size", type=int, default=128, help="测试批次大小")
#     # parser.add_argument("--n_epochs", type=int, default=80, help="训练轮数")
#     parser.add_argument("--model", type=str, choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base"], default="xlnet-base-cased", help="选择模型")
#     parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
#     parser.add_argument("--gradient_accumulation_step", type=int, default=1, help="梯度累积步数")
#     parser.add_argument("--warmup_proportion", type=float, default=0.2, help="预热比例")
#     parser.add_argument("--seed", type=seed, default="random", help="随机种子")
#     # Arguments for multi-stage training configuration
#     parser.add_argument('--n_epochs_stage_1', type=int, default=40, help='Number of epochs for stage 1')
#     parser.add_argument('--lr_stage_1', type=float, default=1e-5, help='Learning rate for stage 1')
#     parser.add_argument('--n_epochs_stage_2', type=int, default=20, help='Number of epochs for stage 2')
#     parser.add_argument('--lr_stage_2', type=float, default=1e-5, help='Learning rate for stage 2')
#     return parser.parse_args()
#
#
# args = parse_arguments()
#
# # 设置日志配置
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# class InputFeatures:
#     def __init__(self, input_ids, input_mask, segment_ids=None, visual=None, acoustic=None, label_id=None):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.visual = visual
#         self.acoustic = acoustic
#         self.label_id = label_id
#
#
# def prepare_input(tokens, visual, acoustic, tokenizer, model_type, max_seq_length):
#     if model_type == "bert-base-uncased":
#         return prepare_bert_input(tokens, visual, acoustic, tokenizer, max_seq_length)
#     elif model_type == "xlnet-base-cased":
#         return prepare_xlnet_input(tokens, visual, acoustic, tokenizer, max_seq_length)
#     elif model_type == "roberta-base":
#         return prepare_roberta_input(tokens, visual, acoustic, tokenizer, max_seq_length)
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}")
#
#
# def _pad_features(features, pad_value, max_seq_length):
#     """Helper function to pad features."""
#     padded_features = []
#     for feature in features:
#         pad_length = max_seq_length - len(feature)
#         if pad_length > 0:
#             padded_feature = feature + [pad_value] * pad_length
#         elif pad_length < 0:
#             padded_feature = feature[:max_seq_length]
#         else:
#             padded_feature = feature
#         padded_features.append(padded_feature)
#     return padded_features
#
#
# def prepare_common_input(tokens, visual, acoustic, tokenizer, bos_token, eos_token, max_seq_length):
#     tokens = [bos_token] + tokens + [eos_token]
#
#     # Pad zero vectors for acoustic / visual to account for BOS/EOS tokens
#     acoustic = np.pad(acoustic, ((1, 1), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((1, 1), (0, 0)), mode='constant')
#
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     attention_mask = [1] * len(input_ids)
#
#     input_ids, attention_mask = _pad_features(
#         [input_ids, attention_mask], tokenizer.pad_token_id, max_seq_length)
#     acoustic = np.pad(acoustic, ((0, max_seq_length - acoustic.shape[0]), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((0, max_seq_length - visual.shape[0]), (0, 0)), mode='constant')
#
#     return input_ids, visual, acoustic, attention_mask
#
#
# def prepare_bert_input(tokens, visual, acoustic, tokenizer, max_seq_length):
#     CLS = tokenizer.cls_token
#     SEP = tokenizer.sep_token
#     input_ids, visual, acoustic, input_mask = prepare_common_input(tokens, visual, acoustic, tokenizer, CLS, SEP, max_seq_length)
#     segment_ids = [0] * len(input_ids)
#     return input_ids, visual, acoustic, input_mask, segment_ids
#
#
# def prepare_xlnet_input(tokens, visual, acoustic, tokenizer, max_seq_length):
#     CLS = tokenizer.cls_token
#     SEP = tokenizer.sep_token
#     PAD_ID = tokenizer.pad_token_id
#
#     tokens = tokens + [SEP] + [CLS]
#     acoustic = np.pad(acoustic, ((0, 2), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((0, 2), (0, 0)), mode='constant')
#
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     input_mask = [1] * len(input_ids)
#     segment_ids = [0] * (len(tokens) - 1) + [2]
#
#     input_ids, input_mask, segment_ids = _pad_features(
#         [input_ids, input_mask, segment_ids], PAD_ID, max_seq_length)
#     acoustic = np.pad(acoustic, ((max_seq_length - acoustic.shape[0], 0), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((max_seq_length - visual.shape[0], 0), (0, 0)), mode='constant')
#
#     return input_ids, visual, acoustic, input_mask, segment_ids
#
#
# def prepare_roberta_input(tokens, visual, acoustic, tokenizer, max_seq_length):
#     input_ids, visual, acoustic, attention_mask = prepare_common_input(tokens, visual, acoustic, tokenizer, tokenizer.bos_token, tokenizer.eos_token, max_seq_length)
#     return input_ids, visual, acoustic, attention_mask, None  # RoBERTa does not use segment_ids
#
#
# def convert_to_features(examples, max_seq_length, tokenizer, model_type):
#     features = []
#     valid_example_count = 0
#     for ex_index, example in enumerate(examples):
#         if isinstance(example, tuple) and len(example) == 3:
#             (words, visual, acoustic), label_id, _ = example
#             tokens, inversions = [], []
#             for idx, word in enumerate(words):
#                 tokenized = tokenizer.tokenize(word)
#                 tokens.extend(tokenized)
#                 inversions.extend([idx] * len(tokenized))
#
#             assert len(tokens) == len(inversions)
#
#             aligned_visual = [visual[idx] for idx in inversions]
#             aligned_acoustic = [acoustic[idx] for idx in inversions]
#
#             visual = np.array(aligned_visual)
#             acoustic = np.array(aligned_acoustic)
#
#             if len(tokens) > max_seq_length - 2:
#                 tokens = tokens[:max_seq_length - 2]
#                 acoustic = acoustic[:max_seq_length - 2]
#                 visual = visual[:max_seq_length - 2]
#
#             input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
#                 tokens, visual, acoustic, tokenizer, model_type, max_seq_length
#             )
#
#             # Ensure all arrays/lists have the correct length
#             assert len(input_ids) == max_seq_length, f"Input IDs length ({len(input_ids)}) does not match max_seq_length ({max_seq_length})."
#             assert len(input_mask) == max_seq_length, f"Input mask length ({len(input_mask)}) does not match max_seq_length ({max_seq_length})."
#             if segment_ids is not None:
#                 assert len(segment_ids) == max_seq_length, f"Segment IDs length ({len(segment_ids)}) does not match max_seq_length ({max_seq_length})."
#             assert acoustic.shape[0] == max_seq_length, f"Acoustic features length ({acoustic.shape[0]}) does not match max_seq_length ({max_seq_length})."
#             assert visual.shape[0] == max_seq_length, f"Visual features length ({visual.shape[0]}) does not match max_seq_length ({max_seq_length})."
#
#             features.append(
#                 InputFeatures(
#                     input_ids=input_ids,
#                     input_mask=input_mask,
#                     segment_ids=segment_ids,  # This can be None for RoBERTa
#                     visual=visual,
#                     acoustic=acoustic,
#                     label_id=label_id,
#                 )
#             )
#             valid_example_count += 1
#         else:
#             logger.error(f"Example {ex_index} has incorrect structure: {example}")
#
#     if valid_example_count == 0:
#         logger.error("No valid examples were found in the input data.")
#     return features
#
#
# def get_tokenizer(model_name, local_dir):
#     """
#     获取与模型类型相匹配的分词器，并强制使用本地文件。
#     """
#     if model_name.startswith('bert'):
#         logger.info(f"Loading BERT tokenizer from {local_dir}")
#         tokenizer = BertTokenizer.from_pretrained(local_dir, local_files_only=True)
#     elif model_name.startswith('xlnet'):
#         logger.info(f"Loading XLNet tokenizer from {local_dir}")
#         tokenizer = XLNetTokenizer.from_pretrained(local_dir, local_files_only=True)
#     elif model_name.startswith('roberta'):
#         logger.info(f"Loading RoBERTa tokenizer from {local_dir}")
#         tokenizer = RobertaTokenizer.from_pretrained(local_dir, local_files_only=True)
#     else:
#         raise ValueError(f"Unsupported model type: {model_name}")
#
#     return tokenizer
#
#
# def get_model(model_name, num_labels, local_dir):
#     """
#     根据提供的模型名称获取相应的预训练模型，并强制使用本地文件。
#     """
#     if model_name == "bert-base-uncased":
#         logger.info(f"Loading BERT model from {local_dir}")
#         model = MAG_BertForSequenceClassification.from_pretrained(
#             local_dir,
#             local_files_only=True,
#             num_labels=num_labels
#         )
#     elif model_name == "xlnet-base-cased":
#         logger.info(f"Loading XLNet model from {local_dir}")
#         model = MAG_XLNetForSequenceClassification.from_pretrained(
#             local_dir,
#             local_files_only=True,
#             num_labels=num_labels
#         )
#     elif model_name == "roberta-base":
#         logger.info(f"Loading RoBERTa model from {local_dir}")
#         model = MAG_RobertaForSequenceClassification.from_pretrained(
#             local_dir,
#             local_files_only=True,
#             num_labels=num_labels
#         )
#     else:
#         raise ValueError("Unsupported model name: {}".format(model_name))
#
#     return model
#
#
# def get_appropriate_dataset(data, model_type, max_seq_length):
#     """
#     根据提供的数据和模型类型生成适当的数据集。
#     """
#     # 确保本地目录存在并且包含所需文件
#     local_model_dir = os.path.join('./pre_trained_model', model_type)
#     if not os.path.exists(local_model_dir):
#         raise FileNotFoundError(f"The specified local directory does not exist: {local_model_dir}")
#
#     logger.info("Initializing tokenizer for %s...", model_type)
#     tokenizer = get_tokenizer(model_type, local_dir=local_model_dir)
#
#     logger.info("Converting data to features...")
#     features = convert_to_features(data, max_seq_length, tokenizer, model_type)
#
#     # 将特征转换为张量
#     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
#
#     # Only create segment_ids tensor if it's not None and model requires it
#     if features[0].segment_ids is not None:
#         all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
#     else:
#         all_segment_ids = None
#
#     all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
#     all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
#     all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
#
#     # Ensure that all tensors have the correct shape
#     assert all_input_ids.size(0) == len(features), "Mismatch in input IDs length"
#     assert all_input_mask.size(0) == len(features), "Mismatch in input mask length"
#     if all_segment_ids is not None:
#         assert all_segment_ids.size(0) == len(features), "Mismatch in segment IDs length"
#     assert all_visual.size(0) == len(features), "Mismatch in visual features length"
#     assert all_acoustic.size(0) == len(features), "Mismatch in acoustic features length"
#     assert all_label_ids.size(0) == len(features), "Mismatch in label IDs length"
#
#     # Create dataset with or without segment_ids based on model requirements
#     if all_segment_ids is not None:
#         dataset = TensorDataset(
#             all_input_ids,
#             all_visual,
#             all_acoustic,
#             all_input_mask,
#             all_segment_ids,
#             all_label_ids
#         )
#     else:
#         dataset = TensorDataset(
#             all_input_ids,
#             all_visual,
#             all_acoustic,
#             all_input_mask,
#             all_label_ids
#         )
#
#     logger.info("Dataset created successfully.")
#
#     return dataset
#
# def set_up_data_loader():
#     """
#     加载数据并设置数据加载器。
#
#     返回:
#     tuple: 包含训练、验证、测试数据加载器以及优化步数和分词器的元组。
#     """
#     args = parse_arguments()
#     # 加载数据集
#     data_path = f"datasets/{args.dataset}.pkl"
#     logger.info(f"Loading dataset from {data_path}")
#     with open(data_path, "rb") as handle:
#         data = pickle.load(handle)
#
#     # 获取训练、验证和测试数据
#     train_data = data["train"]
#     dev_data = data["dev"]
#     test_data = data["test"]
#
#     # 确保本地目录存在并且包含所需文件
#     local_model_dir = os.path.join('./pre_trained_model', args.model)
#     if not os.path.exists(local_model_dir):
#         raise FileNotFoundError(f"The specified local directory does not exist: {local_model_dir}")
#
#     # 初始化分词器
#     logger.info("Initializing tokenizer for %s...", args.model)
#     tokenizer = get_tokenizer(args.model, local_dir=local_model_dir)
#
#     # 创建适当的数据集并传递 model_type 参数
#     logger.info("Creating datasets...")
#     train_dataset = get_appropriate_dataset(train_data, args.model, args.max_seq_length)
#     dev_dataset = get_appropriate_dataset(dev_data, args.model, args.max_seq_length)
#     test_dataset = get_appropriate_dataset(test_data, args.model, args.max_seq_length)
#
#     # 计算每个阶段的训练优化步骤数
#     num_train_optimization_steps_stage_1 = (
#             int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_step) * args.n_epochs_stage_1
#     )
#     num_train_optimization_steps_stage_2 = (
#             int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_step) * args.n_epochs_stage_2
#     )
#     logger.info(f"Number of training optimization steps for stage 1: {num_train_optimization_steps_stage_1}")
#     logger.info(f"Number of training optimization steps for stage 2: {num_train_optimization_steps_stage_2}")
#
#
#     # 设置数据加载器
#     logger.info("Setting up data loaders...")
#     train_dataloader = DataLoader(
#         train_dataset,
#         sampler=RandomSampler(train_dataset),
#         batch_size=args.train_batch_size,
#     )
#
#     dev_dataloader = DataLoader(
#         dev_dataset,
#         sampler=SequentialSampler(dev_dataset),
#         batch_size=args.dev_batch_size,
#     )
#
#     test_dataloader = DataLoader(
#         test_dataset,
#         sampler=SequentialSampler(test_dataset),
#         batch_size=args.test_batch_size,
#     )
#
#     return train_dataloader, dev_dataloader, test_dataloader, \
#            (num_train_optimization_steps_stage_1, num_train_optimization_steps_stage_2), tokenizer
#
#
#
# def set_random_seed(seed: int):
#     print("Seed: {}".format(seed))
#
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.deterministic = True
#
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#
# def prepare_optimizer_and_scheduler(model, num_train_optimization_steps, learning_rate, warmup_proportion):
#     """
#     初始化优化器和学习率调度器。
#     """
#     param_optimizer = list(model.named_parameters())
#     no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#             "weight_decay": 0.01,
#         },
#         {
#             "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#             "weight_decay": 0.0,
#         },
#     ]
#
#     optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=warmup_proportion * num_train_optimization_steps,
#         num_training_steps=num_train_optimization_steps,
#     )
#
#     return optimizer, scheduler
#
#
# def prep_for_training(num_train_optimization_steps_stage_1, num_train_optimization_steps_stage_2):
# # def prep_for_training(num_train_optimization_steps):
#     args = parse_arguments()
#     # 确保本地目录存在并且包含所需文件
#     local_model_dir = os.path.join('./pre_trained_model', args.model)
#     if not os.path.exists(local_model_dir):
#         raise FileNotFoundError(f"The specified local directory does not exist: {local_model_dir}")
#
#     # 获取分词器
#     tokenizer = get_tokenizer(args.model, local_dir=local_model_dir)
#
#     # 获取多模态模型
#     model = get_model(args.model, num_labels=1, local_dir=local_model_dir)
#     model.to(DEVICE)
#
#     # 初始化第一阶段的优化器和调度器
#     optimizer_1, scheduler_1 = prepare_optimizer_and_scheduler(
#         model,
#         num_train_optimization_steps_stage_1,
#         args.lr_stage_1,
#         args.warmup_proportion
#     )
#
#     # 初始化第二阶段的优化器和调度器
#     optimizer_2, scheduler_2 = prepare_optimizer_and_scheduler(
#         model,
#         num_train_optimization_steps_stage_2,
#         args.lr_stage_2,
#         args.warmup_proportion
#     )
#
#
#     return model, (optimizer_1, optimizer_2), (scheduler_1, scheduler_2), tokenizer
#
#
#
# def process_batch(batch):
#     """
#     Process and unpack the batch data.
#     """
#     batch = tuple(t.to(DEVICE) for t in batch)
#     if len(batch) == 6:
#         input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
#     elif len(batch) == 5:
#         input_ids, visual, acoustic, input_mask, label_ids = batch
#         segment_ids = None
#     else:
#         raise ValueError("Unexpected number of items in batch")
#
#     visual = torch.squeeze(visual, 1)
#     acoustic = torch.squeeze(acoustic, 1)
#
#     return input_ids, visual, acoustic, input_mask, segment_ids, label_ids
#
#
# def compute_loss(logits, labels):
#     """
#     Compute the MSE loss.
#     """
#     mse_loss = torch.nn.MSELoss()(logits.view(-1), labels.view(-1))
#     return mse_loss
#
#
# class CosineLoss(torch.nn.Module):
#     def __init__(self):
#         super(CosineLoss, self).__init__()
#
#     def forward(self, feature1, feature2):
#         # Compute cosine similarity between two tensors of shape (batch_size, dimension)
#         cos_sim = torch.nn.functional.cosine_similarity(feature1, feature2, dim=1)  # Shape: (batch_size,)
#
#         # The loss is the mean of (1 - cos_sim) across the batch
#         loss = (1 - cos_sim).mean()
#
#         return loss
#
# def train_epoch(model, dataloader, optimizer, scheduler,
#                 gradient_accumulation_steps=1, temperature=2.0, kd_weight=0.1, stage=1):
#     model.train()
#     tr_loss = 0
#     nb_tr_steps = 0
#     cos_similarity = CosineLoss()
#     kd_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
#
#     # 分层权重配置
#     BASE_WEIGHTS = {
#         'm_loss': 0.7,  # 主任务基础权重（最大）
#         'cos_loss': 0.2,  # 表示学习权重
#         'kd_loss': 0.1  # 知识蒸馏权重
#     }
#
#     for step, batch in enumerate(tqdm(dataloader, desc="Training")):
#         input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)
#
#         # 前向传播
#         outputs, t, v, a = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask,
#                                  labels=None)
#         # outputs, t, v, a = model(input_ids, visual, acoustic, attention_mask=input_mask,
#         #                          labels=None)
#
#         logits = outputs[0]
#
#         # 计算多模态任务损失
#         total_loss_m = compute_loss(logits, label_ids)
#         # total_loss = total_loss_m
#
#         if stage == 1:
#             # 第一阶段只计算多模态任务的MSE损失
#             total_loss = total_loss_m
#         else:
#             # 计算余弦相似度损失
#             total_loss_t = cos_similarity(t, logits.detach())
#             total_loss_v = cos_similarity(v, logits.detach())
#             total_loss_a = cos_similarity(a, logits.detach())
#             cos_loss = total_loss_t * 0.8 + total_loss_v * 0.1 + total_loss_a * 0.1
#             # cos_loss = total_loss_a + total_loss_v
#
#             # 计算KL损失
#             teacher_soft_labels = F.softmax(logits / temperature, dim=1)
#
#             student_soft_logits_t = F.log_softmax(t / temperature, dim=1)
#             student_soft_logits_v = F.log_softmax(v / temperature, dim=1)
#             student_soft_logits_a = F.log_softmax(a / temperature, dim=1)
#
#             kd_loss_t = kd_loss_fn(student_soft_logits_t, teacher_soft_labels) * (temperature ** 2)
#             kd_loss_v = kd_loss_fn(student_soft_logits_v, teacher_soft_labels) * (temperature ** 2)
#             kd_loss_a = kd_loss_fn(student_soft_logits_a, teacher_soft_labels) * (temperature ** 2)
#
#             kd_loss = kd_loss_t * 0.8 + kd_loss_v * 0.1 + kd_loss_a * 0.1
#             # kd_loss = kd_loss_a + kd_loss_v
#
#             # 权重归一化
#             total_base = sum(BASE_WEIGHTS.values())
#             m_weight = BASE_WEIGHTS['m_loss'] / total_base
#             cos_weight = BASE_WEIGHTS['cos_loss'] / total_base
#             kd_weight_actual = (BASE_WEIGHTS['kd_loss'] / total_base) * kd_weight
#
#             # 组合损失
#             total_loss = (
#                     m_weight * total_loss_m +
#                     cos_weight * cos_loss +
#                     kd_weight_actual * kd_loss
#             )
#
#         if gradient_accumulation_steps > 1:
#             total_loss = total_loss / gradient_accumulation_steps
#
#         total_loss.backward()
#
#         tr_loss += total_loss.item()
#         nb_tr_steps += 1
#
#         if (step + 1) % gradient_accumulation_steps == 0:
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()
#
#     avg_loss = tr_loss / nb_tr_steps
#
#     return avg_loss
#
# def eval_epoch(model: torch.nn.Module, dataloader: DataLoader, gradient_accumulation_steps=1, stage=1, temperature=2.0, kd_weight=0.1):
#     """
#     Evaluate model on a given dataloader.
#     """
#     model.eval()
#     eval_loss = 0
#     nb_eval_steps = 0
#     cos_similarity = CosineLoss()
#     kd_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
#
#     # 分层权重配置
#     BASE_WEIGHTS = {
#         'm_loss': 0.7,  # 主任务基础权重（最大）
#         'cos_loss': 0.2,  # 表示学习权重
#         'kd_loss': 0.1  # 知识蒸馏权重
#     }
#
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Evaluating"):
#             input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)
#
#             # 前向传播
#             outputs, t, v, a = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask,
#                                      labels=None)
#             # outputs, t, v, a = model(input_ids, visual, acoustic, attention_mask=input_mask,
#             #                          labels=None)
#             logits = outputs[0]
#
#             # 计算多模态任务损失
#             total_loss_m = compute_loss(logits, label_ids)
#             # total_loss = total_loss_m
#
#             if stage == 1:
#                 # 第一阶段只计算多模态任务的MSE损失
#                 total_loss = total_loss_m
#             else:
#                 # 计算余弦相似度损失
#                 total_loss_t = cos_similarity(t, logits.detach())
#                 total_loss_v = cos_similarity(v, logits.detach())
#                 total_loss_a = cos_similarity(a, logits.detach())
#                 cos_loss = total_loss_t * 0.8 + total_loss_v * 0.1 + total_loss_a * 0.1
#                 # cos_loss = total_loss_a + total_loss_v
#
#                 # 计算KL损失
#                 teacher_soft_labels = F.softmax(logits / temperature, dim=1)
#
#                 student_soft_logits_t = F.log_softmax(t / temperature, dim=1)
#                 student_soft_logits_v = F.log_softmax(v / temperature, dim=1)
#                 student_soft_logits_a = F.log_softmax(a / temperature, dim=1)
#
#                 kd_loss_t = kd_loss_fn(student_soft_logits_t, teacher_soft_labels) * (temperature ** 2)
#                 kd_loss_v = kd_loss_fn(student_soft_logits_v, teacher_soft_labels) * (temperature ** 2)
#                 kd_loss_a = kd_loss_fn(student_soft_logits_a, teacher_soft_labels) * (temperature ** 2)
#
#                 kd_loss = kd_loss_t * 0.8 + kd_loss_v * 0.1 + kd_loss_a * 0.1
#                 # kd_loss = kd_loss_a + kd_loss_v
#
#                 # 权重归一化
#                 total_base = sum(BASE_WEIGHTS.values())
#                 m_weight = BASE_WEIGHTS['m_loss'] / total_base
#                 cos_weight = BASE_WEIGHTS['cos_loss'] / total_base
#                 kd_weight_actual = (BASE_WEIGHTS['kd_loss'] / total_base) * kd_weight
#
#                 # 组合损失
#                 total_loss = (
#                         m_weight * total_loss_m +
#                         cos_weight * cos_loss +
#                     kd_weight_actual * kd_loss
#                 )
#
#             if gradient_accumulation_steps > 1:
#                 total_loss = total_loss / gradient_accumulation_steps
#
#             eval_loss += total_loss.item()
#             nb_eval_steps += 1
#
#     return eval_loss / nb_eval_steps
#
#
# def test_epoch(model: torch.nn.Module, dataloader: DataLoader):
#     """
#     Test model and collect predictions and labels.
#     """
#     model.eval()
#     preds, labels = [], []
#
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Testing"):
#             input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)
#             outputs, *_ = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask,
#                             labels=None)
#             # outputs, t, v, a = model(input_ids, visual, acoustic, attention_mask=input_mask,
#             #                          labels=None)
#             logits = outputs[0].detach().cpu().numpy()
#             label_ids = label_ids.detach().cpu().numpy()
#             preds.extend(np.squeeze(logits).tolist())
#             labels.extend(np.squeeze(label_ids).tolist())
#
#     return np.array(preds), np.array(labels)
#
#
#
# def calculate_metrics(preds, y_test, use_zero=False):
#     """
#     Calculate metrics for test scores.
#     """
#
#     non_zeros = [i for i, e in enumerate(y_test) if e != 0 or use_zero]
#     preds, y_test = preds[non_zeros], y_test[non_zeros]
#
#     mae = np.mean(np.abs(preds - y_test))
#     corr = pearsonr(preds, y_test)[0]
#
#     binary_preds = preds > 0
#     binary_labels = y_test > 0
#
#     f_score = f1_score(binary_labels, binary_preds, average='weighted')
#     acc = accuracy_score(binary_labels, binary_preds)
#
#     return acc, mae, corr, f_score
#
#
# def multiclass_accuracy(preds, y_test):
#     test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
#     test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
#     test_preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
#     test_truth_a5 = np.clip(y_test, a_min=-2., a_max=2.)
#
#     mult_a7 = np.sum(np.round(test_preds_a7) == np.round(test_truth_a7)) / float(len(test_truth_a7))
#     mult_a5 = np.sum(np.round(test_preds_a5) == np.round(test_truth_a5)) / float(len(test_truth_a5))
#
#     return mult_a7, mult_a5
#
# def train_loop(
#         model,
#         train_dataloader,
#         validation_dataloader,
#         test_dataloader,
#         optimizers,
#         schedulers,
#         stage_configs: list,
#         output_file='results.txt',
#         gradient_accumulation_steps=1,
#         temperature=2.0,
#         kd_weight=0.1
# ):
#     best_valid_loss = float('inf')
#     best_metrics = None
#
#     # Ensure the directory exists for saving the output file
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#
#     for stage_i, stage_config in enumerate(stage_configs, start=1):
#         print(f"\nStarting Stage {stage_i}/{len(stage_configs)}")
#
#
#         # 根据阶段选择优化器和调度器
#         optimizer = optimizers[stage_i - 1]
#         scheduler = schedulers[stage_i - 1]
#
#         # Optionally update learning rate for this stage
#         if 'lr' in stage_config:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = stage_config['lr']
#             print(f"Updated learning rate to {stage_config['lr']}")
#             print(f"Updated epoch to {stage_config['n_epochs']}")
#
#         n_epochs = stage_config.get('n_epochs')
#
#         for epoch_i in range(n_epochs):
#             print(f"Stage {stage_i}, Epoch {epoch_i + 1}/{n_epochs}", end='\r')
#
#             train_loss = train_epoch(model, train_dataloader, optimizer, scheduler,
#                                      gradient_accumulation_steps, temperature, kd_weight, stage=stage_i)
#
#             # Validation phase
#             valid_loss = eval_epoch(model, validation_dataloader, gradient_accumulation_steps, stage=stage_i, temperature=temperature, kd_weight=kd_weight)
#
#             # Testing phase (kept for completeness but not used for deciding best metrics)
#             preds, y_test = test_epoch(model, test_dataloader)
#             test_acc, test_mae, test_corr, test_f_score = calculate_metrics(preds, y_test)
#             test_acc_7, test_acc_5 = multiclass_accuracy(preds, y_test)
#
#
#             # Update the best metrics based on validation loss
#             if valid_loss < best_valid_loss:
#                 best_valid_loss = valid_loss
#                 best_metrics = {
#                     "train_loss": train_loss,
#                     "valid_loss": valid_loss,
#                     "test_acc": test_acc,
#                     "test_mae": test_mae,
#                     "test_corr": test_corr,
#                     "test_f_score": test_f_score,
#                     "test_acc7": test_acc_7,
#                     "test_acc5": test_acc_5,
#                     "epoch_with_best_valid_loss": epoch_i,
#                     "stage": stage_i
#                 }
#                 best_model_state = copy.deepcopy(model.state_dict())
#         # 训练结束后加载最佳模型
#         if best_model_state is not None:
#
#             model.load_state_dict(best_model_state)
#             # Print current epoch number and losses to console
#             print(f"\nStage {stage_i}, Epoch {epoch_i + 1}/{n_epochs}")
#             print(f"Training Loss: {train_loss:.4f}")
#             print(f"Validation Loss: {valid_loss:.4f}")
#
#     # After all stages, write the final best metrics at the end of training
#     with open(output_file, 'a') as f:
#         now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         f.write(f"\n--- Training completed on {now} ---\n")
#         f.write("Best Metrics:\n")
#         for key, value in best_metrics.items():
#             f.write(f"{key}: {value}\n")
#
#     # Print final best metrics to console
#     print("\nTraining completed.")
#     print("Best Metrics:")
#     for key, value in best_metrics.items():
#         print(f"{key}: {value}")
#
#     return best_metrics
#
#
# def main():
#     args = parse_arguments()
#     set_random_seed(args.seed)
#
#     # 加载数据集
#     data_path = f"datasets/{args.dataset}.pkl"
#     logger.info(f"Loading dataset from {data_path}")
#     with open(data_path, "rb") as handle:
#         data = pickle.load(handle)
#
#     train_data = data["train"]
#     dev_data = data["dev"]
#     test_data = data["test"]
#
#     # 合并训练集和验证集进行五折交叉验证
#     full_train_data = train_data + dev_data
#
#     # 初始化KFold
#     kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
#     all_metrics = []
#
#     # 新增：用于保存每一折最佳结果的文件路径
#     fold_results_file = f"{os.path.splitext(args.output_file)[0]}_fold_results.txt"
#
#     for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_data), 1):
#         logger.info(f"Processing Fold {fold}")
#
#         # 生成当前折的训练集和验证集
#         current_train = [full_train_data[i] for i in train_idx]
#         current_val = [full_train_data[i] for i in val_idx]
#
#         # 生成数据集
#         train_dataset = get_appropriate_dataset(current_train, args.model, args.max_seq_length)
#         val_dataset = get_appropriate_dataset(current_val, args.model, args.max_seq_length)
#         test_dataset = get_appropriate_dataset(test_data, args.model, args.max_seq_length)
#
#         # 计算训练步数
#         num_train_steps_stage1 = (
#             len(train_dataset) // args.train_batch_size // args.gradient_accumulation_step
#         ) * args.n_epochs_stage_1
#         num_train_steps_stage2 = (
#             len(train_dataset) // args.train_batch_size // args.gradient_accumulation_step
#         ) * args.n_epochs_stage_2
#
#         # 初始化模型和优化器
#         model, optimizers, schedulers, _ = prep_for_training(num_train_steps_stage1, num_train_steps_stage2)
#
#         # 创建数据加载器
#         train_loader = DataLoader(
#             train_dataset,
#             sampler=RandomSampler(train_dataset),
#             batch_size=args.train_batch_size,
#             drop_last=True
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             sampler=SequentialSampler(val_dataset),
#             batch_size=args.dev_batch_size
#         )
#         test_loader = DataLoader(
#             test_dataset,
#             sampler=SequentialSampler(test_dataset),
#             batch_size=args.test_batch_size
#         )
#
#         # 定义阶段配置
#         stage_configs = [
#             {'n_epochs': args.n_epochs_stage_1, 'lr': args.lr_stage_1},
#             {'n_epochs': args.n_epochs_stage_2, 'lr': args.lr_stage_2}
#         ]
#
#         # 训练循环
#         best_metrics = train_loop(
#             model=model,
#             train_dataloader=train_loader,
#             validation_dataloader=val_loader,
#             test_dataloader=test_loader,
#             optimizers=optimizers,
#             schedulers=schedulers,
#             stage_configs=stage_configs,
#             output_file=f"{args.output_file}.fold{fold}",
#             gradient_accumulation_steps=args.gradient_accumulation_step,
#             temperature=args.temperature,
#             kd_weight=args.kd_weight
#         )
#
#         all_metrics.append(best_metrics)
#
#         # 新增：将每一折的最佳结果写入文件
#         with open(fold_results_file, 'a') as f:
#             f.write(f"--- Fold {fold} Best Metrics ---\n")
#             for key, value in best_metrics.items():
#                 f.write(f"{key}: {value}\n")
#             f.write("\n")
#
#     # 计算平均结果
#     avg_metrics = {}
#     for key in best_metrics:
#         if key in ['epoch_with_best_valid_loss', 'stage']:
#             continue
#         values = [m[key] for m in all_metrics]
#         avg_metrics[key] = np.mean(values)
#         avg_metrics[f"{key}_std"] = np.std(values)
#
#     # 输出最终结果
#     logger.info("\n五折交叉验证平均结果：")
#     for key, value in avg_metrics.items():
#         logger.info(f"{key}: {value:.4f} (±{avg_metrics[f'{key}_std']:.4f})")
#
#     # 保存平均结果到文件
#     with open(args.output_file, 'a') as f:
#         f.write("\n=== 五折交叉验证结果 ===\n")
#         for key, value in avg_metrics.items():
#             f.write(f"{key}: {value:.4f} (±{avg_metrics[f'{key}_std']:.4f})\n")
#
#
# if __name__ == "__main__":
#     main()
# from __future__ import absolute_import, division, print_function
# import argparse
# import os
# import random
# import numpy as np
# from pytorch_transformers import AdamW
# from sklearn.metrics import accuracy_score, f1_score
# import torch
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from tqdm import tqdm
# from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, get_linear_schedule_with_warmup
# from bert import MAG_BertForSequenceClassification
# from xlnet import MAG_XLNetForSequenceClassification
# from roberta import MAG_RobertaForSequenceClassification
# from argparse_utils import seed
# from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE
# import warnings
# import logging
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# import pickle
# from scipy.stats import pearsonr
# from sklearn.model_selection import KFold
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
#
#
# warnings.filterwarnings('ignore')
#
# # 全局设备设置
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Multimodal Sentiment Analysis")
#     # Modified: 移除知识蒸馏相关参数
#     parser.add_argument('--output_file', type=str, default='./results.txt',
#                         help='Path to the file where best metrics will be saved.')
#     parser.add_argument("--model_name_or_path", default='./pretrained_model/roberta/', type=str,
#                         help="Path to pre-trained model or shortcut name")
#     parser.add_argument("--dataset", type=str, choices=["mosi", "mosei", "sims"], default="mosi", help="选择数据集")
#     parser.add_argument("--max_seq_length", type=int, default=50, help="最大序列长度")
#     parser.add_argument("--train_batch_size", type=int, default=32, help="训练批次大小")
#     parser.add_argument("--dev_batch_size", type=int, default=128, help="开发批次大小")
#     parser.add_argument("--test_batch_size", type=int, default=128, help="测试批次大小")
#     parser.add_argument("--model", type=str, choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base"], default="xlnet-base-cased", help="选择模型")
#     parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
#     parser.add_argument("--gradient_accumulation_step", type=int, default=1, help="梯度累积步数")
#     parser.add_argument("--warmup_proportion", type=float, default=0.2, help="预热比例")
#     parser.add_argument("--seed", type=seed, default="random", help="随机种子")
#     # Modified: 仅保留单阶段参数
#     parser.add_argument('--n_epochs', type=int, default=40, help='Total training epochs')
#     return parser.parse_args()
#
# args = parse_arguments()
#
#
# # 设置日志配置
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# class InputFeatures:
#     def __init__(self, input_ids, input_mask, segment_ids=None, visual=None, acoustic=None, label_id=None):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.visual = visual
#         self.acoustic = acoustic
#         self.label_id = label_id
#
#
# def prepare_input(tokens, visual, acoustic, tokenizer, model_type, max_seq_length):
#     if model_type == "bert-base-uncased":
#         return prepare_bert_input(tokens, visual, acoustic, tokenizer, max_seq_length)
#     elif model_type == "xlnet-base-cased":
#         return prepare_xlnet_input(tokens, visual, acoustic, tokenizer, max_seq_length)
#     elif model_type == "roberta-base":
#         return prepare_roberta_input(tokens, visual, acoustic, tokenizer, max_seq_length)
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}")
#
#
# def _pad_features(features, pad_value, max_seq_length):
#     """Helper function to pad features."""
#     padded_features = []
#     for feature in features:
#         pad_length = max_seq_length - len(feature)
#         if pad_length > 0:
#             padded_feature = feature + [pad_value] * pad_length
#         elif pad_length < 0:
#             padded_feature = feature[:max_seq_length]
#         else:
#             padded_feature = feature
#         padded_features.append(padded_feature)
#     return padded_features
#
#
# def prepare_common_input(tokens, visual, acoustic, tokenizer, bos_token, eos_token, max_seq_length):
#     tokens = [bos_token] + tokens + [eos_token]
#
#     # Pad zero vectors for acoustic / visual to account for BOS/EOS tokens
#     acoustic = np.pad(acoustic, ((1, 1), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((1, 1), (0, 0)), mode='constant')
#
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     attention_mask = [1] * len(input_ids)
#
#     input_ids, attention_mask = _pad_features(
#         [input_ids, attention_mask], tokenizer.pad_token_id, max_seq_length)
#     acoustic = np.pad(acoustic, ((0, max_seq_length - acoustic.shape[0]), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((0, max_seq_length - visual.shape[0]), (0, 0)), mode='constant')
#
#     return input_ids, visual, acoustic, attention_mask
#
#
# def prepare_bert_input(tokens, visual, acoustic, tokenizer, max_seq_length):
#     CLS = tokenizer.cls_token
#     SEP = tokenizer.sep_token
#     input_ids, visual, acoustic, input_mask = prepare_common_input(tokens, visual, acoustic, tokenizer, CLS, SEP, max_seq_length)
#     segment_ids = [0] * len(input_ids)
#     return input_ids, visual, acoustic, input_mask, segment_ids
#
#
# def prepare_xlnet_input(tokens, visual, acoustic, tokenizer, max_seq_length):
#     CLS = tokenizer.cls_token
#     SEP = tokenizer.sep_token
#     PAD_ID = tokenizer.pad_token_id
#
#     tokens = tokens + [SEP] + [CLS]
#     acoustic = np.pad(acoustic, ((0, 2), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((0, 2), (0, 0)), mode='constant')
#
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     input_mask = [1] * len(input_ids)
#     segment_ids = [0] * (len(tokens) - 1) + [2]
#
#     input_ids, input_mask, segment_ids = _pad_features(
#         [input_ids, input_mask, segment_ids], PAD_ID, max_seq_length)
#     acoustic = np.pad(acoustic, ((max_seq_length - acoustic.shape[0], 0), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((max_seq_length - visual.shape[0], 0), (0, 0)), mode='constant')
#
#     return input_ids, visual, acoustic, input_mask, segment_ids
#
#
# def prepare_roberta_input(tokens, visual, acoustic, tokenizer, max_seq_length):
#     input_ids, visual, acoustic, attention_mask = prepare_common_input(tokens, visual, acoustic, tokenizer, tokenizer.bos_token, tokenizer.eos_token, max_seq_length)
#     return input_ids, visual, acoustic, attention_mask, None  # RoBERTa does not use segment_ids
#
#
# def convert_to_features(examples, max_seq_length, tokenizer, model_type):
#     features = []
#     valid_example_count = 0
#     for ex_index, example in enumerate(examples):
#         if isinstance(example, tuple) and len(example) == 3:
#             (words, visual, acoustic), label_id, _ = example
#             tokens, inversions = [], []
#             for idx, word in enumerate(words):
#                 tokenized = tokenizer.tokenize(word)
#                 tokens.extend(tokenized)
#                 inversions.extend([idx] * len(tokenized))
#
#             assert len(tokens) == len(inversions)
#
#             aligned_visual = [visual[idx] for idx in inversions]
#             aligned_acoustic = [acoustic[idx] for idx in inversions]
#
#             visual = np.array(aligned_visual)
#             acoustic = np.array(aligned_acoustic)
#
#             if len(tokens) > max_seq_length - 2:
#                 tokens = tokens[:max_seq_length - 2]
#                 acoustic = acoustic[:max_seq_length - 2]
#                 visual = visual[:max_seq_length - 2]
#
#             input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
#                 tokens, visual, acoustic, tokenizer, model_type, max_seq_length
#             )
#
#             # Ensure all arrays/lists have the correct length
#             assert len(input_ids) == max_seq_length, f"Input IDs length ({len(input_ids)}) does not match max_seq_length ({max_seq_length})."
#             assert len(input_mask) == max_seq_length, f"Input mask length ({len(input_mask)}) does not match max_seq_length ({max_seq_length})."
#             if segment_ids is not None:
#                 assert len(segment_ids) == max_seq_length, f"Segment IDs length ({len(segment_ids)}) does not match max_seq_length ({max_seq_length})."
#             assert acoustic.shape[0] == max_seq_length, f"Acoustic features length ({acoustic.shape[0]}) does not match max_seq_length ({max_seq_length})."
#             assert visual.shape[0] == max_seq_length, f"Visual features length ({visual.shape[0]}) does not match max_seq_length ({max_seq_length})."
#
#             features.append(
#                 InputFeatures(
#                     input_ids=input_ids,
#                     input_mask=input_mask,
#                     segment_ids=segment_ids,  # This can be None for RoBERTa
#                     visual=visual,
#                     acoustic=acoustic,
#                     label_id=label_id,
#                 )
#             )
#             valid_example_count += 1
#         else:
#             logger.error(f"Example {ex_index} has incorrect structure: {example}")
#
#     if valid_example_count == 0:
#         logger.error("No valid examples were found in the input data.")
#     return features
#
#
# def get_tokenizer(model_name, local_dir):
#     """
#     获取与模型类型相匹配的分词器，并强制使用本地文件。
#     """
#     if model_name.startswith('bert'):
#         logger.info(f"Loading BERT tokenizer from {local_dir}")
#         tokenizer = BertTokenizer.from_pretrained(local_dir, local_files_only=True)
#     elif model_name.startswith('xlnet'):
#         logger.info(f"Loading XLNet tokenizer from {local_dir}")
#         tokenizer = XLNetTokenizer.from_pretrained(local_dir, local_files_only=True)
#     elif model_name.startswith('roberta'):
#         logger.info(f"Loading RoBERTa tokenizer from {local_dir}")
#         tokenizer = RobertaTokenizer.from_pretrained(local_dir, local_files_only=True)
#     else:
#         raise ValueError(f"Unsupported model type: {model_name}")
#
#     return tokenizer
#
#
# def get_model(model_name, num_labels, local_dir):
#     """
#     根据提供的模型名称获取相应的预训练模型，并强制使用本地文件。
#     """
#     if model_name == "bert-base-uncased":
#         logger.info(f"Loading BERT model from {local_dir}")
#         model = MAG_BertForSequenceClassification.from_pretrained(
#             local_dir,
#             local_files_only=True,
#             num_labels=num_labels
#         )
#     elif model_name == "xlnet-base-cased":
#         logger.info(f"Loading XLNet model from {local_dir}")
#         model = MAG_XLNetForSequenceClassification.from_pretrained(
#             local_dir,
#             local_files_only=True,
#             num_labels=num_labels
#         )
#     elif model_name == "roberta-base":
#         logger.info(f"Loading RoBERTa model from {local_dir}")
#         model = MAG_RobertaForSequenceClassification.from_pretrained(
#             local_dir,
#             local_files_only=True,
#             num_labels=num_labels
#         )
#     else:
#         raise ValueError("Unsupported model name: {}".format(model_name))
#
#     return model
#
#
# def get_appropriate_dataset(data, model_type, max_seq_length):
#     """
#     根据提供的数据和模型类型生成适当的数据集。
#     """
#     # 确保本地目录存在并且包含所需文件
#     local_model_dir = os.path.join('./pre_trained_model', model_type)
#     if not os.path.exists(local_model_dir):
#         raise FileNotFoundError(f"The specified local directory does not exist: {local_model_dir}")
#
#     logger.info("Initializing tokenizer for %s...", model_type)
#     tokenizer = get_tokenizer(model_type, local_dir=local_model_dir)
#
#     logger.info("Converting data to features...")
#     features = convert_to_features(data, max_seq_length, tokenizer, model_type)
#
#     # 将特征转换为张量
#     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
#
#     # Only create segment_ids tensor if it's not None and model requires it
#     if features[0].segment_ids is not None:
#         all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
#     else:
#         all_segment_ids = None
#
#     all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
#     all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
#     all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
#
#     # Ensure that all tensors have the correct shape
#     assert all_input_ids.size(0) == len(features), "Mismatch in input IDs length"
#     assert all_input_mask.size(0) == len(features), "Mismatch in input mask length"
#     if all_segment_ids is not None:
#         assert all_segment_ids.size(0) == len(features), "Mismatch in segment IDs length"
#     assert all_visual.size(0) == len(features), "Mismatch in visual features length"
#     assert all_acoustic.size(0) == len(features), "Mismatch in acoustic features length"
#     assert all_label_ids.size(0) == len(features), "Mismatch in label IDs length"
#
#     # Create dataset with or without segment_ids based on model requirements
#     if all_segment_ids is not None:
#         dataset = TensorDataset(
#             all_input_ids,
#             all_visual,
#             all_acoustic,
#             all_input_mask,
#             all_segment_ids,
#             all_label_ids
#         )
#     else:
#         dataset = TensorDataset(
#             all_input_ids,
#             all_visual,
#             all_acoustic,
#             all_input_mask,
#             all_label_ids
#         )
#
#     logger.info("Dataset created successfully.")
#
#     return dataset
#
# # def set_up_data_loader():
# #     """
# #     加载数据并设置数据加载器。
# #
# #     返回:
# #     tuple: 包含训练、验证、测试数据加载器以及优化步数和分词器的元组。
# #     """
# #     args = parse_arguments()
# #     # 加载数据集
# #     data_path = f"datasets/{args.dataset}.pkl"
# #     logger.info(f"Loading dataset from {data_path}")
# #     with open(data_path, "rb") as handle:
# #         data = pickle.load(handle)
# #
# #     # 获取训练、验证和测试数据
# #     train_data = data["train"]
# #     dev_data = data["dev"]
# #     test_data = data["test"]
# #
# #     # 确保本地目录存在并且包含所需文件
# #     local_model_dir = os.path.join('./pre_trained_model', args.model)
# #     if not os.path.exists(local_model_dir):
# #         raise FileNotFoundError(f"The specified local directory does not exist: {local_model_dir}")
# #
# #     # 初始化分词器
# #     logger.info("Initializing tokenizer for %s...", args.model)
# #     tokenizer = get_tokenizer(args.model, local_dir=local_model_dir)
# #
# #     # 创建适当的数据集并传递 model_type 参数
# #     logger.info("Creating datasets...")
# #     train_dataset = get_appropriate_dataset(train_data, args.model, args.max_seq_length)
# #     dev_dataset = get_appropriate_dataset(dev_data, args.model, args.max_seq_length)
# #     test_dataset = get_appropriate_dataset(test_data, args.model, args.max_seq_length)
# #
# #     # 计算总训练步数
# #     num_train_optimization_steps = (
# #             int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_step) * args.n_epochs
# #     )
# #     logger.info(f"Total training optimization steps: {num_train_optimization_steps}")
# #
# #     # 设置数据加载器
# #     logger.info("Setting up data loaders...")
# #     train_dataloader = DataLoader(
# #         train_dataset,
# #         sampler=RandomSampler(train_dataset),
# #         batch_size=args.train_batch_size,
# #     )
# #
# #     dev_dataloader = DataLoader(
# #         dev_dataset,
# #         sampler=SequentialSampler(dev_dataset),
# #         batch_size=args.dev_batch_size,
# #     )
# #
# #     test_dataloader = DataLoader(
# #         test_dataset,
# #         sampler=SequentialSampler(test_dataset),
# #         batch_size=args.test_batch_size,
# #     )
# #
# #     # 返回参数调整
# #     return train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps, tokenizer
#
#
# def set_random_seed(seed: int):
#     print("Seed: {}".format(seed))
#
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.deterministic = True
#
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#
# def prepare_optimizer_and_scheduler(model, num_train_optimization_steps, learning_rate, warmup_proportion):
#     """
#     初始化优化器和学习率调度器。
#     """
#     param_optimizer = list(model.named_parameters())
#     no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#             "weight_decay": 0.01,
#         },
#         {
#             "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#             "weight_decay": 0.0,
#         },
#     ]
#
#     optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=warmup_proportion * num_train_optimization_steps,
#         num_training_steps=num_train_optimization_steps,
#     )
#
#     return optimizer, scheduler
#
#
# def prep_for_training(num_train_optimization_steps_stage_1, num_train_optimization_steps_stage_2):
# # def prep_for_training(num_train_optimization_steps):
#     args = parse_arguments()
#     # 确保本地目录存在并且包含所需文件
#     local_model_dir = os.path.join('./pre_trained_model', args.model)
#     if not os.path.exists(local_model_dir):
#         raise FileNotFoundError(f"The specified local directory does not exist: {local_model_dir}")
#
#     # 获取分词器
#     tokenizer = get_tokenizer(args.model, local_dir=local_model_dir)
#
#     # 获取多模态模型
#     model = get_model(args.model, num_labels=1, local_dir=local_model_dir)
#     model.to(DEVICE)
#
#     # 初始化第一阶段的优化器和调度器
#     optimizer_1, scheduler_1 = prepare_optimizer_and_scheduler(
#         model,
#         num_train_optimization_steps_stage_1,
#         args.lr_stage_1,
#         args.warmup_proportion
#     )
#
#     # 初始化第二阶段的优化器和调度器
#     optimizer_2, scheduler_2 = prepare_optimizer_and_scheduler(
#         model,
#         num_train_optimization_steps_stage_2,
#         args.lr_stage_2,
#         args.warmup_proportion
#     )
#
#     return model, (optimizer_1, optimizer_2), (scheduler_1, scheduler_2), tokenizer
#
#
#
# def process_batch(batch):
#     """
#     Process and unpack the batch data.
#     """
#     batch = tuple(t.to(DEVICE) for t in batch)
#     if len(batch) == 6:
#         input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
#     elif len(batch) == 5:
#         input_ids, visual, acoustic, input_mask, label_ids = batch
#         segment_ids = None
#     else:
#         raise ValueError("Unexpected number of items in batch")
#
#     visual = torch.squeeze(visual, 1)
#     acoustic = torch.squeeze(acoustic, 1)
#
#     return input_ids, visual, acoustic, input_mask, segment_ids, label_ids
#
#
# def compute_loss(logits, labels):
#     """
#     Compute the MSE loss.
#     """
#     mse_loss = torch.nn.MSELoss()(logits.view(-1), labels.view(-1))
#     return mse_loss
#
#
# class CosineLoss(torch.nn.Module):
#     def __init__(self):
#         super(CosineLoss, self).__init__()
#
#     def forward(self, feature1, feature2):
#         # Compute cosine similarity between two tensors of shape (batch_size, dimension)
#         cos_sim = torch.nn.functional.cosine_similarity(feature1, feature2, dim=1)  # Shape: (batch_size,)
#
#         # The loss is the mean of (1 - cos_sim) across the batch
#         loss = (1 - cos_sim).mean()
#
#         return loss
#
#
# def train_epoch(model, dataloader, optimizer, scheduler, gradient_accumulation_steps=1):
#     """Modified: 仅保留多模态任务损失"""
#     model.train()
#     tr_loss = 0
#     nb_tr_steps = 0
#
#     for step, batch in enumerate(tqdm(dataloader, desc="Training")):
#         input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)
#
#         # 前向传播
#         (
#             logits, _, _, _,
#             _, _, _, _, _,
#             _, _, _,
#             text_features,  # 文本特征 [batch, hidden]
#             vision_features,  # 视觉特征
#             audio_features,  # 音频特征
#             multimodal_features  # 多模态融合特征
#         ) = model(
#             input_ids,
#             visual,
#             acoustic,
#             token_type_ids=segment_ids,
#             attention_mask=input_mask,
#             labels=None
#         )
#
#         # 仅计算多模态任务损失
#         total_loss = compute_loss(logits, label_ids)
#
#         if gradient_accumulation_steps > 1:
#             total_loss = total_loss / gradient_accumulation_steps
#
#         total_loss.backward()
#
#         tr_loss += total_loss.item()
#         nb_tr_steps += 1
#
#         if (step + 1) % gradient_accumulation_steps == 0:
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()
#
#     return tr_loss / nb_tr_steps
#
# def eval_epoch(model, dataloader):
#     """Modified: 简化验证过程"""
#     model.eval()
#     eval_loss = 0
#     nb_eval_steps = 0
#
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Evaluating"):
#             input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)
#             (
#                 logits, _, _, _,
#                 _, _, _, _, _,
#                 _, _, _,
#                 text_features,  # 文本特征 [batch, hidden]
#                 vision_features,  # 视觉特征
#                 audio_features,  # 音频特征
#                 multimodal_features  # 多模态融合特征
#             ) = model(
#                 input_ids,
#                 visual,
#                 acoustic,
#                 token_type_ids=segment_ids,
#                 attention_mask=input_mask,
#                 labels=None
#             )
#             eval_loss += compute_loss(logits, label_ids).item()
#             nb_eval_steps += 1
#
#     return eval_loss / nb_eval_steps
#
#
# def test_epoch(model: torch.nn.Module, dataloader: DataLoader):
#     model.eval()
#     preds, labels = [], []
#     features = {
#         'text': [],
#         'vision': [],
#         'audio': [],
#         'multimodal': []
#     }
#
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Testing"):
#             input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)
#             (
#                 logits, _, _, _,
#                 _, _, _, _, _,
#                 _, _, _,
#                 text_features,  # 文本特征 [batch, hidden]
#                 vision_features,  # 视觉特征
#                 audio_features,  # 音频特征
#                 multimodal_features  # 多模态融合特征
#             ) = model(
#                 input_ids,
#                 visual,
#                 acoustic,
#                 token_type_ids=segment_ids,
#                 attention_mask=input_mask,
#                 labels=None
#             )
#
#             # Collect features
#             features['text'].append(text_features.cpu())
#             features['vision'].append(vision_features.cpu())
#             features['audio'].append(audio_features.cpu())
#             features['multimodal'].append(multimodal_features.cpu())
#
#             # Collect labels and predictions
#             logits = logits.detach().cpu().numpy()
#             label_ids = label_ids.detach().cpu().numpy()
#             preds.extend(np.squeeze(logits).tolist())
#             labels.extend(np.squeeze(label_ids).tolist())
#
#     # Concatenate features
#     for key in features:
#         features[key] = torch.cat(features[key], dim=0).numpy()
#
#     # 添加特征尺寸校验
#     assert text_features.shape[1] == model.config.d_model, "特征维度不匹配"
#
#     # 添加内存清理
#     torch.cuda.empty_cache()
#
#     return np.array(preds), np.array(labels), features
#
#
# def calculate_metrics(preds, y_test, use_zero=False):
#     """
#     Calculate metrics for test scores.
#     """
#
#     non_zeros = [i for i, e in enumerate(y_test) if e != 0 or use_zero]
#     preds, y_test = preds[non_zeros], y_test[non_zeros]
#
#     mae = np.mean(np.abs(preds - y_test))
#     corr = pearsonr(preds, y_test)[0]
#
#     binary_preds = preds > 0
#     binary_labels = y_test > 0
#
#     f_score = f1_score(binary_labels, binary_preds, average='weighted')
#     acc = accuracy_score(binary_labels, binary_preds)
#
#     return acc, mae, corr, f_score
#
#
# def multiclass_accuracy(preds, y_test):
#     test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
#     test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
#     test_preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
#     test_truth_a5 = np.clip(y_test, a_min=-2., a_max=2.)
#
#     mult_a7 = np.sum(np.round(test_preds_a7) == np.round(test_truth_a7)) / float(len(test_truth_a7))
#     mult_a5 = np.sum(np.round(test_preds_a5) == np.round(test_truth_a5)) / float(len(test_truth_a5))
#
#     return mult_a7, mult_a5
#
#
# # ... (keep all previous imports and functions unchanged until set_up_data_loader)
#
# def set_up_data_loader():
#     """Modified: 恢复原始单折数据加载"""
#     args = parse_arguments()
#     # 加载数据集
#     data_path = f"datasets/{args.dataset}.pkl"
#     logger.info(f"Loading dataset from {data_path}")
#     with open(data_path, "rb") as handle:
#         data = pickle.load(handle)
#
#     # 获取训练、验证和测试数据
#     train_data = data["train"]
#     dev_data = data["dev"]
#     test_data = data["test"]
#
#     # 创建适当的数据集
#     logger.info("Creating datasets...")
#     train_dataset = get_appropriate_dataset(train_data, args.model, args.max_seq_length)
#     dev_dataset = get_appropriate_dataset(dev_data, args.model, args.max_seq_length)
#     test_dataset = get_appropriate_dataset(test_data, args.model, args.max_seq_length)
#
#     # 计算总训练步数
#     num_train_optimization_steps = (
#         int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_step) * args.n_epochs
#     )
#     logger.info(f"Total training optimization steps: {num_train_optimization_steps}")
#
#     # 设置数据加载器
#     logger.info("Setting up data loaders...")
#     train_dataloader = DataLoader(
#         train_dataset,
#         sampler=RandomSampler(train_dataset),
#         batch_size=args.train_batch_size,
#     )
#
#     dev_dataloader = DataLoader(
#         dev_dataset,
#         sampler=SequentialSampler(dev_dataset),
#         batch_size=args.dev_batch_size,
#     )
#
#     test_dataloader = DataLoader(
#         test_dataset,
#         sampler=SequentialSampler(test_dataset),
#         batch_size=args.test_batch_size,
#     )
#
#     return train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps
#
#
# def train_loop(
#     model,
#     train_dataloader,
#     dev_dataloader,
#     test_dataloader,  # 新增测试集
#     optimizer,
#     scheduler,
#     output_file,
#     gradient_accumulation_steps
# ):
#     """Modified: 恢复单折训练流程"""
#     best_valid_loss = float('inf')
#     best_metrics = None
#     best_model_state_dict = None  # 用于保存最佳模型的状态字典
#
#     for epoch_i in range(args.n_epochs):
#         print(f"Epoch {epoch_i + 1}/{args.n_epochs}")
#
#         # 训练阶段
#         train_loss = train_epoch(
#             model,
#             train_dataloader,
#             optimizer,
#             scheduler,
#             gradient_accumulation_steps
#         )
#
#         # 验证阶段
#         valid_loss = eval_epoch(model, dev_dataloader)
#
#         # 测试阶段 (每个epoch结束后在测试集评估)
#         test_preds, test_labels, features = test_epoch(model, test_dataloader)
#         test_acc, test_mae, test_corr, test_f_score = calculate_metrics(test_preds, test_labels)
#         test_acc_7, test_acc_5 = multiclass_accuracy(test_preds, test_labels)
#
#         # 更新最佳指标
#         if valid_loss < best_valid_loss:
#             best_valid_loss = valid_loss
#             best_metrics = {
#                 "epoch": epoch_i + 1,
#                 "train_loss": train_loss,
#                 "valid_loss": valid_loss,
#                 "test_acc": test_acc,
#                 "test_mae": test_mae,
#                 "test_corr": test_corr,
#                 "test_f_score": test_f_score,
#                 "test_acc7": test_acc_7,
#                 "test_acc5": test_acc_5
#             }
#
#         print(f"Epoch {epoch_i + 1} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
#
#         best_model_state_dict = model.state_dict()  # 保存最佳模型的状态字典
#
#     # 保存最佳模型
#     if best_model_state_dict is not None:
#         model_path = 'best_model.pth'
#         torch.save(best_model_state_dict, model_path)
#         print(f"Best model saved to {model_path}")
#
#         # 加载最佳模型进行最终可视化
#         model.load_state_dict(torch.load('best_model.pth'))
#         preds, labels, features = test_epoch(model, test_dataloader)
#
#         # 可视化多模态特征
#         plot_tsne(features, labels, mode='multimodal')
#
#     return best_metrics
#
#
# def plot_tsne(features_dict, labels, mode='multimodal', n_samples=1000):
#     """
#     参数说明：
#     features_dict - 包含各模态特征的字典
#     labels - 原始连续值标签
#     mode - 选择可视化的特征模态
#     n_samples - 最大采样数量
#     """
#     # ========== 仅修改这部分 ==========
#     # 转换为7分类标签（-3到3）
#     clipped_labels = np.clip(labels, -2., 2.)
#     class_labels = np.round(clipped_labels).astype(int) + 2  # 映射到0-6
#
#     # 设置7色方案（保持原深色风格）
#     DARK_PALETTE = {
#         # 0: "#1a1334",  # 深紫
#         # 1: "#01545a",  # 墨绿
#         # 2: "#017351",  # 深绿
#         # 3: "#03c383",  # 青绿
#         # 4: "#aad962",  # 黄绿
#         # 5: "#fbbf45",  # 橙黄
#         # 6: "#ef6a32"  # 橙红
#         0: "#2F4F4F",
#         1: "#228B22",
#         2: "#808080",
#         3: "#FF8C00",
#         4: "#8B0000"
#     }
#     # ========== 修改结束 ==========
#
#     # 智能采样（保持原逻辑）
#     if len(labels) > n_samples:
#         idx = np.random.choice(len(labels), n_samples, replace=False)
#     else:
#         idx = np.arange(len(labels))
#
#     # 准备数据（保持原逻辑）
#     sampled_labels = class_labels[idx]  # 使用新标签
#     features = {
#         'all': [
#             features_dict['text'][idx],
#             features_dict['vision'][idx],
#             features_dict['audio'][idx],
#             features_dict['multimodal'][idx]
#         ],
#         'single': [features_dict[mode][idx]]
#     }['all' if mode == 'all' else 'single']
#
#     # 创建画布（保持原逻辑）
#     plt.figure(figsize=(15, 10) if mode == 'all' else (8, 6))
#     sns.set(style="white")
#
#     # 绘制TSNE（仅修改循环部分）
#     for i, feat in enumerate(features):
#         tsne = TSNE(n_components=2, perplexity=30, n_iter=500)
#         embed = tsne.fit_transform(feat)
#
#         if mode == 'all':
#             plt.subplot(2, 2, i + 1)
#
#         # ========== 修改绘制逻辑 ==========
#         for lbl in range(5):  # 遍历0-6
#             mask = (sampled_labels == lbl)
#             plt.scatter(embed[mask, 0], embed[mask, 1],
#                         color=DARK_PALETTE[lbl],
#                         label=f'Score {lbl - 2}',  # 显示-3到+3
#                         alpha=0.7,
#                         s=30)
#         # ========== 修改结束 ==========
#
#         # 坐标轴设置（保持原逻辑）
#         plt.xticks([])
#         plt.yticks([])
#         plt.xlabel('')
#         plt.ylabel('')
#
#         # 图例设置（保持原位置）
#         if i == 0:
#             plt.legend(frameon=True,
#                        edgecolor='black',
#                        facecolor='white',
#                        title="Sentiment Score")
#
#     # 保存输出（保持原逻辑）
#     plt.tight_layout()
#     plt.savefig(f'tsne_{mode}.png', dpi=600, bbox_inches='tight')
#     plt.close()
#
#
# def main():
#     """Modified: 恢复单折训练主流程"""
#     set_random_seed(args.seed)
#
#     # 初始化数据加载器
#     train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps = set_up_data_loader()
#
#     # 初始化模型
#     model = get_model(args.model, 1, os.path.join('./pre_trained_model', args.model))
#     model.to(DEVICE)
#
#     # 优化器设置
#     optimizer = AdamW(model.parameters(), lr=args.learning_rate)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=int(args.warmup_proportion * num_train_optimization_steps),
#         num_training_steps=num_train_optimization_steps
#     )
#
#     # 训练模型
#     best_metrics = train_loop(
#         model,
#         train_dataloader,
#         dev_dataloader,
#         test_dataloader,  # 传入测试集
#         optimizer,
#         scheduler,
#         args.output_file,
#         args.gradient_accumulation_step
#     )
#
#     # 保存结果
#     with open(args.output_file, 'w') as f:
#         f.write("=== Final Results ===\n")
#         for k, v in best_metrics.items():
#             f.write(f"{k}: {v}\n")
#
#     # 打印结果
#     print("\nTraining Completed!")
#     print(pd.DataFrame([best_metrics]))
#
# if __name__ == "__main__":
#     main()
# from __future__ import absolute_import, division, print_function
# import argparse
# import os
# import random
# import numpy as np
# from pytorch_transformers import AdamW
# from sklearn.metrics import accuracy_score, f1_score
# import torch
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from tqdm import tqdm
# from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, get_linear_schedule_with_warmup
# from bert import MAG_BertForSequenceClassification
# from xlnet import MAG_XLNetForSequenceClassification
# from roberta import MAG_RobertaForSequenceClassification
# from argparse_utils import seed
# from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE
# import warnings
# import logging
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# import pickle
# from scipy.stats import pearsonr
# from sklearn.model_selection import KFold
# import torch.nn.functional as F
#
#
# warnings.filterwarnings('ignore')
#
# # 全局设备设置
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Multimodal Sentiment Analysis")
#     # 新增多任务参数
#     parser.add_argument('--loss_plot', type=str, default='multitask_loss.png',
#                         help='Path to save the loss curves visualization')
#     parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for knowledge distillation')
#     parser.add_argument('--kd_weight', type=float, default=0.1, help='Weight for knowledge distillation loss')
#     parser.add_argument('--m_loss_weight', type=float, default=0.8, help='Weight for multimodal task loss')
#     parser.add_argument('--cos_loss_weight', type=float, default=0.1, help='Weight for cosine similarity loss')
#
#     parser.add_argument('--output_file', type=str, default='./results.txt',
#                         help='Path to the file where best metrics will be saved.')
#     parser.add_argument("--model_name_or_path", default='./pretrained_model/roberta/', type=str,
#                         help="Path to pre-trained model or shortcut name")
#     parser.add_argument("--dataset", type=str, choices=["mosi", "mosei", "sims"], default="mosi", help="选择数据集")
#     parser.add_argument("--max_seq_length", type=int, default=50, help="最大序列长度")
#     parser.add_argument("--train_batch_size", type=int, default=32, help="训练批次大小")
#     parser.add_argument("--dev_batch_size", type=int, default=128, help="开发批次大小")
#     parser.add_argument("--test_batch_size", type=int, default=128, help="测试批次大小")
#     parser.add_argument("--model", type=str, choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base"],
#                         default="xlnet-base-cased", help="选择模型")
#     parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
#     parser.add_argument("--gradient_accumulation_step", type=int, default=1, help="梯度累积步数")
#     parser.add_argument("--warmup_proportion", type=float, default=0.2, help="预热比例")
#     parser.add_argument("--seed", type=seed, default="random", help="随机种子")
#
#     # 两阶段训练参数
#     parser.add_argument('--n_epochs_stage_1', type=int, default=40, help='Number of epochs for stage 1')
#     parser.add_argument('--lr_stage_1', type=float, default=1e-5, help='Learning rate for stage 1')
#     parser.add_argument('--n_epochs_stage_2', type=int, default=20, help='Number of epochs for stage 2')
#     parser.add_argument('--lr_stage_2', type=float, default=1e-5, help='Learning rate for stage 2')
#
#     return parser.parse_args()
#
# args = parse_arguments()
#
#
# # 设置日志配置
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# class InputFeatures:
#     def __init__(self, input_ids, input_mask, segment_ids=None, visual=None, acoustic=None, label_id=None):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.visual = visual
#         self.acoustic = acoustic
#         self.label_id = label_id
#
#
# def prepare_input(tokens, visual, acoustic, tokenizer, model_type, max_seq_length):
#     if model_type == "bert-base-uncased":
#         return prepare_bert_input(tokens, visual, acoustic, tokenizer, max_seq_length)
#     elif model_type == "xlnet-base-cased":
#         return prepare_xlnet_input(tokens, visual, acoustic, tokenizer, max_seq_length)
#     elif model_type == "roberta-base":
#         return prepare_roberta_input(tokens, visual, acoustic, tokenizer, max_seq_length)
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}")
#
#
# def _pad_features(features, pad_value, max_seq_length):
#     """Helper function to pad features."""
#     padded_features = []
#     for feature in features:
#         pad_length = max_seq_length - len(feature)
#         if pad_length > 0:
#             padded_feature = feature + [pad_value] * pad_length
#         elif pad_length < 0:
#             padded_feature = feature[:max_seq_length]
#         else:
#             padded_feature = feature
#         padded_features.append(padded_feature)
#     return padded_features
#
#
# def prepare_common_input(tokens, visual, acoustic, tokenizer, bos_token, eos_token, max_seq_length):
#     tokens = [bos_token] + tokens + [eos_token]
#
#     # Pad zero vectors for acoustic / visual to account for BOS/EOS tokens
#     acoustic = np.pad(acoustic, ((1, 1), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((1, 1), (0, 0)), mode='constant')
#
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     attention_mask = [1] * len(input_ids)
#
#     input_ids, attention_mask = _pad_features(
#         [input_ids, attention_mask], tokenizer.pad_token_id, max_seq_length)
#     acoustic = np.pad(acoustic, ((0, max_seq_length - acoustic.shape[0]), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((0, max_seq_length - visual.shape[0]), (0, 0)), mode='constant')
#
#     return input_ids, visual, acoustic, attention_mask
#
#
# def prepare_bert_input(tokens, visual, acoustic, tokenizer, max_seq_length):
#     CLS = tokenizer.cls_token
#     SEP = tokenizer.sep_token
#     input_ids, visual, acoustic, input_mask = prepare_common_input(tokens, visual, acoustic, tokenizer, CLS, SEP, max_seq_length)
#     segment_ids = [0] * len(input_ids)
#     return input_ids, visual, acoustic, input_mask, segment_ids
#
#
# def prepare_xlnet_input(tokens, visual, acoustic, tokenizer, max_seq_length):
#     CLS = tokenizer.cls_token
#     SEP = tokenizer.sep_token
#     PAD_ID = tokenizer.pad_token_id
#
#     tokens = tokens + [SEP] + [CLS]
#     acoustic = np.pad(acoustic, ((0, 2), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((0, 2), (0, 0)), mode='constant')
#
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     input_mask = [1] * len(input_ids)
#     segment_ids = [0] * (len(tokens) - 1) + [2]
#
#     input_ids, input_mask, segment_ids = _pad_features(
#         [input_ids, input_mask, segment_ids], PAD_ID, max_seq_length)
#     acoustic = np.pad(acoustic, ((max_seq_length - acoustic.shape[0], 0), (0, 0)), mode='constant')
#     visual = np.pad(visual, ((max_seq_length - visual.shape[0], 0), (0, 0)), mode='constant')
#
#     return input_ids, visual, acoustic, input_mask, segment_ids
#
#
# def prepare_roberta_input(tokens, visual, acoustic, tokenizer, max_seq_length):
#     input_ids, visual, acoustic, attention_mask = prepare_common_input(tokens, visual, acoustic, tokenizer, tokenizer.bos_token, tokenizer.eos_token, max_seq_length)
#     return input_ids, visual, acoustic, attention_mask, None  # RoBERTa does not use segment_ids
#
#
# def convert_to_features(examples, max_seq_length, tokenizer, model_type):
#     features = []
#     valid_example_count = 0
#     for ex_index, example in enumerate(examples):
#         if isinstance(example, tuple) and len(example) == 3:
#             (words, visual, acoustic), label_id, _ = example
#             tokens, inversions = [], []
#             for idx, word in enumerate(words):
#                 tokenized = tokenizer.tokenize(word)
#                 tokens.extend(tokenized)
#                 inversions.extend([idx] * len(tokenized))
#
#             assert len(tokens) == len(inversions)
#
#             aligned_visual = [visual[idx] for idx in inversions]
#             aligned_acoustic = [acoustic[idx] for idx in inversions]
#
#             visual = np.array(aligned_visual)
#             acoustic = np.array(aligned_acoustic)
#
#             if len(tokens) > max_seq_length - 2:
#                 tokens = tokens[:max_seq_length - 2]
#                 acoustic = acoustic[:max_seq_length - 2]
#                 visual = visual[:max_seq_length - 2]
#
#             input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
#                 tokens, visual, acoustic, tokenizer, model_type, max_seq_length
#             )
#
#             # Ensure all arrays/lists have the correct length
#             assert len(input_ids) == max_seq_length, f"Input IDs length ({len(input_ids)}) does not match max_seq_length ({max_seq_length})."
#             assert len(input_mask) == max_seq_length, f"Input mask length ({len(input_mask)}) does not match max_seq_length ({max_seq_length})."
#             if segment_ids is not None:
#                 assert len(segment_ids) == max_seq_length, f"Segment IDs length ({len(segment_ids)}) does not match max_seq_length ({max_seq_length})."
#             assert acoustic.shape[0] == max_seq_length, f"Acoustic features length ({acoustic.shape[0]}) does not match max_seq_length ({max_seq_length})."
#             assert visual.shape[0] == max_seq_length, f"Visual features length ({visual.shape[0]}) does not match max_seq_length ({max_seq_length})."
#
#             features.append(
#                 InputFeatures(
#                     input_ids=input_ids,
#                     input_mask=input_mask,
#                     segment_ids=segment_ids,  # This can be None for RoBERTa
#                     visual=visual,
#                     acoustic=acoustic,
#                     label_id=label_id,
#                 )
#             )
#             valid_example_count += 1
#         else:
#             logger.error(f"Example {ex_index} has incorrect structure: {example}")
#
#     if valid_example_count == 0:
#         logger.error("No valid examples were found in the input data.")
#     return features
#
#
# def get_tokenizer(model_name, local_dir):
#     """
#     获取与模型类型相匹配的分词器，并强制使用本地文件。
#     """
#     if model_name.startswith('bert'):
#         logger.info(f"Loading BERT tokenizer from {local_dir}")
#         tokenizer = BertTokenizer.from_pretrained(local_dir, local_files_only=True)
#     elif model_name.startswith('xlnet'):
#         logger.info(f"Loading XLNet tokenizer from {local_dir}")
#         tokenizer = XLNetTokenizer.from_pretrained(local_dir, local_files_only=True)
#     elif model_name.startswith('roberta'):
#         logger.info(f"Loading RoBERTa tokenizer from {local_dir}")
#         tokenizer = RobertaTokenizer.from_pretrained(local_dir, local_files_only=True)
#     else:
#         raise ValueError(f"Unsupported model type: {model_name}")
#
#     return tokenizer
#
#
# def get_model(model_name, num_labels, local_dir):
#     """
#     根据提供的模型名称获取相应的预训练模型，并强制使用本地文件。
#     """
#     if model_name == "bert-base-uncased":
#         logger.info(f"Loading BERT model from {local_dir}")
#         model = MAG_BertForSequenceClassification.from_pretrained(
#             local_dir,
#             local_files_only=True,
#             num_labels=num_labels
#         )
#     elif model_name == "xlnet-base-cased":
#         logger.info(f"Loading XLNet model from {local_dir}")
#         model = MAG_XLNetForSequenceClassification.from_pretrained(
#             local_dir,
#             local_files_only=True,
#             num_labels=num_labels
#         )
#     elif model_name == "roberta-base":
#         logger.info(f"Loading RoBERTa model from {local_dir}")
#         model = MAG_RobertaForSequenceClassification.from_pretrained(
#             local_dir,
#             local_files_only=True,
#             num_labels=num_labels
#         )
#     else:
#         raise ValueError("Unsupported model name: {}".format(model_name))
#
#     return model
#
#
# def get_appropriate_dataset(data, model_type, max_seq_length):
#     """
#     根据提供的数据和模型类型生成适当的数据集。
#     """
#     # 确保本地目录存在并且包含所需文件
#     local_model_dir = os.path.join('./pre_trained_model', model_type)
#     if not os.path.exists(local_model_dir):
#         raise FileNotFoundError(f"The specified local directory does not exist: {local_model_dir}")
#
#     logger.info("Initializing tokenizer for %s...", model_type)
#     tokenizer = get_tokenizer(model_type, local_dir=local_model_dir)
#
#     logger.info("Converting data to features...")
#     features = convert_to_features(data, max_seq_length, tokenizer, model_type)
#
#     # 将特征转换为张量
#     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
#
#     # Only create segment_ids tensor if it's not None and model requires it
#     if features[0].segment_ids is not None:
#         all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
#     else:
#         all_segment_ids = None
#
#     all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
#     all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
#     all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
#
#     # Ensure that all tensors have the correct shape
#     assert all_input_ids.size(0) == len(features), "Mismatch in input IDs length"
#     assert all_input_mask.size(0) == len(features), "Mismatch in input mask length"
#     if all_segment_ids is not None:
#         assert all_segment_ids.size(0) == len(features), "Mismatch in segment IDs length"
#     assert all_visual.size(0) == len(features), "Mismatch in visual features length"
#     assert all_acoustic.size(0) == len(features), "Mismatch in acoustic features length"
#     assert all_label_ids.size(0) == len(features), "Mismatch in label IDs length"
#
#     # Create dataset with or without segment_ids based on model requirements
#     if all_segment_ids is not None:
#         dataset = TensorDataset(
#             all_input_ids,
#             all_visual,
#             all_acoustic,
#             all_input_mask,
#             all_segment_ids,
#             all_label_ids
#         )
#     else:
#         dataset = TensorDataset(
#             all_input_ids,
#             all_visual,
#             all_acoustic,
#             all_input_mask,
#             all_label_ids
#         )
#
#     logger.info("Dataset created successfully.")
#
#     return dataset
#
# def set_up_data_loader():
#     """
#     加载数据并设置数据加载器。
#
#     返回:
#     tuple: 包含训练、验证、测试数据加载器以及优化步数和分词器的元组。
#     """
#     args = parse_arguments()
#     data_path = f"datasets/{args.dataset}.pkl"
#     logger.info(f"Loading dataset from {data_path}")
#     with open(data_path, "rb") as handle:
#         data = pickle.load(handle)
#
#     # 获取训练、验证和测试数据
#     train_data = data["train"]
#     dev_data = data["dev"]
#     test_data = data["test"]
#
#     # 合并训练集和验证集用于五折交叉验证
#     full_train_data = train_data + dev_data
#
#     return full_train_data, test_data
#
#
# def set_random_seed(seed: int):
#     print("Seed: {}".format(seed))
#
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.deterministic = True
#
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#
# def prepare_optimizer_and_scheduler(model, num_train_optimization_steps, learning_rate, warmup_proportion):
#     """
#     初始化优化器和学习率调度器。
#     """
#     param_optimizer = list(model.named_parameters())
#     no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#             "weight_decay": 0.01,
#         },
#         {
#             "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#             "weight_decay": 0.0,
#         },
#     ]
#
#     optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=warmup_proportion * num_train_optimization_steps,
#         num_training_steps=num_train_optimization_steps,
#     )
#
#     return optimizer, scheduler
#
#
# def prep_for_training(num_train_optimization_steps_stage_1, num_train_optimization_steps_stage_2):
#     args = parse_arguments()
#     local_model_dir = os.path.join('./pre_trained_model', args.model)
#     tokenizer = get_tokenizer(args.model, local_dir=local_model_dir)
#     model = get_model(args.model, num_labels=1, local_dir=local_model_dir)
#     model.to(DEVICE)
#
#     # 初始化两阶段优化器和调度器
#     optimizer_1, scheduler_1 = prepare_optimizer_and_scheduler(
#         model,
#         num_train_optimization_steps_stage_1,
#         args.lr_stage_1,
#         args.warmup_proportion
#     )
#
#     optimizer_2, scheduler_2 = prepare_optimizer_and_scheduler(
#         model,
#         num_train_optimization_steps_stage_2,
#         args.lr_stage_2,
#         args.warmup_proportion
#     )
#
#     return model, (optimizer_1, optimizer_2), (scheduler_1, scheduler_2), tokenizer
#
#
# def process_batch(batch):
#     """
#     Process and unpack the batch data.
#     """
#     batch = tuple(t.to(DEVICE) for t in batch)
#     if len(batch) == 6:
#         input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
#     elif len(batch) == 5:
#         input_ids, visual, acoustic, input_mask, label_ids = batch
#         segment_ids = None
#     else:
#         raise ValueError("Unexpected number of items in batch")
#
#     visual = torch.squeeze(visual, 1)
#     acoustic = torch.squeeze(acoustic, 1)
#
#     return input_ids, visual, acoustic, input_mask, segment_ids, label_ids
#
#
# def compute_loss(logits, labels):
#     """
#     Compute the MSE loss.
#     """
#     mse_loss = torch.nn.MSELoss()(logits.view(-1), labels.view(-1))
#     return mse_loss
#
#
# class CosineLoss(torch.nn.Module):
#     def __init__(self):
#         super(CosineLoss, self).__init__()
#
#     def forward(self, feature1, feature2):
#         # Compute cosine similarity between two tensors of shape (batch_size, dimension)
#         cos_sim = torch.nn.functional.cosine_similarity(feature1, feature2, dim=1)  # Shape: (batch_size,)
#
#         # The loss is the mean of (1 - cos_sim) across the batch
#         loss = (1 - cos_sim).mean()
#
#         return loss
#
#
# def train_epoch(model, dataloader, optimizer, scheduler,
#                 gradient_accumulation_steps=1, temperature=2.0, kd_weight=0.1, stage=1):
#     model.train()
#     tr_loss = 0
#     nb_tr_steps = 0
#     cos_similarity = CosineLoss()
#     kd_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
#     BASE_WEIGHTS = {
#         'm_loss': 0.7,
#         'cos_loss': 0.2,
#         'kd_loss': 0.1
#     }
#
#     # 新增损失记录列表
#     mse_losses = []
#     cos_losses = []
#     kd_losses = []
#
#     for step, batch in enumerate(tqdm(dataloader, desc="Training")):
#         input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)
#
#         # 前向传播
#         outputs, t, a, v = model(
#             input_ids,
#             visual,
#             acoustic,
#             token_type_ids=segment_ids,
#             attention_mask=input_mask,
#             labels=None
#         )
#         logits = outputs[0]
#
#         # 计算多模态任务损失
#         total_loss_m = compute_loss(logits, label_ids)
#
#         if stage == 1:
#             total_loss = total_loss_m
#             mse_losses.append(total_loss.item())
#         else:
#             # 计算余弦相似度损失
#             cos_loss = (
#                     cos_similarity(t, logits.detach()) * 0.8 +
#                     cos_similarity(v, logits.detach()) * 0.1 +
#                     cos_similarity(a, logits.detach()) * 0.1
#             )
#
#             # 计算知识蒸馏损失
#             teacher_soft = F.softmax(logits / temperature, dim=1)
#             student_soft_t = F.log_softmax(t / temperature, dim=1)
#             student_soft_v = F.log_softmax(v / temperature, dim=1)
#             student_soft_a = F.log_softmax(a / temperature, dim=1)
#
#             kd_loss = (
#                               kd_loss_fn(student_soft_t, teacher_soft) * 0.8 +
#                               kd_loss_fn(student_soft_v, teacher_soft) * 0.1 +
#                               kd_loss_fn(student_soft_a, teacher_soft) * 0.1
#                       ) * (temperature ** 2)
#
#             # 组合损失
#             total_base = sum(BASE_WEIGHTS.values())
#             total_loss = (
#                 (BASE_WEIGHTS['m_loss'] / total_base) * total_loss_m +
#                 (BASE_WEIGHTS['cos_loss'] / total_base) * cos_loss +
#                 (BASE_WEIGHTS['kd_loss'] / total_base) * kd_loss
#             )
#
#             # 记录各子损失
#             mse_losses.append(total_loss_m.item())
#             cos_losses.append(cos_loss.item())
#             kd_losses.append(kd_loss.item())
#
#
#         if gradient_accumulation_steps > 1:
#             total_loss = total_loss / gradient_accumulation_steps
#
#         total_loss.backward()
#
#         tr_loss += total_loss.item()
#         nb_tr_steps += 1
#
#         if (step + 1) % gradient_accumulation_steps == 0:
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()
#
#     return tr_loss / nb_tr_steps, mse_losses, cos_losses, kd_losses  # 修改返回值
#
# def eval_epoch(model, dataloader, temperature=2.0, kd_weight=0.1, stage=1):
#     """Modified: 简化验证过程"""
#     model.eval()
#     eval_loss = 0
#     nb_eval_steps = 0
#     cos_similarity = CosineLoss()
#     kd_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
#
#     BASE_WEIGHTS = {
#         'm_loss': 0.7,
#         'cos_loss': 0.2,
#         'kd_loss': 0.1
#     }
#
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Evaluating"):
#             input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)
#
#             # 前向传播
#             outputs, t, a, v = model(
#                 input_ids,
#                 visual,
#                 acoustic,
#                 token_type_ids=segment_ids,
#                 attention_mask=input_mask,
#                 labels=None
#             )
#             logits = outputs[0]
#
#             # 计算多模态任务损失
#             total_loss_m = compute_loss(logits, label_ids)
#
#             if stage == 1:
#                 total_loss = total_loss_m
#             else:
#                 # 计算余弦相似度损失
#                 cos_loss = (
#                     cos_similarity(t, logits.detach()) * 0.8 +
#                     cos_similarity(v, logits.detach()) * 0.1 +
#                     cos_similarity(a, logits.detach()) * 0.1
#                 )
#
#                 # 计算知识蒸馏损失
#                 teacher_soft = F.softmax(logits / temperature, dim=1)
#                 student_soft_t = F.log_softmax(t / temperature, dim=1)
#                 student_soft_v = F.log_softmax(v / temperature, dim=1)
#                 student_soft_a = F.log_softmax(a / temperature, dim=1)
#
#                 kd_loss = (
#                               kd_loss_fn(student_soft_t, teacher_soft) * 0.8 +
#                               kd_loss_fn(student_soft_v, teacher_soft) * 0.1 +
#                               kd_loss_fn(student_soft_a, teacher_soft) * 0.1
#                           ) * (temperature ** 2)
#
#                 # 组合损失
#                 total_base = sum(BASE_WEIGHTS.values())
#                 total_loss = (
#                     (BASE_WEIGHTS['m_loss'] / total_base) * total_loss_m +
#                     (BASE_WEIGHTS['cos_loss'] / total_base) * cos_loss +
#                     (BASE_WEIGHTS['kd_loss'] / total_base) * kd_weight * kd_loss
#                 )
#
#             eval_loss += total_loss.item()
#             nb_eval_steps += 1
#
#     return eval_loss / nb_eval_steps
#
#
# def test_epoch(model: torch.nn.Module, dataloader: DataLoader):
#     """
#     Test model and collect predictions and labels.
#     """
#     model.eval()
#     preds, labels = [], []
#
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Testing"):
#             input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)
#             outputs, *_ = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask,
#                             labels=None)
#             logits = outputs[0].detach().cpu().numpy()
#             label_ids = label_ids.detach().cpu().numpy()
#             preds.extend(np.squeeze(logits).tolist())
#             labels.extend(np.squeeze(label_ids).tolist())
#
#     return np.array(preds), np.array(labels)
#
#
# import matplotlib.pyplot as plt
#
#
# def plot_loss_curves(stage_losses, output_file='loss_curves.png'):
#     plt.figure(figsize=(15, 6))
#
#     # 阶段1/2 MSE对比
#     plt.subplot(1, 2, 1)
#     plt.plot(stage_losses['stage1_mse'], label='Stage 1 MSE Loss')
#     plt.plot(stage_losses['stage2_mse'], label='Stage 2 MSE Loss')
#     plt.title('MSE Loss Comparison Between Stages')
#     plt.xlabel('Training Steps')
#     plt.ylabel('MSE Loss')
#     plt.legend()
#
#     # 阶段2子损失
#     plt.subplot(1, 2, 2)
#     plt.plot(stage_losses['stage2_cos'], label='Cosine Similarity Loss')
#     plt.plot(stage_losses['stage2_kd'], label='Knowledge Distillation Loss')
#     plt.title('Stage 2 Loss Components')
#     plt.xlabel('Training Steps')
#     plt.ylabel('Loss Value')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.savefig(output_file)
#     plt.close()
#
#
# def calculate_metrics(preds, y_test, use_zero=False):
#     """
#     Calculate metrics for test scores.
#     """
#
#     non_zeros = [i for i, e in enumerate(y_test) if e != 0 or use_zero]
#     preds, y_test = preds[non_zeros], y_test[non_zeros]
#
#     mae = np.mean(np.abs(preds - y_test))
#     corr = pearsonr(preds, y_test)[0]
#
#     binary_preds = preds > 0
#     binary_labels = y_test > 0
#
#     f_score = f1_score(binary_labels, binary_preds, average='weighted')
#     acc = accuracy_score(binary_labels, binary_preds)
#
#     return acc, mae, corr, f_score
#
#
# def multiclass_accuracy(preds, y_test):
#     test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
#     test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
#     test_preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
#     test_truth_a5 = np.clip(y_test, a_min=-2., a_max=2.)
#
#     mult_a7 = np.sum(np.round(test_preds_a7) == np.round(test_truth_a7)) / float(len(test_truth_a7))
#     mult_a5 = np.sum(np.round(test_preds_a5) == np.round(test_truth_a5)) / float(len(test_truth_a5))
#
#     return mult_a7, mult_a5
#
#
# def train_loop(
#         model,
#         train_dataloader,
#         validation_dataloader,
#         test_dataloader,
#         optimizers,
#         schedulers,
#         stage_configs,
#         output_file='results.txt',
#         gradient_accumulation_steps=1,
#         temperature=2.0,
#         kd_weight=0.1
# ):
#     best_valid_loss = float('inf')
#     best_metrics = None
#     stage_losses = {  # 新增损失存储结构
#         'stage1_mse': [],
#         'stage2_mse': [],
#         'stage2_cos': [],
#         'stage2_kd': []
#     }
#
#     for stage_i, stage_config in enumerate(stage_configs, start=1):
#         print(f"\nStarting Stage {stage_i}/{len(stage_configs)}")
#         optimizer = optimizers[stage_i - 1]
#         scheduler = schedulers[stage_i - 1]
#         n_epochs = stage_config['n_epochs']
#
#         for epoch_i in range(n_epochs):
#             print(f"Stage {stage_i}, Epoch {epoch_i + 1}/{n_epochs}", end='\r')
#
#             train_loss, mse, cos, kd = train_epoch(  # 接收新返回值
#                 model,
#                 train_dataloader,
#                 optimizer,
#                 scheduler,
#                 gradient_accumulation_steps,
#                 temperature,
#                 kd_weight,
#                 stage=stage_i
#             )
#
#             # 存储损失数据
#             if stage_i == 1:
#                 stage_losses['stage1_mse'].extend(mse)
#             else:
#                 stage_losses['stage2_mse'].extend(mse)
#                 stage_losses['stage2_cos'].extend(cos)
#                 stage_losses['stage2_kd'].extend(kd)
#
#             # 修正后的eval_epoch调用
#             valid_loss = eval_epoch(
#                 model,
#                 validation_dataloader,
#                 temperature,  # 传递temperature
#                 kd_weight,    # 传递kd_weight
#                 stage=stage_i  # 传递stage
#             )
#
#             # 测试阶段
#             preds, y_test = test_epoch(model, test_dataloader)
#             test_acc, test_mae, test_corr, test_f_score = calculate_metrics(preds, y_test)
#             test_acc7, test_acc5 = multiclass_accuracy(preds, y_test)
#
#             # 更新最佳指标
#             if valid_loss < best_valid_loss:
#                 best_valid_loss = valid_loss
#                 best_metrics = {
#                     "train_loss": train_loss,
#                     "valid_loss": valid_loss,
#                     "test_acc": test_acc,
#                     "test_mae": test_mae,
#                     "test_corr": test_corr,
#                     "test_f_score": test_f_score,
#                     "test_acc7": test_acc7,
#                     "test_acc5": test_acc5,
#                     "epoch": epoch_i + 1,
#                     "stage": stage_i
#                 }
#
#     return best_metrics
#
#
# def main():
#     args = parse_arguments()
#     set_random_seed(args.seed)
#
#     # 加载数据集
#     full_train_data, test_data = set_up_data_loader()
#
#     # 初始化KFold
#     kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
#     all_metrics = []
#
#     # 保存每一折结果的文件
#     fold_results_file = f"{os.path.splitext(args.output_file)[0]}_fold_results.txt"
#
#     for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_data), 1):
#         logger.info(f"Processing Fold {fold}")
#
#         # 生成当前折的训练集和验证集
#         current_train = [full_train_data[i] for i in train_idx]
#         current_val = [full_train_data[i] for i in val_idx]
#
#         # 创建数据集
#         train_dataset = get_appropriate_dataset(current_train, args.model, args.max_seq_length)
#         val_dataset = get_appropriate_dataset(current_val, args.model, args.max_seq_length)
#         test_dataset = get_appropriate_dataset(test_data, args.model, args.max_seq_length)
#
#         # 计算训练步数
#         num_train_steps_stage1 = (
#             len(train_dataset) // args.train_batch_size // args.gradient_accumulation_step
#         ) * args.n_epochs_stage_1
#         num_train_steps_stage2 = (
#             len(train_dataset) // args.train_batch_size // args.gradient_accumulation_step
#         ) * args.n_epochs_stage_2
#
#         # 初始化模型和优化器
#         model, optimizers, schedulers, _ = prep_for_training(num_train_steps_stage1, num_train_steps_stage2)
#
#         # 创建数据加载器
#         train_loader = DataLoader(
#             train_dataset,
#             sampler=RandomSampler(train_dataset),
#             batch_size=args.train_batch_size,
#             drop_last=True
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             sampler=SequentialSampler(val_dataset),
#             batch_size=args.dev_batch_size
#         )
#         test_loader = DataLoader(
#             test_dataset,
#             sampler=SequentialSampler(test_dataset),
#             batch_size=args.test_batch_size
#         )
#
#         # 定义阶段配置
#         stage_configs = [
#             {'n_epochs': args.n_epochs_stage_1, 'lr': args.lr_stage_1},
#             {'n_epochs': args.n_epochs_stage_2, 'lr': args.lr_stage_2}
#         ]
#
#         # 训练循环
#         best_metrics, stage_losses = train_loop(
#             model=model,
#             train_dataloader=train_loader,
#             validation_dataloader=val_loader,
#             test_dataloader=test_loader,
#             optimizers=optimizers,
#             schedulers=schedulers,
#             stage_configs=stage_configs,
#             output_file=f"{args.output_file}.fold{fold}",
#             gradient_accumulation_steps=args.gradient_accumulation_step,
#             temperature=args.temperature,
#             kd_weight=args.kd_weight
#         )
#
#         # 生成可视化
#         plot_loss_curves(stage_losses, output_file='multitask_loss_curves.png')
#
#         all_metrics.append(best_metrics)
#
#     # 五折全部完成后保存结果
#     with open(fold_results_file, 'w') as f:  # 使用 'w' 模式覆盖之前的内容
#         f.write("=== 所有五折最佳结果 ===\n")
#         for fold_idx, metrics in enumerate(all_metrics, 1):
#             f.write(f"--- Fold {fold_idx} ---\n")
#             for key, value in metrics.items():
#                 f.write(f"{key}: {value}\n")
#             f.write("\n")
#
#     # 计算平均结果
#     avg_metrics = {}
#     metric_keys = ['train_loss', 'valid_loss', 'test_acc', 'test_mae', 'test_corr', 'test_f_score', 'test_acc7', 'test_acc5']
#     for key in metric_keys:
#         values = [m[key] for m in all_metrics]
#         avg_metrics[key] = np.mean(values)
#         avg_metrics[f"{key}_std"] = np.std(values)
#
#     # 输出最终结果
#     logger.info("\n五折交叉验证平均结果：")
#     with open(args.output_file, 'a') as f:
#         f.write("\n=== 五折交叉验证结果 ===\n")
#         for key in metric_keys:
#             mean = avg_metrics[key]
#             std = avg_metrics[f"{key}_std"]
#             logger.info(f"{key}: {mean:.4f} (±{std:.4f})")
#             f.write(f"{key}: {mean:.4f} (±{std:.4f})\n")
#
# if __name__ == "__main__":
#     main(
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
from modeling import GCAA
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

        # 在train_epoch函数中计算复杂度的部分：
        # t = torch.randint(1, 30000, (1, 50), dtype=torch.long).to(DEVICE)  # 原始形状：(batch=1, seq_len=50)
        # v = torch.randn(1, 50, 47).to(DEVICE)
        # a = torch.randn(1, 50, 74).to(DEVICE)
        # s = torch.randint(0, 2, (1, 50), dtype=torch.long).to(DEVICE)
        # i = torch.randint(0, 2, (1, 50), dtype=torch.long).to(DEVICE)
        #
        # # 调整维度为XLNet所需的(seq_len, batch_size)
        # t = t.transpose(0, 1)  # 形状：(50, 1)
        # s = s.transpose(0, 1)  # token_type_ids形状：(50, 1)
        # i = i.transpose(0, 1)  # attention_mask形状：(50, 1)
        #
        # # 临时禁用记忆机制
        # model.transformer.mem_len = 0
        #
        # # 计算复杂度（此时klen=qlen=50，维度匹配）
        # flops, params = profile(model, inputs=(t, v, a, s, i, None), )
        # flops, params = clever_format([flops, params], "%.3f")
        # print(f"Total FLOPs: {flops}, Params: {params}")

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

# def test_epoch(dataloader: DataLoader, model_config_dir, model_weights_path, num_labels):
#     # 检查是否有可用的 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     try:
#         # 从指定目录加载模型配置
#         model = MAG_XLNetForSequenceClassification.from_pretrained(
#             model_config_dir,
#             local_files_only=True,
#             num_labels=num_labels
#         )
#         # 将模型移动到指定设备
#         model = model.to(device)
#     except Exception as e:
#         print(f"Error loading model configuration from {model_config_dir}: {e}")
#         return None, None, None
#
#     try:
#         # 加载模型权重
#         model.load_state_dict(torch.load(model_weights_path), strict=False)
#     except Exception as e:
#         print(f"Error loading model weights from {model_weights_path}: {e}")
#         return None, None, None
#     model.eval()
#     preds, labels = [], []
#     features = {
#         'text': [],
#         'vision': [],
#         'audio': [],
#         'multimodal': []
#     }
#
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Testing"):
#             input_ids, visual, acoustic, input_mask, segment_ids, label_ids = process_batch(batch)
#             input_ids = input_ids.to(device)
#
#             (
#                 logits, logits_text, logits_audio, logits_vision,
#                 kd_loss, cos_loss, kd_loss_t, kd_loss_v, kd_loss_a,
#                 cos_sim_t, cos_sim_v, cos_sim_a,
#                 text_features,  # 文本特征 [batch, hidden]
#                 vision_features,  # 视觉特征
#                 audio_features,  # 音频特征
#                 multimodal_features  # 多模态融合特征
#             ) = model(
#                 input_ids,
#                 visual,
#                 acoustic,
#                 token_type_ids=segment_ids,
#                 attention_mask=input_mask,
#                 labels=None
#             )
#
#             # 可视化第一个样本的注意力权重
#             if sample_idx == 0:
#                 attn_weights = model.GCAA.attn_weights
#                 tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
#                 visualize_attention_weights(attn_weights, tokens)
#
#
#             # Collect features
#             features['text'].append(text_features.cpu())
#             features['vision'].append(vision_features.cpu())
#             features['audio'].append(audio_features.cpu())
#             features['multimodal'].append(multimodal_features.cpu())
#
#             # Collect labels and predictions
#             logits = logits.detach().cpu().numpy()
#             label_ids = label_ids.detach().cpu().numpy()
#
#             # 处理单样本/单标签情况
#             if len(logits.shape) == 1:
#                 logits = logits[np.newaxis, :]
#             if num_labels == 1:
#                 logits = logits.squeeze(axis=1)
#
#             preds.extend(logits.tolist())
#             labels.extend(label_ids.tolist())
#
#
#     # Concatenate features
#     for key in features:
#         features[key] = torch.cat(features[key], dim=0).numpy()
#
#     # 添加特征尺寸校验
#     assert text_features.shape[1] == model.config.d_model, "特征维度不匹配"
#
#     # 添加内存清理
#     torch.cuda.empty_cache()
#
#     return np.array(preds), np.array(labels), features

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

        # 加载最佳模型进行最终可视化
        # model.load_state_dict(torch.load('best_model.pth'))
        # preds, labels, features = test_epoch(model, test_dataloader)

        # 转换标签为二分类
        # binary_labels = (labels > 0).astype(int)

        # 可视化多模态特征
        # plot_tsne(features, labels, mode='multimodal')

        # 生成动态演化图
        # generate_evolution_animation(epoch_features)

        # 在train_loop函数中
        # stage1_epochs = stage_configs[0]['n_epochs'] if stage_configs else 0

        # 添加边界检查
        # valid_stage1_epochs = min(stage1_epochs, len(train_loss_records['mse']))

        # plot_loss_curves(
        #     train_loss_records,
        #     val_loss_records,
        #     stage1_epochs=valid_stage1_epochs,
        #     output_file='loss_curves.png'
        # )

        # plot_stage2_alignment(alignment_records, "stage2_alignment.png")

        # plot_kl_3d_heatmap(kl_records)

        return best_metrics, train_loss_records, val_loss_records, alignment_records

#
# def generate_evolution_animation(epoch_features, output_file='feature_evolution.gif'):
#     if not epoch_features:
#         return
#
#     # 设置深色配色
#     COLOR_MAP = ['#075F6D', '#752306']
#
#     fig, ax = plt.subplots(figsize=(10, 8))
#     plt.axis('off')  # 移除所有坐标轴
#
#     def update(frame):
#         ax.clear()
#         data = epoch_features[frame]
#         tsne = TSNE(n_components=2)
#         embed = tsne.fit_transform(data['features']['multimodal'])
#
#         # 转换标签
#         binary_labels = (data['labels'] > 0).astype(int)
#
#         # 绘制二分类
#         for lbl in [0, 1]:
#             mask = binary_labels == lbl
#             ax.scatter(embed[mask, 0], embed[mask, 1],
#                        color=COLOR_MAP[lbl],
#                        alpha=0.7, s=30)
#
#         # 添加动态标题
#         ax.set_title(f"Epoch {data['epoch']} (Stage {data['stage']})",
#                      fontsize=12, pad=10)
#
#     ani = FuncAnimation(fig, update, frames=len(epoch_features), interval=800)
#
#     try:
#         ani.save(output_file, writer='pillow', fps=3)
#     except Exception as e:
#         print(f"动画保存失败: {str(e)}")


# def plot_tsne(features_dict, labels, mode='multimodal', n_samples=1000):
#     """
#     参数说明：
#     features_dict - 包含各模态特征的字典
#     labels - 原始连续值标签
#     mode - 选择可视化的特征模态
#     n_samples - 最大采样数量
#     """
#     # ========== 仅修改这部分 ==========
#     # 转换为7分类标签（-3到3）
#     clipped_labels = np.clip(labels, -2., 2.)
#     class_labels = np.round(clipped_labels).astype(int) + 2  # 映射到0-6
#
#     # 设置7色方案（保持原深色风格）
#     DARK_PALETTE = {
#         # 0: "#1a1334",  # 深紫
#         # 1: "#01545a",  # 墨绿
#         # 2: "#017351",  # 深绿
#         # 3: "#03c383",  # 青绿
#         # 4: "#aad962", # 黄绿
#         # 5: "#fbbf45",  # 橙黄
#         # 6: "#ef6a32"  # 橙红
#         0: "#2F4F4F",
#         1: "#228B22",
#         2: "#808080",
#         3: "#FF8C00",
#         4: "#8B0000"
#     }
#     # ========== 修改结束 ==========
#
#     # 智能采样（保持原逻辑）
#     if len(labels) > n_samples:
#         idx = np.random.choice(len(labels), n_samples, replace=False)
#     else:
#         idx = np.arange(len(labels))
#
#     # 准备数据（保持原逻辑）
#     sampled_labels = class_labels[idx]  # 使用新标签
#     features = {
#         'all': [
#             features_dict['text'][idx],
#             features_dict['vision'][idx],
#             features_dict['audio'][idx],
#             features_dict['multimodal'][idx]
#         ],
#         'single': [features_dict[mode][idx]]
#     }['all' if mode == 'all' else 'single']
#
#     # 创建画布（保持原逻辑）
#     plt.figure(figsize=(15, 10) if mode == 'all' else (8, 6))
#     sns.set(style="white")
#
#     # 绘制TSNE（仅修改循环部分）
#     for i, feat in enumerate(features):
#         tsne = TSNE(n_components=2, perplexity=30, n_iter=500)
#         embed = tsne.fit_transform(feat)
#
#         if mode == 'all':
#             plt.subplot(2, 2, i + 1)
#
#         # ========== 修改绘制逻辑 ==========
#         for lbl in range(5):  # 遍历0-6
#             mask = (sampled_labels == lbl)
#             plt.scatter(embed[mask, 0], embed[mask, 1],
#                         color=DARK_PALETTE[lbl],
#                         label=f'Score {lbl - 2}',  # 显示-3到+3
#                         alpha=0.7,
#                         s=30)
#         # ========== 修改结束 ==========
#
#         # 坐标轴设置（保持原逻辑）
#         plt.xticks([])
#         plt.yticks([])
#         plt.xlabel('')
#         plt.ylabel('')
#
#         # 图例设置（保持原位置）
#         if i == 0:
#             plt.legend(frameon=True,
#                        edgecolor='black',
#                        facecolor='white',
#                        title="Sentiment Score")
#
#     # 保存输出（保持原逻辑）
#     plt.tight_layout()
#     plt.savefig(f'tsne_{mode}.png', dpi=600, bbox_inches='tight')
#     plt.close()
# def plot_tsne(features_dict, labels, mode='multimodal', n_samples=1000):
#     """
#     参数说明：
#     features_dict - 包含各模态特征的字典
#     labels - 原始连续值标签（将被转换为二分类）
#     mode - 选择可视化的特征模态
#     n_samples - 最大采样数量
#     """
#     # 转换为二分类标签
#     binary_labels = (labels > 0).astype(int)
#
#     # 设置深色配色方案
#     DARK_PALETTE = {
#         0: "#075F6D",  # 深蓝色
#         1: "#752306"  # 深红色
#     }
#
#     # 智能采样
#     if len(labels) > n_samples:
#         idx = np.random.choice(len(labels), n_samples, replace=False)
#     else:
#         idx = np.arange(len(labels))
#
#     # 准备数据
#     sampled_labels = binary_labels[idx]
#     features = {
#         'all': [
#             features_dict['text'][idx],
#             features_dict['vision'][idx],
#             features_dict['audio'][idx],
#             features_dict['multimodal'][idx]
#         ],
#         'single': [features_dict[mode][idx]]
#     }['all' if mode == 'all' else 'single']
#
#     # 创建画布
#     plt.figure(figsize=(15, 10) if mode == 'all' else (8, 6))
#     sns.set(style="white")
#
#     # 绘制TSNE
#     for i, feat in enumerate(features):
#         tsne = TSNE(n_components=2, perplexity=30, n_iter=500)
#         embed = tsne.fit_transform(feat)
#
#         if mode == 'all':
#             plt.subplot(2, 2, i + 1)
#
#         # 绘制散点图
#         for lbl in [0, 1]:
#             mask = (sampled_labels == lbl)
#             plt.scatter(embed[mask, 0], embed[mask, 1],
#                         color=DARK_PALETTE[lbl],
#                         label=f'Positive' if lbl else 'Non-positive',
#                         alpha=0.7,  # 提高透明度
#                         s=30)  # 增大点尺寸
#
#         # 移除坐标轴
#         plt.xticks([])
#         plt.yticks([])
#         plt.xlabel('')
#         plt.ylabel('')
#
#         # 仅第一个子图显示图例
#         if i == 0:
#             plt.legend(frameon=True,
#                        edgecolor='black',
#                        facecolor='white')
#
#     plt.tight_layout()
#     plt.savefig(f'tsne_{mode}.png', dpi=600, bbox_inches='tight')
#     plt.close()


# def plot_stage2_alignment(alignment_records, output_path="stage2_alignment.png"):
#     """可视化第二阶段各模态与多模态表示的余弦相似度趋势"""
#     plt.figure(figsize=(10, 6))
#     plt.rcParams.update({
#         'font.family': 'DejaVu Sans',
#         'axes.edgecolor': '#333333',
#         'grid.color': '#d8d8d8',
#         'grid.alpha': 0.4,
#         'axes.titlepad': 15
#     })
#
#     # ========== 阶段标记验证 ==========
#     try:
#         stage_markers = alignment_records['stage_markers']
#         if len(stage_markers) < 1:
#             raise ValueError("No stage markers found. Expected at least 1 stage marker.")
#
#         # 获取第二阶段起始索引（假设第一个标记是第一阶段的结束）
#         stage1_end = stage_markers[0]
#         if stage1_end >= len(alignment_records['text']):
#             raise IndexError(f"Invalid stage marker {stage1_end} for data length {len(alignment_records['text'])}")
#     except KeyError as e:
#         raise KeyError(f"Missing required key in alignment_records: {e}") from None
#
#     # ========== 数据准备 ==========
#     mod_config = {
#         'text': {'color': '#2c3e50', 'label': 'Text', 'zorder': 3},
#         'visual': {'color': '#c0392b', 'label': 'Visual', 'zorder': 2},
#         'acoustic': {'color': '#27ae60', 'label': 'Acoustic', 'zorder': 1}
#     }
#
#     # ========== 绘图逻辑 ==========
#     ax = plt.gca()
#
#     for mod in mod_config:
#         # 提取第二阶段数据
#         full_data = alignment_records.get(mod, [])
#         stage2_data = full_data[stage1_end:] if len(full_data) > stage1_end else []
#
#         if not stage2_data:
#             print(f"[Warning] No valid data for {mod} in stage2")
#             continue
#
#         # 生成对应epoch编号（从stage2起始epoch开始）
#         epochs = np.arange(stage1_end, stage1_end + len(stage2_data))
#
#         # 数据平滑处理
#         smooth_window = max(3, int(len(stage2_data) * 0.1))  # 动态窗口大小
#         smooth_data = np.convolve(stage2_data, np.ones(smooth_window) / smooth_window, mode='valid')
#
#         # 计算置信区间（使用滚动标准差）
#         data_series = pd.Series(stage2_data)
#         rolling_std = data_series.rolling(smooth_window, min_periods=1).std().values[smooth_window - 1:]
#
#         # 对齐数据长度
#         valid_epochs = epochs[:len(smooth_data)]
#         lower_bound = smooth_data - rolling_std
#         upper_bound = smooth_data + rolling_std
#
#         # 绘制置信区间（更精细的阴影）
#         ax.fill_between(
#             valid_epochs, lower_bound, upper_bound,
#             color=mod_config[mod]['color'],
#             alpha=0.08,  # 降低透明度
#             linewidth=0,  # 移除阴影边界线
#             zorder=mod_config[mod]['zorder']
#         )
#
#         # 绘制主趋势线
#         ax.plot(
#             valid_epochs, smooth_data,
#             color=mod_config[mod]['color'],
#             lw=2.2,
#             label=mod_config[mod]['label'],
#             zorder=mod_config[mod]['zorder'] + 1
#         )
#
#     # ========== 图表装饰 ==========
#     # 坐标轴设置
#     ax.set_xlabel("Training Epochs", fontsize=12, labelpad=8)
#     ax.set_ylabel("Cosine Similarity", fontsize=12, labelpad=8)
#     ax.set_title("Modality Alignment Progress with Confidence Bands",
#                  fontsize=14, fontweight='semibold', pad=20)
#
#     # 智能设置坐标范围
#     pad_ratio = 0.02
#     y_min, y_max = ax.get_ylim()
#     ax.set_ylim(y_min - (y_max - y_min) * pad_ratio,
#                 y_max + (y_max - y_min) * pad_ratio)
#
#     # 专业图例设置
#     legend = ax.legend(
#         loc='upper left',
#         ncol=1,
#         frameon=True,
#         framealpha=0.95,
#         edgecolor='#333333',
#         facecolor='white',
#         fontsize=11,
#         handlelength=2.5,
#         handletextpad=0.8,
#         columnspacing=1.2,
#         borderaxespad=0.5
#     )
#     legend.get_frame().set_linewidth(0.8)
#
#     # 专业网格设置
#     ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.4)
#     ax.spines['top'].set_visible(True)
#     ax.spines['right'].set_visible(True)
#
#     # ========== 保存输出 ==========
#     plt.tight_layout()
#     plt.savefig(
#         output_path,
#         dpi=600,
#         bbox_inches='tight',
#         facecolor='white',  # 确保白色背景
#         transparent=False
#     )
#     print(f"Stage2 alignment visualization saved to {output_path}")
#
#
# def plot_kl_3d_heatmap(kl_records, save_path="kl_3d_heatmap.png"):
#     # 构建矩阵（模态数 × 训练轮次）
#     kl_matrix = np.array([
#         kl_records['text'],
#         kl_records['vision'],
#         kl_records['audio']
#     ])
#
#     # 数据归一化到0-1
#     kl_matrix_normalized = (kl_matrix - kl_matrix.min()) / (kl_matrix.max() - kl_matrix.min())
#
#     plt.rcParams.update({
#         'font.family': 'sans-serif',
#         'font.sans-serif': 'DejaVu Sans',
#         'font.size': 11,
#         'axes.labelsize': 12,
#         'axes.titlesize': 16
#     })
#
#     num_epochs = len(kl_matrix[0])
#     modalities = ['text', 'acoustic', 'visual']
#     num_modalities = len(modalities)
#
#     fig = plt.figure(figsize=(13, 6))  # 适当增加高度
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 生成网格
#     X, Y = np.meshgrid(np.arange(num_epochs), np.arange(num_modalities))
#     Z = kl_matrix_normalized
#
#     # 绘制3D热力图（曲面图）
#     surf = ax.plot_surface(X, Y, Z, cmap='summer',
#                            linewidth=0, antialiased=False, alpha=0.9,
#                            shade=True)
#
#     # 添加颜色条
#     fig.colorbar(surf, ax=ax, label='Normalized KL Divergence (0-1)')
#
#     # 设置坐标轴标签和标题
#     ax.set_xlabel("Training Epoch", fontsize=12)
#     ax.set_yticklabels(modalities, rotation=-25)  # 旋转模态标签
#
#     # 调整子图边距
#     plt.subplots_adjust(
#         left=0.08,  # 减小左侧边距
#         right=0.82,  # 增大右侧边距
#         top=0.92,  # 调整顶部边距
#         bottom=0.08  # 调整底部边距
#     )
#     ax.set_title("3D KL Divergence Heatmap",
#                  fontsize=16, pad=20)
#
#     # 关键修改：设置横坐标每10轮显示
#     xticks = np.arange(0, num_epochs, 10)
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticks)
#
#     ax.set_yticks(np.arange(num_modalities))
#     ax.set_yticklabels(modalities)
#
#     # 调整Y轴标签位置，避免重叠
#     ax.yaxis.set_label_coords(-0.25, 0.5)  # 调整Y轴标签横向位置
#
#     # 调整视角
#     ax.view_init(elev=35, azim=40)
#
#     # 优化布局并保存
#     plt.tight_layout(pad=2.5)
#     plt.savefig(save_path, dpi=600, bbox_inches='tight')
#     plt.close()
#     print(f"3D KL heatmap saved to {save_path}")
#
#
# def plot_loss_curves(train_loss_records, val_loss_records, stage1_epochs, output_file='loss_curves.png'):
#     plt.figure(figsize=(14, 6))
#     plt.rcParams.update({
#         'font.family': 'DejaVu Sans',
#         'axes.edgecolor': '#404040',
#         'grid.color': '#d9d9d9',
#         'grid.alpha': 0.5
#     })
#
#     # 颜色映射
#     colors = {
#         'train_mse': '#1f77b4',
#         'val_mse': '#d62728',
#         'cos': '#2ca02c',
#         'kd': '#d62728'
#     }
#
#     # 第一阶段
#     ax1 = plt.subplot(1, 2, 1)
#     epochs = range(1, stage1_epochs + 1)
#     ax1.plot(epochs, train_loss_records['mse'][:stage1_epochs],
#              color=colors['train_mse'], lw=2, label='Train MSE')
#     ax1.fill_between(epochs,
#                      np.array(train_loss_records['mse'][:stage1_epochs]) * 0.95,
#                      np.array(train_loss_records['mse'][:stage1_epochs]) * 1.05,
#                      color=colors['train_mse'], alpha=0.15)
#
#     ax1.plot(epochs, val_loss_records['mse'][:stage1_epochs],
#              color=colors['val_mse'], lw=2, linestyle='--', label='Val MSE')
#     ax1.fill_between(epochs,
#                      np.array(val_loss_records['mse'][:stage1_epochs]) * 0.95,
#                      np.array(val_loss_records['mse'][:stage1_epochs]) * 1.05,
#                      color=colors['val_mse'], alpha=0.15)
#
#     # 第二阶段
#     ax2 = plt.subplot(1, 2, 2)
#     stage2_length = min(len(train_loss_records['mse'][stage1_epochs:]),
#                         len(train_loss_records['cos'][stage1_epochs:]),
#                         len(train_loss_records['kd'][stage1_epochs:]))
#     epochs = range(1, stage2_length + 1)
#
#     # 绘制 MSE 线条及阴影
#     mse_values = train_loss_records['mse'][stage1_epochs:stage1_epochs + stage2_length]
#     ax2.plot(epochs, mse_values,
#              color=colors['train_mse'], lw=2, label='MSE')
#     ax2.fill_between(epochs,
#                      np.array(mse_values) * 0.95,
#                      np.array(mse_values) * 1.05,
#                      color=colors['train_mse'], alpha=0.15)
#
#     # 绘制 Cosine 线条及阴影
#     cos_values = train_loss_records['cos'][stage1_epochs:stage1_epochs + stage2_length]
#     ax2.plot(epochs, cos_values,
#              color=colors['cos'], lw=2, label='Cosine')
#     ax2.fill_between(epochs,
#                      np.array(cos_values) * 0.95,
#                      np.array(cos_values) * 1.05,
#                      color=colors['cos'], alpha=0.15)
#
#     # 绘制 KL 线条及阴影
#     kd_values = train_loss_records['kd'][stage1_epochs:stage1_epochs + stage2_length]
#     ax2.plot(epochs, kd_values,
#              color=colors['kd'], lw=2, label='KL')
#     ax2.fill_between(epochs,
#                      np.array(kd_values) * 0.95,
#                      np.array(kd_values) * 1.05,
#                      color=colors['kd'], alpha=0.15)
#
#     # 通用设置
#     for ax in [ax1, ax2]:
#         ax.set_xlabel('Training Epochs')
#         ax.set_ylabel('Loss Value')
#         ax.legend(frameon=True, edgecolor='#d9d9d9')
#         ax.grid(True)
#
#     ax1.set_title('Stage 1: Training & Validation Loss')
#     ax2.set_title('Stage 2: Training Loss Components')
#     plt.tight_layout(pad=3.0)
#     plt.savefig(output_file, dpi=600, bbox_inches='tight')
#     plt.close()


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
