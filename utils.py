import numpy as np
import torch
from PIL import ImageFile
import heapq
from datasets.data_builder import build_dataloader

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import json
# import wandb
import re


# def load_text_prompts(root_path, class_name):
#     json_path = os.path.join(root_path, 'knowledge', 'mechanics_visual_description.json')
#
#     if not os.path.exists(json_path):
#         print(f"[Warning] Knowledge file not found at {json_path}. Using default prompts.")
#         return [f"a photo of a healthy {class_name}"], ["normal"]
#
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     prompts = []
#     anomaly_names = []
#
#     # === 1. 构建正样本 Prompt (Normal) ===
#     # 这是我们希望图像特征靠近的目标
#     normal_prompt = f"a photo of a healthy {class_name}, clean surface, no defects"
#     prompts.append(normal_prompt)
#     anomaly_names.append("normal")
#
#     # === 2. 构建负样本 Prompts (Anomalies) ===
#     # 这是我们希望图像特征远离的目标
#     for item in data:
#         # 过滤非当前类别的描述
#         if item["category"] != class_name:
#             continue
#
#         anomaly_type = item["anomaly"]  # e.g., "mold"
#         desc = item["output"]  # 详细描述字典
#
#         # 核心逻辑：利用 GPT 风格的 Prompt Engineering 将结构化数据转为句子
#         # 模板: "A photo of {class} with {anomaly}, characterized by {color} {shape} on {location}."
#         sentence = (
#             f"a photo of a {class_name} with {anomaly_type}, "
#             f"appearing {desc.get('color', 'discolored')} "
#             f"and {desc.get('shape', 'irregular')} "
#             f"on the {desc.get('location', 'surface')}."
#         )
#
#         prompts.append(sentence)
#         anomaly_names.append(anomaly_type)
#
#     print(f"[*] Loaded {len(prompts)} text prompts for {class_name}.")
#     # 打印前两个看看效果
#     if len(prompts) > 1:
#         print(f"    - Normal: {prompts[0]}")
#         print(f"    - Anomaly 1: {prompts[1]}")
#
#     return prompts, anomaly_names

def load_text_prompts(root_path, class_name):
    import json
    import os

    # [修复] 根据类别名映射到正确的知识文件，避免 maize 读 mechanics 导致 0 条匹配
    CLASS_TO_CATEGORY = {
        # Agriculture
        'maize': 'agriculture', 'rice': 'agriculture', 'wheat': 'agriculture',
        'soybean': 'agriculture', 'corn': 'agriculture',
        # Mechanics
        'terminal': 'mechanics', 'bolt': 'mechanics', 'nut': 'mechanics',
        'screw': 'mechanics', 'connector': 'mechanics',
        # Electronics
        'pcb': 'electronics', 'chip': 'electronics', 'capacitor': 'electronics',
        # Groceries
        'bottle': 'groceries', 'can': 'groceries',
        # Medicine
        'pill': 'medicine', 'capsule': 'medicine',
    }
    category = CLASS_TO_CATEGORY.get(class_name.lower(), 'mechanics')

    # 路径定义
    desc_path = os.path.join(root_path, 'knowledge', f'{category}_visual_description.json')
    qa_path = os.path.join(root_path, 'knowledge', f'{category}_QA.json')

    # 如果映射的文件不存在，遍历所有 knowledge 文件兜底
    knowledge_dir = os.path.join(root_path, 'knowledge')
    if not os.path.exists(desc_path) and os.path.exists(knowledge_dir):
        for fname in os.listdir(knowledge_dir):
            if fname.endswith('_visual_description.json'):
                candidate = os.path.join(knowledge_dir, fname)
                try:
                    with open(candidate, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if any(item.get('category') == class_name for item in data):
                        desc_path = candidate
                        qa_candidate = candidate.replace('_visual_description.json', '_QA.json')
                        if os.path.exists(qa_candidate):
                            qa_path = qa_candidate
                        print(f"[*] Auto-detected knowledge file: {fname} for class '{class_name}'")
                        break
                except Exception:
                    continue

    normal_prompts = set()
    anomaly_prompts = set()

    # =======================================================
    # 辅助函数：强力清洗与过滤
    # =======================================================
    def clean_text(text):
        if not text: return ""
        # 移除多余空格、换行
        text = text.strip().replace("\n", " ")
        # 移除句末标点，方便后续拼接（可选）
        return text

    def is_valid_anomaly(text):
        """
        核心过滤器：确保异常 Prompt 里绝对不包含“正常”的描述
        """
        text_lower = text.lower()
        # 1. 绝对黑名单：如果包含这些词，说明这句在描述正常状态
        blacklist = [
            "is normal", "are normal", "appears normal", "looks normal",
            "no anomalies", "no defects", "no visible anomalies",
            "without defects", "free from defects", "good condition",
            "clean surface", "perfect", "intact"
        ]
        if any(w in text_lower for w in blacklist):
            return False

        # 2. 必须包含至少一个负面词汇（可选，防止提取到废话）
        # valid_keywords = ["crack", "scratch", "dent", "stain", "defect", "damage", "oxidiz", "bent"]
        # if not any(k in text_lower for k in valid_keywords):
        #    return False

        # 3. 过滤非视觉的“推理”废话
        noise_list = ["sales price", "customer acceptance", "lifespan", "bad impression"]
        if any(n in text_lower for n in noise_list):
            return False

        return True

    # =======================================================
    # 策略 1: 基础模板 (修正语法)
    # =======================================================
    base_normal = [
        f"a photo of a healthy {class_name}",
        f"a photo of a perfect {class_name}",
        f"a {class_name} with a clean and smooth surface",
        f"a {class_name} without any scratches or cracks",
        f"a {class_name} showing structural integrity",
        f"a close-up of an undamaged {class_name}",
        f"a flawless {class_name}"
    ]
    normal_prompts.update(base_normal)

    # =======================================================
    # 策略 2: 解析 Visual Description (属性增强)
    # =======================================================
    if os.path.exists(desc_path):
        try:
            with open(desc_path, 'r', encoding='utf-8') as f:
                desc_data = json.load(f)

            for item in desc_data:
                if item.get('category') != class_name: continue

                ano_name = item.get('anomaly', 'defect')
                attrs = item.get('output', {})

                # --- 负向 Prompts (Grammar Fix) ---
                # 避免 "with scraped"，改为 "with scrapes" 或 "showing signs of..."
                anomaly_prompts.add(f"a photo of a {class_name} with {ano_name}")
                anomaly_prompts.add(f"a {class_name} showing signs of {ano_name}")

                # 属性描述
                desc_parts = []
                if 'color' in attrs:
                    desc_parts.append(f"appears {attrs['color']}")
                if 'shape' in attrs:
                    desc_parts.append(f"looks {attrs['shape']}")

                if desc_parts:
                    full_desc = " and ".join(desc_parts)
                    anomaly_prompts.add(f"a {class_name} with {ano_name} which {full_desc}")

                # --- 正向 Prompts (通过否定异常) ---
                # 修正语法: "free from scraped" -> "free from scraping" 或 "no scraping"
                # 简单处理：使用 without + 名词
                normal_prompts.add(f"a {class_name} without {ano_name}")
                normal_prompts.add(f"a {class_name} free from {ano_name}")

        except Exception as e:
            print(f"[Warning] Failed to parse visual_description.json: {e}")

    # =======================================================
    # 策略 3: 解析 QA (利用 Logic + Filter)
    # =======================================================
    if os.path.exists(qa_path):
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)

            for item in qa_data:
                if item.get('category') != class_name: continue

                pair_type = item.get('pair_type', '')

                # 获取 Conclusion
                conclusion_obj = item.get('conclusion', {})
                conclusion_text = ""
                if isinstance(conclusion_obj, dict):
                    conclusion_text = conclusion_obj.get('conclusion_en', '')
                elif isinstance(conclusion_obj, str):
                    conclusion_text = conclusion_obj

                conclusion_text = clean_text(conclusion_text)

                # === N-N (正常样本) ===
                if "N-N" in pair_type:
                    # 只要不是太短，通常都是好的描述
                    if len(conclusion_text) > 10:
                        # 移除 "The image shows..." 这种前缀，让 CLIP 聚焦物体
                        clean = re.sub(r'^(The image shows|The terminal is|This is).*?(that|where)?\s',
                                       f'a {class_name} ', conclusion_text)
                        normal_prompts.add(clean)

                # === N-A (异常样本) ===
                elif "N-A" in pair_type:
                    # [关键修改] 必须通过黑名单检查！
                    # 因为有些 N-A 样本的 Conclusion 可能会说 "Unlike the normal image..."
                    if len(conclusion_text) > 5 and is_valid_anomaly(conclusion_text):
                        anomaly_prompts.add(conclusion_text)

                    # 检查 QA 列表
                    for qa in item.get('question_answer', []):
                        if qa.get('answer') is True:  # 只看 Answer=True 的描述
                            q_text = qa.get('question_en', '')
                            # 如果问题是 "Is there a crack?" -> True
                            # 我们可以提取 "crack" 构造 Prompt，或者使用 qa['conclusion_en']
                            qa_concl = qa.get('conclusion_en', '')
                            if qa_concl and is_valid_anomaly(qa_concl):
                                anomaly_prompts.add(qa_concl)

        except Exception as e:
            print(f"[Warning] Failed to parse QA.json: {e}")

    # === 最终清洗与列表化 ===
    normal_prompts_list = list(normal_prompts)
    anomaly_prompts_list = list(anomaly_prompts)

    # 兜底
    if not anomaly_prompts_list:
        anomaly_prompts_list = [f"a damaged {class_name}", f"a {class_name} with defects"]

    # 打印前5个检查
    print(f"[*] Validated {len(normal_prompts_list)} Normal & {len(anomaly_prompts_list)} Anomaly prompts.")

    return normal_prompts_list, anomaly_prompts_list


def train_dataset(train_function, config):
    if config["wandb"]:
        wandb.init(project=config["project"], config={c: a for c, a in config.items() if c != "data_config"},
                   name=config["prefix"], mode="online", settings=wandb.Settings(start_method='thread'))
        wandb.define_metric("train_loss", step_metric="train_step")
        wandb.define_metric("test_loss", step_metric="test_step")
        wandb.define_metric("NF_samplewise_mean", step_metric="epoch")
        wandb.define_metric("NF_samplewise_max", step_metric="epoch")
        wandb.define_metric("NF_mean_image_roc", step_metric="epoch")
        wandb.define_metric("NF_pixel_roc", step_metric="epoch")
        wandb.define_metric("NF_aupro", step_metric="epoch")
        wandb.define_metric("NF_max_image_roc", step_metric="epoch")

    data_config = config["data_config"]

    train_loader, test_loader = build_dataloader(data_config, distributed=False)
    train_function(train_loader, test_loader, config=config)


class AnomalyTracker:
    """
    A class for tracking the top N anomalies and normal samples based on their anomaly scores.

    Attributes:
        top_n (int): The number of top anomalies and normal samples to track.
        anomalies (list): A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top anomalies.
        normals (list): A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top normal samples.
    """

    def __init__(self, top_n=100):
        """
        Initializes the tracker with a specified top_n value.

        Args:
            top_n (int, optional): The number of top anomalies and normal samples to track. Defaults to 100.
        """
        self.top_n = top_n
        self.anomalies = []  # (anomaly_score, filename, anomaly_map, gt_mask, image)
        self.normals = []  # (anomaly_score, filename, anomaly_map, gt_mask, image)

    def update(self, anomaly_score, filename, anomaly_map, gt_mask, label, image):
        if label == 1:  # 异常样本
            # 堆是最小堆：永远弹出最小的，留下的就是大的
            if len(self.anomalies) < self.top_n:
                heapq.heappush(self.anomalies, (anomaly_score, filename, anomaly_map, gt_mask, image))
            else:
                heapq.heappushpop(self.anomalies, (anomaly_score, filename, anomaly_map, gt_mask, image))
        else:  # 正常样本
            # 对于正常样本，我们也想看分数最高的（误报）
            if len(self.normals) < self.top_n:
                heapq.heappush(self.normals, (anomaly_score, filename, anomaly_map, gt_mask, image))
            else:
                heapq.heappushpop(self.normals, (anomaly_score, filename, anomaly_map, gt_mask, image))

    def get_top_anomalies(self):
        """
        Returns the top N anomalies, sorted in descending order of their anomaly scores.

        Returns:
            list: A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top anomalies.
        """
        return sorted(self.anomalies, key=lambda x: x[0], reverse=True)

    def get_top_normals(self):
        """
        Returns the top N normal samples, sorted in descending order of their anomaly scores.

        Returns:
            list: A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top normal samples.
        """
        return sorted(self.normals, key=lambda x: x[0], reverse=True)

    def clear(self):
        self.anomalies = []
        self.normals = []


def get_instancewise_data(data, config):
    labels, image, features = data["label"], data["image"], data["feature"]
    features = to_device([features], config["device"])[0]
    mask = data["mask"]
    img_in = features if config["pre_extracted"] else image
    cameras = data["camera"]

    return img_in, labels, image, mask, cameras, data["filename"]


# def get_samplewise_data(data, config):
#
#     # here B is the batch_size
#     B = data["feature_0"].shape[0]
#     idx = torch.arange(B * 5)
#     result = (idx % 5) * B + (idx // 5)
#
#
#     labels = torch.cat(data["label"])[result]
#
#     images = torch.cat([data["image_0"],data["image_1"],data["image_2"],data["image_3"],data["image_4"]], dim=0)[result,...]
#     features = to_device([data["feature_0"], data["feature_1"], data["feature_2"], data["feature_3"], data["feature_4"]], config["device"])
#     masks = torch.cat([data["mask_0"],data["mask_1"],data["mask_2"],data["mask_3"],data["mask_4"]], dim=0)[result,...]
#     if config["rem_bg"]:
#         foregrounds = torch.cat([data["foreground_0"],data["foreground_1"],data["foreground_2"],data["foreground_3"],data["foreground_4"]], dim=0)[result,...].to("cuda")
#     else:
#         B, C, H, W = features[0].shape
#         foregrounds = torch.ones((5 * B, H, W)).to("cuda")
#
#     filenames = np.concatenate(data["filename"])[result]
#     cameras = torch.cat(data["cameras"])[result]
#
#     return features, labels, images, masks, cameras, filenames, foregrounds
# utils.py 中的 get_samplewise_data 函数替换版

def get_samplewise_data(data, config):
    # data: (views, labels, filenames, masks)
    if isinstance(data, (list, tuple)) and len(data) >= 4:
        views, labels, filenames_batch, masks_batch = data[0], data[1], data[2], data[3]
    else:
        # Fallback
        views, labels = data[0], data[1]
        filenames_batch = ["unknown"] * labels.shape[0]
        masks_batch = None

    # 处理 Views (Feature) -> List of [B, C, H, W]
    if isinstance(views, torch.Tensor):
        views = [views[:, i, ...] for i in range(views.shape[1])]
    img_in = [v.to(config['device']) for v in views]

    B = img_in[0].shape[0]
    n_views = 5
    total_samples = B * n_views

    # Labels 扩展 -> [B*5]
    labels = labels.repeat_interleave(n_views).to(config['device'])

    # Filenames 扩展 -> [S0, S0, S0, S0, S0, S1...]
    filenames = []
    for f in filenames_batch:
        filenames.extend([f] * n_views)

    # Masks 处理
    input_size = config.get("img_size", (256, 256))

    if masks_batch is not None:
        # masks_batch 来自 dataloader: [B, 5, 1, H, W]
        # 我们需要把它变成 [B*5, H, W] 且顺序为 Sample-Major
        # 1. flatten 前两个维度: [B*5, 1, H, W] -> 顺序是 B0V0, B0V1 ... B1V0
        # 这正好是 Sample-Major，因为 B 维在前
        mask = masks_batch.view(-1, input_size[0], input_size[1])
        mask = mask.to(config['device'])
        mask = (mask > 0.5).float()  # 二值化
    else:
        mask = torch.zeros((total_samples, input_size[0], input_size[1])).to(config['device'])

    image = torch.zeros((total_samples, 3, input_size[0], input_size[1])).cpu()
    cameras = None
    feat_h, feat_w = img_in[0].shape[2], img_in[0].shape[3]
    foregrounds = torch.ones((total_samples, feat_h, feat_w)).to(config['device'])

    return img_in, labels, image, mask, cameras, filenames, foregrounds


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def to_device(tensors, device):
    return [t.to(device) for t in tensors]


class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name, percentage=True):
        self.name = name
        self.max_epoch = 0
        self.best_score = None
        self.last_score = None
        self.percentage = percentage

    def update(self, score, epoch, print_score=False):
        if self.percentage:
            score = score * 100
        self.last_score = score
        improved = False
        if epoch == 0 or score > self.best_score:
            self.best_score = score
            improved = True
        if print_score:
            self.print_score()
        return improved

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t best: {:.2f}'.format(self.name, self.last_score, self.best_score))


def model_size_info(model):
    # Get the number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Get the size of the model in MB
    model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 * 1024)

    # Format the output string
    output = f"**Model Size Info**\n"
    output += f"  * Number of Parameters: {num_params:,}\n"
    output += f"  * Model Size (MB): {model_size_mb:.2f} MB"

    return output


def save_weights(model, class_name, suffix, device="cuda"):
    """
    [Fix-9] 保存完整模型（包括 ica_encoder、pred_decoder、flow_bn 等），
    而非仅保存 model.net。否则加载后 ICA/LOO 相关推理会使用随机权重。
    """
    save_to = "checkpoints"
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    model.to('cpu')
    torch.save(model.state_dict(), os.path.join(save_to, f'{class_name}_{suffix}.pth'))
    print(f'[*] Full model saved to checkpoints/{class_name}_{suffix}.pth')
    model.to(device)


def load_weights(model, class_name, suffix, device="cuda"):
    """
    [Fix-9] 加载完整模型权重。兼容旧版仅保存 model.net 的 checkpoint：
    若 key 不以 'net.' 等已知前缀开头，则尝试按 model.net 子模块加载。
    """
    ckpt_path = os.path.join("checkpoints", f'{class_name}_{suffix}.pth')
    print(f"[*] Loading: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # 兼容旧版：如果 key 都不以 'net.' / 'ica_encoder.' 等开头，说明是仅 net 的旧格式
    sample_key = next(iter(state_dict))
    if not any(sample_key.startswith(p) for p in ('net.', 'ica_encoder.', 'pred_decoder.', 'flow_')):
        print("  [compat] Detected legacy net-only checkpoint, loading into model.net")
        model.net.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    return model