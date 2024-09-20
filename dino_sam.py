'''
Grounding DINO로 텍스트 설명을 기반으로 이미지를 탐지한 후,
SAM을 이용해 해당 부분을 세그먼트하는 코드
'''
import os
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# Segment anything
from segment_anything import sam_model_registry, SamPredictor
from transformers import BertTokenizer, BertModel

import warnings

# 모든 경고를 무시합니다.
warnings.filterwarnings("ignore")

def load_image(image_path):
    """이미지를 로드하고 변환."""
    image_pil = Image.open(image_path).convert("RGB")  # load image
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, tokenizer, bert_model, device):
    """Grounding DINO 모델을 로드."""
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_model = bert_model  # BERT 모델을 전달
    args.tokenizer = tokenizer  # 토크나이저 전달
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    """Grounding DINO로 텍스트 기반 객체 탐지."""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False, opacity=0.9):
    """마스크 시각화."""
    # 첫 번째 차원(채널)을 제거하여 2D 마스크로 변환
    mask = mask.squeeze(0)  # [1, 427, 640] -> [427, 640]

    # 마스크 색상 설정, 기본 색상은 파란색
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([opacity])], axis=0)  # 불투명도 높이기
    else:
        color = np.array([30/255, 144/255, 255/255, opacity])  # Alpha 채널(투명도)을 0.9로 설정
    
    # 마스크 이미지를 시각화
    h, w = mask.shape[-2], mask.shape[-1]  # 높이와 너비 추출
    mask_image = np.zeros((h, w, 4))  # 4채널 (RGB + Alpha) 이미지 생성
    mask_image[:, :, :3] = mask[:, :, None] * color[:3]  # 색상 적용
    mask_image[:, :, 3] = mask * color[3]  # Alpha 채널 적용
    
    ax.imshow(mask_image)

def show_box(box, ax, label):
    """박스 시각화."""
    x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label, color='blue')

def segment_with_grounding_dino_sam(image_path, text_prompt, output_dir):
    """Grounding DINO와 SAM을 이용한 세그먼트."""
    
    # 설정 (수동 설정)
    grounded_checkpoint = '/mnt/nas_drive/Artwork/models/groundingdino_swint_ogc.pth'  # Grounding DINO 가중치 경로
    config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'  # Grounding DINO 설정 파일 경로
    sam_checkpoint_path = '/mnt/nas_drive/Artwork/models/sam_vit_h_4b8939.pth'
    
    # BERT 모델과 토크나이저를 Hugging Face에서 바로 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    box_threshold = 0.3
    text_threshold = 0.3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 이미지 로드
    image_pil, image = load_image(image_path)    
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # Grounding DINO 모델 로드
    model = load_model(config_file, grounded_checkpoint, tokenizer, bert_model, device=device)

    # Grounding DINO로 텍스트 기반 객체 탐지
    boxes_filt, pred_phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device=device)

    # SAM 모델 초기화
    predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path).to(device))

    image_cv = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_rgb.shape[:2]).to(device)
    
    masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes.to(device), multimask_output=False)
    print("masks 형태:", masks.shape) # masks 형태: torch.Size([box 개수, 1, w, h])
    
    num_labels = len(pred_phrases)
    # 마스크와 원본 이미지 출력 함수 호출
    plot_segmented_images(image_rgb, masks, boxes_filt, pred_phrases, num_labels, output_dir)

def plot_segmented_images(image_rgb, masks, boxes_filt, pred_phrases, num_labels, output_dir):
    """마스크가 적용된 이미지를 원본과 함께 subplot으로 표시 및 저장 (마스크 결합 기능 포함)"""

    # 1. 라벨에서 정확도 제거 후 결합된 마스크 생성
    label_to_masks = {}
    label_to_scores = {}
    for mask, label in zip(masks, pred_phrases):
        print(f"label: {label}")
        name, score = label.split('(')  # 정확도를 분리하여 추출
        score = float(score[:-1])  # ')' 제거하고 float으로 변환
        
        # 같은 이름의 라벨에 해당하는 마스크를 결합
        if name not in label_to_masks:
            label_to_masks[name] = mask
            label_to_scores[name] = [score]
        else:
            label_to_masks[name] = torch.logical_or(label_to_masks[name], mask)  # 마스크를 논리합으로 결합
            label_to_scores[name].append(score)
    label_to_avg_score = {label: sum(scores) / len(scores) for label, scores in label_to_scores.items()}
    # print(f"label to mask : {label_to_masks}")
    
    # 2. 모든 마스크를 하나로 결합
    all_mask = torch.zeros_like(masks[0], dtype=torch.bool)
    for mask in masks:
        all_mask = torch.logical_or(all_mask, mask)  # 모든 마스크를 논리합으로 결합

    # 3. 원래 이미지 + 마스크 이미지 subplot 생성 (원래 이미지도 포함하므로 num_labels + 2)
    num_labels = len(label_to_masks)
    print(num_labels)
    fig, axes = plt.subplots(2, num_labels + 2, figsize=(25, 15), gridspec_kw={'height_ratios': [10, 1]})
    
    # 첫 번째 subplot에 원래 이미지 표시
    ax_image = axes[0, 0]
    ax_image.imshow(image_rgb)
    ax_image.set_title("Original Image", fontsize=14, color='black')
    ax_image.axis('off')
    
    ax_label = axes[1, 0]  # 원래 이미지 아래에는 빈 라벨 영역
    ax_label.axis('off')  # 빈 축 제거

    # 4. 각 라벨별로 결합된 마스크에 대한 subplot 생성
    for idx, (label, combined_mask) in enumerate(label_to_masks.items()):
        ax_image = axes[0, idx + 1]  # 첫 번째 row에는 이미지
        ax_label = axes[1, idx + 1]  # 두 번째 row에는 라벨

        # 이미지에 결합된 마스크 적용
        ax_image.imshow(image_rgb)
        show_mask(combined_mask.cpu().numpy(), ax_image, random_color=True)
        ax_image.axis('off')  # 축 제거
        
        # Title을 subplot 상단에 표시 (이미지의 title)
        ax_image.set_title(f"{label} Combined", fontsize=14, color='black')

        # 라벨을 subplot 하단에 표시
        score = round(label_to_avg_score[label], 2)
        ax_label.text(0.5, 0.5, f"{label}({score})", fontsize=20, color='blue', ha='center', va='center')
        ax_label.axis('off')  # 축 제거

    # 5. 마지막 subplot에 모든 마스크 결합하여 시각화
    ax_image = axes[0, num_labels + 1]
    ax_image.imshow(image_rgb)
    show_mask(all_mask.cpu().numpy(), ax_image, random_color=True)
    ax_image.set_title("All Masks Combined", fontsize=14, color='black')
    ax_image.axis('off')  # 축 제거

    ax_label = axes[1, num_labels + 1]  # 결합된 마스크 아래에도 라벨 추가
    ax_label.text(0.5, 0.5, 'All Combined Masks', fontsize=20, color='blue', ha='center', va='center')
    ax_label.axis('off')  # 축 제거

    # 6. 전체 figure의 제목 추가
    plt.suptitle("Segmented Image with Combined Masks", fontsize=30, color='black', weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 전체 제목을 위해 여백 조정
    
    # 저장 경로 지정
    output_image_path = os.path.join(output_dir, "combined_masks/sunflower4.jpg")
    plt.savefig(output_image_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

    print(f"Segmented image saved at: {output_image_path}")

# def plot_segmented_images(image_rgb, masks, boxes_filt, pred_phrases, num_labels, output_dir):
#     """마스크가 적용된 이미지를 원본과 함께 subplot으로 표시 및 저장"""
    
#     # 원래 이미지 + 마스크 이미지 subplot 생성 (원래 이미지도 포함하므로 num_labels + 1)
#     fig, axes = plt.subplots(2, num_labels + 1, figsize=(25, 15), gridspec_kw={'height_ratios': [10, 1]})
    
#     # 첫 번째 subplot에 원래 이미지 표시
#     ax_image = axes[0, 0]
#     ax_image.imshow(image_rgb)
#     ax_image.set_title("Original Image", fontsize=14, color='black')
#     ax_image.axis('off')
    
#     ax_label = axes[1, 0]  # 원래 이미지 아래에는 빈 라벨 영역
#     ax_label.axis('off')  # 빈 축 제거

#     # 각 마스크 이미지에 대한 subplot 생성
#     for idx, (mask, box, label) in enumerate(zip(masks, boxes_filt, pred_phrases)):
#         ax_image = axes[0, idx + 1]  # 첫 번째 row에는 이미지
#         ax_label = axes[1, idx + 1]  # 두 번째 row에는 라벨

#         # 이미지에 마스크 적용
#         ax_image.imshow(image_rgb)
#         show_mask(mask.cpu().numpy(), ax_image, random_color=True)
#         show_box(box.numpy(), ax_image, label)
#         ax_image.axis('off')  # 축 제거
        
#         # Title을 subplot 상단에 표시 (이미지의 title)
#         ax_image.set_title(f"Segment {idx + 1}", fontsize=14, color='black')

#         # 라벨을 subplot 하단에 표시
#         ax_label.text(0.5, 0.5, label, fontsize=20, color='blue', ha='center', va='center')
#         ax_label.axis('off')  # 축 제거

#     # 전체 figure의 제목 추가
#     plt.suptitle("Segmented Image and Original with Labels", fontsize=30, color='black', weight='bold')

#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # 전체 제목을 위해 여백 조정
    
#     # 저장 경로 지정
#     output_image_path = os.path.join(output_dir, "las_meninas.jpg")
#     plt.savefig(output_image_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
#     plt.close()

#     print(f"Segmented image saved at: {output_image_path}")

# 함수 호출 예시
if __name__ == "__main__":
    # image_path = 'data/images/yellow_flower.jpg'
    # image_path = 'data/images/girl_with_a_pearl_earring.jpg'
    image_path = 'data/images/sunflower.jpg'
    # image_path = 'data/images/las_meninas.jpg'
    # image_path = 'data/images/swing.jpg'
    
    
    # text = "This is a yellow flower"
    # text = "'Girl with a Pearl Earring' by Johannes Vermeer is a captivating portrait of a young girl, known for her enigmatic gaze and the striking simplicity of her pearl earring."
    # text = "Vincent van Gogh's 'Sunflowers' is a vibrant series of paintings that celebrates the beauty of sunflowers through expressive brushwork and vivid colors."
    # text = "Diego Velázquez's 'Las Meninas' is a complex and enigmatic painting that masterfully blends portraiture and perspective, highlighting the Spanish royal court."
    # text = "Jean-Honoré Fragonard's 'The Swing' is a quintessential Rococo painting, known for its playful, romantic theme and light, airy brushwork, depicting a woman on a swing amidst lush greenery."
    # text = "he Swing by Jean-Honoré Fragonard depicts a playful and romantic scene of a young woman in a lavish dress soaring on a swing in a lush garden."
    
    # text = "Johannes Vermeer's 'Girl with a Pearl Earring' captivates with its delicate interplay of light and shadow, where the soft gaze of the girl and the luminous pearl evoke an ethereal beauty."
    # text = "Vincent van Gogh's 'Sunflowers' radiates with vibrant yellow hues and dynamic brushstrokes, capturing the raw energy of life and the fleeting beauty of nature."
    # text = "Diego Velázquez's 'Las Meninas' is a masterful exploration of perspective and illusion, where the intimate interactions of royalty and servants create a profound sense of depth and mystery."
    # text = "Jean-Honoré Fragonard's 'The Swing' enchants with its playful elegance, as the soft pastel colors and fluid movement capture the carefree joy and romantic whimsy of a fleeting moment."
    
    # text = "Vincent van Gogh's Sunflowers bursts with vibrant yellows and swirling textures, evoking the fleeting vitality of nature with a raw intensity that celebrates both beauty and impermanence."
    text = "Van Gogh used thick layers of paint to create a textured, three-dimensional effect, simplifying the background and vase with yellow hues to make the sunflowers stand out even more vividly."
    output_dir = "data/sam_outputs"
    
    # 세그먼트된 이미지 가져오기
    segment_with_grounding_dino_sam(image_path, text, output_dir)
