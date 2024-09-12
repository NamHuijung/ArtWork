from transformers import LxmertTokenizer, LxmertForPreTraining, LxmertModel
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import roi_align
import torch.nn as nn
import numpy as np


# 랜덤 시드 고정
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # 비결정적 연산 방지
    torch.backends.cudnn.benchmark = False  # 성능 저하 가능성 있지만 결정적 결과를 보장

# 2048차원 특징 벡터로 변환하는 Linear layer 정의
class FeatureTransformer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeatureTransformer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the feature map
        return self.fc(x)

# Faster R-CNN을 사용해 이미지에서 객체를 탐지하고 특징 벡터를 추출하는 함수
def get_object_features_and_boxes(image_path):
    # Faster R-CNN 모델 로드 (COCO 데이터셋에서 학습된 가중치 사용)
    # 더 많은 객체를 탐지하기 위해 box_detections_per_img 값을 조정
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, box_detections_per_img=100).eval()

    # 이미지 불러오기 및 전처리
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    # 이미지 크기 얻기
    image_width, image_height = image.size

    # 객체 탐지 수행
    with torch.no_grad():
        detections = model(image_tensor)[0]  # 객체 탐지 수행 (bounding box와 scores 반환)

    # top_k 객체만 추출 (confidence 높은 순)
    top_k = min(15, len(detections['boxes']))  # max 50개의 객체만 선택 (top_k)
    boxes = detections['boxes'][:top_k]  # Bounding boxes 추출

    # 모델의 backbone에서 중간 feature map 추출
    feature_map = model.backbone(image_tensor)['0']  # ResNet50의 feature map (C4 block)

    # ROI Align을 사용해 bounding box에 해당하는 feature map에서 특징 벡터 추출
    box_indices = torch.zeros((boxes.shape[0],), dtype=torch.int32)  # 배치 내 index (단일 배치이므로 전부 0)
    aligned_features = roi_align(feature_map, [boxes], output_size=(7, 7), spatial_scale=1.0/feature_map.shape[-1])

    # Feature map을 2048차원으로 변환
    feature_transformer = FeatureTransformer(in_dim=7*7*256, out_dim=2048)
    transformed_features = feature_transformer(aligned_features)  # 각 객체에 대한 2048차원 특징 벡터 추출

    # Bounding box 정규화 (0~1 범위로)
    normalized_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        normalized_box = [x_min / image_width, y_min / image_height, x_max / image_width, y_max / image_height]
        normalized_boxes.append(normalized_box)

    normalized_boxes = torch.tensor(normalized_boxes)
    # print(f"normalized_boxes: {normalized_boxes.shape}") # normalized_boxes: torch.Size([5, 4])
    return normalized_boxes, transformed_features, image

# 텍스트 및 이미지 인코딩(LXMERT 모델 사용)
def encode_text_and_image(text, image_features, image_boxes):
    # LXMERT tokenizer 및 모델 로드
    tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    model = LxmertForPreTraining.from_pretrained("unc-nlp/lxmert-base-uncased", output_attentions=True)
    model.eval()
    # 텍스트 토큰화
    inputs = tokenizer(text, return_tensors="pt")

    # 객체 위치를 LXMERT 모델에 입력
    visual_pos = image_boxes.unsqueeze(0)  # Normalized bounding boxes without area
    print(f"visual_pos: {visual_pos.shape}") # visual_pos: torch.Size([1, 5, 4]) # visual_pos: torch.Size([1, 2, 4])
    output = model(input_ids=inputs.input_ids, visual_feats=image_features.unsqueeze(0), visual_pos=visual_pos)
    return output

# 모든 head의 attention 값을 평균낸 후 시각화
def visualize_attention(image, boxes, output):
    # 마지막 attention layer
    cross_attention = output.cross_encoder_attentions[-1]  # [Batch, Heads, Tokens, Objects]
    # 모든 head에 대해 attention 값을 평균
    attention_scores = cross_attention.mean(dim=1)  # (batch_size, text_tokens, image_objects)
    
    # CLS 토큰과 각 객체 간의 관계 (CLS 토큰은 텍스트의 첫 번째 토큰)
    cls_attention_scores = attention_scores[0, 0, :]  # (image_objects)

    # # 이미지 시각화
    # fig, ax = plt.subplots(1)
    # ax.imshow(image)

    # # 각 객체에 대해 bounding box 그리기
    # for i, box in enumerate(boxes):
    #     x_min, y_min, x_max, y_max = box
    #     score = cls_attention_scores[i].item()
        
    #     edge_color = 'b' if i == torch.argmax(cls_attention_scores) else 'black'  # 가장 높은 주목 객체는 초록색으로 표시
    #     rect = patches.Rectangle((x_min * image.width, y_min * image.height),
    #                              (x_max - x_min) * image.width, (y_max - y_min) * image.height,
    #                              linewidth=2, edgecolor=edge_color, facecolor='none')
    #     ax.add_patch(rect)
        
    #     # Bounding box 위에 attention score 값 표시
    #     ax.text(x_min * image.width, y_min * image.height - 10, f'Score: {score:.2f}',
    #             color='blue', fontsize=10, weight='bold')

    # # 결과 저장 및 출력
    # plt.title(f"Average Attention across all Heads")
    # plt.savefig(f"swing.png")
    # plt.show()
    #  Subplot을 위한 설정
    num_boxes = len(boxes)
    cols = 5
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

    for i, (box, ax) in enumerate(zip(boxes, axes.flat)):
        x_min, y_min, x_max, y_max = box
        score = cls_attention_scores[i].item()
        
        ax.imshow(image)
        
        edge_color = 'b' if i == torch.argmax(cls_attention_scores) else 'black'  # 가장 높은 주목 객체는 파란색
        rect = patches.Rectangle((x_min * image.width, y_min * image.height),
                                 (x_max - x_min) * image.width, (y_max - y_min) * image.height,
                                 linewidth=2, edgecolor=edge_color, facecolor='none')
        ax.add_patch(rect)
        
        # Bounding box 위에 attention score 값 표시
        ax.text(x_min * image.width, y_min * image.height - 10, f'Score: {score:.2f}',
                color='blue', fontsize=10, weight='bold')

        ax.set_title(f"Object {i+1} with Score: {score:.2f}")
        ax.axis('off')

    # 불필요한 subplot 제거
    for j in range(i+1, rows*cols):
        fig.delaxes(axes.flat[j])

    # 전체 subplot 저장
    plt.tight_layout()
    plt.savefig("subplot_objects.png")
    plt.show()

def visualize_attention_map(image, boxes, output):
    # 마지막 attention layer에서 CLS 토큰과 객체 간의 attention 추출
    cross_attention = output.cross_encoder_attentions[-1]  # [Batch, Heads, Tokens, Objects] # torch.Size([1, 12, 51, 15])
    print(f"cross_attention: {cross_attention}, {cross_attention.shape}")
    attention_scores = cross_attention.mean(dim=1)  # (batch_size, text_tokens, image_objects)
    
    # CLS 토큰과 각 객체 간의 관계 (CLS 토큰은 텍스트의 첫 번째 토큰)
    cls_attention_scores = attention_scores[0, 0, :]  # (image_objects)

    # Attention map 생성 (이미지 크기와 맞춤)
    attention_map = np.zeros((image.height, image.width))

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        score = cls_attention_scores[i].item()
        
        # Bounding box 영역을 attention score로 채우기 (픽셀 단위로)
        x_min, y_min, x_max, y_max = int(x_min * image.width), int(y_min * image.height), int(x_max * image.width), int(y_max * image.height)
        attention_map[y_min:y_max, x_min:x_max] = score  # 각 픽셀에 score를 동일하게 할당

    # Attention map을 시각화하기 위해 이미지와 합성
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image, alpha=0.6)  # 원본 이미지
    heatmap = ax.imshow(attention_map, cmap='jet', alpha=0.4)  # Attention map
    
    # Heatmap의 색상 막대 추가
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Attention Score', fontsize=12)
    
    plt.title("Attention Map (Pixel-wise)")
    plt.savefig("attention_map_pixelwise.png")
    plt.show()


# # 각 head를 별도로 시각화
# def visualize_attention(image, boxes, output):
#     cross_attention = output.cross_encoder_attentions[-1]
#     num_heads = cross_attention.shape[1]
    
#     # 각 head에 대해 attention score 시각화
#     for head_idx in range(num_heads):
        
#         attention_scores = cross_attention[0, head_idx, 0, :]  # CLS 토큰과 각 객체 간의 관계
        
#         # 이미지 시각화
#         fig, ax = plt.subplots(1)
#         ax.imshow(image)
        
#         # 각 객체에 대해 bounding box 그리기
#         for i, box in enumerate(boxes):
#             x_min, y_min, x_max, y_max = box
#             score = attention_scores[i].item()
            
#             edge_color = 'g' if i == torch.argmax(attention_scores) else 'r'  # 가장 높은 주목 객체는 초록색으로 표시
#             rect = patches.Rectangle((x_min * image.width, y_min * image.height),
#                                      (x_max - x_min) * image.width, (y_max - y_min) * image.height,
#                                      linewidth=2, edgecolor=edge_color, facecolor='none')
#             ax.add_patch(rect)
            
#             # Bounding box 위에 attention score 값 표시
#             ax.text(x_min * image.width, y_min * image.height - 10, f'Score: {score:.2f}',
#                     color='blue', fontsize=10, weight='bold')

#         plt.title(f"Attention from Head {head_idx}")
#         plt.savefig(f"Attention from Head {head_idx}.png")
#         plt.show()

# 실행 함수
def visualize_segmentation(image_path, text):
    # 랜덤 시드 고정
    set_random_seed()
    # 이미지에서 객체 위치와 특징 추출
    boxes, features, image = get_object_features_and_boxes(image_path)

    # 텍스트 및 이미지 인코딩
    output = encode_text_and_image(text, features, boxes)

    # 시각화
    visualize_attention(image, boxes, output)
    visualize_attention_map(image, boxes, output)
    
# 실행 예시
if __name__ == "__main__":
    # 실행 예시
    # image_path = 'data/images/yellow_flower.jpg'
    # image_path = 'data/images/girl_with_a_pearl_earring.jpg'
    # image_path = 'data/images/sunflower.jpg'
    # image_path = 'data/images/las_meninas.jpg'
    image_path = 'data/images/swing.jpg'
    
    
    # text = "This is a yellow flower"
    # text = "'Girl with a Pearl Earring' by Johannes Vermeer is a captivating portrait of a young girl, known for her enigmatic gaze and the striking simplicity of her pearl earring."
    # text = "Vincent van Gogh's 'Sunflowers' is a vibrant series of paintings that celebrates the beauty of sunflowers through expressive brushwork and vivid colors."
    # text = "Diego Velázquez's 'Las Meninas' is a complex and enigmatic painting that masterfully blends portraiture and perspective, highlighting the Spanish royal court."
    # text = "Jean-Honoré Fragonard's 'The Swing' is a quintessential Rococo painting, known for its playful, romantic theme and light, airy brushwork, depicting a woman on a swing amidst lush greenery."
    text = "he Swing by Jean-Honoré Fragonard depicts a playful and romantic scene of a young woman in a lavish dress soaring on a swing in a lush garden."
    visualize_segmentation(image_path, text)
