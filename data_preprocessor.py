'''
단계
1. Hue saturation 조정
- 이미지의 밝기와 채도를 조정하고, 노란색 톤을 추가
2. Crop
- 이미지를 256x256 크기의 패치로 자르기
3. Add crack
- 각 패치마다 0~10번의 crack 텍스처를 오버랩
- crack 텍스처의 크기와 위치는 랜덤으로 결정
- crack 텍스처의 크기는 32, 64, 128, 256 크기 중 하나로 랜덤하게 resize
- crack 텍스처는 패치 이미지의 임의의 위치에 랜덤하게 오버랩
'''
import os
import random
from PIL import Image, ImageDraw, ImageEnhance
import subprocess


def hue_saturation(input_image_path):
    # 원본 이미지 열기
    image = Image.open(input_image_path)

    # brightness_factor = random.uniform(0, 0.65)
    # saturation_factor = random.uniform(0, 0.65)
    
    # 밝기 조절 
    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(0.9)  # 밝기를 랜덤 값으로 조정

    # 채도 조절 
    enhancer_color = ImageEnhance.Color(image)
    image = enhancer_color.enhance(0.5)  # 채도를 랜덤 값으로 조정

    # 노란색 톤 추가 (알파 블렌딩)
    yellow_overlay = Image.new("RGB", image.size, (255, 255, 0))
    image = Image.blend(image, yellow_overlay, alpha=0.1)  # 노란색을 약간만 추가

    return image

def slice_image(image, output_path, image_name, tile_size=256, save=False):
    width, height = image.size
    count = 1

    cropped_images = []

    # Create a new folder for this image
    image_name = os.path.splitext(os.path.basename(image_name))[0]
    image_folder = os.path.join(output_path, image_name)
    os.makedirs(image_folder, exist_ok=True)

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            new_width = min(tile_size, width - x)
            new_height = min(tile_size, height - y)

            cropped_image = image.crop((x, y, x + new_width, y + new_height))

            cropped_output_path = os.path.join(image_folder, f"{image_name}_{count}.png")
            cropped_image.save(cropped_output_path) if save else None
            cropped_images.append((cropped_output_path, x, y))  # 이미지 경로와 위치를 함께 저장

            count += 1

    print(f"Image size: {width}x{height}, patch count: {count}")
    return cropped_images, width, height

def add_crack_texture(patch_image_path, crack_texture_path, temp_path):
    patch_image = Image.open(patch_image_path)
    patch_width, patch_height = patch_image.size
    
    num_cracks = random.randint(0, 11)
    for _ in range(num_cracks):
        crack_texture = Image.open(crack_texture_path)
        crack_width, crack_height = crack_texture.size
        # 랜덤하게 크랙의 일부 영역 선택
        crop_x1 = random.randint(0, crack_width - patch_width)
        crop_y1 = random.randint(0, crack_height - patch_height)
        crop_x2 = random.randint(crop_x1 + patch_width, crack_width)
        crop_y2 = random.randint(crop_y1 + patch_height, crack_height)
        crack_crop = crack_texture.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        # 리사이즈를 할지 말지 랜덤하게 결정
        if random.choice([True, False]):
            # 선택된 크랙 부분을 32, 64, 128, 256 중 하나로 리사이즈 
            crack_size = random.choice([32, 64, 128, 256])
            crack_crop = crack_crop.resize((crack_size, crack_size))
        else:
            # 크랙의 원본 크기를 유지
            crack_size = crack_crop.size[0]
        # 랜덤한 위치에 오버랩
        paste_x = random.randint(0, max(0, patch_width - crack_size))
        paste_y = random.randint(0, max(0, patch_height - crack_size))
        
        crack_save_path = temp_path + 'crack.png'
        
        crack_crop.save(crack_save_path)
        gimp_command = [
            "gimp",
            "-i",
            "-b",
            f'(let* ((image (car (gimp-file-load RUN-NONINTERACTIVE "{patch_image_path}" "{patch_image_path}"))) \
                    (texture-image (car (gimp-file-load RUN-NONINTERACTIVE "{crack_save_path}" "{crack_save_path}"))) \
                    (texture-layer (car (gimp-layer-new-from-drawable (car (gimp-image-get-active-layer texture-image)) image)))) \
                (gimp-image-add-layer image texture-layer -1) \
                (gimp-item-transform-translate texture-layer {paste_x} {paste_y}) \
                (gimp-layer-set-mode texture-layer SOFTLIGHT-MODE) \
                (gimp-layer-set-opacity texture-layer 100) \
                (gimp-levels-stretch texture-layer) \
                (gimp-image-merge-visible-layers image CLIP-TO-IMAGE) \
                (gimp-file-save RUN-NONINTERACTIVE image (car (gimp-image-get-active-layer image)) "{patch_image_path}" "{patch_image_path}") \
                (gimp-image-delete image) \
                (gimp-image-delete texture-image))',
            "-b", "(gimp-quit 0)"
        ]
        subprocess.run(gimp_command)
    
    
def reconstruct_image(cropped_images, width, height, output_image, line_color="red", line_width=2):
    # 큰 빈 캔버스를 생성
    reconstructed_image = Image.new('RGB', (width, height))

    # 경계선 없는 이미지를 위해 복사본 생성
    reconstructed_image_no_lines = reconstructed_image.copy()

    for cropped_image_path, x, y in cropped_images:
        cropped_image = Image.open(cropped_image_path)
        reconstructed_image.paste(cropped_image, (x, y))  # 원래 위치에 붙이기
        reconstructed_image_no_lines.paste(cropped_image, (x, y))  # 경계선 없는 이미지에 붙이기

    # 경계선을 그리기 위해 Draw 객체 생성
    draw = ImageDraw.Draw(reconstructed_image)

    for cropped_image_path, x, y in cropped_images:
        cropped_image = Image.open(cropped_image_path)

        # 선을 그리기 위한 좌표 계산
        right = x + cropped_image.width
        bottom = y + cropped_image.height

        # 선 그리기 (사각형의 경계선)
        draw.rectangle([x, y, right, bottom], outline=line_color, width=line_width)

    # 경계선 없는 이미지를 먼저 저장
    output_image_no_lines = output_image.replace(".png", "_no_lines.png")
    reconstructed_image_no_lines.save(output_image_no_lines)
    print(f"Reconstructed image without lines saved as: {output_image_no_lines}")

    # 경계선이 있는 이미지를 저장
    reconstructed_image.save(output_image)
    print(f"Reconstructed image with lines saved as: {output_image}")

if __name__ == "__main__":
    folder_path = "Data/Obelisk/oil_on_canvas/"
    output_path = "Data/Obelisk_preprocessed/oil_on_canvas/"
    crack_texture_path = "Data/texture/crack5.png"

    for image_name in os.listdir(folder_path):
        if image_name.endswith(('png')):
            temp_path = output_path + "temp/"
            os.makedirs(temp_path, exist_ok=True)
            
            image_path = folder_path + image_name
            
            # 1. Hue-Saturation 조정 및 노란색 톤 추가
            image = hue_saturation(image_path)
            
            # 2. 이미지를 자르고 경로를 반환
            cropped_images, img_width, img_height = slice_image(image, output_path, image_name)
            
            # 3. 자른 이미지에 균열 텍스처 추가
            for cropped_image_path, x, y in cropped_images:
                add_crack_texture(cropped_image_path, crack_texture_path, temp_path)
                print(cropped_image_path)
                
            # 4. 균열 텍스처가 추가된 자른 이미지를 다시 결합하여 큰 이미지로 만듦
            # reconstructed_image_path = temp_path + "reconstruction.png"
            # reconstruct_image(cropped_images, img_width, img_height, reconstructed_image_path)
            
            break
