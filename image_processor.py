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
from PIL import Image, ImageDraw, ImageEnhance, ImageChops
import subprocess

# GIMP 실행 파일 경로 (GIMP 설치 경로에 맞게 수정)
gimp_path = r"C:\Program Files\GIMP 2\bin\gimp-console-2.10.exe"

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

    # # 결과 이미지 저장
    # image.save(output_image_path)
    # print(f"Processed image saved at {output_image_path}")
    return image

def slice_image(image, output_path, image_file):
    width, height = image.size
    tile_size = 256
    count = 1
    
    cropped_images = []
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            new_width = min(tile_size, width - x)
            new_height = min(tile_size, height - y)
            
            cropped_image = image.crop((x, y, x + new_width, y + new_height))
            
            path = os.path.splitext(os.path.basename(image_file))
            cropped_output_path = os.path.join(output_path, f"{path[0]}_{count}{path[1]}")
            cropped_image.save(cropped_output_path)
            cropped_images.append((cropped_output_path, x, y))  # 이미지 경로와 위치를 함께 저장
            
            count += 1
            
    print(f"Image size: {width}x{height}, count: {count}")
    return cropped_images, width, height

# # Soft Light 블렌딩 함수
# def soft_light_blend(A, B):
#     """Apply Soft Light blending mode"""
#     return B - (1 - 2 * A) * B * (1 - B) + A * (1 - (1 - B) * (1 - B))

# def apply_soft_light(texture, base_image):
#     """Apply the soft light blend between two images"""
#     # Convert images to 'RGBA' to ensure alpha channel is present
#     texture = texture.convert("RGBA")
#     base_image = base_image.convert("RGBA")
    
#     # Extract individual channels
#     r1, g1, b1, a1 = texture.split()
#     r2, g2, b2, a2 = base_image.split()

#     # Apply the soft light blend function
#     r = ImageChops.blend(r2, r1, 0.5)
#     g = ImageChops.blend(g2, g1, 0.5)
#     b = ImageChops.blend(b2, b1, 0.5)

#     # Combine channels back
#     return Image.merge("RGBA", (r, g, b, a2))

def add_crack_texture(patch_image_path, crack_texture_path, temp_path):
    patch_image = Image.open(patch_image_path)
    patch_width, patch_height = patch_image.size
    
    num_cracks = random.randint(0, 11)
    # print(num_cracks)
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
        
        crack_size_x, crack_size_y = crack_crop.size
        # 랜덤한 위치에 오버랩
        paste_x = random.randint(0, max(0, patch_width - crack_size_x))
        paste_y = random.randint(0, max(0, patch_height - crack_size_y))

        crack_save_path = temp_path + 'crack.png'
        
        crack_crop.save(crack_save_path)
        gimp_command = [
            gimp_path,
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

folder_path = "Data/Obelisk/oil_on_canvas/"
output_path = "Data/Obelisk_preprocessed/oil_on_canvas/"
crack_texture_path = "Data/texture/crack5.png"

for image_file in os.listdir(folder_path):
    if image_file.endswith(('png')):
        print(image_file)
        temp_path = output_path + "temp/"
        os.makedirs(temp_path, exist_ok=True)
        
        input_image = folder_path + image_file
        
        # 1. Hue-Saturation 조정 및 노란색 톤 추가
        image = hue_saturation(input_image)
        
        # 2. 이미지를 자르고 경로를 반환
        cropped_images, img_width, img_height = slice_image(image, output_path, image_file)
        
        # 3. 자른 이미지에 균열 텍스처 추가
        for cropped_image_path, x, y in cropped_images:
            add_crack_texture(cropped_image_path, crack_texture_path, temp_path)
            print(cropped_image_path)
            
        # # 4. 균열 텍스처가 추가된 자른 이미지를 다시 결합하여 큰 이미지로 만듦
        reconstructed_image_path = temp_path +"reconstruction.png"
        reconstruct_image(cropped_images, img_width, img_height, reconstructed_image_path)
        
        break
