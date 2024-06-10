import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# YOLOv5 모델 로드 (사전 학습된 모델 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img_path = 'path/to/WSU/image.jpg'

img = Image.open(img_path)

# 객체 탐지
results = model(img)

# 객체 탐지 결과를 DataFrame으로 출력
df = results.pandas().xyxy[0]  # xmin, ymin, xmax, ymax, confidence, class, name

filtered_df = df[df['name'].isin(['dog', 'cat'])]

print(filtered_df)

img_cv2 = cv2.imread(img_path)

for index, row in filtered_df.iterrows():
    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    label = row['name']
    confidence = row['confidence']

    cv2.rectangle(img_cv2, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.putText(img_cv2, f'{label} {confidence:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# 결과 이미지 표시
plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
