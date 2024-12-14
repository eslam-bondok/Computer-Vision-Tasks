#git clone https://github.com/ultralytics/yolov5.git

data_path = "D:\Computer Vision Practical\Alzheimer_Dataset"

import os

with open('alzheimer_dataset.yaml', 'w') as f:
    f.write(
        f"""
        path: {os.path.abspath(data_path)} 
        train: {os.path.abspath(data_path)}
        val: {os.path.abspath(data_path)} 
        nc: {4}
        names: {['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']}
        """
    )

print("alzheimer_dataset.yaml Created Successfully.")

images_path = "D:\Computer Vision Practical\Images"  

# Train using Terminal
#cd yolov5
#python train.py --img 224 --batch 16 --epochs 12 --data alzheimer_dataset.yaml --weights yolov5s.pt --task classify

# Predict using Terminal
# python classify/predict.py \
#     --weights runs/train-cls/exp/weights/best.pt \
#     --source {images_path}