import glob
import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
import openslide

def feature_extract(filename):
    glob_dir = filename + '/*/*/*.jpg'
    images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
    paths = [file for file in glob.glob(glob_dir)]
    if len(images) == 0:
        print(f"Skipping empty directory: {filename}")
        return None, None
    images = np.array(images, dtype=np.float32).reshape(len(images), -1) / 255
    model = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    predictions = model.predict(images.reshape(-1, 224, 224, 3))
    pred_images = predictions.reshape(images.shape[0], -1)
    return pred_images, paths

def update_clusters(pred_images, paths, kmodel):
    """
    根据聚类模型更新每个patch的信息
    """
    patches_info = []
    kpredictions = kmodel.predict(pred_images)
    for path, label in zip(paths, kpredictions):
        # 解析出patch的位置信息
        parts = path.split('/')
        x = int(parts[-2])
        y = int(parts[-1].split('.')[0])
        patches_info.append({'path': path, 'location': (x, y), 'cluster_label': label})
    return patches_info

def draw_clusters_on_slide(slide_path, patches_info, output_path):
    slide = openslide.open_slide(slide_path)
    slide_size = slide.dimensions
    cluster_map = Image.new("RGB", slide_size)

    draw = ImageDraw.Draw(cluster_map)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    for patch in patches_info:
        x, y = patch['location']
        color = colors[patch['cluster_label'] % len(colors)]
        draw.rectangle([(x, y), (x + 256, y + 256)], fill=color)

    cluster_map.save(output_path)

def main():
    data_dir_path = '/mnt/M/WHZ/datasets/patch/picture/'
    out_dir_path = '/mnt/M/WHZ/datasets/patch/output/3_picture/'
    wsi_base_path = "/mnt/M/WHZ/datasets/WSI/"
    k_NUM = 3
    batch_size = 50  # 设置每个批次处理的文件数量
    kmodel = KMeans(n_clusters=k_NUM, random_state=888)

    all_patches_info = []

    for filename in os.listdir(data_dir_path):
        dst_dir = os.path.join(data_dir_path, filename)
        pred_images, paths = feature_extract(dst_dir)

        if pred_images is None:
            continue

        pred_images = np.array(pred_images, dtype=np.float32)
        kmodel.fit(pred_images)
        patches_info = update_clusters(pred_images, paths, kmodel)
        all_patches_info.extend(patches_info)

        # 动态获取对应WSI的.svs文件路径
        wsi_folder = os.path.join(wsi_base_path, filename)
        svs_files = [f for f in os.listdir(wsi_folder) if f.endswith('.svs')]
        if not svs_files:
            print(f"No .svs file found for {filename}")
            continue
        slide_filename = svs_files[0]
        slide_path = os.path.join(wsi_folder, slide_filename)

        # 生成输出路径
        output_path = os.path.join(out_dir_path, f"{filename}_clustered.png")
        draw_clusters_on_slide(slide_path, patches_info, output_path)

if __name__ == '__main__':
    main()