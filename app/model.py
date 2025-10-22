import numpy as np 
import os
import random
import keras
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from PIL import Image
from keras.layers import Conv2D,Flatten,Dense,Dropout,BatchNormalization,MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

path = 'C:/Users/Usuario/OneDrive/Escritorio/Alzheimer Disease Model Predict/dataset/'

def collect_image_paths(directory):
    paths = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
    return paths
    
path1 = collect_image_paths(os.path.join(path, 'Data/Non Demented'))
path2 = collect_image_paths(os.path.join(path, 'Data/Mild Dementia'))
path3 = collect_image_paths(os.path.join(path, 'Data/Moderate Dementia'))
path4 = collect_image_paths(os.path.join(path, 'Data/Very mild Dementia'))

# Set the size of the sample
size = 400

# Set seed for reproducibility
random.seed(42)

# Sample random paths
sample_path1 = random.sample(path1, min(size, len(path1)))
sample_path2 = random.sample(path2, min(size, len(path2)))
sample_path3 = random.sample(path3, min(size, len(path3)))
sample_path4 = random.sample(path4, min(size, len(path4)))

# Output the sample sizes
print(f'Sampled {len(sample_path1)} paths from Non Demented')
print(f'Sampled {len(sample_path2)} paths from Mild Dementia')
print(f'Sampled {len(sample_path3)} paths from Moderate Dementia')
print(f'Sampled {len(sample_path4)} paths from Very mild Dementia')


# ETL

classes = ['Non Demented', 'Mild Dementia', 'Moderate Dementia', 'Very mild Dementia']
for class_name in classes:
    class_path = os.path.join(path, f"Data/{class_name}")
    num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    print(f"Classe '{class_name}': {num_images} imagens")

def show_images_from_class(class_name, num_images=5):
    class_path = os.path.join(path, f"Data/{class_name}")
    images = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(images[:num_images]):
        img = Image.open(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis("off")
    plt.show()

for class_name in classes:
    show_images_from_class(class_name)

data_distribution = []
for class_name in classes:
    class_path = os.path.join(path, f"Data/{class_name}")
    num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    data_distribution.append({"Class": class_name, "Amount": num_images})

df_distribution = pd.DataFrame(data_distribution)

df_distribution['Class'] = ['non', 'mild', 'moderate', 'very']

plt.figure(figsize=(8, 6))
ax = sns.barplot(x="Class", y="Amount", data=df_distribution, palette="viridis")

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), 
                textcoords='offset points')

plt.title("Class Image Distribution", fontsize=14)
plt.xlabel("Tumor type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()