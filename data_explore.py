import os

import imageio
import matplotlib.pyplot as plt

# input path for the images
base_path = "./data/train/"

plt.figure(0, figsize=(12, 20))
cpt = 0

# Show table with several sample images per emotion
for expression in os.listdir(base_path):
    for i in range(1, 6):
        cpt = cpt + 1
        plt.subplot(7, 5, cpt)

        img = imageio.imread(base_path + expression + "/"
                             + os.listdir(base_path + expression)[i])

        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show()

# Print number of images per emotion to console
all_images = []
print("Number of images per emotion :")
print('-' * 30)
for expression in os.listdir(base_path):
    num_images = len(os.listdir(base_path + expression))
    print('{:<10}'.format(expression) + ': ' + str(num_images))
    all_images.append(num_images)

# Show these numbers in bar plot
objects = os.listdir(base_path)
y_pos = range(len(objects))
i_num = all_images

plt.bar(y_pos, i_num, alpha=0.8, zorder=2)
plt.xticks(y_pos, objects)
plt.grid(True, axis='y', color='lightgrey', linewidth=.5, zorder=1)
plt.title('Number of images per Emotion')

plt.tight_layout()
plt.show()
