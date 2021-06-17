from PIL import Image
import os

path = './data/test_rotation11/'
save_path = './data/test_rotation11/'

for file_path in os.listdir(path):
		
		img = Image.open(path + file_path)
		img = img.resize((50, 50), Image.ANTIALIAS) #resize to 50x50
		img.save(save_path + file_path)

