import os

path = "/home/kryc1/s2i_django/stories2image/static/images/education_reference"
name_list = os.listdir(path)
cnt = 0

for i in(name_list):
	splitted = i.split('.')

	if splitted[1] == "jpeg":
		os.rename(path + "/" + i, path + "/" + splitted[0] + ".jpg")
		cnt = cnt + 1
		print(cnt)
