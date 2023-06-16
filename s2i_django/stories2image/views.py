from django.shortcuts import render
from django.http import HttpResponse

import sys
import os

sys.path.append("/home/kryc1/stories2image")
sys.path.append("/home/kryc1/stories2image/src/image/captioning")
from s2i_main import start_process

sys.path.append("/home/kryc1/s2i_django/s2i_django")
import settings
image_dir = os.path.join(settings.BASE_DIR, 'stories2image', 'static', 'images')

def home(request):
    if request.method == 'POST':
        my_data = request.POST.get('my_data')
        split_order = request.POST.get('order')

        print(image_dir)

        splitted_list, image_path_list, genres_list, keywords_new_list, keywords_org_list, summary_list = start_process(my_data, split_order)

        """
        splitted_list = ["There were a boy and girl. They were at the beach. They were enjoying the sea and the sun.", "The girl suddenly got sick and wanted to leave. She took her white bag and left."]
        image_list_1 = ["/home/kryc1/s2i_django/stories2image/static/images/general/667626_18933d713e.jpg", "/home/kryc1/s2i_django/stories2image/static/images/general/3637013_c675de7705.jpg", "/home/kryc1/s2i_django/stories2image/static/images/general/10815824_2997e03d76.jpg", "/home/kryc1/s2i_django/stories2image/static/images/general/12830823_87d2654e31.jpg"]
        image_list_2 = ["/home/kryc1/s2i_django/stories2image/static/images/general/17273391_55cfc7d3d4.jpg", "/home/kryc1/s2i_django/stories2image/static/images/general/19212715_20476497a3.jpg", "/home/kryc1/s2i_django/stories2image/static/images/general/23445819_3a458716c1.jpg", "/home/kryc1/s2i_django/stories2image/static/images/general/27782020_4dab210360.jpg"]
        image_path_list = [image_list_1, image_list_2]
        genres_list = ["general", "general"]
        keywords_new_list = ["ali", "veli", "ebubekir", "sıddık"]
        keywords_org_list = ["ayşe", "fatma", "hayriye", "haydi"]
        summary_list = ["There was a boy and girl", "The girl got sick"]
        """

        context = {
            'parts': splitted_list,
            'images': image_path_list,
            'genres': genres_list,
            'keywords_new': keywords_new_list,
            'keywords_org': keywords_org_list,
            'summaries': summary_list
        }

        return render(request, 'stories2image/index.html', context)
    else:
        return render(request, 'stories2image/index.html')

def about(request):
    return render(request, 'stories2image/about.html')

def contact(request):
    return render(request, 'stories2image/contact.html')
