import os
import requests
from dotenv import load_dotenv

def upload_image_to_imgur(image_path, client_id):
    headers = {
        'Authorization': f'Client-ID {client_id}'
    }
    url = "https://api.imgur.com/3/upload"
    with open(image_path, 'rb') as image:
        response = requests.post(url, headers=headers, files={'image': image})

    if response.status_code == 200:
        data = response.json()
        return data['data']['link']
    else:
        raise Exception(f"Failed to upload image {image_path}. Status code: {response.status_code}")

def upload_images_in_folder(folder_path):
    load_dotenv()
    client_id = os.getenv('IMGUR_CLIENT_ID')

    if not client_id:
        raise Exception("IMGUR_CLIENT_ID not found in .env file")

    image_urls = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                imgur_url = upload_image_to_imgur(file_path, client_id)
                image_urls.append(imgur_url)
                print(f"Uploaded {file_name}: {imgur_url}")
            except Exception as e:
                print(e)
    
    return image_urls

# Example usage:
folder_path = 'images'
image_urls = upload_images_in_folder(folder_path)
print("Uploaded image URLs:", image_urls)