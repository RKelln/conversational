import requests

def download_image(image_url, filename):
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            f.write(response.content)

        print(f"Image downloaded as {filename}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the image: {e}")
