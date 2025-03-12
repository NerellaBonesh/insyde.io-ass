import requests
from PIL import Image
import io
import base64

url = "http://localhost:5000/generate-layout"
data = {
    "room_width": 20.0,
    "room_height": 20.0,
    "furniture": [
        {"name": "Bed", "width": 2, "height": 3},
        {"name": "Table", "width": 2, "height": 2},
        {"name": "Sofa", "width": 3, "height": 2},
        {"name": "Chair", "width": 1, "height": 1},
        {"name": "Wardrobe", "width": 2, "height": 2},
    ]
}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    print("Response:", result)
    if "image" in result:
        img_data = base64.b64decode(result["image"])
        img = Image.open(io.BytesIO(img_data))
        img.show()
    else:
        print("No image in response.")
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")