import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
from PIL import Image
import os

# Define the LayoutPredictor class
class LayoutPredictor(nn.Module):
    def __init__(self, feature_count=14, target_count=12):
        super(LayoutPredictor, self).__init__()
        self.layer1 = nn.Linear(feature_count, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, target_count)
    
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

# Define furniture catalog
furniture_catalog = [
    {"item": "MasterBed", "width": 3, "height": 2},
    {"item": "DiningTable", "width": 2, "height": 1.5},
    {"item": "SectionalSofa", "width": 2.5, "height": 2},
    {"item": "SideChair", "width": 1, "height": 1},
    {"item": "StorageCloset", "width": 2, "height": 2},
    {"item": "OfficeDesk", "width": 1.5, "height": 1},
    {"item": "TallBookshelf", "width": 1, "height": 2.5},
    {"item": "TableLamp", "width": 0.5, "height": 0.5},
    {"item": "ChestDrawer", "width": 1.5, "height": 1},
    {"item": "EntertainmentUnit", "width": 2, "height": 1}
]

# Load the trained model
model = LayoutPredictor()
try:
    model.load_state_dict(torch.load("layout_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("✓ Model loaded successfully from layout_model.pth")
except FileNotFoundError:
    print("⚠ Model file 'layout_model.pth' not found. Falling back to random placement.")
    model = None

# Detect overlaps between items
def detect_collisions(coordinates, items):
    for i in range(len(coordinates)):
        x1, y1 = coordinates[i]
        w1, h1 = items[i]["width"], items[i]["height"]
        for j in range(i + 1, len(coordinates)):
            x2, y2 = coordinates[j]
            w2, h2 = items[j]["width"], items[j]["height"]
            if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
                return True
    return False

# Verify if the bed is near a wall
def verify_bed_position(coordinates, items, room_width, room_height):
    bed_idx = next((i for i, item in enumerate(items) if item["item"].lower() == "masterbed"), None)
    if bed_idx is not None:
        bed_x, bed_y = coordinates[bed_idx]
        bed_w, bed_h = items[bed_idx]["width"], items[bed_idx]["height"]
        return (bed_x == 0 or bed_x + bed_w == room_width or
                bed_y == 0 or bed_y + bed_h == room_height)
    return True

# Optimize item placements
def optimize_placements(coordinates, items, room_width, room_height):
    optimized_coords = coordinates.copy()
    for i in range(len(optimized_coords)):
        w, h = items[i]["width"], items[i]["height"]
        placed = False
        for attempt in range(150):
            x = (optimized_coords[i][0] + (attempt % (room_width // 3))) % (room_width - w + 1)
            y = (optimized_coords[i][1] + (attempt // (room_width // 3))) % (room_height - h + 1)
            optimized_coords[i] = (x, y)
            has_collision = False
            for j in range(i):
                x1, y1 = optimized_coords[j]
                w1, h1 = items[j]["width"], items[j]["height"]
                x2, y2 = optimized_coords[i]
                w2, h2 = items[i]["width"], items[i]["height"]
                if (x2 < x1 + w1 and x2 + w2 > x1 and y2 < y1 + h1 and y2 + h2 > y1):
                    has_collision = True
                    break
            if items[i]["item"].lower() == "masterbed" and not (x == 0 or x + w == room_width or y == 0 or y + h == room_height):
                has_collision = True
            if not has_collision:
                placed = True
                break
        if not placed:
            if items[i]["item"].lower() == "masterbed":
                optimized_coords[i] = (0, 0)
            else:
                base_x = i * 1.5 if i * 1.5 + w <= room_width else room_width - w
                base_y = 0 if i * 1.5 + h <= room_height else room_height - h
                optimized_coords[i] = (base_x, base_y)
    return optimized_coords

# Create a visual representation of the layout
def create_visualization(coordinates, items, room_size, title="Room Arrangement"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, room_size[0])
    ax.set_ylim(0, room_size[1])
    color_map = {
        "MasterBed": "#FF6F61", "DiningTable": "#6B5B95", "SectionalSofa": "#88B04B",
        "SideChair": "#F7CAC9", "StorageCloset": "#92A8D1", "OfficeDesk": "#055A8C",
        "TallBookshelf": "#D4A017", "TableLamp": "#F4A261", "ChestDrawer": "#E76F51",
        "EntertainmentUnit": "#2A9D8F"
    }
    for i, (x, y) in enumerate(coordinates):
        item_name = items[i]["item"]
        width = items[i]["width"]
        height = items[i]["height"]
        edge_color = "red" if detect_collisions(coordinates[:i + 1], items[:i + 1]) else "black"
        ax.add_patch(plt.Rectangle((x, y), width, height, facecolor=color_map.get(item_name, "#A3BFFA"),
                                 alpha=0.7, edgecolor=edge_color, linewidth=1.5))
        ax.text(x + width / 2, y + height / 2, item_name, ha="center", va="center", fontsize=10, color="white")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title(title, fontsize=14, pad=15)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# Main function to generate the layout
def generate_room_arrangement(room_length, room_depth, furniture_list):
    try:
        room_length = float(room_length)
        room_depth = float(room_depth)
        if room_length <= 0 or room_depth <= 0:
            return "Error: Room dimensions must be positive.", None, None

        furniture_items = []
        for item in furniture_list.split("\n"):
            if not item.strip():
                continue
            parts = item.split(",")
            if len(parts) != 3:
                return f"Error: Invalid format for {item}. Use: item,width,height", None, None
            name, width, height = parts
            width = float(width.strip())
            height = float(height.strip())
            if width <= 0 or height <= 0:
                return f"Error: Invalid size for {name}. Width and height must be positive.", None, None
            furniture_items.append({"item": name.strip(), "width": width, "height": height})

        if not furniture_items:
            return "Error: No items specified.", None, None

        if len(furniture_items) > 6:
            return "Error: Limit of 6 items exceeded.", None, None

        input_vector = [room_length, room_depth] + [dim for item in furniture_items for dim in [item["width"], item["height"]]]
        input_vector += [0] * (14 - len(input_vector))

        if model is not None:
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predictions = model(input_tensor).numpy().flatten()
            coordinates = [(int(predictions[i]), int(predictions[i + 1])) for i in range(0, len(predictions), 2)][:len(furniture_items)]
        else:
            coordinates = [(np.random.randint(0, max(1, int(room_length - item["width"]))),
                          np.random.randint(0, max(1, int(room_depth - item["height"]))))
                          for item in furniture_items]

        optimized_coords = optimize_placements(coordinates, furniture_items, room_length, room_depth)

        has_collision = detect_collisions(optimized_coords, furniture_items)
        bed_valid = verify_bed_position(optimized_coords, furniture_items, room_length, room_depth)
        if not has_collision and (bed_valid or "masterbed" not in [item["item"].lower() for item in furniture_items]):
            status = "Arrangement created successfully!"
            title = f"Design for {room_length}x{room_depth}"
        else:
            status = "Arrangement invalid: Collisions or Bed not against wall."
            title = f"Problematic Design for {room_length}x{room_depth}"

        room_dims = (room_length, room_depth)
        visual = create_visualization(optimized_coords, furniture_items, room_dims, title)

        coords_text = "\n".join([f"{item['item']}: ({x}, {y})" for item, (x, y) in zip(furniture_items, optimized_coords)])

        return status, coords_text, visual

    except Exception as e:
        return f"Error: {str(e)}", None, None

# Enhanced Gradio interface
with gr.Blocks(theme=gr.themes.Default(primary_hue="teal", secondary_hue="blue"), title="Room Planner") as app:
    gr.Markdown(
        """
        # 🎨 Room Planner 🎨
        Create your ideal room layout by inputting dimensions and furniture details.  
        Use format: `item,width,height` (one per line).  
        *Tip: Place the MasterBed against a wall for best results!*
        """
    )
    
    with gr.Row(variant="panel", elem_classes=["custom-panel"]):
        with gr.Column(scale=1):
            room_length = gr.Number(label="Room Length (units)", value=10.0, precision=1,
                                  info="Length of the room in units")
            room_depth = gr.Number(label="Room Depth (units)", value=8.0, precision=1,
                                 info="Depth of the room in units")
        
        with gr.Column(scale=2):
            furniture_input = gr.Textbox(
                label="Furniture Details",
                value="MasterBed,3,2\nDiningTable,2,1.5\nSectionalSofa,2.5,2\nSideChair,1,1\nStorageCloset,2,2",
                lines=6,
                placeholder="e.g., MasterBed,3,2\nDiningTable,2,1.5",
                info="Enter each item as: name,width,height"
            )

    with gr.Row():
        generate_button = gr.Button("Generate Layout 🎨", variant="primary", elem_classes=["full-width"])

    with gr.Row():
        status_display = gr.Textbox(label="Status", interactive=False)
        coords_display = gr.Textbox(label="Coordinates", lines=5, interactive=False)
        layout_image = gr.Image(label="Room Visualization", interactive=False, height=400)

    generate_button.click(
        fn=generate_room_arrangement,
        inputs=[room_length, room_depth, furniture_input],
        outputs=[status_display, coords_display, layout_image],
        api_name="generate"
    )

    generate_button.click(
        fn=lambda: gr.Progress(track_tqdm=True),
        inputs=None,
        outputs=None,
        api_name="progress"
    )

# Custom CSS for additional styling
app.css = """
.custom-panel {
    background-color: #f0f4f8;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.full-width {
    width: 100%;
}
"""

# Launch the application
if __name__ == "__main__":
    app.launch(share=False, debug=True)