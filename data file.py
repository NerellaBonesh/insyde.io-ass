import numpy as np
import pandas as pd
import random
import os

# Define a diverse set of furniture items with different dimensions
furniture_catalog = [
    {"item_name": "KingBed", "length": 3, "depth": 2},
    {"item_name": "CoffeeTable", "length": 2, "depth": 1.5},
    {"item_name": "Couch", "length": 2.5, "depth": 2},
    {"item_name": "Armchair", "length": 1, "depth": 1},
    {"item_name": "Closet", "length": 2, "depth": 2},
    {"item_name": "WorkDesk", "length": 1.5, "depth": 1},
    {"item_name": "Bookcase", "length": 1, "depth": 2.5},
    {"item_name": "FloorLamp", "length": 0.5, "depth": 0.5},
    {"item_name": "Chest", "length": 1.5, "depth": 1},
    {"item_name": "MediaUnit", "length": 2, "depth": 1}
]

# Function to generate a valid layout with custom placement logic
def create_valid_layout(sample_count, min_room_dim=8, max_room_dim=12):
    layouts = []
    attempt_limit = sample_count * 15  # Cap on attempts to avoid infinite loops
    current_attempt = 0

    print(f"Beginning layout creation for {sample_count} examples...")

    while len(layouts) < sample_count and current_attempt < attempt_limit:
        # Random room dimensions
        room_length = random.randint(min_room_dim, max_room_dim)
        room_depth = random.randint(min_room_dim, max_room_dim)
        item_count = random.randint(3, 6)  # Select 3 to 6 items
        chosen_items = random.sample(furniture_catalog, item_count)

        # Store item dimensions (with possible rotation)
        item_dims = []
        for item in chosen_items:
            if random.random() > 0.6:  # 40% chance to rotate
                item_dims.append((item["depth"], item["length"]))
            else:
                item_dims.append((item["length"], item["depth"]))

        # Attempt to place items randomly
        placements = {}
        for idx, (length, depth) in enumerate(item_dims):
            max_x = room_length - length
            max_y = room_depth - depth
            if max_x < 0 or max_y < 0:
                continue  # Skip if item too large

            valid_placement = False
            max_tries = 50
            try_count = 0
            while not valid_placement and try_count < max_tries:
                x = random.uniform(0, max_x)
                y = random.uniform(0, max_y)
                placements[chosen_items[idx]["item_name"]] = {"x": x, "y": y, "length": length, "depth": depth}

                # Check for overlaps
                overlap_detected = False
                for prev_name, prev_pos in placements.items():
                    if prev_name != chosen_items[idx]["item_name"]:
                        prev_x, prev_y = prev_pos["x"], prev_pos["y"]
                        prev_l, prev_d = prev_pos["length"], prev_pos["depth"]
                        if not (x + length <= prev_x or x >= prev_x + prev_l or
                                y + depth <= prev_y or y >= prev_y + prev_d):
                            overlap_detected = True
                            break
                if not overlap_detected:
                    valid_placement = True
                try_count += 1

            if not valid_placement:
                placements.clear()
                break

        # Verify bed placement near a wall if present
        bed_near_wall = True
        bed_item = next((item for item in chosen_items if item["item_name"] == "KingBed"), None)
        if bed_item:
            bed_pos = placements["KingBed"]
            bed_x, bed_y = bed_pos["x"], bed_pos["y"]
            bed_l, bed_d = bed_pos["length"], bed_pos["depth"]
            bed_near_wall = (bed_x == 0 or bed_x + bed_l == room_length or
                           bed_y == 0 or bed_y + bed_d == room_depth)

        # Fixed condition: Check for invalid dimensions (length <= 0)
        if placements and not any(v["length"] <= 0 for v in placements.values()) and bed_near_wall:
            layouts.append({
                "room_dimensions": (room_length, room_depth),
                "item_details": item_dims,
                "coordinates": [placements[item["item_name"]] for item in chosen_items],
                "item_names": [item["item_name"] for item in chosen_items]
            })
            print(f"✓ Recorded layout {len(layouts)} / {sample_count} | Room: {room_length}x{room_depth}")

        current_attempt += 1

    print(f"✓ Completed: Generated {len(layouts)} valid layouts after {current_attempt} attempts.")
    return layouts

# Function to format data for CSV
def format_data_for_storage(layout_data):
    X_data, y_data = [], []
    max_items_per_layout = max(len(entry["item_details"]) for entry in layout_data)

    for entry in layout_data:
        room_dims = entry["room_dimensions"]
        item_sizes = entry["item_details"]
        positions = entry["coordinates"]

        # Input features: [room_length, room_depth, item1_length, item1_depth, ..., padded zeros]
        features = [room_dims[0], room_dims[1]]
        for size in item_sizes:
            features.extend(size)
        while len(features) < 2 + (max_items_per_layout * 2):
            features.append(0)

        # Output labels: [item1_x, item1_y, ..., padded zeros]
        labels = []
        for pos in positions:
            labels.extend([pos["x"], pos["y"]])
        while len(labels) < (max_items_per_layout * 2):
            labels.append(0)

        X_data.append(features)
        y_data.append(labels)

    return np.array(X_data), np.array(y_data)

if __name__ == "__main__":
    # Generate the dataset
    layout_collection = create_valid_layout(500)  # Generate 500 samples
    if not layout_collection:
        print("✗ Dataset generation failed. Verify setup or increase attempt limit.")
        exit(1)

    print(f"✓ Generated {len(layout_collection)} samples successfully.")

    # Prepare and save to CSV
    try:
        print("📋 Preparing data for CSV export...")
        X, y = format_data_for_storage(layout_collection)
        data_frame = pd.DataFrame({
            "features": [x.tolist() for x in X],
            "labels": [y.tolist() for y in y],
            "item_list": [entry["item_names"] for entry in layout_collection]
        })

        csv_file = "layout_dataset.csv"
        print(f"💾 Exporting dataset to `{csv_file}`...")
        data_frame.to_csv(csv_file, index=False)

        if os.path.exists(csv_file):
            print(f"✓ Dataset exported successfully: {csv_file}")
        else:
            print(f"✗ Export failed: File `{csv_file}` not created.")
    except Exception as e:
        print(f"✗ Error during CSV export: {e}")
        exit(1)