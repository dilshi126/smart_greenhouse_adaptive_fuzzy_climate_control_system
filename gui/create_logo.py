"""
Create a logo for the Greenhouse Fuzzy Control System GUI.
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_greenhouse_logo(size=64, save_path="logo.png"):
    """Create a greenhouse-themed logo."""
    # Create image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Colors
    greenhouse_green = (34, 139, 34)  # Forest green
    glass_blue = (135, 206, 235, 180)  # Sky blue with transparency
    plant_green = (50, 205, 50)  # Lime green
    soil_brown = (139, 69, 19)  # Saddle brown
    frame_color = (105, 105, 105)  # Dim gray
    
    margin = size // 10
    
    # Draw greenhouse structure (house shape)
    # Base rectangle
    base_top = size // 3
    base_bottom = size - margin - size // 8
    draw.rectangle([margin, base_top, size - margin, base_bottom], 
                   fill=glass_blue, outline=frame_color, width=2)
    
    # Roof (triangle)
    roof_peak = margin
    roof_left = (margin, base_top)
    roof_right = (size - margin, base_top)
    roof_top = (size // 2, roof_peak)
    draw.polygon([roof_left, roof_top, roof_right], 
                 fill=glass_blue, outline=frame_color)
    
    # Draw vertical frame lines
    for i in range(1, 4):
        x = margin + (size - 2 * margin) * i // 4
        draw.line([(x, base_top), (x, base_bottom)], fill=frame_color, width=1)
    
    # Draw horizontal frame line
    mid_y = (base_top + base_bottom) // 2
    draw.line([(margin, mid_y), (size - margin, mid_y)], fill=frame_color, width=1)
    
    # Draw plants inside
    plant_y = base_bottom - size // 10
    for i in range(3):
        px = margin + size // 6 + i * (size // 4)
        # Stem
        draw.line([(px, plant_y), (px, plant_y - size // 6)], fill=plant_green, width=2)
        # Leaves
        draw.ellipse([px - size//12, plant_y - size//5, px + size//12, plant_y - size//10], 
                     fill=plant_green)
    
    # Draw soil/ground
    draw.rectangle([margin, base_bottom, size - margin, size - margin], 
                   fill=soil_brown)
    
    # Save the logo
    img.save(save_path, 'PNG')
    print(f"Logo saved to: {save_path}")
    return save_path


def create_icon(size=32, save_path="icon.ico"):
    """Create an ICO file for window icon."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Simple greenhouse icon
    green = (34, 139, 34)
    blue = (135, 206, 235)
    
    # House shape
    margin = 2
    mid = size // 2
    
    # Roof
    draw.polygon([(margin, mid), (mid, margin), (size - margin, mid)], fill=green)
    
    # Body
    draw.rectangle([margin + 2, mid, size - margin - 2, size - margin], fill=blue, outline=green)
    
    # Plant
    draw.ellipse([mid - 4, mid + 2, mid + 4, size - margin - 2], fill=(50, 205, 50))
    
    img.save(save_path, 'ICO')
    print(f"Icon saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    # Create both logo and icon
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    logo_path = os.path.join(script_dir, "logo.png")
    icon_path = os.path.join(script_dir, "icon.ico")
    
    create_greenhouse_logo(64, logo_path)
    create_icon(32, icon_path)
    
    print("\nLogo files created successfully!")
