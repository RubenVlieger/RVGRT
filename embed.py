# embed_png.py
import os

input_file = "resources/texturepack.png"
output_file = "include/texturepack.h"

# Make sure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(input_file, "rb") as f:
    data = f.read()

with open(output_file, "w") as f:
    f.write("unsigned char texturepack_png[] = {\n")
    for i, b in enumerate(data):
        f.write(f"0x{b:02X}")
        if i != len(data) - 1:
            f.write(", ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"unsigned int texturepack_png_len = {len(data)};\n")

print(f"Header generated at {output_file}")