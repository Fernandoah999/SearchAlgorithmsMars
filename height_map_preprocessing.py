#------------------------------------------------------------------------------------------------------------------
#   Height map pre-processing
#   Processes both mars_map.IMG (route search) and crater_map.IMG (crater descent)
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#   Imports
#------------------------------------------------------------------------------------------------------------------
import os
import copy
import numpy as np
from skimage.transform import downscale_local_mean

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import plotly.graph_objects as px

#------------------------------------------------------------------------------------------------------------------
#   Processing function
#------------------------------------------------------------------------------------------------------------------
def process_img(input_file, output_file):
    """
    Reads a .IMG height map file, applies subsampling (~10 m/pixel),
    and saves the result as a .npy numpy matrix.
    """
    if not os.path.exists(input_file):
        print(f"  [SKIP] '{input_file}' not found.")
        return None, None, None, None, None, None

    data_file = open(input_file, "rb")

    endHeader = False
    while not endHeader:
        line = data_file.readline().rstrip().lower()
        sep_line = line.split(b'=')

        if len(sep_line) == 2:
            itemName = sep_line[0].rstrip().lstrip()
            itemValue = sep_line[1].rstrip().lstrip()

            if itemName == b'valid_maximum':
                maxV = float(itemValue)
            elif itemName == b'valid_minimum':
                minV = float(itemValue)
            elif itemName == b'lines':
                n_rows = int(itemValue)
            elif itemName == b'line_samples':
                n_columns = int(itemValue)
            elif itemName == b'map_scale':
                scale_str = itemValue.split()
                if len(scale_str) > 1:
                    scale = float(scale_str[0])

        elif line == b'end':
            endHeader = True
            char = 0
            while char == 0 or char == 32:
                char = data_file.read(1)[0]
            pos = data_file.seek(-1, 1)

    image_size = n_rows * n_columns
    data = data_file.read(4 * image_size)
    data_file.close()

    image_data = np.frombuffer(data, dtype=np.dtype('f'))
    image_data = image_data.reshape((n_rows, n_columns))
    image_data = np.array(image_data).astype('float64')

    image_data = image_data - minV
    image_data[image_data < -10000] = -1

    # Subsampling
    sub_rate = round(10 / scale)
    image_data = downscale_local_mean(image_data, (sub_rate, sub_rate))
    image_data[image_data < 0] = -1

    new_scale = scale * sub_rate
    print(f"  Original: {n_rows} x {n_columns} px, scale: {scale:.4f} m/px")
    print(f"  Sub-sampling rate: {sub_rate}")
    print(f"  Result: {image_data.shape[0]} x {image_data.shape[1]} px, scale: {new_scale:.4f} m/px")

    np.save(output_file, image_data)
    print(f"  Saved: {output_file}")

    return image_data, n_rows, n_columns, maxV, minV, scale

#------------------------------------------------------------------------------------------------------------------
#   Visualization function
#------------------------------------------------------------------------------------------------------------------
def show_map(image_data, n_rows, n_columns, maxV, minV, scale, title):
    """Shows 3D surface and 2D shaded image of the processed map."""
    new_scale = scale * round(10 / scale)

    # 3D surface
    x = new_scale * np.arange(image_data.shape[1])
    y = new_scale * np.arange(image_data.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = px.Figure(
        data=px.Surface(
            x=X, y=Y, z=np.flipud(image_data), colorscale='hot', cmin=0,
            lighting=dict(ambient=0.0, diffuse=0.8, fresnel=0.02, roughness=0.4, specular=0.2),
            lightposition=dict(x=0, y=n_rows/2, z=2*maxV)
        ),
        layout=px.Layout(
            title=title,
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=n_rows/n_columns, z=max((maxV-minV)/x.max(), 0.2)),
            scene_zaxis_range=[0, maxV-minV]
        )
    )
    fig.show()

    # 2D shaded image
    cmap = copy.copy(plt.cm.get_cmap('autumn'))
    cmap.set_under(color='black')

    ls = LightSource(315, 45)
    rgb = ls.shade(image_data, cmap=cmap, vmin=0, vmax=image_data.max(), vert_exag=2, blend_mode='hsv')

    fig2, ax = plt.subplots()
    im = ax.imshow(rgb, cmap=cmap, vmin=0, vmax=image_data.max(),
                   extent=[0, scale*n_columns, 0, scale*n_rows],
                   interpolation='nearest', origin='upper')
    cbar = fig2.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Height (m)')
    plt.title(title)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

#------------------------------------------------------------------------------------------------------------------
#   Process both maps
#------------------------------------------------------------------------------------------------------------------
maps_to_process = [
    ("mars_map.IMG",   "mars_map.npy",   "Mars Surface (route search)"),
    ("class 7/crater_map.IMG", "class 7/crater_map.npy", "Mars Crater (descent)"),
]

for input_file, output_file, title in maps_to_process:
    print(f"\nProcessing: {input_file}")
    image_data, n_rows, n_columns, maxV, minV, scale = process_img(input_file, output_file)

    if image_data is not None:
        show_map(image_data, n_rows, n_columns, maxV, minV, scale, title)

print("\nDone.")

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
