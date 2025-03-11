import matplotlib.colors as mcolors

STD_colors = [
    "#438086", "#53548A", "#A04DA3",  # Base Colors
    "#71A3A7", "#7B7CB0", "#C178C4",  # Lighter Variations (Tints)
    "#2D5B5E", "#3C3D6A", "#722B74",  # Darker Variations (Shades)
    "#86431E", "#8A7853", "#3AA34D"   # Complementary Accents
]

def make_a_cmap(index):
    # Base color
    base_color = STD_colors[index]

    # Generate a colormap by blending the base color with white and black
    colors = [mcolors.to_rgb('#ffffff'),
              mcolors.to_rgb(STD_colors[index]),
              mcolors.to_rgb('#000000')]
    return mcolors.LinearSegmentedColormap.from_list("custom_purple", colors, N=256)

STD_cmaps = [make_a_cmap(i) for i in range(3)]