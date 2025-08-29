import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

default_transparency = 64  # 25% transparent

def equilateral_triangle(x, y, l):
    h = 0.8660254 * l  # precomputed sqrt(3)/2
    return [
        (x - l//2, y + h//2), # A: bottom-left
        (x + l//2, y + h//2), # B: bottom-right
        (x, y - h//2)         # C: top
    ]

def triangle(dim, x, y, colour, img_size = (256, 256)):
    """
    Create an Image with an equilateral triangle.

    x, y specify the centre of the shape."""
    layer = Image.new("RGBA", img_size, (0, 0, 0, 0))
    corners = equilateral_triangle(x, y, dim)
    ImageDraw.Draw(layer).polygon(corners, fill=(colour, colour, colour, default_transparency))
    return layer

def square(dim, x, y, colour, img_size = (256, 256)):
    """
    Create an Image with a square.

    x, y specify the centre of the shape."""
    layer = Image.new("RGBA", img_size, (0, 0, 0, 0))
    corners = [x - dim//2, y - dim//2, x + dim//2, y + dim//2]
    ImageDraw.Draw(layer).rectangle(corners, fill=(colour, colour, colour, default_transparency))
    return layer

def rectangle(dim1, dim2, x, y, colour, img_size = (256, 256)):
    """
    Create an Image with a rectangle.

    x, y specify the centre of the shape."""
    layer = Image.new("RGBA", img_size, (0, 0, 0, 0))
    corners = [x - dim1//2, y - dim2//2, x + dim1//2, y + dim2//2]
    ImageDraw.Draw(layer).rectangle(corners, fill=(colour, colour, colour, default_transparency))
    return layer

def circle(dim, x, y, colour, img_size = (256, 256)):
    """
    Create an Image with a circle.

    x, y specify the centre of the shape."""
    layer = Image.new("RGBA", img_size, (0, 0, 0, 0))
    corners = [x - dim//2, y - dim//2, x + dim//2, y + dim//2]
    ImageDraw.Draw(layer).ellipse(corners, fill=(colour, colour, colour, default_transparency))
    return layer

def ellipse(dim1, dim2, x, y, colour, img_size = (256, 256)):
    """
    Create an Image with an ellipse.

    x, y specify the centre of the shape."""
    layer = Image.new("RGBA", img_size, (0, 0, 0, 0))
    corners = [x - dim1//2, y - dim2//2, x + dim1//2, y + dim2//2]
    ImageDraw.Draw(layer).ellipse(corners, fill=(colour, colour, colour, default_transparency))
    return layer

def polygon(dims, colour, img_size = (256, 256)):
    """
    Create a polygon with len(dims)//2 sides.

    dims is a list of x, y coordinates for each vertex.
    Sorts the vertices in a clockwise order."""
    layer = Image.new("RGBA", img_size, (0, 0, 0, 0))
    dims = [(dims[i], dims[i+1]) for i in range(0, len(dims), 2)]
    # Sort vertices in clockwise order
    dims = sorted(dims, key=lambda p: (np.arctan2(p[1] - img_size[1] // 2, p[0] - img_size[0] // 2) + 2 * np.pi) % (2 * np.pi))
    ImageDraw.Draw(layer).polygon(dims, fill=(colour, colour, colour, default_transparency))
    return layer

def ssim_score(img1, img2, gray1, gray2):
    '''Compare two images for structural and color similarity.'''

    # Compute SSIM with explicit data range
    data_range = gray1.max() - gray1.min()
    shape_score, _ = ssim(gray1, gray2, data_range=data_range, full=True)
    return shape_score

def colhist_score(img1, img2, gray1, gray2):
    '''Compare two images for structural and color similarity.'''

    # Compute histogram similarity per channel
    hist_score = 0
    for i in range(3):  # R, G, B
        hist1, _ = np.histogram(img1[..., i], bins=256, range=(0, 255), density=True)
        hist2, _ = np.histogram(img2[..., i], bins=256, range=(0, 255), density=True)
        hist_score += np.correlate(hist1, hist2)[0]

    color_score = max(0.0, min(1.0, hist_score / 3))  # normalize

    return color_score

def totpixeldiff_score(img1, img2, gray1, gray2):
    """Compute sum of absolute pixel differences."""
    return np.sum(np.abs(gray1 - gray2))

def avgpixeldiff_score(img1, img2, gray1, gray2):
    """Compute sum of absolute pixel differences."""
    return np.average(np.abs(gray1 - gray2))

def create_image(layers, img_size=(256, 256)):
    """Create an image from a set of layers."""
    img = Image.new("RGBA", img_size, (0, 0, 0, 0))
    for layer in layers:
        img = Image.alpha_composite(img, layer)
    bw = img.convert('L')
    #img = np.array(img, dtype=np.float64)
    #bw = np.array(bw, dtype=np.float64)
    return img, bw

def main():
    """Example usage."""
    # Define the target image
    layers = []
    layers.append(triangle(100, 100, 200, 0))
    layers.append(triangle(100, 150, 200, 200))
    layers.append(triangle(100, 125, 275, 127))
    layers.append(triangle(100, 125, 125, 64))
    layers.append(triangle(100, 175, 175, 255))
    target_img = create_image(layers)
    target_img[0].save("target_image.png")
    # Convert to numpy arrays
    target_col = np.array(target_img[0])
    target_bw = np.array(target_img[1])
    # Define the image to evaluate
    layers = []
    layers.append(triangle(100, 100, 200, 120))
    layers.append(triangle(100, 159, 100, 90))
    layers.append(triangle(100, 155, 175, 227))
    layers.append(triangle(100, 105, 205, 4))
    layers.append(triangle(100, 195, 225, 155))
    img = create_image(layers)
    img[0].save("image.png")
    print("ssim_score:", ssim_score(np.array(img[0]), target_col, np.array(img[1]), target_bw))
    print("colhist_score:", colhist_score(np.array(img[0]), target_col, np.array(img[1]), target_bw))
    print("totpixeldiff_score:", totpixeldiff_score(np.array(img[0]), target_col, np.array(img[1]), target_bw))
    print("avgpixeldiff_score:", avgpixeldiff_score(np.array(img[0]), target_col, np.array(img[1]), target_bw))

if __name__ == "__main__":
    main()

