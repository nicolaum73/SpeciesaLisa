
import numpy as np
from PIL import Image, ImageDraw
Color = tuple[int,int,int,int]

class Point():
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self): return f"P({self.x}, {self.y})"

class Triangle:
    def __init__(self, verts: list[Point], color: Color):
        self.verts = verts
        self.color = color
    
    def make_vert_list(self):
        return list(map(lambda v: (v.x, v.y), self.verts))
    
    def __str__(self): return f"T({','.join(str(v) for v in self.verts)}, C{self.color})"

class Prediction:
    BEST_SCORE = 1
    def __init__(self, tris: list[Triangle], size: tuple[int, int]):
        self.tris = tris
        self.size = size
        self.h = size[1]

    @staticmethod
    def set_target(target: np.ndarray):
        Prediction.target = target

    def render(self):
        self.image = Image.new("RGBA", self.size, (0, 0, 0, 0))  
        for tri in self.tris:
            overlay = Image.new("RGBA", self.image.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).polygon(tri.make_vert_list(), fill=tri.color)
            self.image = Image.alpha_composite(self.image, overlay)
        self.prediction = np.array(self.image)

    def show_error(self):
        image = Image.new(self.image.mode, (self.image.width*3, self.image.height))
        image.paste(self.image, (0,0))
        image.paste(Image.fromarray(self.target), (self.image.width,0))
        diff_img = None
        diff = np.clip(self.alpha_differences(), 0, 255).astype("uint8")
        diff_img = Image.fromarray(diff).convert("RGBA")
            
        image.paste(diff_img, (self.image.width*2, 0))
        return image

    def alpha_differences(self):
        A = self.prediction.astype(np.int16)                        # int16 prevents 8 bit overflow when subtracting
        B = self.target.astype(np.int16)

        diff = np.abs(A - B)
        diff = (diff[:,:,:-1].mean(2) + diff[:,:,-1:].mean(2)) / 2  # averages the difference equally tone vs alpha

        a_A = A[:,:,3] != 0                                         # alpha masks
        a_B = B[:,:,3] != 0

        diff = np.where((a_A ^ a_B), 255, diff)                     # incorrect alpha 
        diff = np.where(~(a_A | a_B), 0, diff)                      # both transparent => no error
        return diff

    def evaluate(self) -> float:
        score = self.alpha_differences().mean() / 255                                   # Average and Normalize to 0-1
        if score < Prediction.BEST_SCORE: 
            Prediction.BEST_SCORE = score
            self.show_error().save(f"./test_images/Error {score:.3f} | Polys {len(self.tris)}.png")
        return score

    def __str__(self): return f"Pred({' | '.join(str(t) for t in self.tris)}, Dim={self.size[0]} x {self.size[1]})"