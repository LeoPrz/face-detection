import numpy as np


class Box():
    '''Represents a rect without position'''
    def __init__(self, w, h):
        self.w = w
        self.h = h
        
    def area(self):
        return self.h * self.w
    
    def scale(self, scale):
        self.w *= scale
        self.h *= scale
        return self
        
    def __repr__(self):
        return f'Box: ({self.x}, {self.y})'


class Rect(Box):
    '''Represents boxes / windows with position'''
    def __init__(self, x, y, w, h):
        super().__init__(w, h)
        self.x = x
        self.y = y
    
    def inter_area(self, other):
        '''Intersection area'''
        a, b = self, other
        x1 = max(min(a.x, a.x + a.w), min(b.x, b.x + b.w))
        y1 = max(min(a.y, a.y + a.h), min(b.y, b.y + b.h))
        x2 = min(max(a.x, a.x + a.w), max(b.x, b.x + b.w))
        y2 = min(max(a.y, a.y + a.h), max(b.y, b.y + b.h))
        if x1 < x2 and y1 < y2:
            return (x2 - x1) * (y2 - y1)
        else:
            return 0
    
    def inter_over_union(self, other):
        inter = self.inter_area(other)
        union = self.area() + other.area() - inter
        return inter / union
            
    def overlap(self, *others, threshold=0.5):
        '''Checks if this rect overlaps (enough) one of the given rects'''
        overlap = False
        for rect in others:
            overlap |= self.inter_over_union(rect) > threshold
        return overlap
    
    def center_is_inside(self, other):
        xc = self.x + self.w / 2
        yc = self.y + self.h / 2
        return xc > other.x and xc < other.x + other.w and yc > other.y and yc < other.y + other.h

    def scale(self, scale):
        super().scale(scale)
        self.x *= scale
        self.y *= scale
        return self
    
    def extract_from_img(self, img):
        img_h, img_w = img.shape[:2]
        if img_w - self.x < self.w or img_h - self.y < self.h:
            raise Exception('Can\'t extract the window, partly outside the image')
        
        return img[int(self.y):int(self.y+self.h), int(self.x):int(self.x+self.w)].astype(np.float32)
        
    def __repr__(self):
        return f'Rect: ({self.x}, {self.y}, {self.w}, {self.h})'

# import here so no circular imports
from params import BOX_SIZE, WINDOW_STEP


def sliding_window(img, step=WINDOW_STEP, box_size=BOX_SIZE):
    for x in range(0, img.shape[1] - box_size.w, step):
        for y in range(0, img.shape[0] - box_size.h, step):
            rect = Rect(x, y, box_size.w, box_size.h)
            extracted = rect.extract_from_img(img)
            yield rect, extracted   
