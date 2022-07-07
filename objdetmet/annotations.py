import numbers
import sys
sys.path.append('..')
from objdetmet.utils import cast_if_different_class

# Class for shape
class Shape2D:

    def __init__(self, w, h):
        try:
            w = cast_if_different_class(w, float)
            h = cast_if_different_class(h, float)
        except :
            raise TypeError('Expected two numbers or other types castable to float.')
        if w < 0 or h < 0:
            raise ValueError('Widths and heights must be non-negative numbers.')
        self._w = w
        self._h = h
    
    # Iterable class to convert to list or tuple
    def __iter__(self):
        yield self.w
        yield self.h
    
    def __eq__(self, other):
        return self.w == other.w and self.h == other.h
    
    def __repr__(self):
        return f"Shape2D(w={self._w}, h={self._h})"
    
    def _get_w(self):
        return self._w
    
    def _set_w(self, value):
        value = cast_if_different_class(value, float)
        if value < 0:
            raise ValueError('Widths and heights must be non-negative numbers.')
        self._w = value

    w = property(_get_w, _set_w, doc='Width')
    
    def _get_h(self):
        return self._h
    
    def _set_h(self, value):
        value = cast_if_different_class(value, float)
        if value < 0:
            raise ValueError('Widths and heights must be non-negative numbers.')
        self._h = value

    h = property(_get_h, _set_h, doc='Height')

# Class for Point2D
class Point2D:

    def __init__(self, x=0.0, y=0.0):
        try:
            self._x = cast_if_different_class(x, float)
            self._y = cast_if_different_class(y, float)
        except :
            raise TypeError('Expected two numbers or other types castable to float.')
    
    # Iterable class to convert to list or tuple
    def __iter__(self):
        yield self._x
        yield self._y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"Point2D(x={self._x}, y={self._y})"
    
    def _get_x(self):
        return self._x
    
    def _set_x(self, value):
        self._x = cast_if_different_class(value, float)

    x = property(_get_x, _set_x, doc='x coordinate')
    
    def _get_y(self):
        return self._y
    
    def _set_y(self, value):
        self._y = cast_if_different_class(value, float)

    y = property(_get_y, _set_y, doc='y coordinate')

# Class for rectangular box
class Box:
    
    def __init__(self, left, top, right, bottom):
        try:
            left = cast_if_different_class(left, float)
            top = cast_if_different_class(top, float)
            right = cast_if_different_class(right, float)
            bottom = cast_if_different_class(bottom, float)
        except Exception as e:
            raise TypeError('Parameters left, right, top and bottom must be float.')
        
        self._l = left
        self._t = top
        self._r = right
        self._b = bottom
        
        self._check_lrtb()
    
    def __eq__(self, other):
        return all([self._l == other._l, self._r == other._r, self._t == other._t, self._b == other._b])
    
    def __iter__(self):
        '''Iterate through the four corners, starting from top-right and going clockwise'''
        yield self.p1
        yield self.p2
        yield self.p3
        yield self.p4
    
    def __repr__(self):
        return f"Box(left={self._l}, top={self._t}, right={self._r}, bottom={self._b})"

    def _check_lr(self):
        '''Ensure r >= l'''
        if self._r < self._l:
            self._l, self._r = self._r, self._l
    
    def _check_tb(self):
        '''Ensure b >= t'''
        if self._b < self._t:
            self._t, self._b = self._b, self._t

    def _check_lrtb(self):
        '''Ensure r >= l and b >= t'''
        self._check_lr()
        self._check_tb()
    
    @property
    def left(self):
        '''Left side x coordinate'''
        return self._l
    
    @property
    def right(self):
        '''Right side x coordinate'''
        return self._r
    
    @property
    def top(self):
        '''Top side y coordinate'''
        return self._t
    
    @property
    def bottom(self):
        '''Bottom side y coordinate'''
        return self._b

    @property
    def coordinates(self):
        '''Tuple of the four box coordinates: (left, top, right, bottom)'''
        return (self._l, self._t, self._r, self._b)
    
    def set_coordinates(self, left=None, top=None, right=None, bottom=None):
        '''Set any of the four box coordinates'''
        if left is not None:
            self._l = cast_if_different_class(left, float)
        if top is not None:
            self._t = cast_if_different_class(top, float)
        if right is not None:
            self._r = cast_if_different_class(right, float)
        if bottom is not None:
            self._b = cast_if_different_class(bottom, float)
        self._check_lrtb()

    @property
    def x1(self):
        '''Top-left corner x coordinate'''
        return self._l
    
    @property
    def y1(self):
        '''Top-left corner y coordinate'''
        return self._t
    
    @property
    def p1(self):
        '''Top-left corner Point2D'''
        return Point2D(self._l, self._t)

    @property
    def x2(self):
        '''Top-right corner x coordinate'''
        return self._r
    
    @property
    def y2(self):
        '''Top-right corner y coordinate'''
        return self._t
    
    @property
    def p2(self):
        '''Top-right corner Point2D'''
        return Point2D(self._r, self._t)

    @property
    def x3(self):
        '''Bottom-right corner x coordinate'''
        return self._r
    
    @property
    def y3(self):
        '''Bottom-right corner y coordinate'''
        return self._b
    
    @property
    def p3(self):
        '''Bottom-right corner Point2D'''
        return Point2D(self._r, self._b)

    @property
    def x4(self):
        '''Bottom-left corner x coordinate'''
        return self._l
    
    @property
    def y4(self):
        '''Bottom-left corner y coordinate'''
        return self._b
    
    @property
    def p4(self):
        '''Bottom-left corner Point2D'''
        return Point2D(self._l, self._b)
    
    def _get_xc(self):
        return (self._l + self._r) / 2
    
    def _set_xc(self, value):
        value = cast_if_different_class(value, float)
        w = self.w
        self._l = value - w / 2
        self._r = value + w / 2
    
    xc = property(_get_xc, _set_xc, doc='Center x coordinate')

    def _get_yc(self):
        return (self._t + self._b) / 2
    
    def _set_yc(self, value):
        value = cast_if_different_class(value, float)
        h = self.h
        self._t = value - h / 2
        self._b = value + h / 2
    
    yc = property(_get_yc, _set_yc, doc='Center y coordinate')

    def _get_pc(self):
        return Point2D(self.xc, self.yc)
    
    def _set_pc(self, value):
        value = cast_if_different_class(value, Point2D)
        self.xc = value.x
        self.yc = value.y
    
    pc = property(_get_pc, _set_pc, doc='Center point2D')

    @property
    def w(self):
        '''Width'''
        return self._r - self._l

    @property
    def h(self):
        '''Height'''
        return self._b - self._t

    @property
    def shape(self):
        '''Width and height as Shpe2D'''
        return Shape2D(self.w, self.h)
    
    def set_w_center(self, value):
        '''Change width keeping the same center coordinate'''
        value = cast_if_different_class(value, float)
        if value < 0:
            raise TypeError('Expected a non-negative width.')
        w_diff = value - self.w
        self._l -= w_diff / 2
        self._r += w_diff / 2
    
    def set_w_left(self, value):
        '''Change width keeping the same top coordinate'''
        value = cast_if_different_class(value, float)
        if value < 0:
            raise TypeError('Expected a non-negative width.')
        w_diff = value - self.w
        self._r += w_diff
    
    def set_h_center(self, value):
        '''Change height keeping the same center coordinate'''
        value = cast_if_different_class(value, float)
        if value < 0:
            raise TypeError('Expected a non-negative height.')
        h_diff = value - self.h
        self._t -= h_diff / 2
        self._b += h_diff / 2
    
    def set_h_top(self, value):
        '''Change height keeping the same top coordinate'''
        value = cast_if_different_class(value, float)
        if value < 0:
            raise TypeError('Expected a non-negative height.')
        h_diff = value - self.h
        self._b += h_diff
    
    def set_shape_center(self, value):
        '''Change shape keeping the same center coordinates'''
        value = cast_if_different_class(value, Shape2D)
        self.set_w_center(value.w)
        self.set_h_center(value.h)
    
    def set_shape_top_left(self, value):
        '''Change shape keeping the same top-left corner coordinates'''
        value = cast_if_different_class(value, Shape2D)
        self.set_w_left(value.w)
        self.set_h_top(value.h)
    
    @property
    def ar(self):
        '''Aspect ratio'''
        try:
            return self.w / self.h
        except ZeroDivisionError as e:
            return float('NaN')
    
    def normalize(self, img_shape):
        '''Normalize coordinates with respect to img_shape (the image shape)'''
        img_shape = cast_if_different_class(img_shape, Shape2D)
        self.set_coordinates(left=self._l / img_shape.w, top=self._t / img_shape.h, right=self._r / img_shape.w, bottom=self._b / img_shape.h)
    
    def denormalize(self, img_shape):
        '''Denormalize coordinates with respect to img_shape (the image shape)'''
        img_shape = cast_if_different_class(img_shape, Shape2D)
        self.set_coordinates(left=self._l * img_shape.w, top=self._t * img_shape.h, right=self._r * img_shape.w, bottom=self._b * img_shape.h)
    
    def to_normalized(self, img_shape):
        '''Returns a new box whose coordinates are normalized with respect to img_shape (the image shape)'''
        img_shape = cast_if_different_class(img_shape, Shape2D)
        return Box(left=self._l / img_shape.w, top=self._t / img_shape.h, right=self._r / img_shape.w, bottom=self._b / img_shape.h)

    def to_denormalized(self, img_shape):
        '''Returns a new box whose coordinates are denormalized with respect to img_shape (the image shape)'''
        img_shape = cast_if_different_class(img_shape, Shape2D)
        return Box(left=self._l * img_shape.w, top=self._t * img_shape.h, right=self._r * img_shape.w, bottom=self._b * img_shape.h)