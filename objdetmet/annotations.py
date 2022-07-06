import numbers
import sys
sys.path.append('..')
from objdetmet.utils import cast_if_different_class

# Class for shape
class Shape2D:

    def __init__(self, *args):
        if len(args) == 2 and isinstance(args[0], numbers.Number) and isinstance(args[1], numbers.Number):
            self.w = args[0]
            self.h = args[1]
        elif len(args) == 1 and isinstance(args[0], Shape2D):
            self.w = args[0].w
            self.h = args[0].h
        elif len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) == 2 and isinstance(args[0][0], numbers.Number) and isinstance(args[0][1], numbers.Number):
            self.w = args[0][0]
            self.h = args[0][1]
        else:
            raise TypeError('Expected Shape2D, tuple/list of two numbers or two numbers.')
    
    def _get_w(self):
        return self._w
    
    def _set_w(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError('Widths and heights must be positive numbers.')
        self._w = value

    w = property(_get_w, _set_w, doc='Width')
    
    def _get_h(self):
        return self._h
    
    def _set_h(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError('Widths and heights must be positive numbers.')
        self._h = value

    h = property(_get_h, _set_h, doc='Height')
    
    # Iterable class to convert to list or tuple
    def __iter__(self):
        yield self.w
        yield self.h
    
    def __eq__(self, other):
        return self.w == other.w and self.h == other.h

# Class for Point2D
class Point2D:

    def __init__(self, *args):
        if len(args) == 2 and isinstance(args[0], numbers.Number) and isinstance(args[1], numbers.Number):
            self._x = args[0]
            self._y = args[1]
        elif len(args) == 1 and isinstance(args[0], Point2D):
            self._x = args[0]._x
            self._y = args[0]._y
        elif len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) == 2 and isinstance(args[0][0], numbers.Number) and isinstance(args[0][1], numbers.Number):
            self._x = args[0][0]
            self._y = args[0][1]
        elif len(args) == 0:
            self._x = 0
            self._y = 0
        else:
            raise TypeError('Expected Point2D, tuple/list of two numbers or two numbers.')
    
    def _get_x(self):
        return self._x
    
    def _set_x(self, value):
        value = float(value)
        self._x = value

    x = property(_get_x, _set_x, doc='x coordinate')
    
    def _get_y(self):
        return self._y
    
    def _set_y(self, value):
        value = float(value)
        self._y = value

    y = property(_get_y, _set_y, doc='y coordinate')
    
    # Iterable class to convert to list or tuple
    def __iter__(self):
        yield self._x
        yield self._y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

# Class for rectangular box
class Box:
    
    def __init__(self, p1=None, p2=None, pc=None, shape=None):
        # p1, p2 and pc can be None, Point2D or any type that can be used to construct a new Point2D
        try:
            p1 = p1 if p1 is None else cast_if_different_class(p1, Point2D)
            p2 = p2 if p2 is None else cast_if_different_class(p2, Point2D)
            pc = pc if pc is None else cast_if_different_class(pc, Point2D)
        except TypeError as e:
            raise TypeError('Parameters p1, p2 and pc must either be None, Point2D or any type that can be used to construct a new Point2D.')
        
        # shape can be None, Shape2D or any type that can be used to construct a new Shape2D
        try:
            shape = shape if shape is None or isinstance(shape, Shape2D) else Shape2D(shape)
        except TypeError as e:
            raise TypeError('Parameter shape must either be None, Shape2D or any type that can be used to construct a new Shape2D.')
       
        # p1 and p2 defined
        if p1 is not None and p2 is not None and pc is None and shape is None:
            self._p1 = p1
            self._p2 = p2
        # p1 and shape defined
        elif p1 is not None and shape is not None and p2 is None and pc is None:
            self._p1 = p1
            self._p2 = Point2D(p1.x + shape.w, p1.y + shape.h)
        # pc and shape defined
        elif pc is not None and shape is not None and p1 is None and p2 is None:
            self._p1 = Point2D(pc.x - shape.w / 2, pc.y - shape.h / 2)
            self._p2 = Point2D(pc.x + shape.w / 2, pc.y + shape.h / 2)
        else:
            raise ValueError('Either parameters p1 and p2 or p1 and shape or pc and shape must be defined.')
        
        self._check_p1_p2()
        self._check_shape()
    
    def __eq__(self, other):
        self.p1 == other.p1
        self.p2 == other.p2

    """Ensure x2 >= x1"""
    def _check_x1_x2(self):
        if self.x2 < self.x1:
            self.x2, self.x1 = self.x1, self.x2
    
    """Ensure y2 >= y1"""
    def _check_y1_y2(self):
        if self.y2 < self.y1:
            self.y2, self.y1 = self.y1, self.y2

    """Ensure x2 >= x1 and y2 >= y1"""
    def _check_p1_p2(self):
        self._check_x1_x2()
        self._check_y1_y2()
    
    """Ensure w > 0"""
    def _check_w(self):
        if self.x2 == self.x1:
            raise ValueError('Width must be positive.')
    
    """Ensure h > 0"""
    def _check_h(self):
        if self.y2 == self.y1:
            raise ValueError('Height must be positive.')
    
    """Ensure w > 0 and h > 0"""
    def _check_shape(self):
        self._check_w()
        self._check_h()

    def _get_x1(self):
        return self.p1.x
    
    def _set_x1(self, value):
        self.p1.x = float(value)
        self._check_x1_x2()
    
    x1 = property(_get_x1, _set_x1, doc='Top-left corner x coordinate')

    def _get_y1(self):
        return self.p1.y
    
    def _set_y1(self, value):
        self.p1.y = float(value)
        self._check_y1_y2()
    
    y1 = property(_get_y1, _set_y1, doc='Top-left corner y coordinate')

    def _get_p1(self):
        return self._p1
    
    def _set_p1(self, value):
        self._p1 = cast_if_different_class(value, Point2D)
    
    p1 = property(_get_p1, _set_p1, doc='Top-left corner point')

    def _get_x2(self):
        return self.p2.x
    
    def _set_x2(self, value):
        self.p2.x = float(value)
        self._check_x1_x2()
    
    x2 = property(_get_x2, _set_x2, doc='Bottom-right corner x coordinate')

    def _get_y2(self):
        return self.p2.y
    
    def _set_y2(self, value):
        self.p2.y = float(value)
        self._check_y1_y2()
    
    y2 = property(_get_y2, _set_y2, doc='Bottom-right corner x coordinate')

    def _get_p2(self):
        return self._p2
    
    def _set_p2(self, value):
        self._p2 = cast_if_different_class(value, Point2D)
    
    p2 = property(_get_p2, _set_p2, doc='Bottom-right corner point')

    def _get_xc(self):
        return (self.x2 + self.x2) / 2
    
    def _set_xc(self, value):
        value = float(value)
        self.x1 = value - self.w / 2
        self.x2 = value + self.w / 2
    
    xc = property(_get_xc, _set_xc, doc='Center point x coordinate')

    def _get_yc(self):
        return (self.y1 + self.y2) / 2
    
    def _set_yc(self, value):
        value = float(value)
        self.y1 = value - self.h / 2
        self.y2 = value + self.h / 2
    
    yc = property(_get_yc, _set_yc, doc='Center point y coordinate')

    def _get_pc(self):
        return Point2D(self.xc, self.yc)
    
    def _set_pc(self, value):
        value = cast_if_different_class(value, Point2D)
        self.xc = value.x
        self.yc = value.y
    
    pc = property(_get_pc, _set_pc, doc='Center point')

    def _get_w(self):
        return self.x2 - self.x1
    
    def resize_w_center(self, value):
        value = float(value)
        if value <= 0:
            raise TypeError('Expected a positive width.')
        w_diff = value - self.w
        self.x1 -= w_diff / 2
        self.x2 += w_diff / 2
    
    def resize_w_left(self, value):
        value = float(value)
        if value <= 0:
            raise TypeError('Expected a positive width.')
        w_diff = value - self.w
        self.x2 += w_diff
    
    w = property(_get_w, doc='Width')

    def _get_h(self):
        return self.y2 - self.y1
    
    def resize_h_center(self, value):
        value = float(value)
        if value <= 0:
            raise TypeError('Expected a positive height.')
        h_diff = value - self.h
        self.y1 -= h_diff / 2
        self.y2 += h_diff / 2
    
    def resize_h_top(self, value):
        value = float(value)
        if value <= 0:
            raise TypeError('Expected a positive width.')
        h_diff = value - self.h
        self.y2 += h_diff
    
    h = property(_get_h, doc='Height')

    def _get_shape(self):
        return Shape2D(self.w, self.h)
    
    def resize_center(self, value):
        value = cast_if_different_class(value, Shape2D)
        self.resize_w_center(value.w)
        self.resize_h_center(value.h)
    
    def resize_top_left(self, value):
        value = cast_if_different_class(value, Shape2D)
        self.resize_w_left(value.w)
        self.resize_h_top(value.h)
    
    shape = property(_get_shape, doc='Shape')
    
    @property
    def ar(self):
        """Aspect ratio"""
        h = self.h()
        return self.w() / h if h != 0 else float('NaN')
    
    def to_rel(self, img_shape):
        img_shape = cast_if_different_class(img_shape, Shape2D)
        return Box(p1=Point2D(self.x1 / img_shape.w, self.y1 / img_shape.h), p2=Point2D(self.x2 / img_shape.w, self.y2 / img_shape.h))

    def to_abs(self, img_shape):
        img_shape = cast_if_different_class(img_shape, Shape2D)
        return Box(p1=Point2D(self.x1 * img_shape.w, self.y1 * img_shape.h), p2=Point2D(self.x2 * img_shape.w, self.y2 * img_shape.h))