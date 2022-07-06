import numbers

# Class for shape
class Shape2D:
    def __init__(self, *args):
        if len(args) == 2 and isinstance(args[0], numbers.Number) and isinstance(args[1], numbers.Number):
            self.w = args[0]
            self.h = args[1]
        elif len(args) == 1 and isinstance(args[0], Shape2D()):
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

# Class for Point2D
class Point2D:    
    def __init__(self, *args):
        if len(args) == 2 and isinstance(args[0], numbers.Number) and isinstance(args[1], numbers.Number):
            self._x = args[0]
            self._y = args[1]
        elif len(args) == 1 and isinstance(args[0], Point2D):
            self._x = args[0].x
            self._y = args[0].y
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

    y = property(_get_y, _set_y, doc='y coordiante')
    
    # Iterable class to convert to list or tuple
    def __iter__(self):
        yield self._x
        yield self._y

# Class for rectangular box
class Box:
    def __init__(self, p1=None, p2=None, pc=None, shape=None):
        # p1, p2 and pc can be None, Point2D or any type that can be used to construct a new Point2D
        try:
            p1 = p1 if p1 is None or isinstance(p1, Point2D) else Point2D(p1)
            p2 = p2 if p2 is None or isinstance(p2, Point2D) else Point2D(p2)
            pc = pc if pc is None or isinstance(pc, Point2D) else Point2D(pc)
        except TypeError as e:
            raise TypeError('Parameters p1, p2 and pc must either be None, Point2D or any type that can be used to construct a new Point2D.')
        
        # shape can be None, Shape2D or any type that can be used to construct a new Shape2D
        try:
            shape = shape if shape is None or isinstance(shape, Shape2D) else Shape2D(shape)
        except TypeError as e:
            raise TypeError('Parameter shape must either be None, Shape2D or any type that can be used to construct a new Shape2D.')

        # p1 and p2 defined
        if p1 is not None and p2 is not None and pc is None and shape is None:
            self._x1 = p1.x
            self._y1 = p1.y
            self._x2 = p2.x
            self._y2 = p2.y
        # p1 and shape defined
        elif p1 is not None and shape is not None and p2 is None and pc is None:
            self._x1 = p1.x
            self._y1 = p1.y
            self._x2 = p1.x + shape.w
            self._y2 = p1.y + shape.h
        # pc and shape defined
        elif pc is not None and shape is not None and p1 is None and p2 is None:
            self._x1 = pc.x - shape.w / 2
            self._y1 = pc.y - shape.h / 2
            self._x2 = pc.x + shape.w / 2
            self._y2 = pc.y + shape.h / 2
        else:
            raise ValueError('Either parameters p1 and p2 or p1 and shape or pc and shape must be defined.')

        self._check_p1_p2()

    """Ensure x2 >= x1"""
    def _check_x1_x2(self):
        if self._x2 < self._x1:
            self._x2, self._x1 = self._x1, self._x2
    
    """Ensure y2 >= y1"""
    def _check_y1_y2(self):
        if self._y2 < self._y1:
            self._y2, self._y1 = self._y1, self._y2

    """Ensure x2 >= x1 and y2 >= y1"""
    def _check_p1_p2(self):
        self._check_x1_x2()
        self._check_y1_y2()

    def _get_x1(self):
        return self._x1
    
    def _set_x1(self, value):
        self._x1 = float(value)
        self._check_x1_x2()
    
    x1 = property(_get_x1, _set_x1, doc='Top-left corner x coordinate')

    def _get_y1(self):
        return self._y1
    
    def _set_y1(self, value):
        self._y1 = float(value)
        self._check_y1_y2()
    
    y1 = property(_get_y1, _set_y1, doc='Top-left corner y coordinate')

    def _get_p1(self):
        return Point2D(self._x1, self._y1)
    
    def _set_p1(self, value):
        value = Point2D(value)
        self.x1 = value.x
        self.y1 = value.y
    
    p1 = property(_get_p1, _set_p1, doc='Top-left corner point')

    def _get_x2(self):
        return self._x2
    
    def _set_x2(self, value):
        self._x2 = float(value)
        self._check_x1_x2()
    
    x2 = property(_get_x2, _set_x2, doc='Bottom-right corner x coordinate')

    def _get_y2(self):
        return self._y2
    
    def _set_y2(self, value):
        self._y2 = float(value)
        self._check_y1_y2()
    
    y2 = property(_get_y2, _set_y2, doc='Bottom-right corner x coordinate')

    def _get_p2(self):
        return Point2D(self._x2, self._y2)
    
    def _set_p2(self, value):
        value = Point2D(value)
        self.x2 = value.x
        self.y2 = value.y
    
    p2 = property(_get_p2, _set_p2, doc='Bottom-right corner point')

    def _get_xc(self):
        return (self._x2 + self._x2) / 2
    
    def _set_xc(self, value):
        value = float(value)
        self._x1 = value - self.w / 2
        self._x2 = value + self.w / 2
    
    xc = property(_get_xc, _set_xc, doc='Center point x coordinate')

    def _get_yc(self):
        return (self._y1 + self._y2) / 2
    
    def _set_yc(self, value):
        value = float(value)
        self._y1 = value - self.h / 2
        self._y2 = value + self.h / 2
    
    yc = property(_get_yc, _set_yc, doc='Center point y coordinate')

    def _get_pc(self):
        return Point2D(self._xc(), self._yc())
    
    def _set_pc(self, value):
        value = Point2D(value)
        self.xc = value.x
        self.yc = value.y
    
    pc = property(_get_pc, _set_pc, doc='Center point')

    def _get_w(self):
        return self._x2 - self._x1
    
    def resize_w_center(self, value):
        value = float(value)
        if value <= 0:
            raise TypeError('Expected a positive width.')
        w_diff = value - self.w
        self._x1 -= w_diff / 2
        self._x2 += w_diff / 2
    
    def resize_w_left(self, value):
        value = float(value)
        if value <= 0:
            raise TypeError('Expected a positive width.')
        w_diff = value - self.w
        self._x2 += w_diff
    
    w = property(_get_w, doc='Width')

    def _get_h(self):
        return self._x2 - self._x1
    
    def resize_h_center(self, value):
        value = float(value)
        if value <= 0:
            raise TypeError('Expected a positive height.')
        h_diff = value - self.h
        self._y1 -= h_diff / 2
        self._y2 += h_diff / 2
    
    def resize_h_top(self, value):
        value = float(value)
        if value <= 0:
            raise TypeError('Expected a positive width.')
        h_diff = value - self.h
        self._y2 += h_diff
    
    h = property(_get_h, doc='Height')

    def _get_shape(self):
        return Shape2D(self.w, self.h)
    
    def resize_center(self, value):
        value = Shape2D(value)
        self.resize_w_center(value.w)
        self.resize_h_center(value.h)
    
    def resize_top_left(self, value):
        value = Shape2D(value)
        self.resize_w_left(value.w)
        self.resize_h_top(value.h)
    
    shape = property(_get_shape, doc='Shape')
    
    @property
    def ar(self):
        """Aspect ratio"""
        h = self.h()
        return self.w() / h if h != 0 else float('NaN')
    
    def to_rel(self, img_shape):
        img_shape = Shape2D(img_shape)
        return Box(p1=Point2D(self._x1 / img_shape.w, self._y1 / img_shape.h), p2=Point2D(self._x2 / img_shape.w, self._y2 / img_shape.h))

    def to_abs(self, img_shape):
        img_shape = Shape2D(img_shape)
        return Box(p1=Point2D(self._x1 * img_shape.w, self._y1 * img_shape.h), p2=Point2D(self._x2 * img_shape.w, self._y2 * img_shape.h))