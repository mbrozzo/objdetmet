import unittest
import math
import sys
sys.path.append('../objdetmet')
from objdetmet.annotations import Shape2D, Point2D, Box

class TestShape2D(unittest.TestCase):

    def test_init_and_tup(self):
        self.assertEqual(tuple(Shape2D(10.2, 100)), (10.2, 100))
        self.assertRaises(ValueError, Shape2D, 10, -10)
        self.assertRaises(ValueError, Shape2D, -10, 10)
        self.assertRaises(TypeError, Shape2D, 'asd', 10)
        self.assertRaises(TypeError, Shape2D, 10, 'asd')
    
    def test_properties(self):
        shape = Shape2D(18, 13)
        self.assertEqual(shape.w, 18)
        self.assertEqual(shape.h, 13)
        shape.w, shape.h = 19.2, 14.3
        self.assertEqual(shape.w, 19.2)
        self.assertEqual(shape.h, 14.3)
        def assign_w(shape, w):
            shape.w = w
        def assign_h(shape, h):
            shape.h = h
        self.assertRaises(ValueError, assign_w, shape, -10)
        self.assertRaises(ValueError, assign_h, shape, -10)

    def test_eq(self):
        s1 = Shape2D(12, 3)
        s2 = Shape2D(12, 3)
        s3 = Shape2D(12, 5)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
    
    def test_repr_str(self):
        self.assertEqual(repr(Shape2D(12, 20)), "Shape2D(w=12.0, h=20.0)")
        self.assertEqual(str(Shape2D(12, 20)), "Shape2D(w=12.0, h=20.0)")

class TestPoint2D(unittest.TestCase):

    def test_init_tup(self):
        self.assertEqual(tuple(Point2D(10.2, -100)), (10.2, -100))
        self.assertRaises(TypeError, Point2D, 'asd', 10)
        self.assertRaises(TypeError, Point2D, 10, 'asd')
    
    def test_properties(self):
        p = Point2D(18, 13)
        self.assertEqual(p.x, 18)
        self.assertEqual(p.y, 13)
        p.x, p.y = 19.2, 14.3
        self.assertEqual(p.x, 19.2)
        self.assertEqual(p.y, 14.3)
    
    def test_eq(self):
        p1 = Point2D(12, 3)
        p2 = Point2D(12, 3)
        p3 = Point2D(12, 5)
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
    
    def test_repr_str(self):
        self.assertEqual(repr(Point2D(12, 20)), "Point2D(x=12.0, y=20.0)")
        self.assertEqual(str(Point2D(12, 20)), "Point2D(x=12.0, y=20.0)")

class TestBox(unittest.TestCase):

    def test_init(self):
        box = Box(12, -20, -45, 50)
        self.assertEqual(box._l, -45)
        self.assertEqual(box._t, -20)
        self.assertEqual(box._r, 12)
        self.assertEqual(box._b, 50)
        def box_wrong_type():
            Box('asd', '12', 'fss', 'a')
        self.assertRaises(TypeError, box_wrong_type)
    
    def test_getters(self):
        box = Box(0, 2, 1, 4)
        self.assertEqual(box.left, 0)
        self.assertEqual(box.top, 2)
        self.assertEqual(box.right, 1)
        self.assertEqual(box.bottom, 4)
        self.assertEqual(box.x1, 0)
        self.assertEqual(box.y1, 2)
        self.assertEqual(box.p1, Point2D(0, 2))
        self.assertEqual(box.x2, 1)
        self.assertEqual(box.y2, 2)
        self.assertEqual(box.p2, Point2D(1, 2))
        self.assertEqual(box.x3, 1)
        self.assertEqual(box.y3, 4)
        self.assertEqual(box.p3, Point2D(1, 4))
        self.assertEqual(box.x4, 0)
        self.assertEqual(box.y4, 4)
        self.assertEqual(box.p4, Point2D(0, 4))
        self.assertEqual(box.coordinates, (0, 2, 1, 4))
        self.assertEqual(box.xc, 0.5)
        self.assertEqual(box.yc, 3)
        self.assertEqual(box.pc, Point2D(0.5, 3))
        self.assertEqual(box.w, 1)
        self.assertEqual(box.h, 2)
        self.assertEqual(box.shape, Shape2D(1, 2))
        self.assertEqual(box.ar, 0.5)
        self.assertTrue(math.isnan(Box(12, 4, 15, 4).ar))
    
    def test_setters(self):
        box = Box(0, 2, 1, 4)
        box.set_coordinates(9, 8, 7, 6)
        self.assertEqual(box.coordinates, (7, 6, 9, 8))
        box.xc = 1
        self.assertEqual(box.coordinates, (0, 6, 2, 8))
        box.yc = 1
        self.assertEqual(box.coordinates, (0, 0, 2, 2))
        box.pc = Point2D(3, 3)
        self.assertEqual(box.coordinates, (2, 2, 4, 4))
        box.set_w_center(4)
        self.assertEqual(box.coordinates, (1, 2, 5, 4))
        box.set_h_center(4)
        self.assertEqual(box.coordinates, (1, 1, 5, 5))
        box.set_shape_center(Shape2D(2, 2))
        self.assertEqual(box.coordinates, (2, 2, 4, 4))
        box.set_w_left(4)
        self.assertEqual(box.coordinates, (2, 2, 6, 4))
        box.set_w_left(4)
        self.assertEqual(box.coordinates, (2, 2, 6, 4))
        box.set_shape_top_left(Shape2D(2, 2))
        self.assertEqual(box.coordinates, (2, 2, 4, 4))
    
    def test_rel_abs(self):
        box = Box(0, 2, 1, 4)
        box.normalize(Shape2D(4, 4))
        self.assertEqual(box.coordinates, (0, 0.5, 0.25, 1))
        box.denormalize(Shape2D(4, 4))
        self.assertEqual(box.coordinates, (0, 2, 1, 4))
        self.assertEqual(box.to_normalized(Shape2D(4, 4)).coordinates, (0, 0.5, 0.25, 1))
        self.assertEqual(box.to_denormalized(Shape2D(4, 4)).coordinates, (0, 8, 4, 16))

if __name__ == '__main__':
    unittest.main()