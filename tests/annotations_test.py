import unittest
import sys
sys.path.append('../objdetmet')
from objdetmet.annotations import Shape2D, Point2D, Box

class TestShape2D(unittest.TestCase):

    def test_init_tup(self):
        self.assertEqual(tuple(Shape2D(10.2, 100)), (10.2, 100))
        self.assertEqual(tuple(Shape2D((10, 100.3))), (10, 100.3))
        self.assertRaises(ValueError, Shape2D, 10, -10)
        self.assertRaises(ValueError, Shape2D, -10, 10)
        self.assertRaises(TypeError, Shape2D, 'asd', 10)
        self.assertRaises(TypeError, Shape2D, 10, 'asd')
        s = Shape2D(12, 24)
        self.assertIsNot(s, Shape2D(s))
    
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
        self.assertRaises(ValueError, assign_h, shape, 0)

    def test_eq(self):
        s1 = Shape2D(12, 3)
        s2 = Shape2D(12, 3)
        s3 = Shape2D(12, 5)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)

class TestPoint2D(unittest.TestCase):

    def test_init_tup(self):
        self.assertEqual(tuple(Point2D(10.2, -100)), (10.2, -100))
        self.assertEqual(tuple(Point2D((-10, 100.3))), (-10, 100.3))
        self.assertEqual(tuple(Point2D()), (0, 0))
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

class TestBox(unittest.TestCase):

    def test_inits(self):
        box = Box((12, -20), Point2D(-45, 50))
        self.assertEqual(box.p1, Point2D(-45, -20))
        self.assertEqual(box.p2, Point2D(12, 50))
        box = Box(p1=(12, -20), shape=Shape2D(45, 50))
        self.assertEqual(box.p1, Point2D(12, -20))
        self.assertEqual(box.p2, Point2D(57, 30))
        box = Box(pc=(20, -20), shape=Shape2D(30, 50))
        self.assertEqual(box.p1, Point2D(5, -45))
        self.assertEqual(box.p2, Point2D(35, 5))
        self.assertRaises(ValueError, Box)

if __name__ == '__main__':
    unittest.main()