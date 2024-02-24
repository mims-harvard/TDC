import unittest

if __name__ == '__main__':
    loader = unittest.TestLoader()
    start_dir = 'tdc/test'
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)