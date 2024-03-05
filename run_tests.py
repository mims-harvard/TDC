import unittest

if __name__ == '__main__':
    loader = unittest.TestLoader()
    start_dir = 'tdc/test'
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    res = runner.run(suite)
    if res.wasSuccessful():
        print("All base tests passed")
    else:
        raise RuntimeError("Some base tests failed")
