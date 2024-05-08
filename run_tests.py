import unittest
import sys


if __name__ == '__main__':
    loader = unittest.TestLoader()
    start_dir = 'tdc/test'
    
    # Check if a specific test is provided as a command-line argument
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        suite = loader.loadTestsFromName(test_name)
    else:
        suite = loader.discover(start_dir)
    
    runner = unittest.TextTestRunner()
    res = runner.run(suite)
    if res.wasSuccessful():
        print("All base tests passed")
    else:
        raise RuntimeError("Some base tests failed")