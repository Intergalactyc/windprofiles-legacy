from classify import TerrainClassifier, SingleClassifier
import math
import numpy as np

def silent_test(test_body):

    def inner(*args, **kwargs):
        try:
            result = test_body(*args, **kwargs)
            return result
        except AssertionError:
            return False
        except:
            print('(non-Assertion error encountered:)')
            return False
    
    return inner

@silent_test
def test_terrain_classifier():
    TERRAIN_CLASSIFIER = TerrainClassifier(complexCenter = 315,
                                          openCenter = 135,
                                          radius = 15,
                                          inclusive = True,
                                          height = 10)
    assert(TERRAIN_CLASSIFIER.get_height() == 10)
    assert(TERRAIN_CLASSIFIER.get_height_column() == 'wd_10m')
    assert(TERRAIN_CLASSIFIER.classify(322) == 'complex')
    assert(TERRAIN_CLASSIFIER.classify(130) == 'open')
    assert(TERRAIN_CLASSIFIER.classify(120) == 'open')
    assert(TERRAIN_CLASSIFIER.classify(150) == 'open')
    assert(TERRAIN_CLASSIFIER.classify(150.7) == 'other')
    assert(TERRAIN_CLASSIFIER.classify(299.9) == 'other')
    return True

@silent_test
def test_single_classifier_stability():
    STABILITY_CLASSIFIER = SingleClassifier(parameter = 'Ri_bulk')
    STABILITY_CLASSIFIER.add_class('unstable', '(-inf,-0.1)')
    STABILITY_CLASSIFIER.add_class('neutral', '[-0.1,0.1]')
    STABILITY_CLASSIFIER.add_class('stable', '(0.1,inf)')
    assert(STABILITY_CLASSIFIER.get_classes() == ['stable', 'neutral', 'unstable', 'other'])
    assert(STABILITY_CLASSIFIER.classify(0) == 'neutral')
    assert(STABILITY_CLASSIFIER.classify(-0.1) == 'neutral')
    assert(STABILITY_CLASSIFIER.classify(0.1) == 'neutral')
    assert(STABILITY_CLASSIFIER.classify(-0.5) == 'unstable')
    assert(STABILITY_CLASSIFIER.classify(0.5) == 'stable')
    assert(STABILITY_CLASSIFIER.classify(np.nan) is None)
    return True

TESTS = {
    'terrain classifier' : test_terrain_classifier,
    'stability classifier' : test_single_classifier_stability,
}

def run_tests():
    failures = 0
    for testName, testFunc in TESTS.items():
        if testFunc():
            print(f"Test '{testName}' successful")
        else:
            print(f"* Test '{testName}' FAILED")
            failures += 1
    if failures == 0:
        print('All tests passed!')
    else:
        print(f'Number of failed tests: {failures}')

if __name__ == '__main__':
    run_tests()
