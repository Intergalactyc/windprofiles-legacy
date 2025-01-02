from classify import TerrainClassifier

def test_terrain_classifier():
    TERRAIN_CLASSIFIER = TerrainClassifier(complexCenter = 315,
                                          openCenter = 135,
                                          radius = 15,
                                          inclusive = True,
                                          height = 10)
    try:
        assert(TERRAIN_CLASSIFIER.get_height() == 10)
        assert(TERRAIN_CLASSIFIER.get_height_column() == 'wd_10m')
        assert(TERRAIN_CLASSIFIER.classify(322) == 'complex')
        assert(TERRAIN_CLASSIFIER.classify(130) == 'open')
        assert(TERRAIN_CLASSIFIER.classify(120) == 'open')
        assert(TERRAIN_CLASSIFIER.classify(150) == 'open')
        assert(TERRAIN_CLASSIFIER.classify(150.7) == 'other')
        assert(TERRAIN_CLASSIFIER.classify(299.9) == 'other')
    except AssertionError as e:
        return False
    return True

TESTS = {
    'terrain classifier' : test_terrain_classifier,
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
