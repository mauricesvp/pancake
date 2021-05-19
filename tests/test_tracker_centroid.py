from pancake.tracker.tracker_centroid import CentroidTracker


def test_main():
    tc = CentroidTracker(11520, 2160) # width, height

    # x_ul, y_ul, x_ur, y_ur, x_lr, y_lr, x_ll, y_ll, confidence, (classid)
    det1 = [(1371, 949, 1403, 925, 1421, 948, 1389, 972), (1411, 1080, 1458, 1043, 1489, 1083, 1442, 1120)]

    det2 = [(1400, 1000, 1500, 1000), (1500, 1100, 1500, 1100, 1500, 1100, 1500, 1200)]

    print( tc.update(det=det1) )

    print( tc.update(det=det2) )

if __name__ == "__main__":
    test_main()

