import argparse
import cProfile
from pstats import SortKey

from pancake import run


def test_run(profile: bool = False):
    if not profile:
        run.main()
        return

    pr = cProfile.Profile()
    pr.enable()

    run.main()

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Use profiler")
    args = parser.parse_args()

    test_run(args.profile)
