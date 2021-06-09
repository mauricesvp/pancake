import argparse
import cProfile
import io
from pstats import SortKey, Stats

from pancake import run


def test_run(profile: bool = False):
    n = 3
    if not profile:
        run.main(n=n)
        return

    pr = cProfile.Profile()
    pr.enable()

    run.main(n=n)

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Use profiler")
    args = parser.parse_args()

    test_run(args.profile)
