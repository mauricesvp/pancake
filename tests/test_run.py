import argparse
from pancake import run


def test_run(cfg_path: str=None):
    run.main(cfg_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='?', type=str, default=None, help="pancake config path")
    args = parser.parse_args()

    test_run(args.cfg)
