""" Function time profiler """
import cProfile
import pstats
from functools import wraps


def profile(
    output_file: str = None,
    sort_by: str = "cumulative",
    lines_to_print: int = None,
    strip_dirs: bool = False,
) -> pstats.Stats:
    """A time profiler decorator.

    Description:
        Measures the time for all subcalls and documents it in a readable fashion.
        Inspired by and modified the profile decorator of Giampaolo Rodola:
        http://code.activestate.com/recipes/577817-profile-decorator/

    Usage:
        Simply import this function and add it as a decorator to the method to be
        profiled. (@profile(...))

    Args:
        output_file (str, optional):
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
            Defaults to None.
        sort_by (str, optional):
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats.
            Defaults to "cumulative".
        lines_to_print (int, optional):
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
            Defaults to None.
        strip_dirs (bool, optional):
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout.
            Defaults to False.

    Returns:
        pstats.Stats: Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + ".prof"
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, "w") as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner
