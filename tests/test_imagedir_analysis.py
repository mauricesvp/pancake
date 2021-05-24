import datetime 
from pancake.utils.datasets import LoadImageDirs, img_formats

def test_order(source):
    DATA = LoadImageDirs(source)
    files = DATA.files

    # RETRIEVE STAMP FROM FILE NAME
    timestamps = [
        list(
            map(
                lambda path: path.split("/")[-1] if (
                    not path.split(".")[-1] in img_formats
                ) else (
                    ".".join(path.split("/")[-1].split(".")[:2])
                ),
                dir
            )
        ) for dir in files
    ]

    # CONVERT TO HUMAN READABLE FORMAT
    converted_timestamps = [
        list(
            map(
                convert_timestamp,
                dir
            )
        ) for dir in timestamps
    ]

    # GET ORDERED SET
    unique_timestamps = [
        list(dict.fromkeys(dir)) 
        for dir in converted_timestamps
    ]

    s = [""] * max([len(entries) 
        for entries in unique_timestamps]
    )

    # COUNT FRAMES PER STAMP
    for i, stamps in enumerate(unique_timestamps):
        for j, stamp in enumerate(stamps):
            s[j] += (f"[{i+1}/{len(files)}]'{stamp}': " 
                     f"{converted_timestamps[i].count(stamp)} \t"
            )

    for entry in s:
        print(entry)


def convert_timestamp(stamp: str):
    return datetime.datetime.fromtimestamp(
        float(stamp)).strftime('%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    source = ["samples/r45/1l", "samples/r45/1c", "samples/r45/1r"]
    test_order(source=source)