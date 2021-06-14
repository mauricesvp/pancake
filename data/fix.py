"""Run this, than run.sh"""
import glob
import os


def main():
    l = sorted(glob.glob("../samples/r45/1l/*jpg"), key=os.path.basename)
    # c = sorted(glob.glob("r45/1c/*jpg"), key=os.path.basename)
    c = sorted(glob.glob("../samples/r45/1c/*jpg"), key=os.path.basename)
    r = sorted(glob.glob("../samples/r45/1r/*jpg"), key=os.path.basename)

    sides = [l, c, r]

    def val(fullpath: str):
        return float(os.path.basename(fullpath).replace(".jpg", ""))

    for i, imgs in enumerate(sides):
        with open(f"{i}.txt", "w+") as f:
            fname = imgs.pop(0)
            prev = val(fname)
            f.write(f"file '{fname}'\n")
            for img in imgs:
                curr = val(img)
                diff = curr - prev
                prev = curr
                f.write(f"duration {diff}\n")
                f.write(f"file '{img}'\n")


if __name__ == "__main__":
    main()
