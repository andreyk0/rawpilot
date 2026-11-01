"""Compare rawpilot's ratings of a folder with a reference selection made from the same shoot.

Usage: python3 -I compare.py FOLDER REFERENCE_JSON [EDITED_LIST]

FOLDER          a folder `rawpilot cull` ran on (its .pp3 and .rpl files are read). It may
                hold symlinks to NEFs elsewhere, e.g. on a card that must not be written to.
REFERENCE_JSON  `exiftool -j -n -FileName -SubSecDateTimeOriginal` over the reference
                selection's NEFs. Frames are matched by file name AND capture time, since
                cameras reuse file names.
EDITED_LIST     optional: one file name per line of reference frames that were also edited
                (anything after ".NEF" in a name is ignored).

Reports at three levels. Frames within a burst are near-identical and bursts of one take are
attempts at the same shot, so the take level is the fair comparison: was anything of the take
in the reference, and how did rawpilot rate the take's best frame. Read-only.
"""
import collections
import json
import pathlib
import re
import sys

folder = pathlib.Path(sys.argv[1])
reference_json = pathlib.Path(sys.argv[2])
edited_list = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None


def capture_key(s):
    """exiftool "2026:01:20 11:27:38.15+00:00" and RFC 3339 "2026-01-20T11:27:38.150+00:00"
    to one comparable form, to 1/100 s."""
    m = re.match(r"(\d{4})[:-](\d\d)[:-](\d\d)[ T](\d\d):(\d\d):(\d\d)(?:\.(\d+))?", s)
    assert m, f"unrecognised capture time {s!r}"
    y, mo, d, h, mi, se, frac = m.groups()
    return f"{y}{mo}{d}{h}{mi}{se}.{(frac or '0')[:2].ljust(2, '0')}"


reference = {(r["FileName"], capture_key(r["SubSecDateTimeOriginal"])) for r in json.load(open(reference_json))}
edited = set()
if edited_list:
    for line in open(edited_list):
        name = line.strip()
        if name:
            edited.add(name[: name.upper().index(".NEF") + 4] if ".NEF" in name.upper() else name)

frames = []
for rpl in sorted(folder.glob("*.NEF.rpl")):
    name = rpl.name.removesuffix(".rpl")
    a = json.loads(rpl.read_text())["analysis"]
    rank = re.search(r"^Rank=(\d+)", (folder / f"{name}.pp3").read_text(), re.M)
    assert rank, f"{name}.pp3 has no Rank"
    in_ref = (name, capture_key(a["taken_at"])) in reference
    frames.append({
        "name": name,
        "stars": int(rank.group(1)),
        "why": a["decision"]["why"],
        "burst": a["burst"]["first"],
        "winner": a["burst"]["winner"],
        "take": a.get("take", a["burst"]["first"]),
        "ref": in_ref,
        "edited": in_ref and name in edited,
    })

matched = sum(f["ref"] for f in frames)
print(f"frames rated: {len(frames)}; reference frames: {len(reference)} (found {matched}); edited: {len(edited)}")


def table(title, groups, stars_of):
    """groups: id → frames. A group is 'in reference' if any frame is."""
    ref = {g for g, fs in groups.items() if any(f["ref"] for f in fs)}
    ed = {g for g, fs in groups.items() if any(f["edited"] for f in fs)}
    print(f"\n{title}: {len(groups)}, of which {len(ref)} in the reference ({len(ref) / max(len(groups), 1):.0%})")
    for t in (5, 4, 3, 2):
        sel = [g for g in groups if stars_of(g) >= t]
        k = sum(g in ref for g in sel)
        e = sum(g in ed for g in sel)
        print(f"  rated ≥{t}★: {len(sel):5d}, {k / max(len(sel), 1):5.1%} in reference; "
              f"covers {k / max(len(ref), 1):5.1%} of reference, {e / max(len(ed), 1):5.1%} of edited")
    return ref, ed


bursts = collections.defaultdict(list)
takes = collections.defaultdict(list)
for f in frames:
    bursts[f["burst"]].append(f)
    takes[f["take"]].append(f)
winner_stars = {b: next(f["stars"] for f in fs if f["winner"]) for b, fs in bursts.items()}
take_stars = {t: max(f["stars"] for f in fs) for t, fs in takes.items()}

frame_stars = {f["name"]: f["stars"] for f in frames}
table("Frames", {f["name"]: [f] for f in frames}, frame_stars.get)
table("Bursts (rated by their winner)", bursts, winner_stars.get)
ref_takes, edited_takes = table("Takes (rated by their best frame)", takes, take_stars.get)


def show(title, rows):
    print(f"\n{title} ({len(rows)})")
    for f in rows:
        print(f"  {f['name']} {f['stars']}★ {'edited ' if f['edited'] else ''}{f['why'][:110]}")


show("Takes rated 5★ with nothing in the reference",
     [max(takes[t], key=lambda f: f["stars"]) for t in takes if take_stars[t] == 5 and t not in ref_takes])
show("Edited takes rated ≤2★",
     [f for t in edited_takes if take_stars[t] <= 2 for f in takes[t] if f["edited"]])
