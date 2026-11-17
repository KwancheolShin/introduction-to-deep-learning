# Reading: hardware -- turn figures/photos/CREDITS.md into the LaTeX credit table
#   python photo_credits.py   ->   ../figures/photos/photo_credits.tex  (\input by main.tex)
# Only the photos that main.tex actually uses are listed.

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(HERE, "..", "figures", "photos")
MAIN = os.path.join(HERE, "..", "main.tex")

# the Artist field of this file is a wiki signature; its Credit field names the photographer
AUTHOR_FIX = {"cpu_ryzen_am4_pins.jpg": "Phillip Voß"}


def tex(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}")]:
        s = s.replace(a, b)
    return s


used = set(re.findall(r"\{([a-z0-9_]+\.(?:jpg|png))\}", open(MAIN, encoding="utf-8").read()))
rows = []
for line in open(os.path.join(PHOTOS, "CREDITS.md"), encoding="utf-8"):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) != 7 or not cells[0].endswith((".jpg", ".png")) or cells[0] not in used:
        continue
    name, _, what, author, lic, url, _ = cells
    author = AUTHOR_FIX.get(name, author)
    url = url.strip("<>").replace("%", chr(92) + "%").replace("#", chr(92) + "#")
    rows.append(f"\\texttt{{{tex(name).replace('_', '_' + chr(92) + 'allowbreak{}')}}} &{tex(what)} & {tex(author)} & {tex(lic)} & \\href{{{url}}}{{Commons}} \\\\")

out = [r"{\scriptsize",
       r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{36mm}>{\raggedright\arraybackslash}p{56mm}"
       r">{\raggedright\arraybackslash}p{26mm}>{\raggedright\arraybackslash}p{15mm}l@{}}",
       r"\toprule",
       r"파일 & 내용 & 저작자 & 라이선스 & 원본 \\",
       r"\midrule",
       r"\endhead",
       r"\bottomrule",
       r"\endfoot"] + rows + [r"\end{longtable}}"]
with open(os.path.join(PHOTOS, "photo_credits.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"{len(rows)} photos of {len(used)} used")
