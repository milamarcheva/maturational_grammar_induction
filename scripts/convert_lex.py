import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    # Example: "1.0 0.3  N --> 'd"
    parts = line.split()
    # last token is the word: "'d"
    word = parts[-1]
    # the tag is the third token: "N"
    tag = parts[2]

    print(f"{word}\t{tag}\t{1}")