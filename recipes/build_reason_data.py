"""Reasoning seed data (2026-11-24 CST): verifiable multi-step puzzles answered with a short <think> chain then a one-line answer."""
import random, json
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NAMES = ["Ann", "Bob", "Cal", "Dina", "Eli", "Fay", "Gus", "Hana", "Ivan", "Jade", "Kurt", "Lena", "Milo", "Nora", "Otto", "Pia", "Quin", "Rita", "Saul", "Tess"]
WORDS = ["strawberry", "banana", "mississippi", "letter", "balloon", "committee", "bookkeeper", "parallel", "coffee", "assessment", "pepper", "grammar", "success", "rhythm", "tomorrow", "occurrence", "necessary", "address", "puppy", "giraffe"]
ADJ = [("taller", "tallest", "shortest"), ("older", "oldest", "youngest"), ("faster", "fastest", "slowest"), ("heavier", "heaviest", "lightest")]
NONCE = ["bloops", "razzies", "glims", "tovs", "wugs", "daxes", "feps", "kibs", "lorps", "zents"]
DIRS = ["north", "east", "south", "west"]
def ex(r):
    f = r.randrange(13)
    if f == 0:
        k = r.randint(3, 5); ppl = r.sample(NAMES, k); a = r.choice(ADJ)
        facts = [f"{ppl[i]} is {a[0]} than {ppl[i+1]}." for i in range(k - 1)]; r.shuffle(facts)
        ask = r.choice([1, 2]); ans = ppl[0] if ask == 1 else ppl[-1]
        q = " ".join(facts) + f" Who is the {a[ask]}?"
        th = "Order from most to least: " + " > ".join(ppl) + f". So the {a[ask]} is {ans}."
        return q, th, f"{ans} is the {a[ask]}."
    if f == 1:
        d = r.randrange(7); n = r.randint(2, 60); fw = r.random() < 0.7
        res = DAYS[(d + n) % 7] if fw else DAYS[(d - n) % 7]
        q = f"Today is {DAYS[d]}. What day {'will it be in' if fw else 'was it'} {n} days{'' if fw else ' ago'}?"
        th = f"{n} days is {n // 7} full weeks and {n % 7} extra days. {'Counting forward' if fw else 'Counting back'} {n % 7} days from {DAYS[d]} gives {res}."
        return q, th, f"It {'will be' if fw else 'was'} {res}."
    if f == 2:
        w = r.choice(WORDS); c = r.choice(sorted(set(w))); n = w.count(c)
        q = f"How many times does the letter \"{c}\" appear in the word \"{w}\"?"
        th = "Spell it out: " + "-".join(w) + f". The letter {c} appears at positions " + ", ".join(str(i + 1) for i, x in enumerate(w) if x == c) + f", so {n} times."
        return q, th, f"The letter \"{c}\" appears {n} time{'s' if n != 1 else ''}."
    if f == 3:
        a, b, c = r.sample(NONCE, 3); nm = r.choice(NAMES); yes = r.random() < 0.6
        if yes:
            q = f"All {a} are {b}. All {b} are {c}. {nm} is one of the {a}. Is {nm} one of the {c}?"
            return q, f"{nm} is one of the {a}, every one of the {a} is one of the {b}, and every one of the {b} is one of the {c}. So yes.", f"Yes, {nm} is one of the {c}."
        q = f"All {a} are {b}. {nm} is one of the {b}. Must {nm} be one of the {a}?"
        return q, f"All {a} are {b}, but not every one of the {b} has to be one of the {a}. Being one of the {b} does not prove {nm} is one of the {a}.", f"No, not necessarily."
    if f == 4:
        s, d, k = r.randint(1, 30), r.randint(2, 12), r.choice(["add", "mul"])
        if k == "add":
            seq = [s + d * i for i in range(5)]; nx = s + d * 5; th = f"Each term goes up by {d}. {seq[-1]} + {d} = {nx}."
        else:
            m = r.choice([2, 3]); seq = [s * m ** i for i in range(4)]; nx = s * m ** 4; th = f"Each term is {m} times the one before. {seq[-1]} * {m} = {nx}."
        return f"What number comes next: {', '.join(map(str, seq))}, ...?", th, f"The next number is {nx}."
    if f == 5:
        a = round(r.uniform(0, 10), r.choice([1, 2])); b = round(r.uniform(0, 10), r.choice([1, 2]))
        if a == b: b = round(b + 0.1, 2)
        big = max(a, b)
        return f"Which is larger, {a} or {b}?", f"Compare {a} and {b}: {big} is larger because {big} - {min(a, b)} = {round(big - min(a, b), 2)} > 0.", f"{big} is larger."
    if f == 6:
        x = r.randint(2, 200); add = r.randint(1, 9)
        par = "even" if x % 2 == 0 else "odd"; res = "even" if (x + add) % 2 == 0 else "odd"
        return f"{x} is {par}. Is {x} + {add} even or odd?", f"{par.capitalize()} plus {'even' if add % 2 == 0 else 'odd'} is {res}. {x} + {add} = {x + add}, which is {res}.", f"{x + add} is {res}."
    if f == 7:
        a, b, c = r.randint(2, 9), r.randint(1, 5), r.randint(1, 6); nm = r.choice(NAMES); b = min(b, a)
        return (f"{nm} has {a} apples, eats {b}, and then picks {c} more. How many apples does {nm} have?", f"Start with {a}. After eating {b}: {a} - {b} = {a - b}. After picking {c}: {a - b} + {c} = {a - b + c}.", f"{nm} has {a - b + c} apples.")
    if f == 8:
        u = r.choice([("hours", "minutes", 60), ("minutes", "seconds", 60), ("days", "hours", 24), ("weeks", "days", 7), ("meters", "centimeters", 100), ("kilograms", "grams", 1000)]); n = r.randint(2, 12)
        return f"How many {u[1]} are in {n} {u[0]}?", f"One {u[0][:-1]} is {u[2]} {u[1]}, so {n} {u[0]} is {n} * {u[2]} = {n * u[2]}.", f"There are {n * u[2]} {u[1]} in {n} {u[0]}."
    if f == 9:
        a, b, c = r.sample(NAMES, 3); rel = r.choice([("mother", "grandmother"), ("father", "grandfather")])
        return (f"{a} is {b}'s {rel[0]}. {b} is {c}'s {rel[0] if r.random() < 0.5 else 'parent'}. What is {a} to {c}?", f"{a} is a parent of {b}, and {b} is a parent of {c}, so {a} is a parent of {c}'s parent.", f"{a} is {c}'s {rel[1]}.")
    if f == 10:
        nm = r.choice(NAMES); fut, now_plus, past = r.randint(2, 10), 0, r.randint(1, 9); age = r.randint(12, 60)
        return (f"In {fut} years, {nm} will be {age + fut}. How old was {nm} {past} years ago?", f"Now {nm} is {age + fut} - {fut} = {age}. {past} years ago: {age} - {past} = {age - past}.", f"{nm} was {age - past} years old.")
    if f == 11:
        s = r.randrange(4); turns = [r.choice(["left", "right"]) for _ in range(r.randint(1, 4))]; d = s; steps = []
        for t in turns:
            d = (d + (1 if t == "right" else -1)) % 4; steps.append(f"turn {t} -> {DIRS[d]}")
        return (f"You face {DIRS[s]}. You turn {', then '.join(turns)}. Which way are you facing now?", f"Start facing {DIRS[s]}; " + "; ".join(steps) + ".", f"You are facing {DIRS[d]}.")
    k = r.randint(3, 6); items = r.sample(["red", "blue", "green", "yellow", "black", "white", "pink"], k); hid = r.choice(items)
    rest = [x for x in items if x != hid]
    return (f"A bag holds exactly one marble of each color: {', '.join(items)}. I take out every marble except the {hid} one. Which colors did I take out?", f"All colors are {', '.join(items)}. Removing everything except {hid} leaves out {', '.join(rest)}.", f"You took out {', '.join(rest)}.")
if __name__ == "__main__":
    r = random.Random(77)
    with open(r"D:\seed_v4\seed_reason1\data\synth_reason.txt", "w", encoding="utf-8", newline="\n") as f:
        for i in range(160000):
            q, th, a = ex(r); f.write(f"<|user|>{q}<|end|>\n<|assistant|><think>{th}</think>{a}<|end|><|endoftext|>")
    print("ok")
