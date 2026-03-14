import argparse
import math
from textwrap import indent

parser = argparse.ArgumentParser()
parser.add_argument("N", type=int)
args = parser.parse_args()

bits = int(math.ceil(math.log2(args.N+1)))

s = ""

index_fn = f"""
fun index(n: int({bits}), xs: list(bool)): bool {{
    if n == int({bits},0) then
        head xs
    else
        index(n - int({bits}, 1), tail xs)
}}\n
"""
s += index_fn

get_balls_fn = f"fun drawurn(N: int({bits})): list(bool)"
get_balls_fn += " {\n"
get_balls_fn_fn_body = f"""if N == int({bits},0) then
    ([] : list(bool))
"""
for i in range(1,args.N+1):
    get_balls_fn_fn_body += f"else if N == int({bits},{i}) then\n" 
    get_balls_fn_fn_body += "    " + "[" + ", ".join(["flip 0.5"]*i) + "]\n"
get_balls_fn_fn_body += f"""else
    ([] : list(bool))
"""
get_balls_fn += indent(get_balls_fn_fn_body, "    ")
get_balls_fn += "}\n"


# s += get_balls_fn

pickball_fn = f"fun pickball(N: int({bits})): int({bits})"
pickball_fn += " {\n"
pickball_fn_body = f"""if N == int({bits},0) then
    int({bits},0)
else if N == int({bits}, 1) then
    int({bits},0)
"""
ps = [0.0] * 2**bits
for i in range(2,args.N+1):
    for j in range(i):
        ps[j] = 1/i
    pickball_fn_body += f"else if N == int({bits},{i}) then\n" 
    pickball_fn_body += "    discrete(" + ", ".join(map(str,ps)) + ")\n"
pickball_fn_body += f"""else
    int({bits}, 0)
"""
pickball_fn += indent(pickball_fn_body, "    ")
pickball_fn += "}\n"

s += pickball_fn

draw_fn = f"""
fun draw(xs: list(bool), obs: bool, N: int({bits})) {{
    let d = pickball(N) in
    let x = index(d, xs) in
    let y = if x then flip 0.8 else flip 0.2 in
    observe(y <=> obs)
}}\n
"""
s += draw_fn


main = ""

pois = [0.0024787508882582188, 0.014872513711452484, 0.04461752995848656, 0.08923499286174774, 0.1338524967432022, 0.1606229990720749, 0.16062292456626892, 0.13767682015895844, 0.10325776040554047, 0.06883830577135086, 0.041302964091300964, 0.02252892032265663, 0.011264455504715443, 0.00519897136837244, 0.002228131052106619, 0.0008912544581107795, 0.0003342198033351451, 0.0001179598766611889, 3.932005347451195e-05, 1.2416859135555569e-05, 3.72504700862919e-06, 1.064304456122045e-06, 2.9026290349065675e-07, 7.572104721020878e-08, 1.8930171208353386e-08, 4.5432755335639285e-09, 1.0484459966875193e-09, 2.3298787898973217e-10, 4.9925868994549205e-11, 1.0329426550215182e-11, 2.06589714953076e-12, 3.998528125357531e-13][:args.N+1]
if any(p < 0.0001 for p in pois):
    p1 = [p if p >= 0.0001 else 0.0 for p in pois]
    n1 = sum(p1)
    p1 = [p / n1 for p in p1]

    p2 = [p if p < 0.0001 else 0.0 for p in pois]
    n2 = sum(p2)
    p2 = [p / n2 for p in p2]

    main += "let N1 = discrete(" + ", ".join(map(str, p1)) + ") in\n"
    main += "let N2 = discrete(" + ", ".join(map(str, p2)) + ") in\n"
    main += f"let N = if flip {n1} then N1 else N2 in\n"
else:
    pois = [p / sum(pois) for p in pois]
    main += "let N = discrete(" + ", ".join(map(str, pois)) + ") in\n"

ps = [0.] * (2**bits)
for i in range(args.N+1):
    ps[i] = 1/(args.N+1)
# main += "let N = discrete(" + ", ".join(map(str, ps)) + ") in\n"

main += f"let tmp = observe(N > int({bits},0)) in\n"
# main += "let xs = drawurn(N) in"
main += "let xs = [" + ", ".join(["flip 0.5"]*(args.N+1)) + "] in\n"


main += """
let tmp = draw(xs, true, N) in
let tmp = draw(xs, false, N) in
let tmp = draw(xs, true, N) in
let tmp = draw(xs, false, N) in
let tmp = draw(xs, true, N) in
let tmp = draw(xs, false, N) in
let tmp = draw(xs, true, N) in
let tmp = draw(xs, false, N) in
let tmp = draw(xs, true, N) in
let tmp = draw(xs, false, N) in
N
"""

s += main

print(s)


# time dice -determinism -eager-eval -flip-lifting -num-recursive-calls -show-size -recursion-limit 21 -max-list-length 21 urnN.dice
# time ./dice -eager-eval -flip-lifting -num-recursive-calls -show-size -recursion-limit 21 -max-list-length 21 urnN.dice