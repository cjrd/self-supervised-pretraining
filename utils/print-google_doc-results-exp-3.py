import json
lines = [line.strip() for line in open(".aaa.aaa")]

lidx = 0
for line in lines:
    if "====" in line:
        print(line)
        continue
    if lidx % 4 == 2:
        res = json.loads(line)
        for item in res:
            print(item)
                
    else:
        print(line)
    lidx += 1