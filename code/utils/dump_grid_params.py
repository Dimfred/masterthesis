from tabulate import tabulate

pretty = [["bs", "loss", "lr"]]

for bs in (32, 64):
    for loss in ("ciou", "eiou1", "eiou0.8"):
        for lr in (0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001):
            pretty += [[str(bs), loss, str(lr)]]

print(tabulate(pretty))
