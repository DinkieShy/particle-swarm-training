import numpy as np
import subprocess
from multiprocessing import Pool
from sys import argv
from json import dump, loads, JSONEncoder
from os.path import exists
from argparse import ArgumentParser


class NpEncoder(JSONEncoder):
    # FROM https://bobbyhadz.com/blog/python-typeerror-object-of-type-int64-is-not-json-serializable
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def runParticle(params, progBar=None):
    print("Running particle with params")
    print(params)
    args = ["python", "runNetwork.py"]
    for (key, value) in params.items():
        args.append(key)
        args.append(str(value))

    output = subprocess.run(args, capture_output=True)
    if progBar is not None:
        progBar.update()
    assert output.stderr == b'', f"Error from subprocess: {output.stderr}"
    result = float(output.stdout)
    return [result, params]


def main():
    parser = ArgumentParser(description='Pytorch grid search test')
    parser.add_argument("--network", "-n", type=str, default="darknet", metavar="network",
                    help="Network to use (default: \"darknet\")")
    parser.add_argument("--training-batch", type=int, default=3, metavar="B",
                        help="input inputs to train at a time (default: 3)")
    args = parser.parse_args()

    MAX_THREADS = 1
    GRID_SIZE = 5

    dimensions = {
        "--lr": (0.0005, 0.005),
        "--lr-drop": (2, 10),
        "--lr2": (0.00001, 0.0005),
        "--momentum": (0.75, 0.95),
        "--decay": (0.00001, 0.0001),
    }

    # MAX_RUNS = GRID_SIZE**len(dimensions)

    grid = {}
    for i in dimensions:
        topBound = float(dimensions[i][1])
        botBound = float(dimensions[i][0])

        grid[i] = [ii for ii in list(np.linspace(botBound, topBound, num=GRID_SIZE,
                                                 dtype=(int if isinstance(dimensions[i][1], (int)) else None)))]

    results = []
    indexSet = [0, 0, 0, 0, 0]
    if "-r" in argv and exists("./result.json"):
        with open('result.json', 'r') as file:
            lines = file.readlines()
            lastResult = loads(lines[-1])
            indexSet = [grid[dim].index(lastResult["position"][dim]) for dim in dimensions]
            if indexSet[-1] == GRID_SIZE - 1:
                indexSet[-2] += 1
                indexSet[-1] = 0
                for ii in range(len(indexSet) - 2, 0, -1):
                    if indexSet[ii] >= GRID_SIZE:
                        indexSet[ii] -= GRID_SIZE
                        indexSet[ii - 1] += 1

    elif not exists("./result.json"):
        with open('result.json', 'w') as file:
            file.write("[\n")
    else:
        assert False, "Results file found, use -r flag to append to file or rename/remove ./results.json"

    print("Begin")

    with Pool(processes=MAX_THREADS) as pool:
        while indexSet[0] < GRID_SIZE:
            indexSet[-1] = 0
            particleSet = []
            for i in range(GRID_SIZE):
                newParticle = {"--network": args.network, "--epochs": 5, "--training-batch": args.training_batch}
                for ii in range(len(dimensions)):
                    dim = list(dimensions.keys())[ii]
                    newParticle[dim] = grid[dim][indexSet[ii]]
                indexSet[-1] += 1
                particleSet.append(newParticle)

            output = np.array([], dtype=np.float32)

            for out in pool.map(runParticle, particleSet):
                output = np.append(output, [out[0]])
                result = {"position": out[1], "result": out[0]}
                print(result)
                results.append(result)

                with open("./result.json", "a") as file:
                    dump(result, file, cls=NpEncoder)
                    file.write("\n")

            indexSet[-2] += 1
            for ii in range(len(indexSet) - 2, 0, -1):
                if indexSet[ii] >= GRID_SIZE:
                    indexSet[ii] -= GRID_SIZE
                    indexSet[ii - 1] += 1


if __name__ == "__main__":
    main()
