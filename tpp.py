def getTimeToStabilize(pipeline: str, failedService: str) -> int:
    time = 0
    pipeline = list(pipeline)

    while True:
        remove_indices = set()
        for i in range(1, len(pipeline)):
            if pipeline[i] == failedService:
                remove_indices.add(i - 1)
        if not remove_indices:
            break
        # remove all affected services
        pipeline = [pipeline[i] for i in range(len(pipeline)) if i not in remove_indices]
        time += 1

    return time
print(getTimeToStabilize("database", "a"))  # 2
print(getTimeToStabilize("acebbbb", "b"))   # 3
print(getTimeToStabilize("abcd", "e"))      # 0

