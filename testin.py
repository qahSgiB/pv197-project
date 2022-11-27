from typing import Dict, List, Tuple



def lion(xy: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    xy_dict: Dict[int, List[int]] = {}

    for (x, y) in xy:
        if x not in xy_dict:
            xy_dict[x] = [y]
        else:
            xy_dict[x].append(y)

    return xy_dict



a = []