import argparse

import torch
from torch.utils.data import Dataset




def progress_bar(iteration, total, bar_length=30):
    progress = float(iteration) / total
    filled_length = int(bar_length * progress)
    bar = '*' * filled_length + '-' * (bar_length - filled_length)
    return f"[{bar}] {progress:.1%}"



def parse_list_arg(arg):
    elements = arg.split(',')
    parsed_elements = []
    for e in elements:
        try:
            parsed_elements.append(int(e))
        except ValueError:
            parsed_elements.append(e)
    return parsed_elements
