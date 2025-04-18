import os
from datetime import datetime


def make_output_path():
    '''
    Create output directory for topic modeling results.
    The directory is named with the current date and time.
    '''
    
    datetime_ = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"./output/{datetime_}/"
    os.makedirs(output_path, exist_ok=True)
    
    return output_path