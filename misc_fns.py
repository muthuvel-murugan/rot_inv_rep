import json

from argparse import Namespace


def read_json(json_fname):
    
    with open(json_fname) as fp:
        jo = json.load(fp, object_hook=lambda d: Namespace(**d))
    
    with open(json_fname) as fp:
        print fp.read()
    
    if hasattr(jo, 'test_data') and type(jo.test_data) != list:
        jo.test_data = [jo.test_data]
    if hasattr(jo, 'op_fname') and type(jo.op_fname) != list:
        jo.op_fname = [jo.op_fname]
    
    return jo

