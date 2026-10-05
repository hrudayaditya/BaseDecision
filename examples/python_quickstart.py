"""python examples/python_quickstart.py --model /path/to/model"""
import argparse
import json
from basedecision import load, decide

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
    model=load(a.model)
    answers=decide(model,context='I was charged twice for this purchase. Please refund the duplicate payment.',questions={
        'request':{'kind':'choice','question':'What does the customer request?',
                   'options':['Refund request','Delivery status','Change shipping address']},
        'duplicate_charge':{'kind':'noul','question':'Does the customer report being charged twice?'},
    })
    print(json.dumps({k:v.to_dict() for k,v in answers.items()},indent=2))

if __name__=='__main__':main()
