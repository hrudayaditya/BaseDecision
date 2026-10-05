"""python examples/decide.py --backend local --model /path/to/model"""
import argparse
import json
from basedecision import BaseDecision
from basedecision.errors import ProviderError

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['local','openai','anthropic'],default='local')
p.add_argument('--model',required=True,help='Local export path or provider model ID')
a=p.parse_args()
try:
    client=(BaseDecision.from_pretrained(a.model) if a.backend=='local' else
            BaseDecision.from_provider(a.backend,a.model))
    try:
        result=client.choose(context='Please refund this purchase.',question='What is requested?',
                             options=['Refund request','Delivery status','Other request'])
        print(json.dumps(result.to_dict(),indent=2))
    finally:
        if a.backend!='local':client.close()
except ProviderError as error:
    print(json.dumps({'error':error.code,'retryable':error.retryable,'status_code':error.status_code}))
    raise SystemExit(1)
