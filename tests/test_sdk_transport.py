"""Official SDK serialization with in-memory HTTP transport. No network or real key."""
import importlib
import importlib.util
import json
import unittest
from basedecision import BaseDecision
from basedecision.errors import ProviderError

AVAILABLE = all(importlib.util.find_spec(x) is not None for x in ('openai','anthropic','httpx'))

@unittest.skipUnless(AVAILABLE, 'Install provider extras for official SDK transport tests')
class TransportTests(unittest.TestCase):
    def run_provider(self, provider, status=200):
        from openai import OpenAI
        from anthropic import Anthropic
        captured=[]
        def handler(request,httpx):
            captured.append((str(request.url),json.loads(request.content)))
            if status!=200:
                return httpx.Response(status,json={'error':{'message':'private body test-secret','type':'authentication_error'}},headers={'request-id':'test'})
            if provider=='openai':
                body={'id':'resp_test','object':'response','created_at':0,'status':'completed','model':'test-model',
                      'output':[{'id':'msg_test','type':'message','status':'completed','role':'assistant','content':[{'type':'output_text','text':'{"option_index":1}','annotations':[]}]}],
                      'usage':{'input_tokens':10,'output_tokens':4,'total_tokens':14}}
            else:
                body={'id':'msg_test','type':'message','role':'assistant','model':'test-model',
                      'content':[{'type':'text','text':'{"option_index":1}'}],'stop_reason':'end_turn','stop_sequence':None,
                      'usage':{'input_tokens':10,'output_tokens':4}}
            return httpx.Response(200,json=body)
        client=BaseDecision.from_provider(provider,'test-model',api_key='test-secret',max_retries=0)
        client._client.close()
        cls=OpenAI if provider=='openai' else Anthropic
        # Newer SDK generations require httpx2 clients, older ones httpx; use whichever the SDK accepts.
        for name in ('httpx2','httpx'):
            try:flavour=importlib.import_module(name)
            except ImportError:continue
            try:
                client._client=cls(api_key='test-secret',max_retries=0,http_client=flavour.Client(
                    transport=flavour.MockTransport(lambda request,flavour=flavour:handler(request,flavour))))
                break
            except TypeError:continue
        else:raise AssertionError('no httpx flavour accepted by the SDK')
        try:
            if status==200:
                result=client.choose(context='source',question='question',options=['A','B'])
                self.assertEqual(result.answer,'B');self.assertEqual(result.usage['input_tokens'],10)
                self.assertEqual(len(captured),1)
                body=captured[0][1]
                if provider=='openai':self.assertIn('format',body['text']);self.assertFalse(body['store'])
                else:self.assertIn('format',body['output_config'])
            else:
                with self.assertRaises(ProviderError) as caught:client.choose(context='source',question='question',options=['A','B'])
                self.assertEqual(caught.exception.code,'authentication_or_permission')
                self.assertNotIn('private body',str(caught.exception));self.assertEqual(len(captured),1)
        finally:client.close()
    def test_openai_http(self):self.run_provider('openai')
    def test_anthropic_http(self):self.run_provider('anthropic')
    def test_openai_auth_error(self):self.run_provider('openai',401)
    def test_anthropic_auth_error(self):self.run_provider('anthropic',401)
