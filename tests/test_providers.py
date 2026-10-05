import importlib.util
import inspect
import json
import sys
import types
import unittest
from unittest.mock import patch
from basedecision import BaseDecision, Request, Option, InputError
from basedecision.errors import ProviderError, ProviderResponseError
from basedecision.providers import parse_selection

class FakeClient:
    def __init__(self, **kwargs):
        self.config=kwargs;self.calls=[];self.closed=False;self.error=None
        self.responses=self.messages=self
        self.reply=types.SimpleNamespace(status='completed',output=[],output_text='{"option_index":1}',
            stop_reason='end_turn',content=[types.SimpleNamespace(type='text',text='{"option_index":1}')],
            usage=types.SimpleNamespace(input_tokens=20,output_tokens=8))
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:raise self.error
        return self.reply
    def close(self):self.closed=True

class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.modules=patch.dict(sys.modules,{'openai':types.SimpleNamespace(OpenAI=FakeClient),'anthropic':types.SimpleNamespace(Anthropic=FakeClient)})
        self.modules.start();self.addCleanup(self.modules.stop)
        self.req=Request('Some source text','Choose',(Option('a','A'),Option('b','B')))
    def engine(self,p='openai',**kwargs):
        return BaseDecision.from_provider(p,'explicit-model',api_key='test-secret',**kwargs)
    def test_both_providers_and_no_fabricated_confidence(self):
        for p in ('openai','anthropic'):
            e=self.engine(p);r=e.predict(self.req)
            self.assertEqual(r.answer,'b');self.assertIsNone(r.probabilities);self.assertIsNone(r.raw_logits)
            self.assertEqual(r.usage,{'input_tokens':20,'output_tokens':8})
            self.assertNotIn('test-secret',repr(r.to_dict()))
            self.assertEqual(e._client.config['max_retries'],2)
    def test_wire_schema(self):
        for p in ('openai','anthropic'):
            e=self.engine(p);e.predict(self.req);call=e._client.calls[0]
            if p=='openai':
                self.assertFalse(call['store']);schema=call['text']['format']['schema']
            else:schema=call['output_config']['format']['schema']
            self.assertEqual(schema['properties']['option_index']['enum'],[0,1])
            self.assertFalse(schema['additionalProperties'])
            self.assertNotIn('tools',call)
    def test_boolean_and_score(self):
        e=self.engine();self.assertTrue(e.check(context='s',question='q').answer)
        r=e.score(context='s',question='q',levels=['low','high'],values=[1,5])
        self.assertEqual(r.selected_value,5);self.assertIsNone(r.expected_value)
    def test_bad_outputs(self):
        for s in ['{}','{"option_index":true}','{"option_index":2}','{"option_index":-1}','{"option_index":0,"extra":1}','{"option_index":0,"option_index":1}','not json']:
            with self.assertRaises(ProviderResponseError):parse_selection(s,2)
    def test_refusal_and_incomplete(self):
        e=self.engine();e._client.reply.status='incomplete'
        with self.assertRaises(ProviderResponseError):e.predict(self.req)
        e=self.engine('anthropic');e._client.reply.stop_reason='refusal'
        with self.assertRaises(ProviderResponseError) as caught:e.predict(self.req)
        self.assertEqual(caught.exception.code,'refusal')
    def test_sanitized_errors(self):
        e=self.engine()
        class Failure(Exception):status_code=401
        e._client.error=Failure('test-secret and private source')
        with self.assertRaises(ProviderError) as caught:e.predict(self.req)
        self.assertEqual(caught.exception.code,'authentication_or_permission')
        self.assertNotIn('test-secret',str(caught.exception));self.assertFalse(caught.exception.retryable)
        e._client.error.status_code=429
        with self.assertRaises(ProviderError) as caught:e.predict(self.req)
        self.assertTrue(caught.exception.retryable)
    def test_prevalidation_before_spending(self):
        e=self.engine()
        with self.assertRaises(InputError):e.predict_batch([self.req,object()])
        self.assertEqual(e._client.calls,[])
    def test_input_bounds_and_configuration(self):
        e=self.engine(max_input_bytes=1)
        with self.assertRaises(InputError):e.predict(self.req)
        for kwargs in [{'timeout':float('nan')},{'max_retries':-1},{'max_retries':True},{'max_output_tokens':0}]:
            with self.assertRaises(InputError):self.engine(**kwargs)
    def test_close(self):
        with self.engine() as e:e.predict(self.req)
        self.assertTrue(e._client.closed)
        with self.assertRaises(InputError):e.predict(self.req)
    def test_hub_download_is_explicit_and_no_remote_code(self):
        calls=[]
        def download(**kwargs):calls.append(kwargs);return '/fake/snapshot'
        with patch.dict(sys.modules,{'huggingface_hub':types.SimpleNamespace(snapshot_download=download)}):
            with patch.object(BaseDecision,'from_pretrained',return_value='loaded'):
                self.assertEqual(BaseDecision.from_hub('owner/model',local_files_only=True),'loaded')
        self.assertTrue(calls[0]['local_files_only']);self.assertIn('*.py',calls[0]['ignore_patterns'])

if __name__=='__main__':unittest.main()
