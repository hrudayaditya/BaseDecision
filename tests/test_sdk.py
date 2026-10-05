import threading
import unittest
from basedecision import BaseDecision, Request, Option, InputError, ContextLengthError
from basedecision.packing import pack, batch_plan

class Tokenizer:
    cls_token_id=1; sep_token_id=2; mask_token_id=3; pad_token_id=0
    def __call__(self,text,add_special_tokens=False,**kwargs):
        return {'input_ids':[ord(c)+10 for c in text]}

class Tests(unittest.TestCase):
    def request(self,context='abc',kind='choice'):
        options=(Option('false','false'),Option('true','true')) if kind=='noul' else (Option('a','Alpha'),Option('b','Beta'))
        return Request(context,'Which?',options,kind)
    def test_default_single_request(self):
        import inspect
        self.assertEqual(inspect.signature(BaseDecision.from_pretrained).parameters['max_batch_size'].default,1)
        self.assertEqual(inspect.signature(BaseDecision).parameters['max_batch_size'].default,1)

    def test_exact_budget(self):
        r=self.request(); n=pack(Tokenizer(),r).tokens
        self.assertEqual(pack(Tokenizer(),r,n).tokens,n)
        with self.assertRaises(ContextLengthError):pack(Tokenizer(),r,n-1)
        r=self.request('x'*(8192-n+3))
        self.assertEqual(pack(Tokenizer(),r).tokens,8192)
        with self.assertRaises(ContextLengthError):pack(Tokenizer(),self.request(r.context+'x'))
    def test_options_not_truncated(self):
        r=Request('state','Q',(Option('a','A'*500),Option('b','B'*500)))
        p=pack(Tokenizer(),r)
        self.assertEqual(p.markers[1]-p.markers[0],502)
    def test_invalid_schema(self):
        for options in [(Option('a','A'),Option('a','B')),(Option('a','A'),Option('b','A'))]:
            with self.assertRaises(InputError):Request('s','q',options)
        with self.assertRaises(InputError):Request({},'q',self.request().options)
        with self.assertRaises(InputError):Request('s','q',self.request().options,'score',(2,1))
    def test_token_collision(self):
        class Collision(Tokenizer):
            def __call__(self,text,**kwargs):return {'input_ids':[99]}
        with self.assertRaises(InputError):pack(Collision(),self.request())
    def test_batch_budget(self):
        lengths=[8000,100,4000,101,4001]
        plan=list(batch_plan(lengths,4,8192))
        self.assertEqual(sorted(sum(plan,[])),list(range(5)))
        for b in plan:self.assertLessEqual(max(lengths[i] for i in b)*len(b),8192)
        with self.assertRaises(InputError):list(batch_plan([8193],8,8192))
    def test_batch_restores_order_and_validates_first(self):
        engine=BaseDecision.__new__(BaseDecision)
        engine._lock=threading.RLock();engine._tokenizer=Tokenizer();engine.maximum_tokens=8192
        engine.max_batch_size=2;engine.max_batch_tokens=8192
        calls=[]
        def forward(items):calls.append(items);return [None]*len(items)
        engine._forward=forward;engine._result=lambda r,p,z:r.context
        req=[self.request('z'*100),self.request('a'),self.request('b'*20)]
        self.assertEqual(engine.predict_batch(req),[r.context for r in req])
        calls.clear()
        with self.assertRaises(ContextLengthError):engine.predict_batch(req+[self.request('x'*9000)])
        self.assertEqual(calls,[])

if __name__=='__main__':unittest.main()
