"""Instance-local CPU attention. ModernBERT adapter derived from Transformers 4.57.6 (Apache-2.0)."""
from types import MethodType
from functools import lru_cache

def install(model):
    import torch
    import torch.nn.functional as F
    from transformers.models.modernbert import modeling_modernbert as mb
    original=mb.MODERNBERT_ATTENTION_FUNCTION['sdpa']
    chunk=256
    @lru_cache(maxsize=8)
    def geometry(length,left,right):
        blocks=(length+chunk-1)//chunk
        starts=torch.arange(blocks)[:,None]*chunk
        keys=starts-left+torch.arange(chunk+left+right)[None,:]
        queries=starts+torch.arange(chunk)[None,:]
        allowed=((keys[:,None,:]>=queries[:,:,None]-left) &
                 (keys[:,None,:]<=queries[:,:,None]+right) &
                 (keys[:,None,:]>=0) & (keys[:,None,:]<length))
        return blocks,keys.clamp(0,length-1).flatten(),allowed[:,None,:,:]

    def tiled(q,k,v,left,right):
        b,h,length,d=q.shape
        blocks,indices,mask=geometry(length,left,right)
        if b!=1:raise ValueError('Fast path currently requires batch=1')
        padded=F.pad(q,(0,0,0,blocks*chunk-length))
        qt=padded.reshape(b,h,blocks,chunk,d).permute(0,2,1,3,4).reshape(b*blocks,h,chunk,d)
        width=chunk+left+right
        def gather(x):
            return x.index_select(2,indices).reshape(b,h,blocks,width,d).permute(0,2,1,3,4).reshape(b*blocks,h,width,d)
        out=F.scaled_dot_product_attention(qt,gather(k),gather(v),attn_mask=mask,dropout_p=0.)
        return out.reshape(b,blocks,h,chunk,d).permute(0,2,1,3,4).reshape(b,h,blocks*chunk,d)[:,:,:length,:]

    counters={'tiled_calls':0,'reference_calls':0}
    def forward(module,qkv,attention_mask,sliding_window_mask,position_ids,local_attention,bs,dim,**kwargs):
        # Original mask construction is retained. Only unpadded, single-input inference
        # with the standard local band uses the optimized compute path.
        use=(not module.training and not torch.is_grad_enabled() and qkv.device.type=='cpu'
             and bs==1 and local_attention!=(-1,-1) and qkv.shape[1]>chunk
             and attention_mask is not None and attention_mask.ndim==4
             and attention_mask.dtype.is_floating_point
             and not bool(attention_mask[0,0,0,:].ne(0).any())
             and sliding_window_mask is not None and not kwargs.get('output_attentions',False))
        if not use:
            counters['reference_calls']+=1
            return original(module,qkv,attention_mask,sliding_window_mask,position_ids,local_attention,bs,dim,**kwargs)
        counters['tiled_calls']+=1
        cos,sin=module.rotary_emb(qkv,position_ids=position_ids)
        query,key,value=qkv.transpose(3,1).unbind(dim=2)
        query,key=mb.apply_rotary_pos_emb(query,key,cos,sin)
        output=tiled(query,key,value,*local_attention)
        return (output.transpose(1,2).contiguous().view(bs,-1,dim),)

    def attention(module, hidden_states, output_attentions=False, **kwargs):
        qkv=module.Wqkv(hidden_states)
        bs=hidden_states.shape[0]
        qkv=qkv.view(bs,-1,3,module.num_heads,module.head_dim)
        outputs=forward(module,qkv=qkv,rotary_emb=module.rotary_emb,
            local_attention=module.local_attention,bs=bs,dim=module.all_head_size,
            output_attentions=output_attentions,**kwargs)
        return (module.out_drop(module.Wo(outputs[0])),)+outputs[1:]
    for module in model.encoder.modules():
        if isinstance(module,mb.ModernBertAttention):module.forward=MethodType(attention,module)
    return counters

def install_head(model):
    # Functional MHA bypasses the fused inference shortcut per instance. No global
    # torch.backends.mha flag changes; need_weights=False uses PyTorch SDPA.
    import torch.nn.functional as F
    def layer_forward(layer,src,src_mask=None,src_key_padding_mask=None,is_causal=False):
        if layer.training:raise RuntimeError('CPU fast mode is inference-only')
        def sa(x):
            m=layer.self_attn;q=x.transpose(0,1)
            out=F.multi_head_attention_forward(q,q,q,m.embed_dim,m.num_heads,
                m.in_proj_weight,m.in_proj_bias,m.bias_k,m.bias_v,m.add_zero_attn,
                m.dropout,m.out_proj.weight,m.out_proj.bias,training=False,
                key_padding_mask=src_key_padding_mask,need_weights=False,
                attn_mask=src_mask,is_causal=is_causal)[0].transpose(0,1)
            return layer.dropout1(out)
        def ff(x):return layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))
        if layer.norm_first:
            x=src+sa(layer.norm1(src));return x+ff(layer.norm2(x))
        x=layer.norm1(src+sa(src));return layer.norm2(x+ff(x))
    if model.head is not None:
        for layer in model.head.layers:layer.forward=MethodType(layer_forward,layer)
