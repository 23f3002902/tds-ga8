import os,re,tempfile,torch
from transformers import LlamaConfig,LlamaForCausalLM
from peft import LoraConfig,get_peft_model

LAYERS={
0:(16,'q v gate up'),1:(32,'q v'),2:(8,'q v gate up'),3:(32,'q v gate up'),4:(8,'q k v o'),
7:(4,'q k v o gate up down'),8:(8,'q v gate up'),9:(4,'q k v o'),10:(32,'q k v o'),11:(8,'q k v o gate up down'),
12:(16,'q v gate up'),13:(32,'q k v o'),15:(32,'q k v o'),16:(16,'q v gate up'),17:(16,'q v gate up'),
18:(16,'q k v o'),19:(16,'q v gate up'),20:(32,'q k v o'),21:(16,'q k v o gate up down'),24:(4,'q k v o'),
26:(4,'q v'),27:(16,'q v gate up'),28:(4,'q k v o gate up down')}
paths=[]; ranks={}; alphas={}
where={'q':'self_attn.q_proj','k':'self_attn.k_proj','v':'self_attn.v_proj','o':'self_attn.o_proj',
       'gate':'mlp.gate_proj','up':'mlp.up_proj','down':'mlp.down_proj'}
for i,(r,mods) in LAYERS.items():
    for m in mods.split():
        p=f'model.layers.{i}.{where[m]}'
        paths.append(p);ranks[p]=r;alphas[p]=2*r
pattern='(?:'+ '|'.join(re.escape(p) for p in paths)+')$'
cfg=LlamaConfig(hidden_size=4096,intermediate_size=16384,num_hidden_layers=29,num_attention_heads=64,
                num_key_value_heads=64,vocab_size=32000)
with torch.device('meta'):
    base=LlamaForCausalLM(cfg)
    model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,target_modules=pattern,rank_pattern=ranks,alpha_pattern=alphas,
                                         lora_dropout=0,bias='none',task_type='CAUSAL_LM'))
# Materialize only adapters; base tensors remain meta and are excluded from adapter save.
for name,module in model.named_modules():
    if hasattr(module,'lora_A'):
        for key,lin in module.lora_A.items(): lin.weight=torch.nn.Parameter(torch.zeros(lin.weight.shape,dtype=torch.float32))
        for key,lin in module.lora_B.items(): lin.weight=torch.nn.Parameter(torch.zeros(lin.weight.shape,dtype=torch.float32))
print(model.print_trainable_parameters())
out='q8_adapter';os.makedirs(out,exist_ok=True)
model.save_pretrained(out,safe_serialization=True)
print({'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),
       'adapter_file_size_bytes':os.path.getsize(os.path.join(out,'adapter_model.safetensors'))})
