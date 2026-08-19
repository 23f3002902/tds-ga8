import json,math,uuid,torch
cfg=json.load(open('../upload/mlflow_config_23f3002902.json'))
X=torch.tensor(cfg['dataset']['X'],dtype=torch.float32); y=torch.tensor(cfg['dataset']['y'],dtype=torch.float32).view(-1,1)
model=torch.nn.Linear(8,1)
with torch.no_grad():
 model.weight.copy_(torch.tensor(cfg['initialization']['initial_weights']['W']).view(1,-1))
 model.bias.copy_(torch.tensor([cfg['initialization']['initial_weights']['b']]))
h=cfg['hyperparameters']; o=h['optimizer']
opt=torch.optim.SGD(model.parameters(),lr=h['lr'],weight_decay=h['weight_decay'],momentum=o['momentum'],dampening=o['dampening'],nesterov=o['nesterov'])
losses=[]; N=len(X)
for i in range(h['num_steps']):
 lr=cfg['lr_schedule']['lr_min']+.5*(h['lr']-cfg['lr_schedule']['lr_min'])*(1+math.cos(i*math.pi/h['num_steps']))
 for g in opt.param_groups:g['lr']=lr
 idx=(i*h['batch_size'])%N; ids=[(idx+j)%N for j in range(h['batch_size'])]
 opt.zero_grad(); loss=torch.nn.functional.mse_loss(model(X[ids]),y[ids]);loss.backward();opt.step();losses.append(float(loss.item()))
ans={'final_loss':losses[-1],'run_id':uuid.uuid4().hex,'mean_last_10_loss':sum(losses[-10:])/10}
print(json.dumps(ans));print(losses[-10:])
