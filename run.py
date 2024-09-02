import torch
from meshgraphnet.model.networks import EncodeProcessDecode

model = EncodeProcessDecode(input_dim_node=16, input_dim_edge=4, hidden_dim=32, output_dim=16, num_layers=4, message_passing_steps=1).cuda()

jmodel = torch.jit.script(model)

torch.jit.save(jmodel, "meshgraphnet.pt")
mnet = torch.jit.load("meshgraphnet.pt")
