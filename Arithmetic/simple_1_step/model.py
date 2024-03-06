import torch.nn as nn
import torch
import random
import time
#from modularity import RIM, SCOFF, RIMv2, SCOFFv2
from RuleNetwork import RuleNetwork

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, intermediate_dim = 32):
        super().__init__()
        self.mlp = nn.Sequential(
                                nn.Linear(in_dim, intermediate_dim),
                                nn.ReLU(),
                                #nn.Linear(intermediate_dim, intermediate_dim),
                                #nn.ReLU(),
                                #nn.Linear(intermediate_dim, intermediate_dim),
                                #nn.ReLU(),
                                nn.Linear(intermediate_dim, out_dim)
                                )

    def forward(self, x):
        return self.mlp(x)



class ArithmeticModel(nn.Module):
    def __init__(self, args, n_tokens = 10):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.algo = args.algo
        self.hidden_dim = args.nhid
        self.num_blocks = args.num_blocks
        self.num_layers = args.nlayers
        
        self.encoder = MLP(2, self.hidden_dim, intermediate_dim = 64)
        self.encoder_operation = MLP(3, self.hidden_dim, intermediate_dim = 64)


        
        self.application_option = args.application_option
        self.num_rules = args.num_rules
        self.design_config = {'comm': True, 'grad': False,
                    'transformer': True, 'application_option': '3.0.1.0', 'selection': 'gumble'}
        
        if self.num_rules > 0:
            self.rule_network = RuleNetwork(self.hidden_dim, num_variables=2, num_transforms = 3, num_rules = args.num_rules, rule_dim = args.rule_emb_dim, query_dim = 16, value_dim = 32, key_dim = 16, num_heads = 4, dropout = 0.35, design_config = self.design_config)


    def forward(self, slot_1, slot_2):
        #x_prev  = self.encoder(torch.cat([x_prev, torch.zeros([x_prev.shape[0], 1]).cuda()], dim=1))
        #x_cur  = self.encoder(torch.cat([x_cur, torch.ones([x_cur.shape[0], 1]).cuda()], dim=1))
        #operation_rep = self.encoder_operation(operation)
        #import ipdb
        #ipdb.set_trace() 

        if self.num_rules > 0:
            rule_out,  _ = self.rule_network(torch.cat([slot_1.unsqueeze(1), slot_2.unsqueeze(1)],dim=1), message_to_rule_network = None) # (batch_size, 2, 4)
            out = rule_out # (batch_size, 4) 
        else:
            #operation = self.operation_encoder(operation)
            intermediate_rep = torch.cat((x_prev, x_cur, operation), dim = 1)
            out = self.decoder(intermediate_rep)



        return out
