
import torch
import torch.nn as nn
import math
import numpy as np
from utilities.GroupLinearLayer import GroupLinearLayer
from utilities.attention_rim import MultiHeadAttention
import itertools
from utilities.attention import SelectAttention

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class Identity(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input):
		return input * 1.0
	def backward(ctx, grad_output):
		#print(torch.sqrt(torch.sum(torch.pow(grad_output,2))))
		print(grad_output)
		return grad_output * 1.0

class ArgMax(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		idx = torch.argmax(input, 1)
		ctx._input_shape = input.shape
		ctx._input_dtype = input.dtype
		ctx._input_device = input.device
		#ctx.save_for_backward(idx)
		op = torch.zeros(input.size()).to(input.device)
		op.scatter_(1, idx[:, None], 1)
		ctx.save_for_backward(op)
		return op

	@staticmethod
	def backward(ctx, grad_output):
		op, = ctx.saved_tensors
		grad_input = grad_output * op
		return grad_input

class GroupMLP(nn.Module):
	def __init__(self, in_dim, out_dim, num):
		super().__init__()
		self.group_mlp1 = GroupLinearLayer(in_dim, 128, num)
		self.group_mlp2 = GroupLinearLayer(128, out_dim, num)
		#self.group_mlp3 = GroupLinearLayer(128, 128, num)
		#self.group_mlp4 = GroupLinearLayer(128, out_dim, num)
		self.dropout = nn.Dropout(p = 0.5)


	def forward(self, x):
		x = torch.relu(self.group_mlp1(x))
		x = self.group_mlp2(x)
		#x = torch.relu(self.dropout(self.group_mlp3(x)))
		#x = torch.relu(self.dropout(self.group_mlp4(x)))
		return x

class MLP(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.mlp1 = nn.Linear(in_dim, 128)
		self.mlp2 = nn.Linear(128, out_dim)
		self.mlp3 = nn.Linear(128, 128)
		self.mlp4 = nn.Linear(128, out_dim)
		#self.dropout = nn.Dropout(p = 0.5)

	def forward(self, x):
		x = torch.relu(self.mlp1(x))
		x = self.mlp2(x)
		#x = torch.relu(self.mlp3(x))
		#x = self.mlp4(x)
		#x = torch.relu(self.mlp3(x))
		#x = self.mlp4(x)
		return x

class Hook():
    def __init__(self, inp):
        self.hook = inp.register_hook(self.hook_fn)
        self.mask = None
    def hook_fn(self, grad):
        grad = grad * self.mask
        return grad
    def close(self):
        self.hook.remove()


class MyRuleNetwork(nn.Module):
	def __init__(self, hidden_dim, num_variables, num_transforms = 3,  num_rules = 4, rule_dim = 64, query_dim = 32, value_dim = 64, key_dim = 32, num_heads = 4, dropout = 0.1, design_config = None):
		super().__init__()
		self.rule_dim = rule_dim
		self.num_heads = num_heads
		self.key_dim = key_dim
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.value_dim = value_dim
		self.query_dim = query_dim
		self.hidden_dim = hidden_dim
		self.design_config = design_config

		self.rule_activation = []
		self.variable_activation = []
		self.softmax = []
		self.masks = []
		import math
		rule_dim = rule_dim
		
		print('RULE DIM:' + str(rule_dim))
		w =   torch.randn(1, num_rules, rule_dim).to(self.device)

		self.share_key_value = False
		self.shared_query = GroupLinearLayer(num_transforms, hidden_dim, 1)
		self.shared_key = GroupMLP(rule_dim, hidden_dim, num_rules)


		self.dummy_transform_rule = nn.Linear(rule_dim, hidden_dim)
		self.rule_embeddings = nn.Parameter(w)
		self.biases = np.zeros((num_rules, num_variables))
		self.use_biases = True
		self.transform_src = nn.Linear(300, 60)

		self.dummy_rule_selector = SelectAttention(num_transforms, rule_dim, d_k = 32, num_read = 1, num_write = num_rules, share_query = True, share_key = True)

		self.dropout = nn.Dropout(p = 0.5)

		

		self.transform_rule = nn.Linear(rule_dim, hidden_dim)
		if hidden_dim % 4 != 0:
			num_heads = 2
		try:
			self.positional_encoding = PositionalEncoding(hidden_dim)
			self.transformer_layer = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = num_heads, dropout = 0.5)

			self.transformer = nn.TransformerEncoder(self.transformer_layer, 3)
			self.multihead_attention = nn.MultiheadAttention(hidden_dim, 4)

		except:
			pass

		

		self.variable_rule_select = SelectAttention(rule_dim, hidden_dim , d_k=32, num_read = num_rules, num_write = num_variables, share_query = True)

		self.encoder_transform = nn.Linear(num_variables * hidden_dim, hidden_dim)
		self.rule_mlp = GroupMLP(hidden_dim, hidden_dim // 2, num_rules)
		self.rule_linear = GroupLinearLayer(rule_dim + hidden_dim, hidden_dim, num_rules)
		self.rule_relevant_variable_mlp = GroupMLP(2 * hidden_dim, hidden_dim, num_rules)
		self.interaction_mlp = GroupMLP(2*hidden_dim, hidden_dim, num_rules)
		self.variables_select = MultiHeadAttention(n_head=4, d_model_read= hidden_dim, d_model_write = hidden_dim , d_model_out = hidden_dim,  d_k=32, d_v=32, num_blocks_read = 1, num_blocks_write = num_variables, topk = 3, grad_sparse = False)

		self.variables_select_1 = SelectAttention(hidden_dim, hidden_dim, d_k = 16, num_read = 1, num_write = num_variables)

		self.phase_1_mha = MultiHeadAttention(n_head = 1, d_model_read = 2 * hidden_dim * num_variables, d_model_write = hidden_dim, d_model_out = hidden_dim, d_k = 64, d_v = 64, num_blocks_read = 1, num_blocks_write = num_rules, topk = num_rules, grad_sparse = False)

		self.variable_mlp = MLP(2 * hidden_dim, hidden_dim)
		num = [i for i in range(num_variables)]
		num_comb = len(list(itertools.combinations(num, r = 2)))
		self.phase_2_mha = MultiHeadAttention(n_head = 1, d_model_read = hidden_dim, d_model_write = hidden_dim, d_model_out = hidden_dim, d_k = 32, d_v = 32, num_blocks_read = num_comb, num_blocks_write = 1, topk = 1, grad_sparse = False )
		self.variable_mlp_2 = GroupMLP(3 * hidden_dim, hidden_dim, num_variables)



		self.mnist_entity_selector = SelectAttention(rule_dim, hidden_dim, d_k = 16, num_read = 1, num_write = 3)


		#--------Compositonal Search Based Rule Application---------------------------------------
		r = 2
		self.rule_probabilities = []
		self.variable_probabilities = []
		self.r = r
		self.variable_combinations = torch.combinations(torch.tensor([i for i in range(num_variables)]), r = r, with_replacement = True)
		self.variable_combinations_mlp = MLP(r * hidden_dim, hidden_dim)
		self.variable_rule_mlp = MLP(3 * hidden_dim, hidden_dim)
		self.selecter = SelectAttention(hidden_dim, hidden_dim, d_k = 16, num_read = num_rules, num_write = len(self.variable_combinations))
		self.use_rules = MLP(num_variables * hidden_dim, 2)
		self.transform_combinations = MLP(len(self.variable_combinations) * hidden_dim, hidden_dim)
		self.selecter_1 = SelectAttention(hidden_dim, hidden_dim, d_k = 16, num_read = 1, num_write = num_rules)
		self.selecter_2 = SelectAttention(hidden_dim, hidden_dim, d_k = 16, num_read = 1, num_write = len(self.variable_combinations))
		self.variable_rule_group_mlp = GroupMLP(3 * hidden_dim, hidden_dim, num_rules)
		if self.design_config['selection'] == 'gumble':
			print('using gumble for rule selection')
		else:
			print('using ArgMax for rule selction')

		print('Using application option ' + str(self.design_config['application_option']))

		self.gumble_temperature = 1.0



		### MULTIMNIST stuff
		self.rule_select_ = SelectAttention(3 * hidden_dim, rule_dim, d_k = 32, num_read = 1, num_write = num_rules, share_query = True, share_key = True)
		self.variables_select_ = SelectAttention(rule_dim, hidden_dim, d_k = 32, num_read = 1, num_write = num_variables, share_key = False)
		self.project_rule_ = nn.Linear(rule_dim, hidden_dim)

	def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
		new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden, message_to_rule_network = None, rule_mask = None):
		#if not self.design_config['grad']:
		#if str(self.design_config['application_option']).split('.')[1] == '0':
		#	hidden = hidden.detach()
		
		batch_size, num_variables, variable_dim = hidden.size()
		num_variables-=1
		op = hidden[:,2,:].unsqueeze(1).repeat(1, 2, 1)
		hidden_ = hidden[:,:2,:]
		hidden = torch.cat((hidden_, op), dim=-1)
		#print('hidden shape', hidden.shape)

		num_rules = self.rule_embeddings.size(1)
		rule_emb_orig = self.rule_embeddings.repeat(batch_size, 1, 1)
		#print("rule shape: ", rule_emb_orig.shape)
		rule_emb = rule_emb_orig
		scores = self.variable_rule_select(rule_emb, hidden)
		#print("scores shape: ", scores.shape)

		if self.training:
			#biases = torch.tensor(self.biases + 1, device = scores.device)
			#biases_mean = torch.sum(biases, dim = 1).unsqueeze(-1)
			#biases = biases / biases_mean
			#biases = biases.unsqueeze(0).repeat(scores.size(0), 1, 1)
			#if False:
			#	scores = torch.clamp(scores, -10., 10.)
			#	scores = scores / biases
			#scores = Identity().apply(scores)
			mask = torch.nn.functional.gumbel_softmax(scores.reshape(batch_size, -1), dim = 1, tau = 1.0, hard = True)
			self.rule_probabilities.append(mask.clone().reshape(batch_size, num_rules, num_variables).detach())
			probs = mask
			mask = mask.reshape(batch_size, num_rules, num_variables)
			stat_mask = torch.sum(mask, dim = 0)
			mask = mask.permute(0, 2, 1)
			#print("mask shape: ", mask.shape)
			
			#print('lol')
			scores = scores.permute(0, 2, 1).float()
			#if self.use_biases:
			#	self.biases += stat_mask.detach().cpu().numpy()

			entropy = 1e-4 * torch.sum(probs * torch.log(probs), dim = 1).mean()
		else:
			mask = ArgMax().apply(scores.reshape(batch_size, -1)).reshape(batch_size, num_rules, num_variables)
			mask = mask.permute(0, 2, 1)
			scores = scores.permute(0, 2, 1).float()
			self.rule_probabilities.append(torch.softmax(scores.reshape(batch_size, -1), dim = 1).reshape(batch_size, num_variables, num_rules).clone().detach())
			entropy = 0
			mask_print = mask
		
		

		variable_mask = torch.sum(mask, dim = 2).unsqueeze(-1)
		rule_mask = torch.sum(mask, dim = 1).unsqueeze(-1)
		
# 			print("Mask:", mask.shape)
# 			print("Variable mask:", variable_mask.shape)
# 			print("Rule mask:", rule_mask.shape)
		
		#if self.training:
		#	hook_hidden.mask = variable_mask
		# using gumbel for training but printing argmax
		rule_mask_print = torch.sum(mask, dim = 1).detach()
		variable_mask_print = torch.sum(mask, dim = 2).detach()
		#print("var mask: ", variable_mask.shape)
		self.rule_activation.append(torch.argmax(rule_mask_print, dim = 1).detach().cpu().numpy())
		
		#print(rule_emb_orig.size())
		#print(rule_mask.size())

		selected_rule = (rule_emb_orig * rule_mask).sum(dim = 1)

# 			print("Selected rule:", selected_rule.size())
# 			print("Hidden size:", hidden.size())

		#variable_score = self.mnist_entity_selector(selected_rule.unsqueeze(1), hidden,)

		#variable_score = variable_score.squeeze(1)
		#variable_score = torch.nn.functional.gumbel_softmax(variable_score, dim = 1, hard = True, tau = 0.5)
		
		self.variable_activation.append(torch.argmax(variable_mask_print.detach(), dim = 1).detach().cpu().numpy())
# 			rule_mlp_input = (hidden * variable_score.unsqueeze(-1)).sum(dim = 1) #torch.cat((rule_emb_orig, selected_variable), dim = 2)
		
		#primary = (hidden * variable_score.unsqueeze(-1)).sum(dim = 1)
		#context = (hidden[:,:2] * (torch.ones((batch_size, 2, 1), dtype=int, device= self.device) -  variable_score[:,:2].unsqueeze(-1))  ).sum(dim=1)
		primary = (hidden * variable_mask).sum(dim = 1)
		context = (hidden * (torch.ones((batch_size, 2, 1), dtype=int, device= self.device) -  variable_mask)  ).sum(dim=1)
		#print("primary, cont: ", primary.shape, context.shape)
		rule_mlp_input = torch.cat((primary[:,:100], context[:,:100]), dim=1)
		#print("rule input: ", rule_mlp_input.shape)
		rule_mlp_input = rule_mlp_input.unsqueeze(1).repeat(1, rule_mask.size(1), 1)
		rule_mlp_output = self.rule_mlp(rule_mlp_input)
		rule_mlp_output = torch.sum(rule_mlp_output * rule_mask, dim = 1).unsqueeze(1)

		return rule_mlp_output, rule_mask
		

	
	def reset_activations(self):
		self.rule_activation = []
		self.variable_activation = []
		self.rule_probabilities = []
		self.variable_probabilities = []

	def reset_bias(self):
		self.biases = np.zeros((num_rules, num_variables))

if __name__ == '__main__':
	model = RuleNetwork(6, 4).cuda()


	hiddens = torch.autograd.Variable(torch.randn(3, 4, 6), requires_grad = True).cuda()
	new_hiddens = model(hiddens)


	hiddens.retain_grad()
	new_hiddens.backward(torch.ones(hiddens.size()).cuda())

	#print(model.rule_embeddings.grad)
	#print(model.query_layer.w.grad)



