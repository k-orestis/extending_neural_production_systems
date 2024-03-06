import torch
import argparse
import numpy as np
from data import ArithmeticData, ArithmeticDataSeq
from torch.utils.data import DataLoader
from model import ArithmeticModel
import torch.nn as nn
from tqdm import tqdm
#from utilities.rule_stats import get_stats
import random
import os

def none_or_str(value):
    if value == 'None':
        return None
    return value

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--noise', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--tied', default=False, action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--cudnn', action='store_true',
                    help='use cudnn optimized version. i.e. use RNN instead of RNNCell with for loop')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='sparse_factor_graphs',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--resume', type=str, default=None,
                    help='if specified with the 1-indexed global epoch, loads the checkpoint and resumes training')
parser.add_argument('--algo', type=str, choices=('rim', 'lstm', 'SCOFF'))
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--nhid', type=int, default=300)
parser.add_argument('--topk', type=int, default=4)
parser.add_argument('--block_dilation', nargs='+', type=int, default=-1)
parser.add_argument('--layer_dilation', nargs='+', type=int, default=-1)
parser.add_argument('--train_len', type=int, default=500)
parser.add_argument('--test_len', type=int, default=1000)
parser.add_argument('--read_input', type=int, default=2)
parser.add_argument('--memory_slot', type=int, default=4)
parser.add_argument('--memory_heads', type=int, default=4)
parser.add_argument('--memory_head_size', type=int, default=16)
parser.add_argument('--gate_style', type=none_or_str, default=None)
parser.add_argument('--use_inactive', action='store_true',
                    help='Use inactive blocks for higher level representations too')
parser.add_argument('--blocked_grad', action='store_true',
                    help='Block Gradients through inactive blocks')
parser.add_argument('--scheduler', action='store_true',
                    help='Scheduler for Learning Rate')
parser.add_argument('--adaptivesoftmax', action='store_true',
                    help='use adaptive softmax during hidden state to output logits.'
                         'it uses less memory by approximating softmax of large vocabulary.')
parser.add_argument('--cutoffs', nargs="*", type=int, default=[10000, 50000, 100000],
                    help='cutoff values for adaptive softmax. list of integers.'
                         'optimal values are based on word frequencey and vocabulary size of the dataset.')
parser.add_argument('--use_attention', action ='store_true')
#parser.add_argument('--name', type=str, default=None,
#                    help='name for this experiment. generates folder with the name if specified.')

## Rule Network Params
parser.add_argument('--use_rules', action ='store_true')
parser.add_argument('--rule_time_steps', type = int, default = 1)
parser.add_argument('--num_rules', type = int, default = 4) 
parser.add_argument('--rule_emb_dim', type = int, default = 64)
parser.add_argument('--rule_query_dim', type = int, default = 32)
parser.add_argument('--rule_value_dim', type = int, default = 64)
parser.add_argument('--rule_key_dim', type = int, default = 32)
parser.add_argument('--rule_heads', type = int, default = 4)

parser.add_argument('--comm', type = str2bool, default = True)
parser.add_argument('--grad', type = str, default = "yes")
parser.add_argument('--transformer', type = str, default = "yes")
parser.add_argument('--application_option', type = str, default = '3')
parser.add_argument('--training_interval', type = int, default = 2)
parser.add_argument('--alternate_training', type = str, default = "yes")

parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--split', type = str, default = 'mcd1')
parser.add_argument('--perm_inv', type=str2bool, default=False)
parser.add_argument('--n_templates', type=int, default=2)
parser.add_argument('--gumble_anneal_rate', type = float, default = 0.00003)
parser.add_argument('--use_entropy', type = str2bool, default = True)
parser.add_argument('--use_biases', type = str2bool, default = True)
parser.add_argument('--generalize', type = str2bool, default = False)
args = parser.parse_args()


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

set_seed(args.seed)

print('TRAINING NORMAL')
train_dataset = ArithmeticData(10000)
val_dataset = ArithmeticData(2000)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size,
                shuffle = True, num_workers = 4)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = ArithmeticModel(args, 10).to(device)

if args.use_biases and args.num_rules > 0:
    model.rule_network.use_biases = True
elif args.num_rules > 0:
    model.rule_network.use_biases = False

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
objective = nn.MSELoss() # nn.L1Loss()

def get_stats(rule_selections, variable_selections, variable_selections_1, variable_rules, application_option, num_rules = 5, num_blocks = 1):
    if isinstance(application_option, str):
        application_option = int(application_option.split('.')[0])
    for b in range(rule_selections[0].shape[0]):
        for w in range(len(rule_selections)):
            #if application_option == 0 or application_option == 3:
            #    try:
            #        tup = (rule_selections[w][b][0], variable_selections[w][b][0])
            #    except:
            tup = (rule_selections[w][b], variable_selections[w][b], variable_selections_1[w][b])
            #elif application_option == 1:
            #    y = rule_selections[w][b]

            #    r1 = y[0] % num_rules
            #    v1 = y[0] % num_blocks
            #    r2  = y[1] % num_rules
            #    v2 = y[1] % num_blocks
            #    tup = (r1, v1, r2, v2)
            if tup not in variable_rules:
                variable_rules[tup] = 1
            else:
                variable_rules[tup] += 1
    return variable_rules

def operation_to_rule_train(operations, rule_selections, vault, inverse_vault):
    operations = operations.cpu().numpy()
    #operations = torch.argmax(operations, dim = 2).cpu().numpy()
    #t = 0
    for t in range(1):
        for b in range(operations.shape[0]):
            if rule_selections[0][t][b] not in inverse_vault:
                inverse_vault[rule_selections[0][t][b]] = {'x-addition':0, 'y-addition':0, 'x-subtraction':0, 'y-subtraction':0}
            if operations[b][t][0] == 1:
                if rule_selections[0][t][b] not in vault['x-addition']:
                    vault['x-addition'].append(rule_selections[0][t][b])
                inverse_vault[rule_selections[0][t][b]]['x-addition'] += 1
            elif operations[b][t][1] == 1:
                if rule_selections[0][t][b] not in vault['y-addition']:
                    vault['y-addition'].append(rule_selections[0][t][b])
                inverse_vault[rule_selections[0][t][b]]['y-addition'] += 1
            elif operations[b][t][2] == 1:
                if rule_selections[0][t][b] not in vault['x-subtraction']:
                    vault['x-subtraction'].append(rule_selections[0][t][b])
                inverse_vault[rule_selections[0][t][b]]['x-subtraction'] += 1
            elif operations[b][t][3] == 1:
                if rule_selections[0][t][b] not in vault['y-subtraction']:
                    vault['y-subtraction'].append(rule_selections[0][t][b])
                inverse_vault[rule_selections[0][t][b]]['y-subtraction'] += 1


    return vault, inverse_vault


def operation_to_rule(operations, rule_selections, vault, inverse_vault):
    operations = operations.cpu().numpy()
    #operations = torch.argmax(operations, dim = 2).cpu().numpy()
    #print(operations)
    #t = 0
    for t in range(1):
        for b in range(operations.shape[0]):
            if rule_selections[0][t][b] not in inverse_vault:
                inverse_vault[rule_selections[0][t][b]] = {'x-addition':0, 'y-addition':0, 'x-subtraction':0, 'y-subtraction':0}
            if operations[b][t][0] == 1:
                if rule_selections[0][t][b] not in vault['x-addition']:
                    vault['x-addition'].append(rule_selections[0][t][b])
                inverse_vault[rule_selections[0][t][b]]['x-addition'] += 1
            elif operations[b][t][1] == 1:
                if rule_selections[0][t][b] not in vault['y-addition']:
                    vault['y-addition'].append(rule_selections[0][t][b])
                inverse_vault[rule_selections[0][t][b]]['y-addition'] += 1
            elif operations[b][t][2] == 1:
                if rule_selections[0][t][b] not in vault['x-subtraction']:
                    vault['x-subtraction'].append(rule_selections[0][t][b])
                inverse_vault[rule_selections[0][t][b]]['x-subtraction'] += 1
            elif operations[b][t][3] == 1:
                if rule_selections[0][t][b] not in vault['y-subtraction']:
                    vault['y-subtraction'].append(rule_selections[0][t][b])
                inverse_vault[rule_selections[0][t][b]]['y-subtraction'] += 1
                
    return vault, inverse_vault
    
best_eval_mse = 10

def eval_epoch(epoch):
    global best_eval_mse
    model.eval()
    correct = 0
    total = 0
    num_examples = 0
    variable_rule = {}
    inverse_vault = {}
    correct_sentences = 1
    total_sentences = 0
    loss = 0

    with torch.no_grad():
        vault = {'x-addition':[], 'y-addition':[], 'x-subtraction':[], 'y-subtraction':[]}
        for i, data in enumerate(val_dataloader):
            if args.num_rules > 0:
                model.rule_network.reset_activations()
            
            
            data = data.float().to(device)
            
            ### edit
            slot_1, slot_2, slot_3, operation = data[:, 0:1, :], data[:, 1:2,:], data[:,2:3, :],  data[:,3:, :] 
                
            target_1 = slot_1[: , :, 2:]
            target_2 = slot_2[: , :, 2:]
            target_3 = slot_2[: , :, 2:]
            targets = torch.cat((target_1, target_2, target_3), dim=2).squeeze(1)
    
            slot_1 = slot_1.squeeze(1) # (batch_size, 4)
            slot_2 = slot_2.squeeze(1) # (batch_size, 4)
            slot_3 = slot_3.squeeze(1)
            
            out = model(slot_1, slot_2, slot_3, train=False)
            
            loss = objective(out, targets)
            op = operation
            ###############
            if (i == 17):
                idx = random.randrange(50)
                ex1 = (slot_1[idx], slot_2[idx], slot_3[idx], out[idx])
            if (i == 32):
                idx = random.randrange(50)
                ex2 = (slot_1[idx], slot_2[idx], slot_3[idx], out[idx])
            if (i == 25):
                idx = random.randrange(50)
                ex3 = (slot_1[idx], slot_2[idx], slot_3[idx], out[idx])
            if (i == 3):
                idx = random.randrange(50)
                ex4 = (slot_1[idx], slot_2[idx], slot_3[idx], out[idx])
            if (i == 12):
                idx = random.randrange(50)
                ex5 = (slot_1[idx], slot_2[idx], slot_3[idx], out[idx])
            if (i == 38):
                idx = random.randrange(50)
                ex6 = (slot_1[idx], slot_2[idx], slot_3[idx], out[idx])
            ###############
            
            num_examples += 1
            if args.num_rules > 0:
                rule_selections = model.rule_network.rule_activation
                variable_selections = model.rule_network.variable_activation
                variable_selections_1 = model.rule_network.variable_activation_1

                #variable_rule = get_stats(rule_selections, variable_selections, variable_selections_1, variable_rule, args.application_option, args.num_rules, args.num_blocks)
                vault, inverse_vault = operation_to_rule(op, rule_selections, vault, inverse_vault)
            
            total_sentences += 1





            
    #print(model.encoder.rimcell[0].bc_lst[0].iatt_log)
    print('eval_mse:'+str(loss/total_sentences))
    
    eval_mse = loss / total_sentences
    
    if eval_mse < best_eval_mse:
        best_eval_mse = eval_mse
        torch.save(model.state_dict(), args.save_dir + '/model_best.pt')
    
    print('eval stats')
    # for v in variable_rule:
    #     print(v, end = ' : ')
    #     print(variable_rule[v])
    for v in vault:
        print(v, end = ' : ')
        print(vault[v])
    for v in inverse_vault:
        print(v, end = ' : ')
        print(inverse_vault[v])
    ############
    if epoch % 5 == 0:
        print('')
        print('two random samples')
        print('EX1:')
        print('input points and expected targets')
        print(ex1[0].tolist(), ex1[1].tolist(), ex1[2].tolist())
        print('predicted targets')
        print(ex1[3].tolist())
        print('EX2:')
        print('input points and expected targets')
        print(ex2[0].tolist(), ex2[1].tolist(), ex2[2].tolist())
        print('predicted targets')
        print(ex2[3].tolist())
        print('EX3:')
        print('input points and expected targets')
        print(ex3[0].tolist(), ex3[1].tolist(), ex3[2].tolist())
        print('predicted targets')
        print(ex3[3].tolist())
        print('EX4:')
        print('input points and expected targets')
        print(ex4[0].tolist(), ex4[1].tolist(), ex4[2].tolist())
        print('predicted targets')
        print(ex4[3].tolist())
        print('EX5:')
        print('input points and expected targets')
        print(ex5[0].tolist(), ex5[1].tolist(), ex5[2].tolist())
        print('predicted targets')
        print(ex5[3].tolist())
        print('EX6:')
        print('input points and expected targets')
        print(ex6[0].tolist(), ex6[1].tolist(), ex6[2].tolist())
        print('predicted targets')
        print(ex6[3].tolist())
    ############
    print('')


def train_epoch(epoch):
    global best_eval_mse
    loss_ = 0
    model.train()
    correct = 0
    total = 0
    num_examples = 0
    vault = {'x-addition':[], 'y-addition':[], 'x-subtraction':[], 'y-subtraction':[]}
    inverse_vault = {}
    variable_rule = {}
    for i, data in tqdm(enumerate(train_dataloader)):
        if args.num_rules > 0:
            model.rule_network.reset_activations()
            
        #### edit
        data = data.float().to(device)
        slot_1, slot_2, slot_3, operation = data[:, 0:1, :], data[:, 1:2,:], data[:,2:3, :],  data[:,3:, :] 
        
        target_1 = slot_1[: , :, 2:]
        target_2 = slot_2[: , :, 2:]
        target_3 = slot_2[: , :, 2:]
        targets = torch.cat((target_1, target_2), dim=2).squeeze(1)

        slot_1 = slot_1.squeeze(1) # (batch_size, 4)
        slot_2 = slot_2.squeeze(1) # (batch_size, 4)
        slot_3 = slot_3.squeeze(1)
        
        out = model(slot_1, slot_2, slot_3)
 
        loss = objective(out, targets)
        op = operation # (bs, seq, 4)
        
            
        model.zero_grad()
        (loss).backward()
        loss_ += loss
        optimizer.step()
        num_examples += 1
        
        if args.num_rules > 0:
            rule_selections = model.rule_network.rule_activation
            variable_selections = model.rule_network.variable_activation
            variable_selections_1 = model.rule_network.variable_activation_1
            #variable_rule = get_stats(rule_selections, variable_selections, variable_selections_1, variable_rule, args.application_option, args.num_rules, args.num_blocks)
            vault, inverse_vault = operation_to_rule_train(op, rule_selections, vault, inverse_vault)
            

        

        
        if i % args.log_interval == 1 and i != 1:
            
            print('epoch:' + str(epoch), end = ' ')
            print('loss: '+str(loss_/num_examples), end = ' ')
            
            loss_ = 0
            correct = 0
            total = 0
            num_examples = 0
            
            print('train stats')
            # for v in variable_rule:
            #     print(v, end = ' : ')
            #     print(variable_rule[v])
            for v in vault:
                print(v, end = ' : ')
                print(vault[v])
            for v in inverse_vault:
                print(v, end = ' : ')
                print(inverse_vault[v])
            inverse_vault = {}
            vault = {'x-addition':[], 'y-addition':[], 'x-subtraction':[], 'y-subtraction':[]}
            
            print('')
            eval_epoch(epoch)
            print('best_eval_mse:' + str(best_eval_mse))
            model.train()
    torch.save(model.state_dict(), args.save_dir + '/model_latest.pt')
for epoch in range(1, args.epochs):
    train_epoch(epoch)

