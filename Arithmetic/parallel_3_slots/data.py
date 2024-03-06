import torch
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ArithmeticData(Dataset):
  def __init__(self, num_examples):
    self.num_examples = num_examples
    self.example_length = 2

    self.point_1 = []
    self.point_2 = []
    self.point_3 = []
    self.target_1 = []
    self.target_2 = []
    self.target_3 = []
    self.operation = []

    for i in range(self.num_examples):
      
      points = []
      for i in range(3):
          points.append(np.array([random.uniform(1, 2), random.uniform(1, 2)]))
          
      op = np.zeros((3,4))  
      
      target = []
      for var in range(3):
          range_index = random.randint(0, 3)
          op[var][range_index] = 1
          if var == 0: secondary = random.choices([1, 2], [0.5, 0.5])[0]
          elif var == 1: secondary = random.choices([0, 2], [0.5, 0.5])[0]
          else: secondary = random.choices([0, 1], [0.5, 0.5])[0]
          if range_index == 0: # x-addition
            target.append(points[var] + points[secondary]*np.array([1,0]))
        
          ###
          elif range_index == 1:  # y-addition 
            target.append(points[var] + points[secondary]*np.array([0,1]))
        
          ###
          elif range_index == 2:  # x-sub
            target.append(points[var] - points[secondary]*np.array([1,0]))
          ###
          else: # y-sub
            target.append(points[var] - points[secondary]*np.array([0,1]))
        
      self.target_1.append(target[0])
      self.target_2.append(target[1])
      self.target_3.append(target[2])
      self.point_1.append(points[0])
      self.point_2.append(points[1])
      self.point_3.append(points[2])
      self.operation.append(op)

  def __len__(self):
      return self.num_examples

  def __getitem__(self, i):
      p1 = self.point_1[i]
      p2 = self.point_2[i]
      p3 = self.point_3[i]
      tar1 = self.target_1[i]
      tar2 = self.target_2[i] 
      tar3 = self.target_3[i]
      op = self.operation[i]
      slot_1 = np.concatenate((p1, tar1), axis = 0)
      slot_2 = np.concatenate((p2, tar2), axis = 0)
      slot_3 = np.concatenate((p3, tar3), axis = 0)
      inp = np.vstack((slot_1, slot_2, slot_3, op))
      return inp #  (3, 4)
      
      
class ArithmeticDataSeq(Dataset):
  def __init__(self, num_examples):
    self.num_examples = num_examples
    self.example_length = 2

    self.point_1 = []
    self.point_2 = []
    self.target_1 = []
    self.target_2 = []
    self.operation = []
    self.prim = []
    self.intermid_1 = []
    self.intermid_2 = []

    for i in range(self.num_examples):
      self.point_1.append(np.array([random.uniform(2, 3), random.uniform(2, 3)]))
      self.point_2.append(np.array([random.uniform(2, 3), random.uniform(2, 3)]))
      dig_index = random.randint(0, 1)
      self.prim.append(dig_index)
      t1 = self.point_1[i]
      t2 = self.point_2[i]
      #while ( np.all(t1 == self.point_1[i]) and np.all(t2 == self.point_2[i]) ):
      op = np.zeros((2,4)) 
      t1 = self.point_1[i]
      t2 = self.point_2[i]
      for j in range(2):
          range_index = random.randint(0, 3)
          op[j][range_index] = 1
          self.operation.append(op)
          if range_index == 0: # x-addition
            #self.prim.append(dig_index)
            if dig_index == 0:
              t1 = t1 + t2*np.array([1,0])
            else:
              t2 = t2 + t1*np.array([1,0])
          ###
          elif range_index == 1:  # y-addition 
            #self.prim.append(dig_index)
            if dig_index == 0:
                t1 = t1 + t2*np.array([0,1])
            else:
                t2 = t2 + t1*np.array([0,1])       
          ###
          elif range_index == 2:  # x-sub
            if dig_index == 0:  # if( (self.point_1[-1] - self.point_2[-1]*np.array([1,0]))[0] >=0 ):
                t1 = t1 - t2*np.array([1,0])
              #self.prim.append(0)
            else:
                t2 = t2 - t1*np.array([1,0])
              #self.prim.append(1)
          ###
          else: # y-sub
            if dig_index == 0: # if( (self.point_1[-1] - self.point_2[-1]*np.array([1,0]))[1] >=0 ):
                t1 = t1 - t2*np.array([0,1])
              #self.prim.append(0)
            else:
                t2 = t2 - t1*np.array([0,1])
              #self.prim.append(1)
          if (j == 0):
              t_int_1 = t1
              t_int_2 = t2
              
      self.intermid_1.append(np.concatenate((t_int_1, t1), axis=0))
      self.intermid_2.append(np.concatenate((t_int_2, t2), axis=0))
      self.operation.append(op)
      self.target_1.append(t1)
      self.target_2.append(t2)

  def __len__(self):
      return self.num_examples

  def __getitem__(self, i):
      p1 = self.point_1[i]
      p2 = self.point_2[i]
      tar1 = self.target_1[i]
      tar2 = self.target_2[i] 
      op = self.operation[i]
      slot_1 = np.concatenate((p1, tar1), axis = 0)
      slot_2 = np.concatenate((p2, tar2), axis = 0)
      int1 = self.intermid_1[i]
      int2 = self.intermid_2[i]
      inp = np.vstack((slot_1, slot_2, int1, int2, op, np.array([self.prim[i], 0, 0, 0])))
      return inp #  (3, 4)
      
      
      
if __name__ == '__main__':
	data = ArithmeticData(500)
	for i in range(len(data)):
		print(data[i])
		print('----------------') 

