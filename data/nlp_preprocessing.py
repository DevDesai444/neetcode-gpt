import torch
import torch.nn as nn
# from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        vocabulary = set()
        for sen in positive :
            for w in sen.split() :
                vocabulary.add(w)
        
        for sen in negative :
            for w in sen.split() :
                vocabulary.add(w)
        
        sorted_vocab = sorted(vocabulary)

        word_to_int = {}

        for i in range(len(sorted_vocab)):
            word_to_int[sorted_vocab[i]] = i+1
        
        tensons = []
        for sen in positive :
            curr_list = []
            for w in sen.split() :
                curr_list.append(word_to_int[w])
            tensons.append(torch.tensor(curr_list))
        
        
        for sen in negative :
            curr_list = []
            for w in sen.split() :
                curr_list.append(word_to_int[w])
            tensons.append(torch.tensor(curr_list))

        return nn.utils.rnn.pad_sequence(tensons, batch_first = True)
