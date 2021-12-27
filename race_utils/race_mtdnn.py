# general imports
import os
from tqdm import tqdm, trange

# mtdnn imports
from tosmaster_mtdnn import MTDNNModel

# tosmaster imports
from utils.tokenization import BertTokenizer
from utils.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from utils.modeling import BertForMultipleChoice, BertConfig

import torch

class race_mtdnn(MTDNNModel):
    '''
    this class implements the optimization routine of the MTDNN model, but on the race-bert tosmaster model.
    '''
    def __init__(self, config, device, state_dict, num_train_step, seed=30):
        MTDNNModel.__init__(self, config, device, state_dict, num_train_step)
        
        # device is set automatically 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
                torch.cuda.manual_seed_all(seed)
        
        # define tokenizer
        bert_model = "bert-base-uncased"
        do_lower_case = config.get('do_lower_case', True)
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        
        # define model
        model = BertForMultipleChoice.from_pretrained(bert_model,
                                                      cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)),
                                                      num_choices=3)
        for name, param in model.named_parameters():
            ln = 24
            if name.startswith('bert.encoder'):
                l = name.split('.')
                ln = int(l[3])

            if name.startswith('bert.embeddings') or ln < 6:
                print(name)  
                param.requires_grad = False
        
#         self.model = model.to(device)
#         del model
        # continue initialization
        self._model_init(model, state_dict, num_train_step)
        
    def update(self, train_dataloader):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.network(input_ids, segment_ids, input_mask, label_ids)
        