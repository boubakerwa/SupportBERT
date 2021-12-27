from os import sep
from .supportdata_utils import SD
import glob

import tarfile

import json

import s3fs

def read_data(bucket:str,
             split:str):
    '''
    this function is inspired from the tomaster repo.
    '''
    assert split in ['train', 'dev', 'test'], \
    'split has to be either train, dev or test.'
    
    samples = []    
#     dir_name = input_dir + sep + split

    fs = s3fs.S3FileSystem()
    data_s3fs_location = "s3://{}/{}/".format(bucket, split)
    files_list = fs.ls(data_s3fs_location)
    files_list = sorted(files_list, key=lambda x: int((x.split(sep)[-1]).split('.')[0]))
    print("After sorted:",files_list[0])

    for file_name in files_list:        
        with fs.open(file_name) as f:
            sample = json.load(f)
        answers = sample['answers']
        text = sample["article"]
        questions = sample['questions']
        options = sample['options']
        rid = file_name[:-4] 
        for i in range(len(answers)):
            samples.append(SD(
                race_id = rid+":"+str(i),
	    		context_sentence = text,
	    		start_ending = questions[i], 
	    		ending_0 = options[i][0],
	    		ending_1 = options[i][1],
	    		ending_2 = options[i][2],
	    		label = ord(answers[i])-65
	    	))

    return samples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training, debug=False):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task like Swag. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Race example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    print(len(examples),examples[0])
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)
        if debug:
            print(f'length question: {len(start_ending_tokens)}')

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(start_ending_tokens)) +[2] * (len(tokenizer.tokenize(ending)) + 1)
            # correction of original implementation
            segment_ids = segment_ids[:len(tokens)]
            if debug:
                count_d = {0: len(context_tokens_choice), 1: len(start_ending_tokens), 2: len(tokens)-len(context_tokens_choice)-len(start_ending_tokens)-3, 'original length of option': len(tokenizer.tokenize(ending))}
                print(f'count segment_ids: {count_d}')
                print(tokenizer.tokenize(ending))
                print(f'length segment_ids={len(segment_ids)}')

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            
            if len(segment_ids)!=len(input_ids):
                print(f'length segment_ids = {len(segment_ids)}, input_ids={len(input_ids)}, input_mask={len(input_mask)}')
                print(f'length segment before truncation={len(context_tokens)+len(start_ending_tokens)+len(tokenizer.tokenize(ending))}')
                raise Exception(f'len(segment_ids)!=len(input_ids)')
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length, f'length segment_ids = {len(segment_ids)}, \
            length max_seq_length = {max_seq_length}'

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index < 5:
            # logger.info("*** Example ***")
            # logger.info("race_id: {}".format(example.race_id))
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                # logger.info("choice: {}".format(choice_idx))
                # logger.info("tokens: {}".format(' '.join(tokens)))
                # logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                # logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                # logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                pass
            if is_training:
                # logger.info("label: {}".format(label))
                pass
        if (example_index%5000 ==0): print(example_index)	
        features.append(
            InputFeatures(
                example_id = example.race_id,
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    #print(outputs,outputs == labels)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label
        

def build_tensor(features):
    all_input_ids = torch.tensor(select_field(features, 'input_ids'),
                                 dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'),
                                  dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'),
                                   dtype=torch.long)
    all_label = torch.tensor([f.label for f in features],
                             dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)