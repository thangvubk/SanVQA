import h5py
import torch
import numpy as np
import misc.utils as utils

class CDATA(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
    def __init__(self, opt, phase, transform=None, quiet=False):
        if not quiet:
            print('DataLoader loading h5 question file: ' + opt['h5_ques_file'])
        h5_file = h5py.File(opt['h5_ques_file'], 'r')
        if phase is 'train':
            if not quiet:
                print('DataLoader loading h5 image train file: ' + opt['h5_img_file_train'])
            self.image = np.array(h5py.File(opt['h5_img_file_train'], 'r')['/images_train'])
            self.ques = np.array(h5_file['/ques_train'])
            self.ques_len = np.array(h5_file['/ques_len_train'])
            self.img_pos = np.array(h5_file['/img_pos_train'])
            self.ques_id = np.array(h5_file['/ques_id_train'])
            self.ans = np.array(h5_file['/answers'])
            self.split = np.array(h5_file['/split_train'])
        else: # valid or test
            if not quiet:
                print('DataLoader loading h5 image test file: ' + opt['h5_img_file_test'])
            self.image = np.array(h5py.File(opt['h5_img_file_test'], 'r')['/images_test'])
            self.ques = np.array(h5_file['/ques_test'])
            self.ques_len = np.array(h5_file['/ques_len_test'])
            self.img_pos = np.array(h5_file['/img_pos_test'])
            self.ques_id = np.array(h5_file['/ques_id_test'])
            self.ans = np.array(h5_file['/ans_test'])
            self.split = np.array(h5_file['/split_test'])

        self.feature_type = opt['feature_type']
        self.phase = phase
        self.transform = transform

        if not quiet:
            print('DataLoader loading json file: %s'% opt['json_file'])
        json_file = utils.read_json(opt['json_file'])
        self.ix_to_word = json_file['ix_to_word']
        self.ix_to_ans = json_file['ix_to_ans']

        self.vocab_size = utils.count_key(self.ix_to_word)
        self.seq_length = self.ques.shape[1]

    def __len__(self):
        if self.phase is not 'valid':
            return self.split.shape[0]
        else:
            return 20000 # Num of validation samples

    def __getitem__(self, idx):

        img_idx = self.img_pos[idx] - 1
        img = self.image[img_idx]
        #print(img_idx, idx)
        #if True:
        #    if self.phase is 'train':
        #        if self.feature_type == 'VGG':
        #            img = self.image[img_idx, 0:14, 0:14, 0:512]  # [14, 14, 512]
        #        elif self.feature_type == 'Residual':
        #            img = self.image[img_idx, 0:14, 0:14, 0:2048] # [14, 14, 2048]
        #        else:
        #            print("Error(train): feature type error")
        #    else:
        #        if self.feature_type == 'VGG':
        #            img = self.image[img_idx, 0:14, 0:14, 0:512] # [14, 14, 512]
        #        elif self.feature_type == 'Residual':
        #            img = self.image[img_idx, 0:14, 0:14, 0:2048] # [14, 14, 2048]
        #        else:
        #            print("Error(test): feature type error")

        question = np.array(self.ques[idx], dtype=np.int32)                 # vector of size 26
        ques_len = self.ques_len[idx].astype(int) # scalar integer
        answer = self.ans[idx].astype(int) - 1    # scalar integer
        ques_id = self.ques_id[idx].astype(int)
        if self.transform is not None:
            img = self.transform(img)
            question = self.transform(question)

        return (img, question, ques_id, ques_len, answer)

    def getVocabSize(self):
        return self.vocab_size

    def getSeqLength(self):
		return self.seq_length
