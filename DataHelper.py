# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:27:31 2019

@author: wangtf

Email: wangtfcyj@163.com

"""

import os
import sys
import random
import pickle

class DataHelper(object):
    def __init__(self, batch_size = 64, readFileDir = None, writeFileDir = None, getChar2Id = True, splitStr = " "):
        self.batch_size = batch_size
        self.splitStr = splitStr
        self.all_data_path = os.path.join(readFileDir, "all.txt")
        self.train_data_path = os.path.join(readFileDir, "train.txt")
        self.test_data_path = os.path.join(readFileDir, "test.txt")
        self.vail_data_path = os.path.join(readFileDir, "vail.txt")
        self.writeFileDir = writeFileDir
        self.char2idPath = os.path.join(writeFileDir, "char2id.txt")
        self.tag2idPath = os.path.join(writeFileDir, "tag2id.txt")
        if getChar2Id:
            if os.path.exists(self.all_data_path):
                self.char2id, self.tag2id = self.writeChar2idAndTag2id()
            else:
                print(self.all_data_path + " is not exits!!!")
                sys.exit()
        else:
            self.char2id = self.readDict(self.char2idPath)
            self.tag2id = self.readDict(self.tag2idPath)
        self.id2char = self.reverseDict(self.char2id)
        self.id2tag = self.reverseDict(self.id2char)
        
        self.generateDatas()
        
    def generateDatas(self):
        if os.path.exists(self.train_data_path):
            self.genPickles(self.train_data_path, "train")
        if os.path.exists(self.test_data_path):
            self.genPickles(self.test_data_path, "test")
        if os.path.exists(self.vail_data_path):
            self.genPickles(self.vail_data_path, "vail")
        
    def writeChar2idAndTag2id(self):
        wordDict = dict()
        tagDict = dict()
        wordDict["<pad>"] = 0
        wordDict["<unk>"] = 1
        for one in self.readFile(self.all_data_path):
            one_list = one.strip("\n").split(self.splitStr)
            if len(one_list) == 2:
                word, tag = one_list
                if word not in wordDict:
                    wordDict[word] = len(wordDict)
                if tag not in tagDict:
                    tagDict[tag] = len(tagDict)
        self.writeDict(self.char2idPath, wordDict)
        self.writeDict(self.tag2idPath, tagDict)
        
        return wordDict, tagDict
        
    def reverseDict(self, oneDict):
        reverse = dict()
        for one in oneDict:
            reverse[oneDict[one]] = one
        return reverse
        
    def readDict(self, path):
        result = dict()
        for one in self.readFile(path):
            key, value = one.strip().split("\t")
            result[key] = int(value)
        return result
        
    def readFile(self, filePath):
        return open(filePath, "r")
        
    def writeDict(self, filepath, oneDict):
        f = open(filepath, "w")
        for one in oneDict:
            f.write("{0}\t{1}\n".format(one, oneDict[one]))
        f.close()
        print("write %s success!!!"%(filepath))
        
    def dealOneLine(self, query):
        wordIds = list()
        tagIds = list()
        queryList = query.split("\n")
        for one in queryList:
            tmp = one.split(self.splitStr)
            if len(tmp) == 2:
                word, label = tmp
            else:
                print(queryList)
                sys.exit()
                
            if one in self.char2id:
                wordIds.append(self.char2id[one])
            else:
                wordIds.append(self.char2id["<unk>"])
            if label in self.tag2id:
                tagIds.append(self.tag2id[label])
            else:
                tagIds.append(self.tag2id["O"])
        return wordIds, tagIds
        
    def genPickles(self, filepath, mode = "train"):
        datas = "".join(open(filepath, "r").readlines()).split("\n\n")
        datas = [x for x in datas if x]
        if not os.path.exists(os.path.join(self.writeFileDir, mode)):
            os.mkdir(os.path.join(self.writeFileDir, mode))
        random.shuffle(datas)
        num_pickles = int(len(datas) / self.batch_size)
        print("{0} has num pickles is : {1}".format(mode, num_pickles))
        for i in range(num_pickles):
            tmp_datas = datas[i*self.batch_size: (i + 1)*self.batch_size]
            batch_datas = list()
            batch_tags = list()
            for one in tmp_datas:
                oneWordId, oneTagId = self.dealOneLine(one)
                batch_datas.append(oneWordId)
                batch_tags.append(oneTagId)
            f = open(os.path.join(self.writeFileDir, mode, "pickle_%d.pkl"%(i)), "wb")
            pickle.dump([batch_datas, batch_tags], f)
            f.close()
        print(mode + " pickles write sucess!!!")
            

if __name__ == "__main__":
    dataHelper = DataHelper(64, "./NER_IDCNN_CRF/TMP", "./TESTDataHelper")
            
        