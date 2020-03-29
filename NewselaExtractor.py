# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:14:37 2020

@author: mohiu
"""

import os
from sklearn.utils import shuffle
import math
from nltk.tokenize import word_tokenize, sent_tokenize
from massalign.core import *
simVals = [0.3,0.4,0.5,0.6,0.7]

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file
#formatting Newsela raw data into format needed by aligner
rawDataPath = 'NewselaRaw'
formattedDataPath = 'NewselaFormatted'
ignoreFiles = []
for f in files(rawDataPath):
    print(f)
    if f in ignoreFiles:
        continue
    fileLines = [line.rstrip('\n') for line in open(rawDataPath+ '/' +f)]
    firstSent = fileLines[0]
    if firstSent.split(' ')[0].isupper():
        fileLines[0] = firstSent[firstSent.find("—")+3:].strip()
        
    file_content = ''
    for l in fileLines:
        if l != '' and (not l.startswith("##")):
            file_content += l.strip() + ' '
    
    try:
        fileSentences = sent_tokenize(file_content)

        outFile = open(formattedDataPath + '/' + f, "w")
        for s in fileSentences:
            # write line to output file
            outFile.write(s)
            outFile.write("\n")
            outFile.write("\n")
        outFile.close()
    except:
        ignoreFiles.append(f)

len(ignoreFiles)

#get list of all articles in the corpus
articles = [line.rstrip('\n') for line in open('articleNames.csv')]
artDict = {}
for a in articles:
    tmpA = a.split(',')
    if tmpA[0] in artDict.keys():
        artDict[tmpA[0]].append(tmpA[5])
    else:
        artDict[tmpA[0]] = [tmpA[5]]
def getAlignedParagraphs(f1, f2, similarity = 0.3, showVisual=False):
    model = TFIDFModel([f1, f2], 'https://ghpaetzold.github.io/massalign_data/stop_words.txt')
    #Get paragraph aligner:
    paragraph_aligner = VicinityDrivenParagraphAligner(similarity_model=model, acceptable_similarity=similarity)

    #Get MASSA aligner for convenience:
    m = MASSAligner()

    #Get paragraphs from the document:
    p1s = m.getParagraphsFromDocument(f1)
    p2s = m.getParagraphsFromDocument(f2)
    
    #Align paragraphs:
    alignments, aligned_paragraphs = m.getParagraphAlignments(p1s, p2s, paragraph_aligner)
    
    if showVisual:
        #m.visualizeParagraphAlignments(p1s, p2s, alignments)
        m.visualizeListOfParagraphAlignments([p1s, p1s], [p2s, p2s], [alignments, alignments])
    return aligned_paragraphs

for sim in simVals:
    #for each similarity value from 0.3 to 0.7
    fileOutput = open("alignedSentences-" + str(sim) + ".txt","w")
    for k in artDict.keys():
        #combine data into aligned sentence pairs
        print k
        for i in range(len(artDict[k])-1):
            for j in range(i+1, len(artDict[k])):
                f1 = 'NewselaFormatted/' + artDict[k][i]
                f2 = 'NewselaFormatted/' + artDict[k][j]

                try:
                    aps = getAlignedParagraphs(f1, f2, sim)
                    for a in aps:
                        s = str(a[0]) + "\t" + str(a[1]) + "\n"
                        fileOutput.write(s)
                except:
                    continue
    fileOutput.close()
#For each file from aligned 0.3 - 0.7
path = 'alignedSentences-0.5.txt'
#output separated files for normal and simplified sentences
outSimpFile = open('alignedSentencesSimp-0.5.txt', "w")
outNormFile = open('alignedSentencesNorm-0.5.txt', "w")
lines = [line.rstrip('\n') for line in open(path)]
for l in lines:
    outNorm = l.split('\t')[0][3:-2] + "\n"
    outSimp = l.split('\t')[1][3:-2] + "\n"
    outNormFile.write(outNorm)
    outSimpFile.write(outSimp)
outNormFile.close()
outSimpFile.close()









