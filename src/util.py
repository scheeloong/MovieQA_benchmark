import numpy as np
import math

"""
This file represents all utilities that can be used by any class
"""
def cosineSimilarity(vecA, vecB):
    """
    OBSOLETE: No longer used, direct computation of cosineSimilarity in main instead
    This method computes the cosine similarity between 2 vectors 
    represented in python's vector class
    vecA and vecB are sets
    """
    # Cosine similarity
    sumA = 0.0
    for keyA in vecA:
        sumA += math.pow(vecA[keyA], 2) * 1.0
    normA = math.sqrt(sumA)

    sumB = 0.0
    for keyB in vecB:
        sumB += math.pow(vecB[keyB], 2) * 1.0
    normB = math.sqrt(sumB)
    if normA == 0.0:
        return 0.0
    if normB == 0.0:
        return 0.0
    sumDot = 0.0
    for keyAB in vecA:
        if keyAB in vecB.keys():
            sumDot += vecA[keyAB] * vecB[keyAB]
    cosineSimilarity = sumDot / ((normA * normB) * (1.0))
    return cosineSimilarity
