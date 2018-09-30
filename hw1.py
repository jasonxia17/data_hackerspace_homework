#
# CS 196 Data Hackerspace
# Assignment 1: Data Parsing and NumPy
# Due September 24th, 2018
#

import json
import csv
import numpy as np

def histogram_times(filename):

    def extractFirstInt(inputString):
        from string import digits

        startingIndex = None
        endingIndex = None

        for i in range(0, len(inputString)):
            if inputString[i] in digits:
                startingIndex = i
                break
        if startingIndex is None:
            raise ValueError('The input string contains no digits.')

        for i in range(startingIndex, len(inputString)):
            if inputString[i] not in digits:
                endingIndex = i
                break

        if endingIndex is None:
            return int(inputString[startingIndex:])
        else:
            return int(inputString[startingIndex:endingIndex])

    with open(filename) as file:
        crashList = list(csv.DictReader(file))

    hourHistogram = [0] * 24
    for crash in crashList:
        try:
            hour = extractFirstInt(crash['Time'])
            hourHistogram[hour] += 1
        except (IndexError, ValueError):
            # throw away crashes without time or where hour is not between 0 and 23
            continue
    return hourHistogram

def weigh_pokemons(filename, weight):
    with open(filename) as file:
        pokemonList = json.load(file)['pokemon']

    weightMatchList = []
    for pokemon in pokemonList:
        kgWeight = float(pokemon['weight'].split()[0])
        if kgWeight == weight:
            weightMatchList.append(pokemon['name'])
    return weightMatchList

def single_type_candy_count(filename):
    with open(filename) as file:
        pokemonList = json.load(file)['pokemon']

    candyCount = 0
    for pokemon in pokemonList:
        #print(pokemon['candy_count'])
        if len(pokemon['type']) == 1 and 'candy_count' in pokemon:
            candyCount += pokemon['candy_count']
    return candyCount

def reflections_and_projections(points):
    newPoints = np.copy(points)
    newPoints[1] = 2 - newPoints[1]
    newPoints = np.array([[0, -1], [1, 0]]) @ newPoints

    m = 3
    newPoints = 1/(m**2 + 1) * np.array([[1, m], [m, m**2]]) @ newPoints
    return newPoints

def normalize(image):
    maxValue, minValue = np.amax(image), np.amin(image)
    return np.rint(255/(maxValue - minValue) * (image - minValue)).astype(int)

def sigmoid_normalize(image, a):
    # "a" is the variance parameter
    from math import e
    exponent = -a**(-1) * (image - 128)
    return np.rint(255 * (1 + e**exponent)**(-1)).astype(int)

def testCases():
    print(histogram_times('airplane_crashes.csv'))
    print(weigh_pokemons('pokedex.json', 10.0))
    print(single_type_candy_count('pokedex.json'))
    print(reflections_and_projections(np.arange(0,6).reshape(2,-1)))

    dim = 3
    testImage = np.random.randint(0, 256, (dim,dim))
    print("--------------------------")
    print(testImage)
    print(normalize(testImage))
    print(sigmoid_normalize(testImage, 80))
