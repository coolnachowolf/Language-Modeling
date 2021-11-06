"""
Language Modeling Project
Name:
Roll No:
"""

import language_tests as test

project = "Language" # don't edit this

### WEEK 1 ###

'''
loadBook(filename)
#1 [Check6-1]
Parameters: str
Returns: 2D list of strs
'''
def loadBook(filename):
    file = open(filename, "r")
    words = []
    for line in file:
        if len(line) > 1:
            line = line.strip()
            wordString = line.split()
            words.append(wordString)
    file.close()
    return words


'''
getCorpusLength(corpus)
#2 [Check6-1]
Parameters: 2D list of strs
Returns: int
'''
def getCorpusLength(corpus):
    count = 0
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            count += 1
    return count


'''
buildVocabulary(corpus)
#3 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def buildVocabulary(corpus):
    vocab_list = []
    for i in range(0,len(corpus)):
        for j in range(len(corpus[i])):
            if corpus[i][j] not in vocab_list:
                vocab_list.append(corpus[i][j])
    return vocab_list


'''
countUnigrams(corpus)
#4 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countUnigrams(corpus):
    count_dictionary = {}
    for line in corpus:
        for word in line:
            if word in count_dictionary:
                count_dictionary[word] += 1
            else:
                count_dictionary[word] = 1
    return count_dictionary


'''
getStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def getStartWords(corpus):
    start_list = []
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            if(j!=0):
                pass
            else:
                if(corpus[i][j] in start_list):
                    pass 
                else:
                    start_list.append(corpus[i][j])
    return start_list


'''
countStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countStartWords(corpus):
    start_list = getStartWords(corpus)
    count_dictionary = {}
    for s in start_list:
        count = 0
        for i in corpus:
            if(i[0]==s):
                count += 1
        count_dictionary[s] = count
    return count_dictionary


'''
countBigrams(corpus)
#6 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def countBigrams(corpus):
    bigram_dict = {}
    for line in corpus:
        for word in range(len(line) - 1):
            if line[word] not in bigram_dict:
                bigram_dict[line[word]] = {}
            if line[word + 1] not in bigram_dict[line[word]]:
                bigram_dict[line[word]][line[word + 1]] = 1
            else:
                bigram_dict[line[word]][line[word + 1]] += 1
    return bigram_dict
    


### WEEK 2 ###

'''
buildUniformProbs(unigrams)
#1 [Check6-2]
Parameters: list of strs
Returns: list of floats
'''
def buildUniformProbs(unigrams):
    prob_list = []
    for i in unigrams:
        prob_list.append(1/len(unigrams))
    return prob_list


'''
buildUnigramProbs(unigrams, unigramCounts, totalCount)
#2 [Check6-2]
Parameters: list of strs ; dict mapping strs to ints ; int
Returns: list of floats
'''
def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    prob_list = []
    for i in unigrams:
        prob_list.append(unigramCounts[i]/totalCount)
    return prob_list


'''
buildBigramProbs(unigramCounts, bigramCounts)
#3 [Check6-2]
Parameters: dict mapping strs to ints ; dict mapping strs to (dicts mapping strs to ints)
Returns: dict mapping strs to (dicts mapping strs to (lists of values))
'''
def buildBigramProbs(unigramCounts, bigramCounts):
    bigram_prob = {}
    for i in bigramCounts:
        words_list = []
        probs_list = []
        temp_dict = {}
        for j in bigramCounts[i]:
            words_list.append(j)
            probs_list.append(bigramCounts[i][j]/unigramCounts[i])
            temp_dict["words"] = words_list
            temp_dict["probs"] = probs_list
        bigram_prob[i] = temp_dict
    return bigram_prob


'''
getTopWords(count, words, probs, ignoreList)
#4 [Check6-2]
Parameters: int ; list of strs ; list of floats ; list of strs
Returns: dict mapping strs to floats
'''
def getTopWords(count, words, probs, ignoreList):
    highProb_words = {}
    for i in range(len(probs)):
        if(words[i] not in ignoreList):
            highProb_words[words[i]] = probs[i]
    topWords = {}
    while(len(topWords) < count):
        maximum = 0
        for i in highProb_words:
            if i not in topWords:
                if highProb_words[i]>maximum:
                    maximum = highProb_words[i]
                    keys = i
        topWords[keys] = maximum
    return topWords


'''
generateTextFromUnigrams(count, words, probs)
#5 [Check6-2]
Parameters: int ; list of strs ; list of floats
Returns: str
'''
from random import choices
def generateTextFromUnigrams(count, words, probs):
    output_string = ''
    for i in range(count):
        word_list = choices(words, weights=probs) 
        output_string = output_string + word_list[0] + ' '
    return output_string


'''
generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs)
#6 [Check6-2]
Parameters: int ; list of strs ; list of floats ; dict mapping strs to (dicts mapping strs to (lists of values))
Returns: str
'''
def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    words_lst = []
    for i in range(count):
        if (len(words_lst) == 0 or words_lst[-1] == "."):
            word = choices(startWords, startWordProbs)
            words_lst.append(word[0])
        else:
            last_word = words_lst[-1]
            word_prob_dict = bigramProbs[last_word]
            word = choices(word_prob_dict['words'], word_prob_dict['probs'])
            words_lst.append(word[0])
    sentence = " "
    return (sentence.join(words_lst))


### WEEK 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]

'''
graphTop50Words(corpus)
#3 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTop50Words(corpus):
    total_count = getCorpusLength(corpus)
    unigrams = buildVocabulary(corpus)
    unigram_count = countUnigrams(corpus)
    probability = buildUnigramProbs(unigrams, unigram_count, total_count)
    dictionary = getTopWords(50, unigrams, probability, ignore)
    barPlot(dictionary, "Top 50 Common Words")
    return None


'''
graphTopStartWords(corpus)
#4 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTopStartWords(corpus):
    start_words = getStartWords(corpus)
    startWord_count = countStartWords(corpus)
    total_count = sum(startWord_count.values())
    probability = buildUnigramProbs(start_words, startWord_count, total_count)
    dictionary = getTopWords(50, start_words, probability, ignore)
    barPlot(dictionary, "Top 50 Common Start Words")
    return None


'''
graphTopNextWords(corpus, word)
#5 [Hw6]
Parameters: 2D list of strs ; str
Returns: None
'''
def graphTopNextWords(corpus, word):
    total_count = countUnigrams(corpus)
    word_count = countBigrams(corpus)
    words_probability = buildBigramProbs(total_count, word_count)
    dictionary = words_probability[word]
    words = dictionary['words']
    probs = dictionary['probs']
    plot_data = getTopWords(10, words, probs, ignore)
    barPlot(plot_data, "Top 10 Occurences of a Word")
    return None


'''
setupChartData(corpus1, corpus2, topWordCount)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int
Returns: dict mapping strs to (lists of values)
'''
def setupChartData(corpus1, corpus2, topWordCount):
    total_count1 = getCorpusLength(corpus1)
    unigrams1 = buildVocabulary(corpus1)
    unigram_count1 = countUnigrams(corpus1)
    probability1 = buildUnigramProbs(unigrams1, unigram_count1, total_count1)
    dictionary1 = getTopWords(topWordCount, unigrams1, probability1, ignore)
    total_count2 = getCorpusLength(corpus2)
    unigrams2 = buildVocabulary(corpus2)
    unigram_count2 = countUnigrams(corpus2)
    probability2 = buildUnigramProbs(unigrams2, unigram_count2, total_count2)
    dictionary2 = getTopWords(topWordCount, unigrams2, probability2, ignore)
    dictionary = {}
    topWords = []
    for i in dictionary1:
        topWords.append(i)
    for i in dictionary2:
        if i not in topWords:
            topWords.append(i)
    corpus1Probs = []
    for i in topWords:
        if i in dictionary1:
            corpus1Probs.append(dictionary1[i])
        else:
            corpus1Probs.append(0)
    corpus2Probs = []
    for i in range(len(topWords)):
        if topWords[i] in dictionary2:
            corpus2Probs.append(dictionary2[topWords[i]])
        else:
            corpus2Probs.append(0)
    dictionary['topWords'] = topWords
    dictionary['corpus1Probs'] = corpus1Probs
    dictionary['corpus2Probs'] = corpus2Probs
    return dictionary


'''
graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; str ; 2D list of strs ; str ; int ; str
Returns: None
'''
def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    result_dict = setupChartData(corpus1, corpus2, numWords)
    sideBySideBarPlots(result_dict['topWords'], result_dict['corpus1Probs'], result_dict['corpus2Probs'], name1, name2, title)
    return None


'''
graphTopWordsInScatterplot(corpus1, corpus2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int ; str
Returns: None
'''
def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    result_dict = setupChartData(corpus1, corpus2, numWords)
    scatterPlot(result_dict['corpus1Probs'], result_dict['corpus2Probs'], result_dict['topWords'], title)
    return None


### WEEK 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt

    names = []
    values = []
    for k in dict:
        names.append(k)
        values.append(dict[k])

    plt.bar(names, values)

    plt.xticks(rotation='vertical')
    plt.title(title)

    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt

    w = 0.35  # the width of the bars

    plt.bar(xValues, values1, width=-w, align='edge', label=category1)
    plt.bar(xValues, values2, width= w, align='edge', label=category2)

    plt.xticks(rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)

    # a bit of advanced code to draw a y=x line
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()
    
    ## Uncomment these for Week 2 ##

    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()


    ## Uncomment these for Week 3 ##

    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
