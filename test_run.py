from hmm import SUTDHMM
import os
import sys
from EvalScript.evalResult import get_observed, get_predicted, compare_observed_to_predicted
import matplotlib.pyplot as plt
import math


# example usage for the emission prediction
# model = SUTDHMM()
# print(model.load_data('a O\nb O\na I\n c O\n'))
# print(model.calculate_emission())
# print(model.get_emission_param(label='O', word='b'))
# print(model.y_labels)
# print(model.predict_label_using_emission('a'))
# print(model.calculate_transition())


model = SUTDHMM()
model.train(raw_string='''#DUM# START
b X
c X
a Z
b X
#DUM# STOP
#DUM# START
a X
b Z
a Y
#DUM# STOP
#DUM# START
b Z
c Y
a X
b Z
d Y
#DUM# STOP
#DUM# START
c Z
b Z
a Y
#DUM# STOP
#DUM# START
c X
a X
#DUM# STOP
#DUM# START
d Z
#DUM# STOP
#DUM# START
d Z
b Z
#DUM# STOP
''')
# languages = ['EN', 'SG', 'CN', 'FR']

# for l in languages:
#     model = SUTDHMM()
#     model.train(input_filename='./{}/train'.format(l))
#     print("Finish training for {}".format(l))


# print(model.max_marginal('a d'))

languages = ['EN', 'FR']

for l in languages:
    i = 0.1
    x_array = []
    entity_array = []
    sentiment_array = []
    while i > 0.0000000000001:
        model = SUTDHMM(default_emission=i)
        model.train(input_filename='./{}/train'.format(l))

        # print("Finish training for {}".format(l))

        # print("----------Predict Using Emission for {0}------------".format(l))
        # with open("./{}/dev.in".format(l)) as in_file, open("./{}/dev.p2.out".format(l), 'w+') as out_file:
        #     for line in in_file:
        #         word = line.strip()
        #         if (word == ''):
        #             out_file.write("\n")
        #         else:
        #             out_file.write("{} {}\n".format(
        #                 word, model.predict_label_using_emission(word)))
        # print("Emission Finished: {}".format(l))

        # output = os.popen(
        #     "python3 EvalScript/evalResult.py {0}/dev.out {0}/dev.p2.out".format(l)).read()
        # print("Language: {}".format(l))
        # print(output)

        print("----------Viterbi for {0}------------".format(l))
        with open("./{}/dev.in".format(l)) as in_file, open("./{}/dev.p3.out".format(l), 'w+') as out_file:
            read_data = in_file.read()
            sentences = list(
                filter(lambda x: len(x) > 0, read_data.split('\n\n')))
            sentences = list(map(lambda x: ' '.join(x.split('\n')), sentences))
            for sentence in sentences:
                sentence_labels, chance = model.viterbi(sentence)
                for idx, word in enumerate(sentence.split()):
                    out_file.write("{} {}\n".format(
                        word, sentence_labels[idx]))
                out_file.write('\n')
            out_file.close()
            in_file.close()

        print("Viterbi Finished: {}".format(l))

        gold = open('{0}/dev.out'.format(l), "r", encoding='UTF-8')
        prediction = open('{0}/dev.p3.out'.format(l), "r", encoding='UTF-8')

        # column separator
        separator = ' '

        # the column index for tags
        outputColumnIndex = 1
        # Read Gold data
        observed = get_observed(gold)

        # Read Predction data
        predicted = get_predicted(prediction)

        # Compare
        x_array.append(-math.log(i))
        entity_f, sen_f = compare_observed_to_predicted(observed, predicted)
        entity_array.append(entity_f)
        sentiment_array.append(sen_f)

        i /= 10

    print(x_array, entity_array, sentiment_array)
    fig = plt.figure()

    plt.plot(x_array, entity_array, label='Entity')
    plt.plot(x_array, sentiment_array, label='Sentiment')
    plt.ylabel('score')
    plt.xlabel('negative of log default emission')
    plt.savefig('{0}/score.png'.format(l))

# output = os.popen(
#     "python3 EvalScript/evalResult.py {0}/dev.out {0}/dev.p3.out".format(l)).read()
# print("Language: {}".format(l))
# print(output)

# print("----------Max Marginal for {0}------------".format(l))
# with open("./{}/dev.in".format(l)) as in_file, open("./{}/dev.p4.out".format(l), 'w+') as out_file:
#     read_data = in_file.read()
#     sentences = list(filter(lambda x: len(x) > 0, read_data.split('\n\n')))
#     sentences = list(map(lambda x: ' '.join(x.split('\n')), sentences))
#     for sentence in sentences:
#         sentence_labels = model.max_marginal(sentence=sentence)
#         for idx, word in enumerate(sentence.split()):
#             out_file.write("{} {}\n".format(word, sentence_labels[idx]))
#         out_file.write('\n')
# print("Max Marginal Finished: {}".format(l))

# output = os.popen(
#     "python3 EvalScript/evalResult.py {0}/dev.out {0}/dev.p4.out".format(l)).read()
# print("Language: {}".format(l))
# print(output)
