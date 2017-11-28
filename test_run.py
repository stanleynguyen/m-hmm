from hmm import SUTDHMM
import os
import sys
# from EvalScript.evalResult import get_observed, get_predicted, compare_observed_to_predicted
# import matplotlib.pyplot as plt
# import math


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

languages = ['EN', 'SG', 'CN', 'FR']

# for l in languages:
#     i = 0.1
#     x_array = []
#     entity_array = []
#     sentiment_array = []
#     while i > 0.0000000000001:
#         model = SUTDHMM(default_emission=i)
#         model.train(input_filename='./{}/train'.format(l))

#         # print("Finish training for {}".format(l))

#         # print("----------Predict Using Emission for {0}------------".format(l))
#         # with open("./{}/dev.in".format(l)) as in_file, open("./{}/dev.p2.out".format(l), 'w+') as out_file:
#         #     for line in in_file:
#         #         word = line.strip()
#         #         if (word == ''):
#         #             out_file.write("\n")
#         #         else:
#         #             out_file.write("{} {}\n".format(
#         #                 word, model.predict_label_using_emission(word)))
#         # print("Emission Finished: {}".format(l))

#         # output = os.popen(
#         #     "python3 EvalScript/evalResult.py {0}/dev.out {0}/dev.p2.out".format(l)).read()
#         # print("Language: {}".format(l))
#         # print(output)

#         print("----------Viterbi for {0}------------".format(l))
#         with open("./{}/dev.in".format(l)) as in_file, open("./{}/dev.p3.out".format(l), 'w+') as out_file:
#             read_data = in_file.read()
#             sentences = list(
#                 filter(lambda x: len(x) > 0, read_data.split('\n\n')))
#             sentences = list(map(lambda x: ' '.join(x.split('\n')), sentences))
#             for sentence in sentences:
#                 sentence_labels, chance = model.viterbi(sentence)
#                 for idx, word in enumerate(sentence.split()):
#                     out_file.write("{} {}\n".format(
#                         word, sentence_labels[idx]))
#                 out_file.write('\n')
#             out_file.close()
#             in_file.close()

#         print("Viterbi Finished: {}".format(l))

#         gold = open('{0}/dev.out'.format(l), "r", encoding='UTF-8')
#         prediction = open('{0}/dev.p3.out'.format(l), "r", encoding='UTF-8')

#         # column separator
#         separator = ' '

#         # the column index for tags
#         outputColumnIndex = 1
#         # Read Gold data
#         observed = get_observed(gold)

#         # Read Predction data
#         predicted = get_predicted(prediction)

#         # Compare
#         x_array.append(-math.log(i))
#         entity_f, sen_f = compare_observed_to_predicted(observed, predicted)
#         entity_array.append(entity_f)
#         sentiment_array.append(sen_f)

#         i /= 10

#     print(x_array, entity_array, sentiment_array)
#     fig = plt.figure()

#     plt.plot(x_array, entity_array, label='Entity')
#     plt.plot(x_array, sentiment_array, label='Sentiment')
#     plt.ylabel('score')
#     plt.xlabel('negative of log default emission')
#     plt.savefig('{0}/score.png'.format(l))

for l in languages:
    model = SUTDHMM()
    model.load_data(data_filename='./{}/train'.format(l))
    with open('./{}/train.ent'.format(l), 'w+') as ent_in_file:
        for token in model.tokens_list:
            word = token[0]
            tag = token[1].split(
                '-')[0] if token[1] not in ['O', 'START', 'STOP'] else token[1]
            ent_in_file.write('{} {}\n'.format(word, tag))
            if token[1] == 'STOP':
                ent_in_file.write('\n')
        ent_in_file.close()
    with open('./{}/train.sen'.format(l), 'w+') as sen_in_file:
        for token in model.tokens_list:
            word = token[0]
            tag = token[1].split(
                '-')[1] if token[1] not in ['O', 'START', 'STOP'] else token[1]
            sen_in_file.write('{} {}\n'.format(word, tag))
            if token[1] == 'STOP':
                sen_in_file.write('\n')
        sen_in_file.close()

    ent_model = SUTDHMM(default_emission=0.0000001)
    ent_model.train(input_filename='./{}/train.ent'.format(l))
    sen_model = SUTDHMM(default_emission=0.0000001)
    sen_model.train(input_filename='./{}/train.sen'.format(l))
    with open('./{}/dev.in'.format(l)) as in_file, open('./{}/dev.p6.out'.format(l), 'w+') as out_file:
        read_data = in_file.read()
        sentences = list(filter(lambda x: len(x) > 0, read_data.split('\n\n')))
        sentences = list(map(lambda x: ' '.join(x.split('\n')), sentences))
        for sentence in sentences:
            sentence_ent, prob = ent_model.viterbi(sentence=sentence)
            sentence_sen, prob = sen_model.viterbi(sentence=sentence)
            for idx in range(0, len(sentence_ent)):
                entity = sentence_ent[idx]
                sentiment = sentence_sen[idx]
                if entity not in ['O', 'START', 'STOP'] and sentiment not in ['O', 'START', 'STOP']:
                    out_file.write(
                        "{} {}-{}\n".format(word, entity, sentiment))
                elif entity in ['O', 'START', 'STOP']:
                    out_file.write("{} {}\n".format(word, entity))
                else:
                    out_file.write('{} {}\n'.format(word, sentiment))
            out_file.write('\n')

    output = os.popen(
        "python3 EvalScript/evalResult.py {0}/dev.out {0}/dev.p6.out".format(l)).read()
    print("Language: {}".format(l))
    print(output)

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
