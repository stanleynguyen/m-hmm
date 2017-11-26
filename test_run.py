from hmm import SUTDHMM
import os
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
languages = ['EN']  # , 'SG', 'CN', 'FR']

for l in languages:
    model = SUTDHMM()
    model.train(input_filename='./{}/train'.format(l))
    print("Finish training for {}".format(l))


print(model.max_marginal('a d'))

languages = ['EN', 'SG', 'CN', 'FR']

for l in languages:
    model = SUTDHMM(k=1)
    model.train(input_filename='./{}/train'.format(l))

    print("Finish training for {}".format(l))

    print("----------Predict Using Emission for {0}------------".format(l))
    with open("./{}/dev.in".format(l)) as in_file, open("./{}/dev.p2.out".format(l), 'w+') as out_file:
        for line in in_file:
            word = line.strip()
            if (word == ''):
                out_file.write("\n")
            else:
                out_file.write("{} {}\n".format(
                    word, model.predict_label_using_emission(word)))
    print("Emission Finished: {}".format(l))

    output = os.popen(
        "python3 EvalScript/evalResult.py {0}/dev.out {0}/dev.p2.out".format(l)).read()
    print("Language: {}".format(l))
    print(output)

    # print("----------Viterbi for {0}------------".format(l))
    # with open("./{}/dev.in".format(l)) as in_file, open("./{}/dev.p3.out".format(l), 'w+') as out_file:
    #     read_data = in_file.read()
    #     sentences = list(filter(lambda x: len(x) > 0, read_data.split('\n\n')))
    #     sentences = list(map(lambda x: ' '.join(x.split('\n')), sentences))
    #     for sentence in sentences:
    #         sentence_labels, chance = model.viterbi(sentence)
    #         for idx, word in enumerate(sentence.split()):
    #             out_file.write("{} {}\n".format(word, sentence_labels[idx]))
    #         out_file.write('\n')
    #     out_file.close()
    #     in_file.close()

    # print("Viterbi Finished: {}".format(l))

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
