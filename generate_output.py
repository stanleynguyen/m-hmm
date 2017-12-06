#!/usr/bin/python3

# pylint: disable-all

import sys
import getopt
from hmm import SUTDHMM


def main(argv):
    inputfile = ''
    outputfile = ''
    trainfile = ''
    try:
        opts, _ = getopt.getopt(
            argv, "hi:o:t:", ["help", "ifile=", "ofile=", "tfile="])
    except getopt.GetoptError:
        print('generate_output.py -t <trainfile> -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('generate_output.py -t <trainfile> -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ('-t', '--tfile'):
            trainfile = arg
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    if trainfile == '' or inputfile == '' or outputfile == '':
        raise getopt.GetoptError(
            "Usage: generate_output.py -t <trainfile> -i <inputfile> -o <outputfile>")

    model = SUTDHMM(default_emission=0.000001)
    model.train(input_filename=trainfile)

    print("Finish training")
    print('Predicting...')
    with open(inputfile) as in_file, open(outputfile, 'w+') as out_file:
        read_data = in_file.read()
        sentences = list(filter(lambda x: len(x) > 0, read_data.split('\n\n')))
        sentences = list(map(lambda x: ' '.join(x.split('\n')), sentences))
        for sentence in sentences:
            sentence_labels = model.max_marginal(sentence)
            for idx, word in enumerate(sentence.split()):
                out_file.write("{} {}\n".format(word, sentence_labels[idx]))
            out_file.write('\n')
        out_file.close()
        in_file.close()

    print("Output Generated!")


if __name__ == "__main__":
    main(sys.argv[1:])
