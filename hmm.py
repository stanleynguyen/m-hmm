import copy
import math
import os


class SUTDHMM:
    def __init__(self, k=1, special_word='#UNK#', dummy_word='#DUM#', pre_prob={}):
        # 2 layer dictionary depth-0 key is the label, depth-1 key is the word
        self.emission_params = {}
        self.y_count = {}
        self.y_labels = []
        self.x_words = [special_word, dummy_word]
        self.x_given_y_count = {}
        # 2 layer dictionary depth-0 key is the (i-1)-label, depth-1 key is the i-label
        self.y_given_prev_y_count = {}
        self.transition_params = {}
        self.label_prob = pre_prob
        self.special_word = special_word
        self.dummy_word = dummy_word
        if k > 0:
            self.k = k
        else:
            self.k = 1
        self.tokens_list = []

    def load_data(self, raw_string=None, data_filename=None):
        tokens_list = []
        if raw_string != None:
            data = os.linesep.join([s if s else self.dummy_word + ' START\n' +
                                    self.dummy_word + ' STOP' for s in raw_string.splitlines()])
            tokens_list = list(
                map(lambda x: x.rsplit(' ', 1), data.split('\n')))
        elif data_filename != None:
            with open(data_filename) as f:
                data = f.read()
                data = os.linesep.join([s if s else self.dummy_word + ' STOP\n' + self.dummy_word + ' START'
                                        for s in data.splitlines()])
                tokens_list = list(
                    map(lambda x: x.rsplit(' ', 1), data.split('\n')))
                f.close()
        else:
            raise Exception('No Data Input Provided!')

        token_freq = {}
        for token in tokens_list:
            if token[0] not in token_freq:
                token_freq[token[0]] = 1
            else:
                token_freq[token[0]] += 1
        for i in range(len(tokens_list)):
            if token_freq[tokens_list[i][0]] < self.k:
                tokens_list[i][0] = self.special_word

        for token in tokens_list:
            if token[0] not in self.x_words:
                self.x_words.append(token[0])
            if token[1] not in self.y_labels:
                self.y_labels.append(token[1])

        # intialise counts and emission params
        for label in self.y_labels:
            if label not in self.emission_params:
                self.y_count[label] = 0
                self.x_given_y_count[label] = {}
                self.emission_params[label] = {}
                for word in self.x_words:
                    self.x_given_y_count[label][word] = 0
                    self.emission_params[label][word] = 0

        # initialise count and transition params
        for label in self.y_labels:
            if label not in self.transition_params:
                self.y_given_prev_y_count[label] = {}
                self.transition_params[label] = {}
                for next_label in self.y_labels:
                    self.y_given_prev_y_count[label][next_label] = 0
                    self.transition_params[label][next_label] = 0

        all_labels = list(map(lambda x: x[1], tokens_list))
        self.y_count = {}
        for label in all_labels:
            if label not in self.y_count:
                self.y_count[label] = 1
            else:
                self.y_count[label] += 1
        for label in self.y_count:
            self.label_prob[label] = float(
                self.y_count[label]) / len(all_labels)

        self.tokens_list += tokens_list
        return self.tokens_list, self.y_labels, self.x_words, self.label_prob

    def calculate_emission(self):
        for token in self.tokens_list:
            self.x_given_y_count[token[1]][token[0]] += 1

        self.emission_params = copy.deepcopy(self.x_given_y_count)
        for label in self.emission_params:
            for word in self.emission_params[label]:
                self.emission_params[label][word] = float(
                    self.x_given_y_count[label][word]) / self.y_count[label]

        return self.emission_params

    def get_emission_param(self, label: str, word: str):
        if word not in self.x_words:
            word = self.special_word

        return self.emission_params[label][word]

    def predict_label_using_emission(self, word: str):
        score = 0.0
        predicted_label = None
        for label in self.y_labels:
            label_score = self.get_emission_param(label, word)
            if label_score > score:
                predicted_label = label
                score = label_score

        return predicted_label

    def calculate_transition(self):
        ordered_labels_list = list(map(lambda x: x[1], self.tokens_list))

        for idx, label in enumerate(ordered_labels_list):
            if idx < len(ordered_labels_list) - 1:
                next_label = ordered_labels_list[idx + 1]
                self.y_given_prev_y_count[label][next_label] += 1

        # calculate trans_params
        trans_params = copy.deepcopy(self.y_given_prev_y_count)
        for given_label in trans_params:
            if given_label == ordered_labels_list[-1]:
                adjusted_count = self.y_count[given_label] - 1
            else:
                adjusted_count = self.y_count[given_label]

            for label in trans_params[given_label]:
                trans_params[given_label][label] /= float(adjusted_count)

        self.transition_params = trans_params
        return self.transition_params

    def train(self, raw_string=None, input_filename=None):
        self.load_data(raw_string=raw_string, data_filename=input_filename)
        self.calculate_emission()
        self.calculate_transition()

    def clean_input_data(self, input_data: str):
        data = input_data.split()
        for idx, word in enumerate(data):
            if word not in self.x_words:
                data[idx] = self.special_word

        return data

    def viterbi(self, sentence: str):
        '''pre-requisite: train must be run before this function'''
        observed_words = self.clean_input_data(sentence)
        cache = [{}]

        # first layer
        for l in self.y_labels:
            trans_param = self.transition_params['START'][l]
            emission_param = self.emission_params[l][observed_words[0]
                                                     ] if observed_words[0] in self.emission_params[l] else self.emission_params[l]['#UNK#']
            cache[0][l] = {"chance": trans_param *
                           emission_param, "prev": None}

        # handle middle layers
        for i in range(1, len(observed_words)):
            cache.append({})
            for l in self.y_labels:
                max_prob = -math.inf
                max_prev_l = None
                for prev_l in self.y_labels:
                    trans_param = self.transition_params[prev_l][l]

                    emission_param = self.emission_params[l][observed_words[i]
                                                             ] if observed_words[i] in self.emission_params[l] else emission_p[l]['#UNK#']
                    prob = cache[i - 1][prev_l]['chance'] * \
                        trans_param * emission_param
                    if prob > max_prob:
                        max_prob = prob
                        max_prev_l = prev_l

                cache[i][l] = {'chance': max_prob, 'prev': max_prev_l}

        # handle the end layer
        cache.append({})
        max_end_prob = -math.inf
        max_end_l = None
        for l in self.y_labels:
            trans_param = self.transition_params[l]['STOP']
            end_prob = cache[len(observed_words) -
                             1][l]['chance'] * trans_param
            if end_prob > max_end_prob:
                max_end_prob = end_prob
                max_end_l = l
        cache[len(observed_words)]['STOP'] = {
            'chance': max_end_prob, 'prev': max_end_l}

        # backtrack for optimal path
        optimal_prob = cache[len(observed_words)]['STOP']['chance']
        previous_l = cache[len(observed_words)]['STOP']['prev']
        optimal = [previous_l]
        for i in range(len(observed_words) - 1, 0, -1):
            optimal.insert(0, cache[i][previous_l]['prev'])
            previous = cache[i][previous_l]['prev']
        return (optimal, optimal_prob)

    def fwd_bwd(self, sentence: str):
        observed_words = self.clean_input_data(sentence)

        # forward part
        forward = []
        prev_forward = {}
        for i, word in enumerate(observed_words):
            curr_forward = {}
            for l in self.y_labels:
                prev_f_sum = 0
                if i == 0:
                    trans_prob = self.transition_params['START'][l]
                    prev_f_sum = trans_prob
                else:
                    for prev_l in self.y_labels:
                        trans_prob = self.transition_params[prev_l][l]
                        prev_f_sum += prev_forward[prev_l] * trans_prob

                emission_prob = self.emission_params[l][word] if word in self.emission_params[
                    l] else self.emission_params[l]['#UNK#']
                curr_forward[l] = emission_prob * prev_f_sum

            forward.append(curr_forward)
            prev_forward = copy.deepcopy(curr_forward)

        forward_prob = 0
        for l in self.y_labels:
            trans_prob = self.transition_params[l]['STOP']
            forward_prob += curr_forward[l] * trans_prob

        # backward part
        backward = []
        prev_backward = {}
        for i, word in enumerate(observed_words[::-1]):
            curr_backward = {}
            for l in self.y_labels:
                curr_backward[l] = 0
                if i == 0:
                    trans_prob = self.transition_params[l]['STOP']
                    emiss_prob = self.emission_params[l][word] if word in self.emission_params[
                        l] else self.emission_params[l][self.special_word]
                    curr_backward[l] = trans_prob
                else:
                    for next_l in self.y_labels:
                        trans_prob = self.transition_params[l][next_l]
                        emm_prob = self.emission_params[l][word] if word in self.emission_params[
                            l] else self.emission_params[l]['#UNK#']
                        curr_backward[l] += trans_prob * \
                            emm_prob * prev_backward[next_l]

            backward.insert(0, curr_backward)
            prev_backward = copy.deepcopy(curr_backward)

        backward_prob = 0
        for l in self.y_labels:
            trans_prob = self.transition_params['START'][l]
            emm_prob = self.emission_params[l][observed_words[0]
                                               ] if observed_words[0] in self.emission_params[l] else self.emission_params[l]['#UNK#']
            backward_prob += trans_prob * emm_prob * curr_backward[l]

        # print(forward)
        # print(backward)
        return forward, backward

    def max_marginal(self, sentence: str):
        forward_p, backward_p = self.fwd_bwd(sentence)
        predictions = []
        for i in range(len(forward_p)):
            product_p = {l: forward_p[i][l] * backward_p[i][l]
                         for l in self.y_labels}
            predictions.append(max(product_p, key=product_p.get))

        return predictions
