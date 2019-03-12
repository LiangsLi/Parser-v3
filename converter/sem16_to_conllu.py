from collections import defaultdict

root = 'root'


# ===============================================================
def to_conllu(sdp_filename, conllu_filename):
    """
sem16示例：
1	妈妈	妈妈	NN	NN	_	5	Agt	_	_
2	把	把	P	P	_	4	mPrep	_	_
3	旧	旧	JJ	JJ	_	4	Desc	_	_
4	窗帘	窗帘	NN	NN	_	5	Pat	_	_
5	撕成	撕成	VV	VV	_	9	Root	_	_
6	了	了	AS	AS	_	5	mTime	_	_
7	抹布	抹布	NN	NN	_	5	Clas	_	_
8	。	。	PU	PU	_	5	mPunc	_	_

    :param sdp_filename:
    :param conllu_filename:
    :return:
    """

    with open(sdp_filename, 'r', encoding='utf-8') as f:
        with open(conllu_filename, 'w', encoding='utf-8') as g:
            buff = []
            sents = f.read().strip().split('\n\n')
            # print ("sents: ", len(sents))
            for sent in sents:
                conllu_form = []
                words = []
                lines = sent.strip().split('\n')
                for line in lines:
                    if line.startswith('#'):
                        conllu_form.append(line)
                        continue
                    items = line.strip().split('\t')
                    if int(items[0]) == len(words) + 1:
                        if items[6] is not '_' and items[7] is not '_':
                            items[8] = items[6] + ':' + items[7]
                        words.append(items)
                    elif int(items[0]) == len(words):
                        words[-1][8] += '|' + items[6] + ':' + items[7]
                    else:
                        print("Error:{}".format(line))
                for word in words:
                    conllu_form.append('\t'.join(word))
                g.write('\n'.join(conllu_form) + '\n\n')


# ***************************************************************
if __name__ == '__main__':
    """"""

    import sys

    to_conllu(sys.argv[1], sys.argv[2])
