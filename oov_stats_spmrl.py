from collections import defaultdict

ID, FORM, LEMMA, CPOS, FPOS, MORPH, HEAD, REL, PHEAD, PREL = range(10)

PATH="SPMRL_SHARED_2014_NO_ARABIC/{LANG}_SPMRL/gold/conll/{type}/{type}.{Lang}.gold.conll"
LANGUAGES = "ARABIC BASQUE FRENCH GERMAN HEBREW HUNGARIAN KOREAN SWEDISH POLISH".split() + ["ENGLISH"]

def read_conll(filename):
    with open(filename, "r", encoding = "utf8") as f:
        sentences = f.read().split("\n\n")
        sentences = [[line.split("\t") for line in sentence.strip().split("\n")] for sentence in sentences if sentence.strip()]
        return sentences

def print_conll(corpus):
    for sentence in corpus:
        for line in sentence:
            print("\t".join(line))
        print()


def compute_oov(train, dev):
    occ_train = 0
    
    voctrain = defaultdict(int)
    for sentence in train:
        for word in sentence:
            if len(word) != 10 :
                print(sentence, word)
            voctrain[word[FORM]] += 1
            occ_train += 1

    vocdev = defaultdict(int)
    N = 0
    oov_occ = 0
    for sentence in dev:
        for word in sentence:
            vocdev[word[FORM]] += 1
            N += 1
            if word[FORM] not in voctrain:
                oov_occ += 1
    
    oov_type = len([word for word in vocdev if word not in voctrain])

    
    train_hapaxes = {w for w in voctrain if voctrain[w] == 1}
    dev_hapaxes = {w for w in vocdev if w not in voctrain or voctrain[w] == 1}
    

    return [occ_train, len(voctrain), N, len(vocdev), oov_occ, oov_type, len(train_hapaxes) / len(voctrain), len(dev_hapaxes) / N]


def main():
    results = {}
    for language in LANGUAGES:
        trainfile = PATH.format(LANG = language, Lang = language.capitalize(), type="train")
        devfile = PATH.format(LANG = language, Lang = language.capitalize(), type="dev")
        
        train = read_conll(trainfile)
        dev   = read_conll(devfile)
        
        stats = compute_oov(train, dev)
        
        results[language] = stats
    
    
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{lrrrrrrrrrr}")
    print("\\toprule")
    print("  & {} \\\\".format(" & ".join([lang.capitalize() for lang in LANGUAGES])))
    print("\\midrule")
    print("Train tokens  & {}      \\\\ ".format(" & ".join(map(str, [results[lang][0] for lang in LANGUAGES]))))
    print("Train types   & {}      \\\\ ".format(" & ".join(map(str, [results[lang][1] for lang in LANGUAGES]))))
    print("Train hapaxes (types)& {}      \\\\ ".format(" & ".join(map(str, [round(results[lang][6] * 100, 1) for lang in LANGUAGES]))))
    
    print("Dev tokens & {}            \\\\ ".format(" & ".join(map(str, [results[lang][2] for lang in LANGUAGES]))))
    print("Dev types & {}             \\\\ ".format(" & ".join(map(str, [results[lang][3] for lang in LANGUAGES]))))
    print("OOV rate (occurrences) & {}\\\\ ".format(" & ".join(map(str, [round(results[lang][4] / results[lang][2] * 100, 1) for lang in LANGUAGES]))))
    print("OOV rate (types) & {}      \\\\ ".format(" & ".join(map(str, [round(results[lang][5] / results[lang][3] * 100, 1) for lang in LANGUAGES]))))
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")





if __name__ == "__main__":
    main()

