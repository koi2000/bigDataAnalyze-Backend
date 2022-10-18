import os

from bigdata.GPT2_Chinese.generate import userGenerate


class args:
    nsamples = 4
    length = 0,
    prefix = ""
    save_samples = True
    segment = True
    save_samples_path = "./checkpoint"
    device = "0"
    batch_size = 1
    temperature = 1
    topk = 8
    topp = 0
    repetition_penalty = 1.0
    tokenizer_path = "model/vocab.txt"
    model_path = 'model/'
    fast_pattern = False


def generatePoemByGPT(word, senNumber, nsample=4):
    # cmd = "python ./generate.py --length=" + str(num) + " --nsamples=4 --prefix=" + word + "， --save_samples --save_samples_path=./checkpoint"
    args.prefix = word
    args.length = int(senNumber) * 8 + 1
    args.nsamples = int(nsample)
    text = userGenerate(args)
    print("------------------------------------------------------")
    print(text)
    return text


if __name__ == '__main__':
    generatePoemByGPT("一行白鹭上青天", 3, 2)
