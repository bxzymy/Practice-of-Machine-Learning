'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
'''


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


if __name__ == '__main__':
    convert(".\Mnist\\t10k-images.idx3-ubyte", ".\Mnist\\t10k-labels.idx1-ubyte",
            ".\Mnist\\mnist_test.csv", 10000)
    convert(".\Mnist\\train-images.idx3-ubyte", ".\Mnist\\train-labels.idx1-ubyte",
            ".\Mnist\mnist_train.csv", 60000)
