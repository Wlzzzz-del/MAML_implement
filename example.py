from scipy.io import loadmat
annots = loadmat("./Mnist/mnist-original.mat")
mat = [[elemtt for elemtt in elem]for elem in annots['data']]
label = annots['label']

print(len(mat))
print(label.shape)