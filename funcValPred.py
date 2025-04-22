import networkClass
import random

#generates function dataset
def functionGeneration():
    dataSet = []
    r = 11
    for i in range(r):
        for j in range(r):
            dataSet.append((networkClass.Value(i-(r-1)/2), networkClass.Value(j-(r-1)/2), networkClass.Value((i-(r-1)/2)**2 + (j-(r-1)/2)**2)))
    return dataSet

def calcLoss(result, label):
    loss = networkClass.Value(0)
    for i in range(len(result)):
        loss += (label[i]-result[i]) ** 2

    return loss / len(result)

mlp = networkClass.MLP(2, [32, 32, 1])
#initialize dataset and labels
dataSet = []
labels = []
for d in functionGeneration():
    dataSet.append((d[0], d[1]))
    labels.append(d[2])

learning_rate = 0.001
epochs = 100

for epoch in range(epochs):
    mlp.zero_grad()
    #this needs changing, something to do with loss
    #testing all data, and i think the end loss is because its only outputting the end loss, therefore not improving
    #since output is size of 1
    #change this
    results = []
    
    #32 batch size rather than going through entire dataset (takes 32 random data points for training)
    batchSize = 32
    indices = random.sample(range(len(dataSet)), batchSize)
    batch_data = [dataSet[i] for i in indices]
    batch_labels = [labels[i] for i in indices]

    for data in batch_data:
        x = (data[0], data[1])
        results.append(mlp(x)[0])
    
    #BECAUSE ONLY USING 32 BATCHSIZE WITH RANDOM GENS, HAVE TO KEEP TRACK OF WHICH ONES TO CALC LOSS, CREATE LIST OF TUPLES
        #OF WHICH DATA IS USED< THEN TAKE THE CORRESPONDING LABELS AND CALCLOSS
    loss = calcLoss(results, batch_labels)
    #calc the gradients of each param, and if theyre neg(with respect to loss, therefore, if neg, means increasing will decrease loss)
    #if then increase or decrease depending on effects on loss
    loss.backward()
    for param in mlp.parameters():
        param -= learning_rate * param.grad
    

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
        print(f"result: {results[0].data} vs {batch_labels[0].data}")