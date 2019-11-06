
from torchvision import datasets, transforms

## YOUR CODE HERE ##
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_train = datasets.MNIST(root = 'hw8_data', transform = transformations, train = True, download = True)
mnist_test = datasets.MNIST(root = 'hw8_data', transform = transformations, train = False, download = True)



from torch.utils.data import DataLoader

## YOUR CODE HERE ##
train_loader = DataLoader(dataset = mnist_train, batch_size = 32, shuffle = True)
test_loader = DataLoader(dataset = mnist_test, batch_size = 32, shuffle = True)

from torch import nn


class OneLayerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OneLayerModel, self).__init__()
        ## YOUR CODE HERE ##
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        ## YOUR CODE HERE ##
        out = self.linear(x)

        return out



## YOUR CODE HERE ##
model = OneLayerModel(input_dim = 28*28, output_dim = 10)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
!rm -r 'logs/expt1'
writer = SummaryWriter('logs/expt1')

from torch.autograd import Variable


def test(model, loss_func, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = Variable(data.view(-1, 28 * 28))
            target = Variable(target)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += loss_func(output, target)
    rate = correct / len(test_loader.dataset)
    #     loss_avg = test_loss.item() / len(test_loader.dataset)

    return rate, test_loss.item()


def train(model, train_loader, val_loader, loss_func, opt, num_epochs=10, writer=None):
    index = 0
    #     batch_size = 32
    for epoch in range(num_epochs):
        model.train()
        print('current epoch = %d' % epoch)
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)
            #             images = images.view(-1, 28 * 28)
            #             labels = Variable(labels

            opt.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            opt.step()

            # validation
            total = 0
            correct = 0
            pred = outputs.max(1, keepdim=True)[1]
            correct = pred.eq(labels.view_as(pred)).sum().item()
            #             _, predicts = torch.max(outputs.data, 1)
            total = labels.size(0)
            #             print(correct)
            #             print(total)
            #             correct = (predicts == labels).sum()
            rate = correct / total
            #             print(epoch*total+i)
            #             print(index)
            writer.add_scalar('Train(loss)', loss, index)
            writer.add_scalar('Train(accuracy rate)', rate, index)
            index += 1
        #             writer.add_scalar('Train', {'loss':loss}, epoch*batch_size+i)
        #             writer.add_scalar('Train', {'accuracy rate':rate}, epoch*batch_size+i)
        #             writer.add_scalar('Train', loss.item(), rate)
        #             print(loss.item())
        #             print(rate)

        rate_val, loss_val = test(model, loss_func, val_loader)
        writer.add_scalar('Validation(loss)', loss_val, epoch)
        writer.add_scalar('Validation(accuracy rate)', rate_val, epoch)
        #         writer.add_scalar('Validation', {'total loss':loss_val}, epoch)
        #         writer.add_scalar('Validation', {'accuracy rate':rate_val}, epoch)
        print(loss_val)
        print(rate_val)
#         total_val = 0
#         correct_val = 0
#         for j, (images_val, labels_val) in enumerate(val_loader):
#             images_val = Variable(images.view(-1, 28 * 28))
#             labels_val = Variable(labels)
#             outputs_val = model(images_val)
#             loss_val = loss_func(outputs_val, labels_val)
# #             _, predicts_val = torch.max(outputs_val.data, 1)
#             pred_val = outputs.max(1, keepdim=True)[1]
#             total_val += labels_val.size(0)
#             correct_val = pred_val.eq(labels_val.view_as(pred_val)).sum().item()
#         rate_val = correct_val / total_val
#         rate_check = correct_val/10000
#         writer.add_scalar('Validation', loss_val.data.item(), rate_val)
#         print(loss_val.item())
#         print(rate_val)
#         print(rate_check)
#             if i % 100 == 0:
#                 print('current loss = %.5f' % loss.data[0])


class TwoLayerModel(nn.Module):
    ## YOUR CODE HERE ##
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerModel, self).__init__()
        #         input_dim = 28*28
        #         hidden_dim = 100
        #         output_dim = 10
        #         self.conv1 = nn.Sequential(
        #                      nn.Linear(28 * 28, 100),
        #                      nn.ReLU(),
        #                      nn.Linear(100, 10)
        #                      )
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

## YOUR CODE HERE ##
model2 = TwoLayerModel(28*28, 100, 10)
loss2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001)
!rm -r 'logs/expt2'
writer2 = SummaryWriter('logs/expt2')








