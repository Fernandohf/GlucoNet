# Pytorch
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# Prepare the data as equaly spaced data
#data = bg_data.glucose.resample('5Min').mean()
#data = data.interpolate('linear')
data = pd.read_csv('time_series.csv')
print(data.isnull().sum())
print(data.shape)

data.head()


# In[131]:


def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''
    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


# In[136]:


batches = get_batches(data.values, 32, 288)
x, y = next(batches)


# In[137]:


x = x[:,:,np.newaxis]
print(x.shape)
x[1,:5,:]


# In[117]:


y = y[:,:,np.newaxis]
print(y.shape)
y[1,:5,:]


# In[174]:


# Model
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.n_hidden = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.hidden_dim = hidden_dim

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        
        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        
        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.contiguous().view(-1, self.hidden_dim)  
        
        # get final output 
        output = self.fc(r_out)
        #output.view(batch_size, -1, self.output_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden


# In[175]:


# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')


# In[176]:


test_rnn = RNN(input_size=1, output_size=1, hidden_dim=32, n_layers=2)
test_rnn


# In[177]:


batches = get_batches(data.values, 32, 288)
x, y = next(batches)
x = x[:,:,np.newaxis]

# test that dimensions are as expected00
test_rnn = RNN(input_size=1, output_size=1, hidden_dim=128, n_layers=2)

test_input = torch.Tensor(x) # give it a batch_size of 1 as first dimension

print('Input size: ', test_input.size())

# test out rnn sizes
test_out, test_h = test_rnn(test_input, None)
print('Output size: ', test_out.size())
print('Hidden state size: ', test_h.size())


# In[208]:


def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 
    
        Arguments
        ---------
        
        net: RNN network
        data: data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Sequence length
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    
    '''
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if(train_on_gpu):
        net.cuda()
    
    counter = 0
    for e in range(epochs):
        # initialize hidden state
        h = None#net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            
            # Reshape inputs and targets
            x = x[:, :, np.newaxis]
            y = y[:, :, np.newaxis]
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            output, h = net(inputs.float(), h)
            
            
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = h.data
            
            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), targets.view(batch_size*seq_length).float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = None#net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = x[:, :, np.newaxis]
                    y = y[:, :, np.newaxis]
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs.float(), val_h.float())
                    
                    val_h = val_h.data
                    
                    val_loss = criterion(output.squeeze(), targets.view(batch_size*seq_length).float())
                
                    val_losses.append(val_loss.item())
                
                net.train() # reset to train mode after iterationg through validation data
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(val_loss.items()))


# In[209]:


# define and print the net
input_size = 1
output_size = 1
n_hidden = 512
n_layers = 3

net = RNN(input_size, output_size, n_hidden, n_layers)
print(net)


# In[210]:


batch_size = 128
seq_length = 288
n_epochs = 250 # start smaller if you are just testing initial behavior

# train the model
train(net, data.values, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=50)

