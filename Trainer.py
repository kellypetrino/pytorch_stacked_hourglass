import torch
import numpy as np

class Trainer():
   def __init__(self,net=None,optim=None,loss_function=None, train_loader=None, device=None, filename=None):
      self.net = net
      self.optim = optim
      self.loss_function = loss_function
      self.train_loader = train_loader
      self.device=device
      self.filename = filename

   def train1(self,epochs):
      losses = []
      acc = []
      for epoch in range(epochs):
         epoch_loss = 0.0
         epoch_steps = 0
         tot_cor = 0
         tot = 0
         for data in self.train_loader:
            # Moves batch to (ideally) GPU
            X = data[0].to(self.device)
            y = data[1].to(self.device)
            
            # Zero the gradient in the optimizer, i.e. self.optim
            self.optim.zero_grad()

            # Get output of network!
            output = self.net(X)

            # Get loss (prbly using cross entropy loss function)
            loss = self.loss_function(output, y)

            # Backpropagate on the loss to compute gradients of parameters
            loss.backward()

            # Step optimizer aka update the gradients
            self.optim.step()


            _, pred = torch.max(output, 1)
            b = len(y[y==pred])
            c = len(y)
            tot_cor += b
            tot += c
            a = loss.detach()

            epoch_loss += a
            epoch_steps += 1
            print(f'loss, acc: {a}, {b/c*100}')
            losses.append(a)
            acc.append(b/c*100)
         # average loss of epoch
         print("epoch [%d]: finished")
         tot_acc = tot_cor/tot*100
         #tot_loss = epoch_loss / epoch_steps
         torch.save(self.net.state_dict(), f'our_models/{self.filename}_eps{epoch}_{tot_acc:.3f}.pt') # Save network labeled with val accuracy
         np.save(f'train_acc/{self.filename}_eps{epoch}_{tot_acc:.3f}.npy', np.array(acc))   # Save training accuracy
         np.save(f'train_loss/{self.filename}_eps{epoch}_{tot_acc:.3f}.npy', np.array(losses))   # Save training loss
      return losses, acc

   def train2(self,epochs):
      losses = []
      acc = []
      for epoch in range(epochs):
         epoch_loss = 0.0
         epoch_steps = 0
         tot_cor = 0
         tot = 0
         for data in self.train_loader:
            # Moves batch to (ideally) GPU
            X = data[0].to(self.device)
            Y = data[1].to(self.device)
            z = data[2].to(self.device)
            
            # Zero the gradient in the optimizer, i.e. self.optim
            self.optim.zero_grad()

            # Get output of network!
            output = self.net(X, Y)

            # Get loss (prbly using cross entropy loss function)
            loss = self.loss_function(output, z)

            # Backpropagate on the loss to compute gradients of parameters
            loss.backward()

            # Step optimizer aka update the gradients
            self.optim.step()


            _, pred = torch.max(output, 1)
            b = len(z[z==pred])
            c = len(z)
            tot_cor += b
            tot += c
            a = loss.detach()

            epoch_loss += a
            epoch_steps += 1
            print(f'loss, acc: {a}, {b/c*100}')
            losses.append(a)
            acc.append(b/c*100)
         # average loss of epoch
         print("epoch [%d]: finished")
         tot_acc = tot_cor/tot*100
         #tot_loss = epoch_loss / epoch_steps
         torch.save(self.net.state_dict(), f'our_models/{self.filename}_eps{epoch}_{tot_acc:.3f}.pt') # Save network labeled with val accuracy
         np.save(f'train_acc/{self.filename}_eps{epoch}_{tot_acc:.3f}.npy', np.array(acc))   # Save training accuracy
         np.save(f'train_loss/{self.filename}_eps{epoch}_{tot_acc:.3f}.npy', np.array(losses))   # Save training loss
      return losses, acc