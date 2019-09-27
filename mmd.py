import torch

def mmd_linear(f_of_X, f_of_Y):               #(source, target) (SmallMAE, BigMAE)
#    print('X size',f_of_X.size())
#    print('Y size',f_of_Y.size())
#    print('X type',f_of_X.type())
#    print('Y type',f_of_Y.type())a
    f_of_X = f_of_X.squeeze()
    f_of_Y = f_of_Y.squeeze()
#    f_of_X = f_of_X.reshape(512,1024)
#    f_of_Y = f_of_Y.reshape(512,1024)
    delta = f_of_X - f_of_Y
#    print('delta shape =',delta.shape)
#    delta = delta.detach().cpu().numpy()
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss
