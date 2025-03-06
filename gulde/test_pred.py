#
from anomaly_detection import mnist_train, modelB
from anomaly_detection import predict, predict_probs

import torch
from torch.utils.data import DataLoader

dataset = mnist_train
dl_b_1 = DataLoader(dataset, batch_size=1, shuffle=False)
dl_b_32 = DataLoader(dataset, batch_size=32, shuffle=False)
dl_b_all = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

t_b_1 = torch.cat([y for _, y in dl_b_1],dim=0)
t_b_32 = torch.cat([y for _, y in dl_b_32],dim=0)
t_b_all = torch.cat([y for _, y in dl_b_all],dim=0)

t_b_1.eq(dataset.targets).sum()
t_b_32.eq(dataset.targets).sum()
t_b_all.eq(dataset.targets).sum()

device = "cpu"
mdl_orig = modelB()
mdl_orig.load_state_dict(torch.load(path_in,map_location=device))
mdl_orig.to(device)
mdl_orig.eval()

pred_probs_b_1 = predict_probs(mdl_orig,dl_b_1,device)
pred_probs_b_32 = predict_probs(mdl_orig,dl_b_32,device)
pred_probs_b_all = predict_probs(mdl_orig,dl_b_all,device)
   
pred_probs_b_1 = predict_probs(mdl_orig,dl_b_1,device)
pred_probs_b_32 = predict_probs(mdl_orig,dl_b_32,device)
pred_probs_b_all = predict_probs(mdl_orig,dl_b_all,device)

pred_b_32.eq(t_b_32).sum()
pred_probs_b_32.argmax(dim=1).eq(pred_b_32.squeeze()).sum()
pred_probs_b_32.argmax(dim=1).eq(t_b_32).sum()


def testmodel_probs(model, testdataX):
    ''' - testdataX holds all the images in the test dataset
        - testdataY holds all the labels for the images in the test dataset
    '''
    model.eval()
    testing = testdataX.to(torch.float32)
    with torch.no_grad():
        y_eval = model.forward(testing)
        return torch.softmax(y_eval,dim=1)

test = torch.cat([x for x, _ in dl_b_32],dim=0)

dataset.data[:32,:,:].eq(next(iter(dl_b_32))[0].view_as(dataset.data[:32,:,:])).sum()



def train_val(model_class, dataset, frac_train=0.8, num_epochs=100, batch_size=32, device='cuda', path_out_model=None):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), train_size=frac_train)
    train_loader, val_loader = create_data_loaders(dataset, (train_idx, val_idx), batch_size=batch_size)

    model = model_class().to(device)
    loss_fn, optimizer = get_loss_and_optimizer(model)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    if path_out_model is not None:
        safe_torch_save(model,path_out_model)

    return model, (val_loss, val_accuracy)


from anomaly_detection import modelB, mnist_train, mnist_test
# from example import train, test
from anomaly_detection import train, test, predict_probs_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "mps"
loss_fn = nn.NLLLoss() # reduction="sum") # nn.CrossEntropyLoss() # 
mdl_orig = modelB()
mdl_orig.load_state_dict(torch.load("mnist_cnn.pt",map_location=device))
mdl_orig.to(device)
mdl_orig.eval()
loader_train = torch.utils.data.DataLoader(mnist_train, batch_size=32, shuffle=False)
test(mdl_orig, device, loader_train, loss_fn)
loader_test = torch.utils.data.DataLoader(mnist_test, batch_size=32, shuffle=False)
test(mdl_orig, device, loader_test, loss_fn)

probs = predict_probs_loader(mdl_orig, loader_test, device)
