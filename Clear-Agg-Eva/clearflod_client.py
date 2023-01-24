from Common.Node.workerbase import WorkerBase
from Common.Grpc.fl_grpc_pb2 import signSGD_Request
from Common.Utils.edcode import encode, decode
import torch
from torch import nn
import numpy as np
import Common.config as config

from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Utils.data_loader import load_data_mnist, load_data_cifar10
from Common.Utils.set_log import setup_logging
from Common.Utils.options import args_parser
import grpc
from Common.Grpc.fl_grpc_pb2_grpc import FL_GrpcStub



class ClearSignSGDClient(WorkerBase):
    def __init__(self, client_id, model, loss_func, train_iter, test_iter, config, optimizer, device, grad_stub):
        super(ClearSignSGDClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer, device=device)
        self.client_id = client_id
        self.grad_stub = grad_stub

    def update(self):
        gradients = np.array(super().get_gradients())
        sgn = np.where(gradients>=0, 0, 1).tolist()
        if self.client_id > 10:
            #gradients = np.random.normal(0, 0.1, self._grad_len)
            sgn = np.random.randint(0,2,self._grad_len).tolist()
        res_sgn_upd = self.grad_stub.Update_SignSGD.future(signSGD_Request(id=self.client_id, sgn_ori=sgn))
        res_sgn = res_sgn_upd.result().sgn_upd
        assert len(res_sgn) == self._grad_len
        res = res_sgn
        #res = np.where(np.array(res_sgn)==1, -1.0, 1.0).tolist()
        super().set_gradients(gradients=res)


if __name__ == '__main__':
    args = args_parser()
    if args.id < 5:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    yaml_path = 'Log/log.yaml'
    setup_logging(default_path=yaml_path)

    PATH = './Model/ResNet20'
    model = ResNet(BasicBlock, [3,3,3]).to(device)
    model.load_state_dict(torch.load(PATH))
    if args.id == 0:
        train_iter, test_iter = load_data_cifar10(id=args.id, batch = args.batch_size, path = args.path)
    else:
        train_iter, test_iter = load_data_cifar10(id=args.id, batch = args.batch_size, path = args.path), None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    server_grad = config.server1_address + ":" + str(config.port1)

    with grpc.insecure_channel(server_grad, options=config.grpc_options) as grad_channel:
        print("connect success!")

        sgn_stub = FL_GrpcStub(grad_channel)

        client = ClearSignSGDClient(client_id=args.id, model=model, loss_func=loss_func, train_iter=train_iter,
                                  test_iter=test_iter, config=config, optimizer=optimizer, device=device, grad_stub=sgn_stub)

        client.fl_train(times=args.E)
        client.write_acc_record(fpath="Eva/clear_signtrust_acc_cifar10.txt", info="clear_signtrust_acc_worker")
