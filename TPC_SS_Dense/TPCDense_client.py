from Common.Node.workerbase import WorkerBase
from Common.Utils.edcode import encode, decode
from Common.Grpc.fl_grpc_pb2 import IdxRequest_uint32, GradRequest_int32

import numpy as np

import torch
from torch import nn

import Common.config as config

from Common.Model.LeNet import LeNet
from Common.Utils.data_loader import load_data_fashion_mnist
from Common.Utils.set_log import setup_logging

import grpc
from Common.Grpc.fl_grpc_pb2_grpc import FL_GrpcStub

import argparse


class TPCDenseClient(WorkerBase):
    def __init__(self, client_id, model, loss_func, train_iter, test_iter, config, optimizer, stub1, stub2):
        super(TPCDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                       test_iter=test_iter, config=config, optimizer=optimizer)
        self.client_id = client_id
        self.stub1 = stub1
        self.stub2 = stub2

    def update(self):

        rst_grad1, rst_grad2 = self.update_grad()

        agg_gradients = np.array(np.array(rst_grad1) + np.array(rst_grad2), dtype='float') / self.config.gradient_frac

        super().set_gradients(agg_gradients.tolist())
        print("update once!")

  

    def update_grad(self):
        grad_float = super().get_gradients()

        grad_int = np.array(grad_float * self.config.gradient_frac, dtype='int')
        share_grad1 = np.random.randint(2**32, size=len(grad_float))
        share_grad2 = grad_int - share_grad1

        grad_upd_res1 = self.stub1.UpdateGrad_int32.future(
            GradRequest_int32(id=self.client_id, grad_ori=share_grad1.tolist()))
        grad_upd_res2 = self.stub2.UpdateGrad_int32(GradRequest_int32(id=self.client_id, grad_ori=share_grad2.tolist()))
        return grad_upd_res1.result().grad_upd, grad_upd_res2.grad_upd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clear_dense_client')
    parser.add_argument('-i', type=int, help="client's id")
    parser.add_argument('-t', type=int, default=10, help="train times locally")

    args = parser.parse_args()

    yaml_path = 'Log/log.yaml'
    setup_logging(default_path=yaml_path)

    model = LeNet()
    batch_size = 512
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, root='Data/MNIST')
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    target1 = config.server1_address + ":" + str(config.port1)
    target2 = config.server2_address + ":" + str(config.port2)

    with grpc.insecure_channel(target1) as channel1:
        with grpc.insecure_channel(target2) as channel2:
            print("connect success!")

            stub1 = FL_GrpcStub(channel1)
            stub2 = FL_GrpcStub(channel2)

            client = TPCDenseClient(client_id=args.i, model=model, loss_func=loss_func, train_iter=train_iter,
                              test_iter=test_iter, config=config, optimizer=optimizer, stub1=stub1, stub2=stub2)

            client.fl_train(times=args.t)
            client.write_acc_record(fpath="Eva_Res/kd_acc.txt", info="TPCSS_worker")
