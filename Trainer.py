from DataLoader import DataLoader
from Model import Seq2SeqModel
import torch


class Trainer(object):
    def __init__(self,
                 json_folder='/home/goktug/projects/tracking_predictor/waymo_valid_0',
                 input_timestamp_length=20,
                 output_timestamp_length=10,
                 ratio_train_set=0.7,
                 ratio_valid_set=0.15,
                 batch_size_train=32,
                 batch_size_valid=32,
                 batch_size_test=32,
                 epoch=100):
        data_loader = DataLoader(input_timestamp_length, output_timestamp_length, json_folder)
        X, Y = data_loader.LoadDateset()

        self.epoch = epoch
        self.input_timestamp_length = input_timestamp_length
        self.output_timestamp_length = output_timestamp_length

        dataset = []
        for sample_id in range(X.shape[0]):
            x_and_y = [X[sample_id], Y[sample_id]]
            dataset.append(x_and_y)

        length_dataset = len(dataset)
        self.ratio_train_set = 0.7
        self.ratio_valid_set = 0.15

        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.batch_size_test = batch_size_test

        size_train_set = int(length_dataset * ratio_train_set)
        size_valid_set = int(length_dataset * ratio_valid_set)
        size_test_set = length_dataset - size_train_set - size_valid_set

        train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [size_train_set, size_valid_set,
                                                                                      size_test_set])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True, drop_last=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size_valid, shuffle=True,
                                                        drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True, drop_last=True)

        # Make list loaders:
        self.train_loader = [batch for batch in train_loader]
        self.train_batch_count = len(train_loader)

        self.validation_loader = [batch for batch in validation_loader]
        self.valid_batch_count = len(validation_loader)

        self.test_loader = [batch for batch in test_loader]
        self.test_batch_count = len(test_loader)

        print("Batch size train:", batch_size_train, "| Batch size valid: ", batch_size_valid, "| Batch size test: ",
              batch_size_test)
        print("Batch count train:", self.train_batch_count, "| Batch count valid: ", self.valid_batch_count,
              "| Batch count test: ", self.test_batch_count)

    def Train(self, use_gpu=True,
              lstm_output_feature_size=100,
              num_lstm_layer=24):
        model = -1
        device = -1
        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("Device: ", device)
            if torch.cuda.is_available():
                model = Seq2SeqModel(input_dim=4,
                                     lstm_output_feature_size=lstm_output_feature_size,
                                     num_lstm_layers=num_lstm_layer,
                                     output_dim=4 * self.output_timestamp_length,
                                     use_gpu=use_gpu,
                                     device=device).to(device)
                print("GPU is available.")
                print(torch.cuda.device_count())
                print(torch.cuda.get_device_name())

        else:
            model = Seq2SeqModel(input_dim=4,
                                 lstm_output_feature_size=lstm_output_feature_size,
                                 num_lstm_layers=num_lstm_layer,
                                 output_dim=4 * self.output_timestamp_length,
                                 use_gpu=use_gpu,
                                 device=device)
            print("CPU")

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(self.epoch):
            print("Epoch:", epoch)
            loss_training = 0
            for batch_id in range(len(self.train_loader)):
                #print("Batch id:", batch_id)
                x, y = self.train_loader[batch_id]
                x, y = x.float(), y.float()
                if use_gpu:
                    if torch.cuda.is_available():
                        x, y = x.to(device), y.to(device)
                # zero the optimizer gradients
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                loss_training += loss.item()
                #print("Loss: ", loss)
            print("Training loss per batch: ", loss_training/self.train_batch_count)

            loss_valid = 0
            for batch_id in range(len(self.validation_loader)):
                x, y = self.validation_loader[batch_id]
                x, y = x.float(), y.float()
                if use_gpu:
                    if torch.cuda.is_available():
                        x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    #model.eval() ahah it doesn't work with GPU, but it is fast enough with no grad :)
                    pred = model(x)
                    loss = criterion(pred, y)
                    loss_valid += loss.item()
            print("Validation loss per batch: ", loss_valid/self.valid_batch_count)
            print("*********************************************************************")


if __name__ == "__main__":
    trainer = Trainer(json_folder='/home/goktug/projects/tracking_predictor/waymo_valid_0',
                      input_timestamp_length=20,
                      output_timestamp_length=10,
                      ratio_train_set=0.7,
                      ratio_valid_set=0.15,
                      batch_size_train=24,
                      batch_size_valid=24,
                      batch_size_test=32)
    trainer.Train(use_gpu=True,
                  lstm_output_feature_size=500,
                  num_lstm_layer=32)
