import datetime
import json
import os.path

import torch
import torch.nn as nn

CNN_BASIC = [
    {
        "layer": "Conv2d",
        "params": {
            "in_channels": 3,
            "out_channels": 32,
            "kernel_size": 4,
            "stride": 1,
            "padding": 0,
        },
    },
    {"layer": "ReLU", "params": {}},
    {
        "layer": "MaxPool2d",
        "params": {
            "kernel_size": 2,
            "stride": 1,
            "padding": 0,
        }
    },
    {
        "layer": "Conv2d",
        "params": {
            "in_channels": 32,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
        },
    },
    {"layer": "ReLU", "params": {}},
    {
        "layer": "Conv2d",
        "params": {
            "in_channels": 64,
            "out_channels": 128,
            "kernel_size": 3,
            "stride": 2,
            "padding": 0,
        },
    },
    {
        "layer": "BatchNorm2d",
        "params": {
            "num_features": 128
        }
    },
    {"layer": "ReLU", "params": {}},
    {
        "layer": "Conv2d",
        "params": {
            "in_channels": 128,
            "out_channels": 256,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
        },
    },
    {"layer": "ReLU", "params": {}},
    {
        "layer": "MaxPool2d",
        "params": {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
        }
    },
]


def parse_crrn_config(config):
    layers = []
    for layer_info in config:
        layer_type = layer_info["layer"]
        params = layer_info["params"]
        if layer_type == "Conv2d":
            layers.append(nn.Conv2d(**params))
        elif layer_type == "ReLU":
            layers.append(nn.ReLU())
        elif layer_type == "MaxPool2d":
            layers.append(nn.MaxPool2d(**params))
        elif layer_type == "Flatten":
            layers.append(nn.Flatten())
        elif layer_type == "Linear":
            layers.append(nn.Linear(**params))
        elif layer_type == "BatchNorm2d":
            layers.append(nn.BatchNorm2d(**params))
        elif layer_type == "Dropout":
            layers.append(nn.Dropout(**params))
        elif layer_type == "LSTM":
            layers.append(nn.LSTM(**params))
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    return nn.Sequential(*layers)


def calc_dimensions(input_size, kernel_size, stride, padding, dilation=1):
    return int(
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def calc_dimensions_final(in_dim, model_config: list):
    for layer in model_config:
        l_name = layer["layer"]
        if l_name == "Dropout" or l_name == "ReLU":
            continue
        elif l_name == "Conv2d" or l_name == "MaxPool2d":
            k = layer["params"].get("kernel_size", 1)
            p = layer["params"].get("padding", 0)
            s = layer["params"].get("stride", 1)
            in_dim = calc_dimensions(in_dim, k, s, p)
        elif l_name == "AdaptiveAvgPool2d":
            output_size = layer["params"]["output_size"]
            in_dim = output_size
        elif l_name == "BatchNorm2d":
            in_dim = in_dim
        else:
            raise ValueError("Unsupported layer type in CNN config: {}".format(l_name))

    return in_dim


def get_final_out_channels(model_config: list):
    out_channels = 0
    for layer in model_config:
        l_name = layer["layer"]
        if l_name == "Conv2d":
            out_channels = layer["params"]["out_channels"]

    return out_channels


class CRNN(nn.Module):
    def __init__(
        self,
        img_h,
        img_w,
        max_seq_len=6,
        in_dimensions_rnn=265,
        hidden_dimensions_rnn=128,
        num_classes=36,
    ):
        super().__init__()
        self.img_w_final = calc_dimensions_final(img_w, CNN_BASIC)
        self.img_h_final = calc_dimensions_final(img_h, CNN_BASIC)
        self.out_channels_cnn = get_final_out_channels(CNN_BASIC)
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.hidden_dimensions_rnn = hidden_dimensions_rnn

        self.cnn = parse_crrn_config(CNN_BASIC)
        self.flattening_layers = nn.ModuleList(
            [nn.Linear(self.out_channels_cnn * self.img_h_final, in_dimensions_rnn), nn.SiLU()]
        )
        self.rnn = nn.LSTM(in_dimensions_rnn, self.hidden_dimensions_rnn, bidirectional=True)
        # hidden_dimensions_rnn * 2 because of bidirectional RNN
        self.fc_out = nn.Linear(
            self.hidden_dimensions_rnn * 2, self.num_classes
        )

    def forward(self, x):
        x = self.cnn(x)
        # Bring the width dimension to the front for RNN processing
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        for m in self.flattening_layers:
            x = m(x)
        # TODO: add Dropout here
        x, (_, _) = self.rnn(x)
        x = self.fc_out(x)

        return x

    def save_status(self, data_dir, params: dict):
        datetime_stamp = datetime.datetime.now().isoformat()
        folderpath = os.path.join(data_dir, "models", datetime_stamp)
        os.makedirs(folderpath, exist_ok=True)

        state_dict_filepath = os.path.join(folderpath, "crnn_model.pth")
        torch.save(self.state_dict(), state_dict_filepath)

        params_filepath = os.path.join(folderpath, "params.json")
        with open(params_filepath, "w") as f:
            if params is not None:
                json.dump(params, f)


if __name__ == "__main__":
    # Example usage
    model = CRNN(
        img_h=60,
        img_w=160,
        in_dimensions_rnn=256,
        hidden_dimensions_rnn=128,
        num_classes=64,
    )

    # Dummy input tensor with shape (batch_size, channels, height, width)
    dummy_input = torch.randn(64, 3, 60, 160)  # Batch size of 8
    output = model(dummy_input)
    assert model.img_w_final == output.shape[0], f"Expected output width {model.img_w_final}, got {output.shape[0]}"
