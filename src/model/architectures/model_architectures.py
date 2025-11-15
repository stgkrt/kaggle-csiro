import torch
from torch import nn
from torch.nn import functional as F

from src.configs import ModelConfig
from src.model.architectures.branch_cnn_aux_meta_model import BranchCNNAuxMetaModel
from src.model.architectures.branch_cnn_half_aux_model import (
    BranchCNNHalfAuxModel,
    BranchCNNHalfAuxModel_2,
)
from src.model.architectures.branch_cnn_half_splithead_aux_model import (
    BranchCNNHalfAuxSplitHeadModel,
)
from src.model.architectures.branch_cnn_lstm_aux_model import BranchCNNLSTMAuxModel
from src.model.architectures.branch_cnn_lstm_model import BranchCNNLSTMModel
from src.model.architectures.branch_cnn_model import BranchCNNModel
from src.model.architectures.branch_trans_cnn_aux_model import BranchTransCNNAuxModel
from src.model.architectures.cnn_lstm_model import CNNLSTMModel
from src.model.architectures.cnn_transformer_model import CNNTransformerModel
from src.model.architectures.each_branch_cnn_aux_model import EachBranchCNNAuxModel
from src.model.architectures.each_branch_cnn_model import EachBranchCNNModel
from src.model.architectures.each_branch_trans_model import EachBranchTransModel
from src.model.architectures.imu_cnn_half_aux_model import ImuCNNHalfAuxModel
from src.model.architectures.many_branch_cnn_aux_model import ManyBranchCNNAuxModel
from src.model.architectures.many_branch_cnn_model import ManyBranchCNNModel
from src.model.architectures.public_model import PublicIMUModel, PublicModel
from src.model.architectures.simple_model import SimpleCNNModel
from src.model.architectures.spec_model import SpecModel


class ModelArchitectures(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(ModelArchitectures, self).__init__()
        self.config = model_config
        self.model_name = model_config.model_name
        self.model = self._get_model()

    def _get_model(self):
        if self.model_name == "public_model":
            model = PublicModel(
                pad_len=self.config.pad_len,
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                n_classes=self.config.n_classes,
            )
        elif self.model_name == "public_imu_model":
            model = PublicIMUModel(
                pad_len=self.config.pad_len,
                imu_dim=self.config.imu_dim,
                n_classes=self.config.n_classes,
            )
        elif self.model_name == "simple_cnn_model":
            model = SimpleCNNModel(
                imu_dim=self.config.imu_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "cnn_lstm_model":
            model = CNNLSTMModel(
                imu_dim=self.config.imu_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "cnn_transformer_model":
            model = CNNTransformerModel(
                imu_dim=self.config.imu_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "each_branch_cnn_model":
            model = EachBranchCNNModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "each_branch_trans_model":
            model = EachBranchTransModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "branch_cnn_model":
            model = BranchCNNModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "each_branch_cnn_aux_model":
            model = EachBranchCNNAuxModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "branch_cnn_half_aux_model":
            model = BranchCNNHalfAuxModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "branch_cnn_half_aux_model_2":
            model = BranchCNNHalfAuxModel_2(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )

        elif self.model_name == "spec_model":
            model = SpecModel(
                imu_dim=self.config.imu_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
                height=64,  # Example height, adjust as needed
                hop_length=None,  # Example hop_length, adjust as needed
                win_length=None,  # Example win_length, adjust as needed
            )
        elif self.model_name == "branch_cnn_aux_meta_model":
            model = BranchCNNAuxMetaModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                meta_dim=7,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "many_branch_cnn_model":
            model = ManyBranchCNNModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "many_branch_cnn_aux_model":
            model = ManyBranchCNNAuxModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "branch_cnn_lstm_model":
            model = BranchCNNLSTMModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "branch_cnn_lstm_aux_model":
            model = BranchCNNLSTMAuxModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "branch_trans_cnn_aux_model":
            model = BranchTransCNNAuxModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "branch_cnn_half_aux_split_head_model":
            model = BranchCNNHalfAuxSplitHeadModel(
                imu_dim=self.config.imu_dim,
                tof_dim=self.config.tof_dim,
                thm_dim=self.config.thm_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        elif self.model_name == "imu_cnn_half_aux_model":
            model = ImuCNNHalfAuxModel(
                imu_dim=self.config.imu_dim,
                n_classes=self.config.n_classes,
                default_emb_dim=self.config.default_emb_dim,
                layer_num=self.config.layer_num,
            )
        else:
            raise NotImplementedError
        return model

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    config = ModelConfig(
        _target_="src.model.architectures.model_architectures.ModelArchitectures",
        # model_name="each_branch_trans_model",  # Example model name
        # model_name="branch_cnn_model",  # Example model name
        # model_name="each_branch_cnn_aux_model",  # Example model name
        # model_name="many_branch_cnn_model",  # Example model name
        # model_name="branch_cnn_half_aux_model",  # Example model name
        # model_name="branch_cnn_lstm_model",
        # model_name="branch_cnn_lstm_aux_model",
        # model_name="branch_trans_cnn_aux_model",
        # model_name="branch_cnn_half_aux_split_head_model",
        model_name="imu_cnn_half_aux_model",
        pad_len=127,  # Example pad length
        imu_dim=11,  # Example IMU dimension
        tof_dim=25,  # Example ToF dimension
        thm_dim=5,  # Example thermal dimension
        n_classes=18,  # Example number of classes
        loss_config=None,  # type: ignore
        optimizer=None,  # type: ignore
        scheduler=None,  # type: ignore
        default_emb_dim=64,  # Example custom default_emb_dim
        layer_num=3,  # Example custom layer_num
    )
    batch_size = 32
    model = ModelArchitectures(config)

    if config.model_name == "public_model":
        print("-- Using PublicModel with IMU and ToF features --")
        print("Using PublicModel with IMU and ToF features")
        # Example input tensor with shape (batch_size, pad_len, imu_dim + tof_dim)
        # Assuming pad_len=100, imu_dim=6, tof_dim=4
        # Adjust the dimensions according to your model's requirements
        input = torch.randn(batch_size, config.pad_len, config.imu_dim + config.tof_dim)
        input = {"features": input}  # Wrap in a dictionary as expected by the model
        print(f"Input shape: {input['features'].shape}")

        output = model(input)

        print(f"Output shape: {output['logits'].shape}")
    elif config.model_name == "public_imu_model":
        print("-- Using PublicIMUModel with IMU features only --")
        print("Using PublicIMUModel with IMU features only")
        # Example input tensor with shape (batch_size, pad_len, imu_dim)
        input_imu = torch.randn(batch_size, config.pad_len, config.imu_dim)
        input_imu = {"features": input_imu}
        print(f"Input IMU shape: {input_imu['features'].shape}")
        output_imu = model(input_imu)
        print(f"Output IMU shape: {output_imu['logits'].shape}")
    elif config.model_name == "simple_cnn_model":
        print("-- Using SimpleCNNModel with IMU features --")
        print("Using SimpleCNNModel with IMU features")
        # Example input tensor with shape (batch_size, pad_len, imu_dim)
        input_simple = torch.randn(batch_size, config.pad_len, config.imu_dim)
        input_simple = {"features": input_simple}
        print(f"Input Simple shape: {input_simple['features'].shape}")
        output_simple = model(input_simple)
        print(f"Output Simple shape: {output_simple['logits'].shape}")
    elif config.model_name == "cnn_lstm_model":
        print("-- Using CNNLSTMModel with IMU features --")
        print("Using CNNLSTMModel with IMU features")
        # Example input tensor with shape (batch_size, pad_len, imu_dim)
        input_cnn_lstm = torch.randn(batch_size, config.pad_len, config.imu_dim)
        input_cnn_lstm = {"features": input_cnn_lstm}
        print(f"Input CNN-LSTM shape: {input_cnn_lstm['features'].shape}")
        output_cnn_lstm = model(input_cnn_lstm)
        print(f"Output CNN-LSTM shape: {output_cnn_lstm['logits'].shape}")
    elif config.model_name == "cnn_transformer_model":
        print("-- Using CNNTransformerModel with IMU features --")
        print("Using CNNTransformerModel with IMU features")
        # Example input tensor with shape (batch_size, pad_len, imu_dim)
        input_cnn_transformer = torch.randn(batch_size, config.pad_len, config.imu_dim)
        input_cnn_transformer = {"features": input_cnn_transformer}
        print(f"Input CNN-Transformer shape: {input_cnn_transformer['features'].shape}")
        output_cnn_transformer = model(input_cnn_transformer)
        print(f"Output CNN-Transformer shape: {output_cnn_transformer['logits'].shape}")
    elif config.model_name == "spec_model":
        print("-- Using SpecModel with IMU features --")
        print("Using SpecModel with IMU features")
        # Example input tensor with shape (batch_size, imu_dim, sequence_length)
        input_spec = torch.randn(
            batch_size, config.pad_len, config.imu_dim
        )  # Example sequence length
        input_spec = {"features": input_spec}
        output_spec = model(input_spec)
        print(f"Output Spec shape: {output_spec['logits'].shape}")
    elif config.model_name == "each_branch_cnn_model":
        print("-- Using EachBranchCNNModel with IMU, ToF, and Thm features --")
        print("Using EachBranchCNNModel with IMU, ToF, and Thm features")
        # Example input tensors for each branch
        imu_input = torch.randn(batch_size, config.pad_len, config.imu_dim)
        tof_input = torch.randn(batch_size, config.pad_len, config.tof_dim)
        thm_input = torch.randn(batch_size, config.pad_len, config.thm_dim)
        dummy_input = {
            "imu_features": imu_input,
            "tof_features": tof_input,
            "thm_features": thm_input,
        }
        output_each_branch_cnn = model(dummy_input)
        print(f"Output EachBranchCNN shape: {output_each_branch_cnn['logits'].shape}")
    elif config.model_name == "each_branch_trans_model":
        print("-- Using EachBranchTransModel with IMU, ToF, and Thm features --")
        print("Using EachBranchTransModel with IMU, ToF, and Thm features")
        # Example input tensors for each branch
        imu_input = torch.randn(batch_size, config.pad_len, config.imu_dim)
        tof_input = torch.randn(batch_size, config.pad_len, config.tof_dim)
        thm_input = torch.randn(batch_size, config.pad_len, config.thm_dim)
        dummy_input = {
            "imu_features": imu_input,
            "tof_features": tof_input,
            "thm_features": thm_input,
        }
        output_each_branch_trans = model(dummy_input)
        print(
            f"Output EachBranchTrans shape: {output_each_branch_trans['logits'].shape}"
        )
    elif config.model_name == "branch_cnn_model":
        print("-- Using BranchCNNModel with IMU, ToF, and Thm features --")
        print("Using BranchCNNModel with IMU, ToF, and Thm features")
        # Example input tensors for each branch
        imu_input = torch.randn(batch_size, config.pad_len, config.imu_dim)
        tof_input = torch.randn(batch_size, config.pad_len, config.tof_dim)
        thm_input = torch.randn(batch_size, config.pad_len, config.thm_dim)
        dummy_input_branch_cnn = {
            "imu_features": imu_input,
            "tof_features": tof_input,
            "thm_features": thm_input,
        }
        output_branch_cnn = model(dummy_input_branch_cnn)
        print(f"Output BranchCNN shape: {output_branch_cnn['logits'].shape}")
    elif config.model_name == "each_branch_cnn_aux_model":
        print("-- Using EachBranchCNNAuxModel with IMU, ToF, and Thm features --")
        print("Using EachBranchCNNAuxModel with IMU, ToF, and Thm features")
        # Example input tensors for each branch
        imu_input = torch.randn(batch_size, config.pad_len, config.imu_dim)
        tof_input = torch.randn(batch_size, config.pad_len, config.tof_dim)
        thm_input = torch.randn(batch_size, config.pad_len, config.thm_dim)
        dummy_input_each_branch_cnn_aux = {
            "imu_features": imu_input,
            "tof_features": tof_input,
            "thm_features": thm_input,
        }
        output_each_branch_cnn_aux = model(dummy_input_each_branch_cnn_aux)
        print(
            "Output EachBranchCNNAux shape:"
            f" {output_each_branch_cnn_aux['logits'].shape}"
        )
        print(
            "Output Orientation shape:", output_each_branch_cnn_aux["orientation"].shape
        )
        print("Output Behavior shape:", output_each_branch_cnn_aux["behavior"].shape)
    elif config.model_name == "branch_cnn_aux_model":
        print("-- Using BranchCNNAuxMetaModel with IMU, ToF, Thm, and Meta features --")
        print("Using BranchCNNAuxMetaModel with IMU, ToF, Thm, and Meta features")
        # Example input tensors for each branch
        imu_input = torch.randn(batch_size, config.pad_len, config.imu_dim)
        tof_input = torch.randn(batch_size, config.pad_len, config.tof_dim)
        thm_input = torch.randn(batch_size, config.pad_len, config.thm_dim)
        meta_input = torch.randn(batch_size, 7)
        dummy_input_branch_cnn_aux_meta = {
            "imu_features": imu_input,
            "tof_features": tof_input,
            "thm_features": thm_input,
            "meta_features": meta_input,
        }
        output_branch_cnn_aux_meta = model(dummy_input_branch_cnn_aux_meta)
        print(
            "Output BranchCNNAuxMeta shape:"
            f" {output_branch_cnn_aux_meta['logits'].shape}"
        )
        print(
            "Output Orientation shape:", output_branch_cnn_aux_meta["orientation"].shape
        )
        print("Output Behavior shape:", output_branch_cnn_aux_meta["behavior"].shape)
    elif config.model_name == "many_branch_cnn_model":
        print(
            "-- Using ManyBranchCNNModel with IMU, ToF, Thm, and Other IMU features --"
        )
        print("Using ManyBranchCNNModel with IMU, ToF, Thm, and Other IMU features")
        # Example input tensors for each branch
        imu_input = torch.randn(batch_size, config.pad_len, 29)
        tof_input = torch.randn(batch_size, config.pad_len, config.tof_dim)
        thm_input = torch.randn(batch_size, config.pad_len, config.thm_dim)
        dummy_input_many_branch_cnn = {
            "imu_features": imu_input,
            "tof_features": tof_input,
            "thm_features": thm_input,
        }
        output_many_branch_cnn = model(dummy_input_many_branch_cnn)
        print(f"Output ManyBranchCNN shape: {output_many_branch_cnn['logits'].shape}")
    elif config.model_name == "branch_cnn_half_aux_model":
        print(
            "- Using BranchCNNHalfAuxModel with IMU, ToF, Thm, and Half IMU features -"
        )
        print("Using BranchCNNHalfAuxModel with IMU, ToF, Thm, and Half IMU features")
        # Example input tensors for each branch
        imu_input = torch.randn(batch_size, config.pad_len, config.imu_dim)
        tof_input = torch.randn(batch_size, config.pad_len, config.tof_dim)
        thm_input = torch.randn(batch_size, config.pad_len, config.thm_dim)
        dummy_input_branch_cnn_half_aux = {
            "imu_features": imu_input,
            "tof_features": tof_input,
            "thm_features": thm_input,
        }
        output_branch_cnn_half_aux = model(dummy_input_branch_cnn_half_aux)
        print(
            "Output BranchCNNHalfAux shape:"
            f" {output_branch_cnn_half_aux['logits'].shape}"
        )
        print(
            "Output Orientation shape:", output_branch_cnn_half_aux["orientation"].shape
        )
        print("Output Behavior shape:", output_branch_cnn_half_aux["behavior"].shape)
    elif config.model_name == "branch_cnn_lstm_model":
        print("-- Using BranchCNNLSTMModel with IMU, ToF, Thm features --")
        print("Using BranchCNNLSTMModel with IMU, ToF, Thm features")
        # Example input tensors for each branch
        imu_input = torch.randn(batch_size, config.pad_len, config.imu_dim)
        tof_input = torch.randn(batch_size, config.pad_len, config.tof_dim)
        thm_input = torch.randn(batch_size, config.pad_len, config.thm_dim)
        dummy_input_branch_cnn_lstm = {
            "imu_features": imu_input,
            "tof_features": tof_input,
            "thm_features": thm_input,
        }
        output_branch_cnn_lstm = model(dummy_input_branch_cnn_lstm)
        print(f"Output BranchCNNLSTM shape: {output_branch_cnn_lstm['logits'].shape}")
    elif config.model_name == "branch_cnn_lstm_aux_model":
        print("-- Using BranchCNNLSTMAuxModel with IMU, ToF, Thm features --")
        print("Using BranchCNNLSTMAuxModel with IMU, ToF, Thm features")
        # Example input tensors for each branch
        imu_input = torch.randn(batch_size, config.pad_len, config.imu_dim)
        tof_input = torch.randn(batch_size, config.pad_len, config.tof_dim)
        thm_input = torch.randn(batch_size, config.pad_len, config.thm_dim)
        dummy_input_branch_cnn_lstm_aux = {
            "imu_features": imu_input,
            "tof_features": tof_input,
            "thm_features": thm_input,
        }
        output_branch_cnn_lstm_aux = model(dummy_input_branch_cnn_lstm_aux)
        print(f"Output shape: {output_branch_cnn_lstm_aux['logits'].shape}")

    elif config.model_name == "branch_trans_cnn_aux_model":
        print("-- Using BranchTransCNNAuxModel with IMU, ToF, Thm features --")
        print("Using BranchTransCNNAuxModel with IMU, ToF, Thm features")
        # Example input tensors for each branch
        imu_input = torch.randn(batch_size, config.pad_len, config.imu_dim)
        tof_input = torch.randn(batch_size, config.pad_len, config.tof_dim)
        thm_input = torch.randn(batch_size, config.pad_len, config.thm_dim)
        dummy_input_branch_trans_cnn_aux = {
            "imu_features": imu_input,
            "tof_features": tof_input,
            "thm_features": thm_input,
        }
        output_branch_trans_cnn_aux = model(dummy_input_branch_trans_cnn_aux)
        print(f"Output shape: {output_branch_trans_cnn_aux['logits'].shape}")
    elif config.model_name == "branch_cnn_half_aux_split_head_model":
        print("-- Using BranchCNNHalfAuxSplitHeadModel with IMU, ToF, Thm features --")
        print("Using BranchCNNHalfAuxSplitHeadModel with IMU, ToF, Thm features")
        # Example input tensors for each branch
        imu_input = torch.randn(batch_size, config.pad_len, config.imu_dim)
        tof_input = torch.randn(batch_size, config.pad_len, config.tof_dim)
        thm_input = torch.randn(batch_size, config.pad_len, config.thm_dim)
        dummy_input_branch_cnn_half_aux_split_head = {
            "imu_features": imu_input,
            "tof_features": tof_input,
            "thm_features": thm_input,
        }
        output_branch_cnn_half_aux_split_head = model(
            dummy_input_branch_cnn_half_aux_split_head
        )
        print(f"Output shape: {output_branch_cnn_half_aux_split_head['logits'].shape}")
