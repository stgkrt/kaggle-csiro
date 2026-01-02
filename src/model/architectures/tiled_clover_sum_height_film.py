import timm
import torch
from model.architectures.model_blocks import FiLM
from torch import nn


class TiledCloverSumHeightFilmModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_channels: int,
        n_classes: int,
        emb_dim: int = 32,
        dropout: float = 0.2,
        drop_path_rate: float = 0.3,
        head_connection_type: str = "direct",
    ):
        super(TiledCloverSumHeightFilmModel, self).__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.head_connection_type = head_connection_type
        self.film = FiLM(self.model.num_features)
        self.target_emb = nn.Sequential(
            nn.Linear(self.model.num_features, emb_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
        )
        self.target_head = nn.Sequential(
            nn.Linear(emb_dim, n_classes - 2),
            nn.Softplus(),
        )
        self.clover_classification_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
        )
        self.height_head = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Softplus(),
        )

    def _grid_split(
        self, img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """画像をn×mのグリッドに分割し、各グリッドを別々に処理するための関数

        Args:
            img (torch.Tensor): 入力画像テンソル (B, C, H, W)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
              分割されたグリッド画像テンソルのタプル
        """
        # n = 2
        # m = 2
        # B, C, H, W = img.shape
        # grid_h = H // n
        # grid_w = W // m
        # grids = []
        # for i in range(n):
        #     for j in range(m):
        #         grid = img[
        #             :, :, i * grid_h : (i + 1) * grid_h, j * grid_w : (j + 1) * grid_w
        #         ]
        #         grids.append(grid)
        # 4分割に特化した実装
        left_top = img[:, :, : img.shape[2] // 2, : img.shape[3] // 2]
        right_top = img[:, :, : img.shape[2] // 2, img.shape[3] // 2 :]
        left_bottom = img[:, :, img.shape[2] // 2 :, : img.shape[3] // 2]
        right_bottom = img[:, :, img.shape[2] // 2 :, img.shape[3] // 2 :]
        return left_top, right_top, left_bottom, right_bottom
        # left_img = img[:, :, :, : img.shape[3] // 2]
        # right_img = img[:, :, :, img.shape[3] // 2 :]
        # return left_img, right_img

    def _get_embedding(self, img: torch.Tensor) -> torch.Tensor:
        emb = self.model(img)
        gemma, beta = self.film(emb)
        emb = gemma * emb + beta
        return emb

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img = input["image"]
        # 画像の左半分をleft_img、右半分をright_imgとして扱う
        left_top, right_top, left_bottom, right_bottom = self._grid_split(img)
        left_top_emb = self._get_embedding(left_top)
        right_top_emb = self._get_embedding(right_top)
        left_bottom_emb = self._get_embedding(left_bottom)
        right_bottom_emb = self._get_embedding(right_bottom)
        emb_sum = left_top_emb + right_top_emb + left_bottom_emb + right_bottom_emb
        # left_emb = self._get_embedding(left_top)
        # right_emb = self._get_embedding(right_top)
        # embの結合方法を選択
        # emb_sum = left_emb + right_emb
        emb = self.target_emb(emb_sum)

        output = self.target_head(emb)
        # gdmとtotalは和から求める
        # gdm = green + clover
        gdm = output[:, 0] + output[:, 2]
        # total = gdm + dead
        total = torch.sum(output, dim=1, keepdim=True)
        output = torch.cat(
            [output[:, 0:1], output[:, 1:2], output[:, 2:3], gdm.unsqueeze(1), total],
            dim=1,
        )
        # clover classification head
        clover_output = self.clover_classification_head(emb)
        height = self.height_head(emb)

        output = {
            "logits": output,
            "include_clover_preds": clover_output,
            "height": height,
        }
        return output


if __name__ == "__main__":
    batch_size = 4
    model = TiledCloverSumHeightFilmModel(
        backbone_name="tf_efficientnet_b0",
        pretrained=True,
        in_channels=3,
        n_classes=5,
    )

    sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}
    output = model(sample_input)
    print(output["logits"].shape)  # Expected output shape: (4, 5)
    print(output["include_clover_preds"].shape)  # Expected output shape: (4, 1)
