import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=64,
        heads=4,
        dropout=0.2
    ):
        super().__init__()

        # =====================
        # Layer 1
        # =====================
        self.conv1 = HeteroConv(
            {
                # -------- forward biological relations --------
                ("circRNA", "interacts", "miRNA"):
                    GATConv(in_channels, hidden_channels, heads=heads,
                            dropout=dropout, add_self_loops=False),

                ("miRNA", "interacts", "disease"):
                    GATConv(in_channels, hidden_channels, heads=heads,
                            dropout=dropout, add_self_loops=False),

                ("circRNA", "associated", "disease"):
                    GATConv(in_channels, hidden_channels, heads=heads,
                            dropout=dropout, add_self_loops=False),

                # -------- reverse biological relations --------
                ("miRNA", "rev_interacts", "circRNA"):
                    GATConv(in_channels, hidden_channels, heads=heads,
                            dropout=dropout, add_self_loops=False),

                ("disease", "rev_interacts", "miRNA"):
                    GATConv(in_channels, hidden_channels, heads=heads,
                            dropout=dropout, add_self_loops=False),

                ("disease", "rev_associated", "circRNA"):
                    GATConv(in_channels, hidden_channels, heads=heads,
                            dropout=dropout, add_self_loops=False),
                

            },
            aggr="mean"
        )

        # =====================
        # Layer 2
        # =====================
        self.conv2 = HeteroConv(
            {
                # -------- forward biological relations --------
                ("circRNA", "interacts", "miRNA"):
                    GATConv(hidden_channels * heads, out_channels,
                            heads=1, concat=False, add_self_loops=False),

                ("miRNA", "interacts", "disease"):
                    GATConv(hidden_channels * heads, out_channels,
                            heads=1, concat=False, add_self_loops=False),

                ("circRNA", "associated", "disease"):
                    GATConv(hidden_channels * heads, out_channels,
                            heads=1, concat=False, add_self_loops=False),

                # -------- reverse biological relations --------
                ("miRNA", "rev_interacts", "circRNA"):
                    GATConv(hidden_channels * heads, out_channels,
                            heads=1, concat=False, add_self_loops=False),

                ("disease", "rev_interacts", "miRNA"):
                    GATConv(hidden_channels * heads, out_channels,
                            heads=1, concat=False, add_self_loops=False),

                ("disease", "rev_associated", "circRNA"):
                    GATConv(hidden_channels * heads, out_channels,
                            heads=1, concat=False, add_self_loops=False),
                
            },
            aggr="mean"
        )


        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    # =====================
    # Forward
    # =====================
    def forward(self, x_dict, edge_index_dict):

        x0 = x_dict  # save input for residual

        # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {
            k: self.dropout(self.act(v))
            for k, v in x_dict.items()
        }

        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)

        # Normalize for dot-product link prediction 
        x_dict = { k: F.normalize(v, p=2, dim=1) 
                   for k, v in x_dict.items() }


        return x_dict
