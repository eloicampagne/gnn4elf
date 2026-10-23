"""Extract the latent spaces of the WEAVE-500 shared MLP and SAGE GNN.

This reuses the exact full-panel cell of ``eval_metrics.py``: 481 forecast
origins, the last fifth held out for validation, 145 test origins, hidden width
64, early stopping, and seed zero.  The archive contains one matched row for
each (test origin, feeder) pair.
"""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA, CACHE  # noqa: E402  (puts every experiment folder on sys.path)
HERE = Path(ROOT)

import eval_metrics as M  # noqa: E402
import ltsf_nodes as E  # noqa: E402

WINDOW = 481
SEED = 0
OUTPUT = HERE / "results_ltsf_nodes" / "weave500_latent_mlp_gnn.npz"


def fitted_model(kind, x, y, n_tr, n_ev, edge_index):
    """Fit one of the two models under the diagnostic's unchanged protocol."""
    torch.manual_seed(SEED)
    if kind == "shared":
        model = E.Shared(x.shape[2], E.HIDDEN, out=y.shape[2])
    elif kind == "sage":
        model = E.Graph(
            x.shape[2], E.HIDDEN, "SAGEConv", edge_index, None,
            out=y.shape[2],
        )
    else:
        raise ValueError(kind)
    E.fit(model, x, y, n_tr, n_ev, SEED)
    return model.eval()


def shared_latent(model, x_test):
    with torch.no_grad():
        return torch.relu(model.net[0](x_test)).cpu().numpy()


def sage_latent(model, x_test):
    chunks = []

    def capture(_module, inputs):
        chunks.append(inputs[0].detach().cpu())

    handle = model.gnn.fc.register_forward_pre_hook(capture)
    with torch.no_grad():
        model(x_test)
    handle.remove()
    return torch.stack(chunks).numpy()


def main():
    x, y, _my, _sy, n_tr, n_ev, edge_index = M.cell(
        "weave", WINDOW, SEED
    )
    mlp = fitted_model("shared", x, y, n_tr, n_ev, edge_index)
    gnn = fitted_model("sage", x, y, n_tr, n_ev, edge_index)
    x_test = x[n_ev:]
    z_mlp = shared_latent(mlp, x_test)
    z_gnn = sage_latent(gnn, x_test)
    if z_mlp.shape != z_gnn.shape:
        raise RuntimeError(f"Latent shapes differ: {z_mlp.shape} vs {z_gnn.shape}")

    n_times, n_nodes, width = z_mlp.shape
    node = np.tile(np.arange(n_nodes), n_times)
    origin = np.repeat(np.arange(n_times), n_nodes)
    period = np.minimum(3, 4 * origin.copy() // n_times).astype(np.int16)
    adjacency = M.W.panel()["A"].astype(np.float32)
    np.savez_compressed(
        OUTPUT,
        shared_mlp=z_mlp.reshape(-1, width),
        sage_gnn=z_gnn.reshape(-1, width),
        node=node,
        origin=origin,
        period=period,
        adjacency=adjacency,
    )
    print(f"{n_times * n_nodes} matched representations -> {OUTPUT}")


if __name__ == "__main__":
    main()
